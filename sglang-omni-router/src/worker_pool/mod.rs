mod health;
mod permit;
pub(crate) mod profile;
mod resolver;
mod selection;

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, RwLock};

use tokio::sync::{Notify, Semaphore};

use crate::config::{Config, RoutingStrategy};

pub(crate) use health::{HealthState, HealthSupervisor, HealthTaskError};
pub(crate) use permit::{
    AdmissionError, AdmissionLease, DispatchError, EnvelopeLease, RequestLease,
};
pub(crate) use profile::{
    BatchFeature, CapacityClass, ChatAudioFormat, MediaPlacement, MediaProfile, MessageContentForm,
    ModelSelection, ProfileRequirement, ReferenceForm, RouteRequirement, ServiceClass,
    SpeechResponseFormat, SpeechTask, SpeechToTextTask, StreamMode, TranscriptionResponseFormat,
    TrustDomain,
};
pub(crate) use resolver::ResolvedTarget;

use health::HealthCell;
use permit::{AdmissionController, Gate};
use profile::{MAX_WORKERS, RegistrationId, ServiceProfile, WorkerCapacityConfig, WorkerId};
use resolver::{StaticResolver, build_health_client, build_http_client};
use selection::Selector;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum Disposition {
    Serving = 0,
    Draining = 1,
}

struct CapacitySlot {
    limit: usize,
    semaphore: Arc<Semaphore>,
}

/// One immutable startup registration plus independent health, disposition,
/// and exact route-capacity owners.
pub(super) struct WorkerRecord {
    worker_id: WorkerId,
    default_model_id: Option<String>,
    registration_id: RegistrationId,
    target: ResolvedTarget,
    trust_domain: TrustDomain,
    profiles: Vec<ServiceProfile>,
    capacity: [Option<CapacitySlot>; 4],
    health: HealthCell,
    disposition: AtomicU8,
    immediate_probe: Notify,
}

impl WorkerRecord {
    fn disposition(&self) -> Disposition {
        if self.disposition.load(Ordering::Acquire) == Disposition::Serving as u8 {
            Disposition::Serving
        } else {
            Disposition::Draining
        }
    }

    fn mark_draining(&self) {
        self.disposition
            .store(Disposition::Draining as u8, Ordering::Release);
    }

    fn slot(&self, class: CapacityClass) -> Option<&CapacitySlot> {
        self.capacity.get(class.index()).and_then(Option::as_ref)
    }

    fn occupancy_snapshot(&self, class: CapacityClass) -> Option<usize> {
        self.slot(class)
            .map(|slot| slot.limit - slot.semaphore.available_permits())
    }

    fn has_profile(&self, requirement: &RouteRequirement) -> bool {
        self.profiles
            .iter()
            .any(|profile| profile.matches(&requirement.profile, self.default_model_id.as_deref()))
    }

    fn available_for_dispatch(&self) -> bool {
        self.health.load() == HealthState::Healthy && self.disposition() == Disposition::Serving
    }
}

/// Immutable worker pool with bounded admission, exact capacity,
/// deterministic policy state, and independently owned health.
pub(crate) struct WorkerPool {
    records: Vec<Arc<WorkerRecord>>,
    gate: Arc<RwLock<Gate>>,
    admission: AdmissionController,
    selector: Selector,
    homogeneous_generation_http: Vec<HomogeneousGenerationCohort>,
    homogeneous_media_http: Vec<HomogeneousMediaCohort>,
    health_client: reqwest::Client,
    generation_client: Option<reqwest::Client>,
    media_client: Option<reqwest::Client>,
}

struct HomogeneousGenerationCohort {
    trust_domain: TrustDomain,
}

struct HomogeneousMediaCohort {
    trust_domain: TrustDomain,
    service: ServiceClass,
    task: Option<SpeechToTextTask>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DefaultModelResolution<'a> {
    NoService,
    Unique(&'a str),
    Ambiguous,
}

/// Startup proof that chat body inspection cannot change the route cohort.
pub(crate) struct ContentBlindGenerationHttp<'a> {
    pool: &'a WorkerPool,
    trust: &'a TrustDomain,
}

pub(crate) struct ContentBlindMediaHttp<'a> {
    pool: &'a WorkerPool,
    cohort: &'a HomogeneousMediaCohort,
}

impl WorkerPool {
    pub(crate) fn build(config: &Config) -> Result<Self, crate::error::RouterError> {
        let targets: Vec<_> = config
            .workers
            .iter()
            .map(ResolvedTarget::from_worker)
            .collect::<Option<_>>()
            .ok_or(crate::error::RouterError::WorkerPoolInvariant)?;
        let resolver = Arc::new(
            StaticResolver::from_targets(&targets)
                .ok_or(crate::error::RouterError::WorkerPoolInvariant)?,
        );
        let health_client = build_health_client(
            Arc::clone(&resolver),
            config.health.timeout(),
            config.health.interval(),
        )
        .map_err(crate::error::RouterError::HealthClient)?;
        let generation_client = config
            .http_generation
            .as_ref()
            .map(|generation| {
                build_http_client(
                    Arc::clone(&resolver),
                    generation.connect_timeout(),
                    generation.pool_idle_timeout(),
                    generation.pool_max_idle_per_host,
                )
            })
            .transpose()
            .map_err(crate::error::RouterError::GenerationClient)?;
        let media_client = config
            .http_media
            .as_ref()
            .map(|media| {
                build_http_client(
                    Arc::clone(&resolver),
                    media.connect_timeout(),
                    media.pool_idle_timeout(),
                    media.pool_max_idle_per_host,
                )
            })
            .transpose()
            .map_err(crate::error::RouterError::MediaClient)?;
        let gate = Arc::new(RwLock::new(Gate::open()));
        let admission_limit = |limit: Option<u32>| {
            limit
                .map(usize::try_from)
                .transpose()
                .map_err(|_| crate::error::RouterError::WorkerPoolInvariant)
        };
        let admission = AdmissionController::new(
            Arc::clone(&gate),
            usize::try_from(config.admission.global)
                .map_err(|_| crate::error::RouterError::WorkerPoolInvariant)?,
            [
                admission_limit(config.admission.generation_http)?,
                admission_limit(config.admission.speech_http)?,
                admission_limit(config.admission.speech_batch)?,
                admission_limit(config.admission.transcription_http)?,
            ],
        );
        let mut records = Vec::with_capacity(config.workers.len());
        for (ordinal, (worker, target)) in config.workers.iter().zip(targets).enumerate() {
            records.push(Arc::new(WorkerRecord {
                worker_id: WorkerId::new(worker.worker_id.clone()),
                default_model_id: worker.default_model_id.clone(),
                registration_id: RegistrationId::from_startup_ordinal(ordinal),
                target,
                trust_domain: TrustDomain::new(worker.trust_domain.clone()),
                profiles: worker.service_profiles.clone(),
                capacity: build_capacity(&worker.capacity)?,
                health: HealthCell::unknown(),
                disposition: AtomicU8::new(Disposition::Serving as u8),
                immediate_probe: Notify::new(),
            }));
        }
        let homogeneous_generation_http = build_content_blind_generation_cohorts(&records);
        let homogeneous_media_http = build_content_blind_media_cohorts(&records);
        Ok(Self {
            records,
            gate,
            admission,
            selector: Selector::new(config.router.strategy),
            homogeneous_generation_http,
            homogeneous_media_http,
            health_client,
            generation_client,
            media_client,
        })
    }

    pub(crate) fn start_health(&self, config: &Config) -> HealthSupervisor {
        HealthSupervisor::start(
            &self.records,
            self.health_client.clone(),
            config.health.interval(),
            config.health.success_threshold(),
            config.health.failure_threshold(),
            config.health.max_concurrent_probes(),
        )
    }

    pub(crate) fn generation_client(&self) -> Option<reqwest::Client> {
        self.generation_client.clone()
    }

    pub(crate) fn media_client(&self) -> Option<reqwest::Client> {
        self.media_client.clone()
    }

    pub(crate) fn try_admit(
        &self,
        class: CapacityClass,
        credits: u32,
    ) -> Result<AdmissionLease, AdmissionError> {
        self.admission.try_admit(class, credits)
    }

    pub(crate) fn try_admit_envelope(&self) -> Result<EnvelopeLease, AdmissionError> {
        self.admission.try_admit_envelope()
    }

    pub(crate) fn try_admit_class(
        &self,
        envelope: EnvelopeLease,
        class: CapacityClass,
        credits: u32,
    ) -> Result<AdmissionLease, AdmissionError> {
        self.admission.try_admit_class(envelope, class, credits)
    }

    pub(crate) fn dispatch(
        &self,
        admission: AdmissionLease,
        requirement: &RouteRequirement,
    ) -> Result<RequestLease, DispatchError> {
        if !requirement.profile.is_well_formed() {
            return Err(DispatchError::NoEligibleProfile);
        }
        if admission.class() != requirement.capacity_class() {
            return Err(DispatchError::Internal);
        }
        let class = admission.class();
        let credits = admission.credits();
        let profile_found = self
            .records
            .iter()
            .any(|record| record.has_profile(requirement));
        self.dispatch_matching(admission, class, credits, profile_found, |record| {
            &record.trust_domain == requirement.trust_domain() && record.has_profile(requirement)
        })
    }

    fn dispatch_matching(
        &self,
        admission: AdmissionLease,
        class: CapacityClass,
        credits: u32,
        profile_found: bool,
        matches: impl Fn(&WorkerRecord) -> bool,
    ) -> Result<RequestLease, DispatchError> {
        if !profile_found {
            return Err(DispatchError::NoEligibleProfile);
        }
        let eligible_count = self
            .records
            .iter()
            .filter(|record| matches(record) && record.available_for_dispatch())
            .count();
        if eligible_count == 0 {
            return Err(DispatchError::Unavailable);
        }
        let policy_guard = self.selector.least_requests_guard();
        let gate = self.gate.read().map_err(|_| DispatchError::Internal)?;
        if !gate.open {
            return Err(DispatchError::Draining);
        }
        let selected = match self.selector.strategy() {
            RoutingStrategy::RoundRobin => {
                let start = self.selector.start(eligible_count);
                self.reserve_round_robin(start, class, credits, &matches)
            }
            RoutingStrategy::LeastRequests => {
                let start = self.selector.start(self.records.len());
                self.reserve_least_requests(start, class, credits, &matches)
            }
        };
        drop(gate);
        drop(policy_guard);
        match selected {
            Some((record, exact)) => Ok(RequestLease::new(admission, exact, record)),
            None if self
                .records
                .iter()
                .any(|record| matches(record) && record.available_for_dispatch()) =>
            {
                Err(DispatchError::Overloaded)
            }
            None => Err(DispatchError::Unavailable),
        }
    }

    fn reserve_round_robin(
        &self,
        start: usize,
        class: CapacityClass,
        credits: u32,
        matches: &impl Fn(&WorkerRecord) -> bool,
    ) -> Option<(Arc<WorkerRecord>, tokio::sync::OwnedSemaphorePermit)> {
        for pass in 0..2 {
            let mut eligible_ordinal = 0;
            for record in &self.records {
                if !matches(record) || !record.available_for_dispatch() {
                    continue;
                }
                let in_pass = if pass == 0 {
                    eligible_ordinal >= start
                } else {
                    eligible_ordinal < start
                };
                eligible_ordinal += 1;
                if in_pass
                    && let Some(slot) = record.slot(class)
                    && let Ok(exact) = Arc::clone(&slot.semaphore).try_acquire_many_owned(credits)
                {
                    return Some((Arc::clone(record), exact));
                }
            }
        }
        None
    }

    fn reserve_least_requests(
        &self,
        start: usize,
        class: CapacityClass,
        credits: u32,
        matches: &impl Fn(&WorkerRecord) -> bool,
    ) -> Option<(Arc<WorkerRecord>, tokio::sync::OwnedSemaphorePermit)> {
        let mut snapshots = [usize::MAX; MAX_WORKERS];
        let mut attempted = [false; MAX_WORKERS];
        for (index, record) in self.records.iter().enumerate() {
            if matches(record) && record.available_for_dispatch() {
                *snapshots.get_mut(index)? = record.occupancy_snapshot(class)?;
            }
        }
        for _ in 0..self.records.len() {
            let mut best: Option<(usize, usize, usize)> = None;
            for (index, occupancy) in snapshots
                .iter()
                .copied()
                .enumerate()
                .take(self.records.len())
            {
                if occupancy == usize::MAX || attempted.get(index).is_none_or(|value| *value) {
                    continue;
                }
                let ordinal = self.records.get(index)?.registration_id.startup_ordinal();
                let rank = if ordinal >= start {
                    ordinal - start
                } else {
                    self.records.len() - start + ordinal
                };
                let key = (occupancy, rank);
                if best
                    .is_none_or(|(_, best_occupancy, best_rank)| key < (best_occupancy, best_rank))
                {
                    best = Some((index, occupancy, rank));
                }
            }
            let (index, _, _) = best?;
            *attempted.get_mut(index)? = true;
            let record = self.records.get(index)?;
            if let Some(slot) = record.slot(class)
                && let Ok(exact) = Arc::clone(&slot.semaphore).try_acquire_many_owned(credits)
            {
                return Some((Arc::clone(record), exact));
            }
        }
        None
    }

    pub(crate) fn resolve_default_model_id(
        &self,
        trust: &TrustDomain,
        service: ServiceClass,
        task: Option<SpeechToTextTask>,
    ) -> DefaultModelResolution<'_> {
        let mut resolved = None;
        for record in &self.records {
            if &record.trust_domain != trust
                || !record.profiles.iter().any(|profile| {
                    profile.service_class() == service
                        && match (profile, task) {
                            (
                                ServiceProfile::TranscriptionHttp { task: row_task, .. },
                                Some(required),
                            ) => *row_task == required,
                            (ServiceProfile::TranscriptionHttp { .. }, None) => false,
                            (_, None) => true,
                            (_, Some(_)) => false,
                        }
                })
            {
                continue;
            }
            let Some(default) = record.default_model_id.as_deref() else {
                return DefaultModelResolution::Ambiguous;
            };
            match resolved {
                None => resolved = Some(default),
                Some(current) if current == default => {}
                Some(_) => return DefaultModelResolution::Ambiguous,
            }
        }
        resolved.map_or(
            DefaultModelResolution::NoService,
            DefaultModelResolution::Unique,
        )
    }

    pub(crate) fn content_blind_generation_http(
        &self,
        trust: &TrustDomain,
    ) -> Option<ContentBlindGenerationHttp<'_>> {
        self.homogeneous_generation_http
            .iter()
            .find(|cohort| &cohort.trust_domain == trust)
            .map(|cohort| ContentBlindGenerationHttp {
                pool: self,
                trust: &cohort.trust_domain,
            })
    }

    pub(crate) fn content_blind_media_http(
        &self,
        trust: &TrustDomain,
        route: crate::config::HttpMediaRoute,
    ) -> Option<ContentBlindMediaHttp<'_>> {
        self.homogeneous_media_http
            .iter()
            .find(|cohort| {
                &cohort.trust_domain == trust
                    && cohort.service == route.service_class()
                    && cohort.task == route.speech_to_text_task()
            })
            .map(|cohort| ContentBlindMediaHttp { pool: self, cohort })
    }

    pub(crate) fn generation_http_ready(&self, trust: &TrustDomain) -> bool {
        self.gate.read().is_ok_and(|gate| {
            gate.open
                && self.records.iter().any(|record| {
                    &record.trust_domain == trust
                        && record.available_for_dispatch()
                        && record
                            .profiles
                            .iter()
                            .any(|profile| profile.service_class() == ServiceClass::GenerationHttp)
                })
        })
    }

    pub(crate) fn media_http_ready(
        &self,
        trust: &TrustDomain,
        routes: &[crate::config::HttpMediaRoute],
    ) -> bool {
        self.gate.read().is_ok_and(|gate| {
            gate.open
                && routes.iter().all(|route| {
                    self.records.iter().any(|record| {
                        &record.trust_domain == trust
                            && record.available_for_dispatch()
                            && record
                                .profiles
                                .iter()
                                .any(|profile| route.matches_profile(profile))
                    })
                })
        })
    }

    pub(crate) fn drain(&self) -> Result<(), DispatchError> {
        let mut gate = self.gate.write().map_err(|_| DispatchError::Internal)?;
        if !gate.open {
            return Ok(());
        }
        gate.open = false;
        self.admission.close();
        for record in &self.records {
            record.mark_draining();
            for slot in record.capacity.iter().flatten() {
                slot.semaphore.close();
            }
        }
        Ok(())
    }
}

impl ContentBlindGenerationHttp<'_> {
    pub(crate) fn dispatch(self, admission: AdmissionLease) -> Result<RequestLease, DispatchError> {
        self.pool.dispatch_matching(
            admission,
            CapacityClass::GenerationHttp,
            1,
            true,
            |record| {
                &record.trust_domain == self.trust
                    && record
                        .profiles
                        .iter()
                        .any(|profile| profile.service_class() == ServiceClass::GenerationHttp)
            },
        )
    }
}

impl ContentBlindMediaHttp<'_> {
    pub(crate) fn dispatch(self, admission: AdmissionLease) -> Result<RequestLease, DispatchError> {
        let class = self.cohort.service.capacity();
        self.pool
            .dispatch_matching(admission, class, 1, true, |record| {
                record.trust_domain == self.cohort.trust_domain
                    && record.profiles.iter().any(|profile| {
                        profile.service_class() == self.cohort.service
                            && match (profile, self.cohort.task) {
                                (
                                    ServiceProfile::TranscriptionHttp { task, .. },
                                    Some(required),
                                ) => *task == required,
                                (ServiceProfile::TranscriptionHttp { .. }, None) => false,
                                (_, None) => true,
                                (_, Some(_)) => false,
                            }
                    })
            })
    }
}

fn build_content_blind_generation_cohorts(
    records: &[Arc<WorkerRecord>],
) -> Vec<HomogeneousGenerationCohort> {
    let mut result = Vec::new();
    for record in records {
        if result
            .iter()
            .any(|cohort: &HomogeneousGenerationCohort| cohort.trust_domain == record.trust_domain)
        {
            continue;
        }
        let mut members = records.iter().filter(|candidate| {
            candidate.trust_domain == record.trust_domain
                && candidate
                    .profiles
                    .iter()
                    .any(|profile| profile.service_class() == ServiceClass::GenerationHttp)
        });
        let Some(first) = members.next() else {
            continue;
        };
        if members.all(|candidate| {
            candidate.default_model_id == first.default_model_id
                && generation_rows_equal(&candidate.profiles, &first.profiles)
        }) {
            result.push(HomogeneousGenerationCohort {
                trust_domain: record.trust_domain.clone(),
            });
        }
    }
    result
}

fn build_content_blind_media_cohorts(records: &[Arc<WorkerRecord>]) -> Vec<HomogeneousMediaCohort> {
    let mut result = Vec::new();
    for record in records {
        for profile in &record.profiles {
            let service = profile.service_class();
            if service == ServiceClass::GenerationHttp || service == ServiceClass::SpeechBatch {
                continue;
            }
            let task = match profile {
                ServiceProfile::TranscriptionHttp { task, .. } => Some(*task),
                _ => None,
            };
            if result.iter().any(|cohort: &HomogeneousMediaCohort| {
                cohort.trust_domain == record.trust_domain
                    && cohort.service == service
                    && cohort.task == task
            }) {
                continue;
            }
            let members: Vec<_> = records
                .iter()
                .filter(|candidate| {
                    candidate.trust_domain == record.trust_domain
                        && candidate.profiles.iter().any(|row| {
                            row.service_class() == service
                                && match (row, task) {
                                    (
                                        ServiceProfile::TranscriptionHttp {
                                            task: row_task, ..
                                        },
                                        Some(required),
                                    ) => *row_task == required,
                                    (ServiceProfile::TranscriptionHttp { .. }, None) => false,
                                    (_, None) => true,
                                    (_, Some(_)) => false,
                                }
                        })
                })
                .collect();
            let Some(first) = members.first() else {
                continue;
            };
            if first.default_model_id.is_some()
                && members.iter().all(|candidate| {
                    candidate.default_model_id == first.default_model_id
                        && service_rows_equal(&candidate.profiles, &first.profiles, service, task)
                })
            {
                result.push(HomogeneousMediaCohort {
                    trust_domain: record.trust_domain.clone(),
                    service,
                    task,
                });
            }
        }
    }
    result
}

fn service_rows_equal(
    left: &[ServiceProfile],
    right: &[ServiceProfile],
    service: ServiceClass,
    task: Option<SpeechToTextTask>,
) -> bool {
    let relevant = |profile: &&ServiceProfile| {
        profile.service_class() == service
            && match (*profile, task) {
                (ServiceProfile::TranscriptionHttp { task: row_task, .. }, Some(required)) => {
                    *row_task == required
                }
                (ServiceProfile::TranscriptionHttp { .. }, None) => false,
                (_, None) => true,
                (_, Some(_)) => false,
            }
    };
    left.iter().filter(relevant).count() == right.iter().filter(relevant).count()
        && left.iter().filter(relevant).all(|profile| {
            right
                .iter()
                .filter(relevant)
                .any(|other| profile.semantically_eq(other))
        })
}

fn generation_rows_equal(left: &[ServiceProfile], right: &[ServiceProfile]) -> bool {
    service_rows_equal(left, right, ServiceClass::GenerationHttp, None)
}

fn build_capacity(
    config: &WorkerCapacityConfig,
) -> Result<[Option<CapacitySlot>; 4], crate::error::RouterError> {
    let values = [
        config.generation_http,
        config.speech_http,
        config.speech_batch,
        config.transcription_http,
    ];
    let mut result: [Option<CapacitySlot>; 4] = std::array::from_fn(|_| None);
    for (index, value) in values.into_iter().enumerate() {
        if let Some(value) = value {
            let limit = usize::try_from(value)
                .map_err(|_| crate::error::RouterError::WorkerPoolInvariant)?;
            result[index] = Some(CapacitySlot {
                limit,
                semaphore: Arc::new(Semaphore::new(limit)),
            });
        }
    }
    Ok(result)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::{Arc, Barrier, RwLock};
    use std::thread;

    use super::profile::{
        InputModality, MessageContentForm, ModelSelection, OutputModality, ProfileRequirement,
        ServiceProfile, StreamMode,
    };
    use super::*;

    fn profile(model: &str) -> ServiceProfile {
        ServiceProfile::GenerationHttp {
            model_ids: vec![model.to_owned()],
            message_content_forms: vec![MessageContentForm::String],
            media_placements: Vec::new(),
            input_modalities: vec![InputModality::Text],
            output_modalities: vec![OutputModality::Text],
            chat_audio_formats: Vec::new(),
            stream_modes: vec![StreamMode::NonStreaming],
        }
    }

    fn requirement(model: &str, trust: &str) -> RouteRequirement {
        RouteRequirement::new(
            ProfileRequirement::GenerationHttp {
                model: ModelSelection::Explicit(model.to_owned()),
                message_content_forms: vec![MessageContentForm::String],
                media_placements: Vec::new(),
                input_modalities: vec![InputModality::Text],
                output_modalities: vec![OutputModality::Text],
                audio_format: None,
                stream_mode: StreamMode::NonStreaming,
            },
            TrustDomain::new(trust.to_owned()),
        )
    }

    fn record_with_profile(
        ordinal: usize,
        trust: &str,
        model: &str,
        limit: usize,
        service_profile: ServiceProfile,
    ) -> Arc<WorkerRecord> {
        let health = HealthCell::unknown();
        health.store(HealthState::Healthy);
        Arc::new(WorkerRecord {
            worker_id: WorkerId::new(format!("worker-{ordinal}")),
            default_model_id: Some(model.to_owned()),
            registration_id: RegistrationId::from_startup_ordinal(ordinal),
            target: ResolvedTarget::from_parts(
                &format!("http://127.0.0.1:{}/", 10_000 + ordinal),
                "/health",
                None,
            )
            .expect("test target"),
            trust_domain: TrustDomain::new(trust.to_owned()),
            profiles: vec![service_profile],
            capacity: [
                Some(CapacitySlot {
                    limit,
                    semaphore: Arc::new(Semaphore::new(limit)),
                }),
                None,
                None,
                None,
            ],
            health,
            disposition: AtomicU8::new(Disposition::Serving as u8),
            immediate_probe: Notify::new(),
        })
    }

    fn record(ordinal: usize, trust: &str, model: &str, limit: usize) -> Arc<WorkerRecord> {
        record_with_profile(ordinal, trust, model, limit, profile(model))
    }

    fn pool(
        strategy: RoutingStrategy,
        records: Vec<Arc<WorkerRecord>>,
        admission: usize,
    ) -> WorkerPool {
        let gate = Arc::new(RwLock::new(Gate::open()));
        let targets: Vec<_> = records.iter().map(|record| record.target.clone()).collect();
        let resolver = Arc::new(StaticResolver::from_targets(&targets).expect("test resolver"));
        let client = build_health_client(
            resolver,
            std::time::Duration::from_secs(1),
            std::time::Duration::from_secs(1),
        )
        .expect("test client");
        WorkerPool {
            homogeneous_generation_http: build_content_blind_generation_cohorts(&records),
            homogeneous_media_http: build_content_blind_media_cohorts(&records),
            records,
            gate: Arc::clone(&gate),
            admission: AdmissionController::new(
                gate,
                admission,
                [Some(admission), None, None, None],
            ),
            selector: Selector::new(strategy),
            health_client: client.clone(),
            generation_client: Some(client),
            media_client: None,
        }
    }

    #[test]
    fn direct_proof_requires_equal_defaults_profiles_and_trust_scopes() {
        let local = TrustDomain::new(String::from("local"));
        let sole = pool(
            RoutingStrategy::RoundRobin,
            vec![record(0, "local", "omni", 1)],
            4,
        );
        assert!(sole.content_blind_generation_http(&local).is_some());

        let replicas = pool(
            RoutingStrategy::RoundRobin,
            vec![record(0, "local", "omni", 1), record(1, "local", "omni", 2)],
            4,
        );
        assert!(replicas.content_blind_generation_http(&local).is_some());

        let defaults_differ = pool(
            RoutingStrategy::RoundRobin,
            vec![
                record(0, "local", "omni", 1),
                record(1, "local", "other", 1),
            ],
            4,
        );
        assert!(
            defaults_differ
                .content_blind_generation_http(&local)
                .is_none()
        );

        let mutations: [fn(&mut ServiceProfile); 6] = [
            |profile| {
                if let ServiceProfile::GenerationHttp { model_ids, .. } = profile {
                    model_ids.push(String::from("other"));
                }
            },
            |profile| {
                if let ServiceProfile::GenerationHttp {
                    message_content_forms,
                    ..
                } = profile
                {
                    message_content_forms.push(MessageContentForm::TypedParts);
                }
            },
            |profile| {
                if let ServiceProfile::GenerationHttp {
                    media_placements, ..
                } = profile
                {
                    media_placements.push(MediaPlacement::TypedParts);
                }
            },
            |profile| {
                if let ServiceProfile::GenerationHttp {
                    input_modalities, ..
                } = profile
                {
                    input_modalities.push(InputModality::Image);
                }
            },
            |profile| {
                if let ServiceProfile::GenerationHttp {
                    output_modalities,
                    chat_audio_formats,
                    ..
                } = profile
                {
                    output_modalities.push(OutputModality::Audio);
                    chat_audio_formats.push(ChatAudioFormat::Wav);
                }
            },
            |profile| {
                if let ServiceProfile::GenerationHttp { stream_modes, .. } = profile {
                    stream_modes.push(StreamMode::Streaming);
                }
            },
        ];
        for mutate in mutations {
            let mut different = profile("omni");
            mutate(&mut different);
            let heterogeneous = pool(
                RoutingStrategy::RoundRobin,
                vec![
                    record(0, "local", "omni", 1),
                    record_with_profile(1, "local", "omni", 1, different),
                ],
                4,
            );
            assert!(
                heterogeneous
                    .content_blind_generation_http(&local)
                    .is_none()
            );
        }

        let mut extra_row = record(1, "local", "omni", 1);
        Arc::get_mut(&mut extra_row)
            .expect("new test record is uniquely owned")
            .profiles
            .push(profile("other"));
        let row_count_differs = pool(
            RoutingStrategy::RoundRobin,
            vec![record(0, "local", "omni", 1), extra_row],
            4,
        );
        assert!(
            row_count_differs
                .content_blind_generation_http(&local)
                .is_none()
        );

        let separate = pool(
            RoutingStrategy::RoundRobin,
            vec![
                record(0, "local", "omni", 1),
                record(1, "remote", "other", 1),
            ],
            4,
        );
        assert!(separate.content_blind_generation_http(&local).is_some());
    }

    #[test]
    fn round_robin_balances_and_skips_full_unhealthy_and_draining_workers() {
        let records = vec![record(0, "local", "omni", 1), record(1, "local", "omni", 1)];
        let pool = pool(RoutingStrategy::RoundRobin, records.clone(), 8);
        let first = pool
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp, 1)
                    .expect("admit first"),
                &requirement("omni", "local"),
            )
            .expect("first dispatch");
        let second = pool
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp, 1)
                    .expect("admit second"),
                &requirement("omni", "local"),
            )
            .expect("second dispatch");
        assert_ne!(first.registration_ordinal(), second.registration_ordinal());
        drop(first);
        drop(second);
        records[0].health.store(HealthState::Unhealthy);
        records[1].mark_draining();
        let unavailable = pool.dispatch(
            pool.try_admit(CapacityClass::GenerationHttp, 1)
                .expect("admit unavailable"),
            &requirement("omni", "local"),
        );
        assert!(matches!(unavailable, Err(DispatchError::Unavailable)));
    }

    #[test]
    fn round_robin_rotates_over_sparse_eligible_workers_without_bias() {
        let records = vec![
            record(0, "local", "omni", 1),
            record(1, "local", "other", 1),
            record(2, "local", "omni", 1),
        ];
        let pool = pool(RoutingStrategy::RoundRobin, records, 8);
        let mut selected = Vec::new();
        for _ in 0..6 {
            let lease = pool
                .dispatch(
                    pool.try_admit(CapacityClass::GenerationHttp, 1)
                        .expect("admit sparse round robin"),
                    &requirement("omni", "local"),
                )
                .expect("dispatch sparse round robin");
            selected.push(lease.registration_ordinal());
            drop(lease);
        }
        assert_eq!(selected, [0, 2, 0, 2, 0, 2]);
    }

    #[test]
    fn least_requests_choose_and_reserve_is_linearized() {
        const REQUESTS: usize = 32;
        let records = vec![
            record(0, "local", "omni", REQUESTS),
            record(1, "local", "omni", REQUESTS),
        ];
        let pool = Arc::new(pool(RoutingStrategy::LeastRequests, records, REQUESTS));
        let start = Arc::new(Barrier::new(REQUESTS + 1));
        let mut threads = Vec::new();
        for _ in 0..REQUESTS {
            let pool = Arc::clone(&pool);
            let start = Arc::clone(&start);
            threads.push(thread::spawn(move || {
                let admission = pool
                    .try_admit(CapacityClass::GenerationHttp, 1)
                    .expect("concurrent admission");
                start.wait();
                pool.dispatch(admission, &requirement("omni", "local"))
                    .expect("concurrent dispatch")
            }));
        }
        start.wait();
        let leases: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().expect("join dispatcher"))
            .collect();
        let first = leases
            .iter()
            .filter(|lease| lease.registration_ordinal() == 0)
            .count();
        assert_eq!(first, REQUESTS / 2);
    }

    #[test]
    fn heterogeneous_default_model_routes_to_the_correlated_capable_worker() {
        let mut multimodal = profile("omni");
        if let ServiceProfile::GenerationHttp {
            message_content_forms,
            media_placements,
            input_modalities,
            ..
        } = &mut multimodal
        {
            message_content_forms.push(MessageContentForm::TypedParts);
            media_placements.push(MediaPlacement::TypedParts);
            input_modalities.push(InputModality::Image);
        }
        let pool = pool(
            RoutingStrategy::RoundRobin,
            vec![
                record(0, "local", "omni", 1),
                record_with_profile(1, "local", "omni", 1, multimodal),
            ],
            2,
        );
        let requirement = RouteRequirement::new(
            ProfileRequirement::GenerationHttp {
                model: ModelSelection::WorkerDefault {
                    expected_model_id: String::from("omni"),
                },
                message_content_forms: vec![MessageContentForm::TypedParts],
                media_placements: vec![MediaPlacement::TypedParts],
                input_modalities: vec![InputModality::Text, InputModality::Image],
                output_modalities: vec![OutputModality::Text],
                audio_format: None,
                stream_mode: StreamMode::NonStreaming,
            },
            TrustDomain::new(String::from("local")),
        );
        let lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp, 1)
                    .expect("admit heterogeneous request"),
                &requirement,
            )
            .expect("dispatch heterogeneous default");
        assert_eq!(lease.registration_ordinal(), 1);
    }

    #[test]
    fn exact_class_and_global_permits_release_on_every_drop() {
        let pool = pool(
            RoutingStrategy::RoundRobin,
            vec![record(0, "local", "omni", 1)],
            1,
        );
        let lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp, 1)
                    .expect("admit"),
                &requirement("omni", "local"),
            )
            .expect("dispatch");
        assert_eq!(pool.admission.available(), (0, [Some(0), None, None, None]));
        assert_eq!(
            pool.records[0]
                .slot(CapacityClass::GenerationHttp)
                .expect("generation slot")
                .semaphore
                .available_permits(),
            0
        );
        drop(lease);
        assert_eq!(pool.admission.available(), (1, [Some(1), None, None, None]));
        assert_eq!(
            pool.records[0]
                .slot(CapacityClass::GenerationHttp)
                .expect("generation slot")
                .semaphore
                .available_permits(),
            1
        );
    }

    #[test]
    fn readiness_starts_unknown_and_tracks_health_and_disposition() {
        let record = record(0, "local", "omni", 1);
        record.health.store(HealthState::Unknown);
        let pool = pool(RoutingStrategy::RoundRobin, vec![Arc::clone(&record)], 1);
        let trust = TrustDomain::new(String::from("local"));
        assert!(!pool.generation_http_ready(&trust));
        record.health.store(HealthState::Healthy);
        assert!(pool.generation_http_ready(&trust));
        record.mark_draining();
        assert!(!pool.generation_http_ready(&trust));
    }

    fn media_record(
        ordinal: usize,
        class: CapacityClass,
        limit: usize,
        profile: ServiceProfile,
    ) -> Arc<WorkerRecord> {
        let health = HealthCell::unknown();
        health.store(HealthState::Healthy);
        let mut capacity = std::array::from_fn(|_| None);
        capacity[class.index()] = Some(CapacitySlot {
            limit,
            semaphore: Arc::new(Semaphore::new(limit)),
        });
        Arc::new(WorkerRecord {
            worker_id: WorkerId::new(format!("media-{ordinal}")),
            default_model_id: Some(String::from("tts")),
            registration_id: RegistrationId::from_startup_ordinal(ordinal),
            target: ResolvedTarget::from_parts(
                &format!("http://127.0.0.1:{}/", 12_000 + ordinal),
                "/health",
                None,
            )
            .expect("media target"),
            trust_domain: TrustDomain::new(String::from("local")),
            profiles: vec![profile],
            capacity,
            health,
            disposition: AtomicU8::new(Disposition::Serving as u8),
            immediate_probe: Notify::new(),
        })
    }

    fn media_pool(records: Vec<Arc<WorkerRecord>>) -> WorkerPool {
        let gate = Arc::new(RwLock::new(Gate::open()));
        let targets: Vec<_> = records.iter().map(|record| record.target.clone()).collect();
        let resolver = Arc::new(StaticResolver::from_targets(&targets).expect("media resolver"));
        let client = build_health_client(
            resolver,
            std::time::Duration::from_secs(1),
            std::time::Duration::from_secs(1),
        )
        .expect("media client");
        WorkerPool {
            homogeneous_generation_http: build_content_blind_generation_cohorts(&records),
            homogeneous_media_http: build_content_blind_media_cohorts(&records),
            records,
            gate: Arc::clone(&gate),
            admission: AdmissionController::new(gate, 8, [Some(8), Some(8), Some(8), Some(8)]),
            selector: Selector::new(RoutingStrategy::RoundRobin),
            health_client: client.clone(),
            generation_client: None,
            media_client: Some(client),
        }
    }

    fn batch_profile() -> ServiceProfile {
        ServiceProfile::SpeechBatch {
            model_ids: vec![String::from("tts")],
            response_formats: vec![SpeechResponseFormat::Wav],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::None],
            managed_voice: false,
            max_batch_size: 8,
            effective_features: Vec::new(),
        }
    }

    fn speech_profile() -> ServiceProfile {
        ServiceProfile::SpeechHttp {
            model_ids: vec![String::from("tts")],
            response_formats: vec![SpeechResponseFormat::Wav],
            stream_modes: vec![StreamMode::NonStreaming],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::None],
            managed_voice: false,
        }
    }

    #[test]
    fn homogeneous_media_uses_existing_policy_and_skips_unhealthy_or_full_workers() {
        let first = media_record(0, CapacityClass::SpeechHttp, 1, speech_profile());
        let second = media_record(1, CapacityClass::SpeechHttp, 1, speech_profile());
        let pool = media_pool(vec![Arc::clone(&first), Arc::clone(&second)]);
        let trust = TrustDomain::new(String::from("local"));
        let route = crate::config::HttpMediaRoute::Speech;
        let first_lease = pool
            .content_blind_media_http(&trust, route)
            .expect("speech cohort")
            .dispatch(
                pool.try_admit(CapacityClass::SpeechHttp, 1)
                    .expect("first admission"),
            )
            .expect("first dispatch");
        assert_eq!(first_lease.registration_ordinal(), 0);
        let second_lease = pool
            .content_blind_media_http(&trust, route)
            .expect("speech cohort")
            .dispatch(
                pool.try_admit(CapacityClass::SpeechHttp, 1)
                    .expect("second admission"),
            )
            .expect("full-worker fallback");
        assert_eq!(second_lease.registration_ordinal(), 1);
        drop(first_lease);
        first.health.store(HealthState::Unhealthy);
        drop(second_lease);
        let healthy = pool
            .content_blind_media_http(&trust, route)
            .expect("speech cohort")
            .dispatch(
                pool.try_admit(CapacityClass::SpeechHttp, 1)
                    .expect("healthy admission"),
            )
            .expect("unhealthy-worker fallback");
        assert_eq!(healthy.registration_ordinal(), 1);
    }

    #[test]
    fn unrelated_media_only_worker_does_not_change_chat_cohort_or_readiness() {
        let generation = record(0, "local", "omni", 1);
        let media = media_record(1, CapacityClass::SpeechHttp, 1, speech_profile());
        let pool = media_pool(vec![media, generation]);
        let trust = TrustDomain::new(String::from("local"));
        assert!(pool.generation_http_ready(&trust));
        let lease = pool
            .content_blind_generation_http(&trust)
            .expect("generation cohort")
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp, 1)
                    .expect("generation admission"),
            )
            .expect("generation dispatch");
        assert_eq!(lease.registration_ordinal(), 0);
    }

    #[test]
    fn batch_reserves_all_item_credits_atomically_and_releases_once() {
        let record = media_record(0, CapacityClass::SpeechBatch, 4, batch_profile());
        let pool = media_pool(vec![Arc::clone(&record)]);
        let requirement = RouteRequirement::new(
            ProfileRequirement::SpeechBatch {
                models: vec![
                    ModelSelection::Explicit(String::from("tts")),
                    ModelSelection::Explicit(String::from("tts")),
                    ModelSelection::Explicit(String::from("tts")),
                ],
                response_formats: vec![SpeechResponseFormat::Wav],
                tasks: vec![SpeechTask::TextToSpeech],
                reference_forms: vec![ReferenceForm::None],
                managed_voice: false,
                batch_size: 3,
                effective_features: Vec::new(),
            },
            TrustDomain::new(String::from("local")),
        );
        let lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::SpeechBatch, 3)
                    .expect("batch admission"),
                &requirement,
            )
            .expect("batch dispatch");
        assert_eq!(
            record
                .slot(CapacityClass::SpeechBatch)
                .expect("batch slot")
                .semaphore
                .available_permits(),
            1
        );
        drop(lease);
        assert_eq!(
            record
                .slot(CapacityClass::SpeechBatch)
                .expect("batch slot")
                .semaphore
                .available_permits(),
            4
        );
        assert!(
            pool.try_admit(CapacityClass::SpeechBatch, 5).is_ok(),
            "class admission remains independent from smaller worker capacity"
        );
        let oversized = RouteRequirement::new(
            ProfileRequirement::SpeechBatch {
                models: vec![ModelSelection::Explicit(String::from("tts")); 5],
                response_formats: vec![SpeechResponseFormat::Wav],
                tasks: vec![SpeechTask::TextToSpeech],
                reference_forms: vec![ReferenceForm::None],
                managed_voice: false,
                batch_size: 5,
                effective_features: Vec::new(),
            },
            TrustDomain::new(String::from("local")),
        );
        assert!(matches!(
            pool.dispatch(
                pool.try_admit(CapacityClass::SpeechBatch, 5)
                    .expect("oversized class admission"),
                &oversized,
            ),
            Err(DispatchError::Overloaded)
        ));
    }
}
