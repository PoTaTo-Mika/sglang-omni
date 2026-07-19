#![allow(dead_code)] // Branch 2 installs the lease API before route consumers land.

mod health;
mod permit;
pub(crate) mod profile;
mod resolver;
mod selection;

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, RwLock};

use tokio::sync::{Notify, Semaphore};

use crate::config::Config;

pub(crate) use health::{HealthState, HealthSupervisor, HealthTaskError};
#[allow(unused_imports)] // Branch-owned lease API is consumed by later route modules.
pub(crate) use permit::{AdmissionError, AdmissionLease, DispatchError, RequestLease};
#[allow(unused_imports)] // Correlated requirement types are consumed by later route modules.
pub(crate) use profile::{
    BatchFeature, CapacityClass, ChatAudioFormat, MediaPlacement, MediaProfile, MessageContentForm,
    ModelSelection, ProfileRequirement, RealtimeProtocol, ReferenceForm, RegistrationId,
    RouteRequirement, ServiceClass, SpeechResponseFormat, SpeechTask, SpeechWebsocketInput,
    StreamMode, TranscriptionResponseFormat, TrustDomain, WorkerId,
};
#[allow(unused_imports)]
// Exact target is consumed by later route modules through RequestLease.
pub(crate) use resolver::ResolvedTarget;

use health::HealthCell;
use permit::{AdmissionController, Gate, class_index};
use profile::{ServiceProfile, WorkerCapacityConfig};
use resolver::{StaticResolver, build_generation_client, build_health_client};
use selection::{PolicyCandidate, Selector};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum Disposition {
    Serving = 0,
    Draining = 1,
}

impl Disposition {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Serving => "serving",
            Self::Draining => "draining",
        }
    }
}

struct CapacitySlot {
    limit: usize,
    semaphore: Arc<Semaphore>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CapacitySnapshot {
    pub(crate) class: CapacityClass,
    pub(crate) limit: usize,
    pub(crate) in_flight: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AdmissionClass {
    Global,
    Capacity(CapacityClass),
}

impl AdmissionClass {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Capacity(class) => class.label(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AdmissionSnapshot {
    pub(crate) class: AdmissionClass,
    pub(crate) limit: usize,
    pub(crate) in_flight: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WorkerSnapshot {
    pub(crate) worker_id: String,
    pub(crate) registration_ordinal: usize,
    pub(crate) health: HealthState,
    pub(crate) disposition: Disposition,
    pub(crate) capacity: Vec<CapacitySnapshot>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OperationsSnapshot {
    pub(crate) admission: [AdmissionSnapshot; 8],
    pub(crate) workers: Vec<WorkerSnapshot>,
}

/// Immutable manifest identity/profile data plus narrow mutable runtime cells.
///
/// Membership, target, trust, and correlated profiles never change. Health and
/// disposition are independent atomics; exact semaphores alone own capacity.
struct WorkerRecord {
    worker_id: WorkerId,
    default_model_id: Option<String>,
    registration_id: RegistrationId,
    target: ResolvedTarget,
    trust_domain: TrustDomain,
    profiles: Vec<ServiceProfile>,
    capacities: [Option<CapacitySlot>; 7],
    health: HealthCell,
    disposition: AtomicU8,
    immediate_probe: Notify,
}

impl WorkerRecord {
    fn default_model_id(&self) -> Option<&str> {
        self.default_model_id.as_deref()
    }

    fn disposition(&self) -> Disposition {
        if self.disposition.load(Ordering::Acquire) == Disposition::Serving as u8 {
            Disposition::Serving
        } else {
            Disposition::Draining
        }
    }

    fn mark_draining(&self) {
        // Release publishes disposition before the exclusive drain guard is
        // released; concurrent dispatch readers recheck it with Acquire.
        self.disposition
            .store(Disposition::Draining as u8, Ordering::Release);
    }

    fn slot(&self, class: CapacityClass) -> Option<&CapacitySlot> {
        self.capacities[class_index(class)].as_ref()
    }

    fn occupancy_snapshot(&self, class: CapacityClass) -> usize {
        self.slot(class).map_or(usize::MAX, |slot| {
            slot.limit - slot.semaphore.available_permits()
        })
    }

    fn has_profile(&self, requirement: &RouteRequirement) -> bool {
        self.profiles
            .iter()
            .any(|profile| profile.matches(&requirement.profile, self.default_model_id()))
    }

    fn immutable_eligible(
        &self,
        requirement: &RouteRequirement,
        voice_owner: Option<&WorkerId>,
    ) -> bool {
        &self.trust_domain == requirement.trust_domain()
            && (!requirement.requires_voice_owner()
                || voice_owner.is_some_and(|owner| owner == &self.worker_id))
    }

    fn available_for_dispatch(&self) -> bool {
        self.health.load() == HealthState::Healthy && self.disposition() == Disposition::Serving
    }

    fn serves_class(&self, class: ServiceClass, voice_owner: Option<&WorkerId>) -> bool {
        self.health.load() == HealthState::Healthy
            && self.disposition() == Disposition::Serving
            && self.profiles.iter().any(|profile| {
                profile.service_class() == class
                    && (!profile.requires_owner()
                        || voice_owner.is_some_and(|owner| owner == &self.worker_id))
            })
    }
}

/// Immutable static worker pool and all router-owned admission/dispatch state.
pub(crate) struct WorkerPool {
    records: Vec<Arc<WorkerRecord>>,
    required_services: Vec<ServiceClass>,
    voice_owner: Option<WorkerId>,
    gate: Arc<RwLock<Gate>>,
    admission: AdmissionController,
    selector: Selector,
    homogeneous_generation_http: Vec<HomogeneousGenerationCohort>,
    homogeneous_media_http: Vec<HomogeneousMediaCohort>,
    realtime_defaults_unambiguous: bool,
    health_client: reqwest::Client,
    generation_client: Option<reqwest::Client>,
    media_client: Option<reqwest::Client>,
}

struct HomogeneousGenerationCohort {
    trust_domain: TrustDomain,
    records: Vec<Arc<WorkerRecord>>,
}

struct HomogeneousMediaCohort {
    trust_domain: TrustDomain,
    service: ServiceClass,
    records: Vec<Arc<WorkerRecord>>,
}

/// Immutable service/trust-scoped worker-default identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DefaultModelResolution<'a> {
    NoService,
    Unique(&'a str),
    Ambiguous,
}

/// Opaque proof that generation request classification cannot alter eligibility.
pub(crate) struct HomogeneousGenerationHttp<'a> {
    pool: &'a WorkerPool,
    cohort: &'a HomogeneousGenerationCohort,
}

impl WorkerPool {
    pub(crate) fn build(config: &Config) -> Result<Self, crate::error::RouterError> {
        let targets: Vec<_> = config
            .workers
            .iter()
            .map(ResolvedTarget::from_worker)
            .collect::<Option<_>>()
            .ok_or(crate::error::RouterError::WorkerPoolInvariant)?;
        let resolver = StaticResolver::from_targets(&targets)
            .ok_or(crate::error::RouterError::WorkerPoolInvariant)?;
        let resolver = Arc::new(resolver);
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
                build_generation_client(
                    Arc::clone(&resolver),
                    generation.connect_timeout(),
                    generation.pool_idle_timeout(),
                    generation.pool_max_idle_per_host,
                )
            })
            .transpose()
            .map_err(crate::error::RouterError::GenerationClient)?;
        let media_client = (config.http_media.is_some()
            || config.router.voice_owner_worker_id.is_some())
        .then(|| {
            let media = config.http_media.clone().unwrap_or_default();
            build_generation_client(
                Arc::clone(&resolver),
                media.connect_timeout(),
                media.pool_idle_timeout(),
                media.pool_max_idle_per_host,
            )
        })
        .transpose()
        .map_err(crate::error::RouterError::MediaClient)?;
        let gate = Arc::new(RwLock::new(Gate::open()));
        let classes = [
            config.admission.generation_http,
            config.admission.speech_http,
            config.admission.transcription_http,
            config.admission.speech_batch,
            config.admission.speech_websocket,
            config.admission.realtime_websocket,
            config.admission.control,
        ];
        let mut class_limits = [0_usize; 7];
        for (destination, source) in class_limits.iter_mut().zip(classes) {
            *destination = usize::try_from(source)
                .map_err(|_| crate::error::RouterError::WorkerPoolInvariant)?;
        }
        let admission = AdmissionController::new(
            Arc::clone(&gate),
            usize::try_from(config.admission.global)
                .map_err(|_| crate::error::RouterError::WorkerPoolInvariant)?,
            class_limits,
        );
        let mut records = Vec::with_capacity(config.workers.len());
        for (registration_ordinal, (worker, target)) in
            config.workers.iter().zip(targets).enumerate()
        {
            let capacities = build_capacities(&worker.capacity)?;
            records.push(Arc::new(WorkerRecord {
                worker_id: WorkerId::new(worker.worker_id.clone()),
                default_model_id: worker.default_model_id.clone(),
                registration_id: RegistrationId::from_startup_ordinal(registration_ordinal),
                target,
                trust_domain: TrustDomain::new(worker.trust_domain.clone()),
                profiles: worker.service_profiles.clone(),
                capacities,
                health: HealthCell::unknown(),
                disposition: AtomicU8::new(Disposition::Serving as u8),
                immediate_probe: Notify::new(),
            }));
        }
        let homogeneous_generation_http = build_homogeneous_generation_cohorts(&records);
        let homogeneous_media_http = build_homogeneous_media_cohorts(&records);
        let realtime_defaults_unambiguous = compute_realtime_defaults_unambiguous(&records);

        Ok(Self {
            records,
            required_services: config.router.required_services.clone(),
            voice_owner: config
                .router
                .voice_owner_worker_id
                .clone()
                .map(WorkerId::new),
            gate,
            admission,
            selector: Selector::new(config.router.strategy),
            homogeneous_generation_http,
            homogeneous_media_http,
            realtime_defaults_unambiguous,
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

    pub(crate) fn try_admit(&self, class: CapacityClass) -> Result<AdmissionLease, AdmissionError> {
        self.admission.try_admit(class)
    }

    pub(crate) fn dispatch(
        &self,
        admission: AdmissionLease,
        requirement: &RouteRequirement,
    ) -> Result<RequestLease, DispatchError> {
        let class = requirement.capacity_class();
        if admission.class() != class {
            return Err(DispatchError::AdmissionClassMismatch);
        }
        if !requirement.profile.is_well_formed() {
            return self.dispatch_candidates(admission, class, false, Vec::new());
        }
        let (profile_found, candidates) =
            build_policy_candidates(&self.records, requirement, self.voice_owner.as_ref());
        self.dispatch_candidates(admission, class, profile_found, candidates)
    }

    pub(crate) fn voice_state_enabled(&self) -> bool {
        self.voice_owner.is_some()
    }

    pub(crate) fn dispatch_voice_control(
        &self,
        admission: AdmissionLease,
    ) -> Result<RequestLease, DispatchError> {
        let class = CapacityClass::Control;
        if admission.class() != class {
            return Err(DispatchError::AdmissionClassMismatch);
        }
        let owner = self.voice_owner.as_ref().ok_or(DispatchError::Internal)?;
        let record = self
            .records
            .iter()
            .find(|record| &record.worker_id == owner)
            .ok_or(DispatchError::Internal)?;
        if !record
            .profiles
            .iter()
            .any(|profile| profile.service_class() == ServiceClass::VoiceControl)
        {
            return Err(DispatchError::Internal);
        }
        let gate = self.gate.read().map_err(|_| DispatchError::Internal)?;
        if !gate.open {
            return Err(DispatchError::Draining);
        }
        if !record.available_for_dispatch() {
            return Err(DispatchError::Unavailable);
        }
        let slot = record.slot(class).ok_or(DispatchError::Internal)?;
        let exact = Arc::clone(&slot.semaphore)
            .try_acquire_owned()
            .map_err(|_| DispatchError::Overloaded)?;
        let record = Arc::clone(record);
        drop(gate);
        Ok(RequestLease::new(admission, exact, record, class))
    }

    pub(crate) fn voice_owner_ready(&self) -> bool {
        let Some(owner) = self.voice_owner.as_ref() else {
            return true;
        };
        self.gate.read().is_ok_and(|gate| {
            gate.open
                && self.records.iter().any(|record| {
                    &record.worker_id == owner
                        && record.available_for_dispatch()
                        && record
                            .profiles
                            .iter()
                            .any(|profile| profile.service_class() == ServiceClass::VoiceControl)
                })
        })
    }

    fn dispatch_candidates(
        &self,
        admission: AdmissionLease,
        class: CapacityClass,
        profile_found: bool,
        candidates: Vec<PolicyCandidate>,
    ) -> Result<RequestLease, DispatchError> {
        let ordered = self.selector.order(candidates, class, self.records.len());

        let gate = self.gate.read().map_err(|_| DispatchError::Internal)?;
        if !gate.open {
            return Err(DispatchError::Draining);
        }
        if !profile_found {
            return Err(DispatchError::NoEligibleProfile);
        }
        if ordered.is_empty() {
            return Err(DispatchError::Unavailable);
        }
        let mut still_eligible = false;
        for candidate in ordered {
            let record = candidate.record;
            if !record.available_for_dispatch() {
                continue;
            }
            still_eligible = true;
            let Some(slot) = record.slot(class) else {
                return Err(DispatchError::Internal);
            };
            if let Ok(exact) = Arc::clone(&slot.semaphore).try_acquire_owned() {
                drop(gate);
                return Ok(RequestLease::new(admission, exact, record, class));
            }
        }
        drop(gate);
        if still_eligible {
            Err(DispatchError::Overloaded)
        } else {
            Err(DispatchError::Unavailable)
        }
    }

    pub(crate) fn resolve_default_model_id(
        &self,
        trust: &TrustDomain,
        service: ServiceClass,
    ) -> DefaultModelResolution<'_> {
        if service == ServiceClass::VoiceControl {
            return DefaultModelResolution::NoService;
        }
        let mut resolved = None;
        for record in &self.records {
            if &record.trust_domain != trust
                || !record
                    .profiles
                    .iter()
                    .any(|profile| profile.service_class() == service)
            {
                continue;
            }
            let Some(default) = record.default_model_id() else {
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

    pub(crate) fn homogeneous_generation_http(
        &self,
        trust: &TrustDomain,
    ) -> Option<HomogeneousGenerationHttp<'_>> {
        self.homogeneous_generation_http
            .iter()
            .find(|cohort| &cohort.trust_domain == trust)
            .map(|cohort| HomogeneousGenerationHttp { pool: self, cohort })
    }

    pub(crate) fn homogeneous_media_http(
        &self,
        trust: &TrustDomain,
        service: ServiceClass,
    ) -> Option<HomogeneousMediaHttp<'_>> {
        if !matches!(
            service,
            ServiceClass::SpeechHttp | ServiceClass::SpeechBatch | ServiceClass::TranscriptionHttp
        ) {
            return None;
        }
        self.homogeneous_media_http
            .iter()
            .find(|cohort| &cohort.trust_domain == trust && cohort.service == service)
            .map(|cohort| HomogeneousMediaHttp { pool: self, cohort })
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.gate.read().is_ok_and(|gate| {
            gate.open
                && self.required_services.iter().all(|class| {
                    if *class == ServiceClass::RealtimeWebsocket
                        && !self.realtime_defaults_unambiguous
                    {
                        return false;
                    }
                    self.records
                        .iter()
                        .any(|record| record.serves_class(*class, self.voice_owner.as_ref()))
                })
        })
    }

    pub(crate) fn operations_snapshot(
        &self,
    ) -> Result<OperationsSnapshot, crate::error::RouterError> {
        let _gate = self
            .gate
            .read()
            .map_err(|_| crate::error::RouterError::WorkerPoolInvariant)?;
        let raw_admission = self.admission.snapshot();
        let admission = std::array::from_fn(|index| AdmissionSnapshot {
            class: if index == 0 {
                AdmissionClass::Global
            } else {
                AdmissionClass::Capacity(CapacityClass::ALL[index - 1])
            },
            limit: raw_admission[index].0,
            in_flight: raw_admission[index].1,
        });
        let workers = self
            .records
            .iter()
            .map(|record| WorkerSnapshot {
                worker_id: record.worker_id.as_str().to_owned(),
                registration_ordinal: record.registration_id.startup_ordinal(),
                health: record.health.load(),
                disposition: record.disposition(),
                capacity: CapacityClass::ALL
                    .into_iter()
                    .filter_map(|class| {
                        record.slot(class).map(|slot| CapacitySnapshot {
                            class,
                            limit: slot.limit,
                            in_flight: slot.limit - slot.semaphore.available_permits(),
                        })
                    })
                    .collect(),
            })
            .collect();
        Ok(OperationsSnapshot { admission, workers })
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
        enabled_services: &[ServiceClass],
    ) -> bool {
        self.gate.read().is_ok_and(|gate| {
            gate.open
                && enabled_services.iter().all(|service| {
                    self.records.iter().any(|record| {
                        &record.trust_domain == trust
                            && record.available_for_dispatch()
                            && record
                                .profiles
                                .iter()
                                .any(|profile| profile.service_class() == *service)
                    })
                })
        })
    }

    pub(crate) fn service_ready(&self, trust: &TrustDomain, service: ServiceClass) -> bool {
        self.gate.read().is_ok_and(|gate| {
            gate.open
                && self.records.iter().any(|record| {
                    &record.trust_domain == trust
                        && record.available_for_dispatch()
                        && record
                            .profiles
                            .iter()
                            .any(|profile| profile.service_class() == service)
                })
        })
    }

    pub(crate) fn drain(&self) -> Result<(), DispatchError> {
        let mut gate = self.gate.write().map_err(|_| DispatchError::Internal)?;
        if !gate.open {
            return Ok(());
        }
        gate.open = false;
        self.admission.close_semaphores();
        for record in &self.records {
            record.mark_draining();
            for slot in record.capacities.iter().flatten() {
                slot.semaphore.close();
            }
        }
        Ok(())
    }
}

/// Opaque proof that media request classification cannot alter eligibility.
pub(crate) struct HomogeneousMediaHttp<'a> {
    pool: &'a WorkerPool,
    cohort: &'a HomogeneousMediaCohort,
}

impl HomogeneousMediaHttp<'_> {
    pub(crate) fn dispatch(self, admission: AdmissionLease) -> Result<RequestLease, DispatchError> {
        let class = self.cohort.service.capacity();
        if admission.class() != class {
            return Err(DispatchError::AdmissionClassMismatch);
        }
        let candidates = self
            .cohort
            .records
            .iter()
            .filter(|record| record.available_for_dispatch())
            .cloned()
            .map(PolicyCandidate::new)
            .collect();
        self.pool
            .dispatch_candidates(admission, class, true, candidates)
    }
}

impl HomogeneousGenerationHttp<'_> {
    pub(crate) fn dispatch(self, admission: AdmissionLease) -> Result<RequestLease, DispatchError> {
        let class = CapacityClass::GenerationHttp;
        if admission.class() != class {
            return Err(DispatchError::AdmissionClassMismatch);
        }
        let candidates = self
            .cohort
            .records
            .iter()
            .filter(|record| record.available_for_dispatch())
            .cloned()
            .map(PolicyCandidate::new)
            .collect();
        self.pool
            .dispatch_candidates(admission, class, true, candidates)
    }
}

fn build_homogeneous_generation_cohorts(
    records: &[Arc<WorkerRecord>],
) -> Vec<HomogeneousGenerationCohort> {
    let mut result = Vec::new();
    let mut processed_scopes = Vec::with_capacity(records.len());
    for record in records {
        if !record
            .profiles
            .iter()
            .any(|profile| profile.service_class() == ServiceClass::GenerationHttp)
            || processed_scopes
                .iter()
                .any(|trust_domain| *trust_domain == &record.trust_domain)
        {
            continue;
        }
        processed_scopes.push(&record.trust_domain);
        let members: Vec<_> = records
            .iter()
            .filter(|candidate| {
                candidate.trust_domain == record.trust_domain
                    && candidate
                        .profiles
                        .iter()
                        .any(|profile| profile.service_class() == ServiceClass::GenerationHttp)
            })
            .cloned()
            .collect();
        let first = &members[0];
        let Some(shared_default) = first.default_model_id() else {
            continue;
        };
        if members.iter().all(|candidate| {
            candidate.default_model_id() == Some(shared_default)
                && generation_rows_equal(&candidate.profiles, &first.profiles)
        }) {
            result.push(HomogeneousGenerationCohort {
                trust_domain: record.trust_domain.clone(),
                records: members,
            });
        }
    }
    result
}

fn build_homogeneous_media_cohorts(records: &[Arc<WorkerRecord>]) -> Vec<HomogeneousMediaCohort> {
    let mut result = Vec::new();
    for service in [
        ServiceClass::SpeechHttp,
        ServiceClass::SpeechBatch,
        ServiceClass::TranscriptionHttp,
    ] {
        let mut processed_scopes = Vec::with_capacity(records.len());
        for record in records {
            if !record
                .profiles
                .iter()
                .any(|profile| profile.service_class() == service)
                || processed_scopes
                    .iter()
                    .any(|trust_domain| *trust_domain == &record.trust_domain)
            {
                continue;
            }
            processed_scopes.push(&record.trust_domain);
            let members: Vec<_> = records
                .iter()
                .filter(|candidate| {
                    candidate.trust_domain == record.trust_domain
                        && candidate
                            .profiles
                            .iter()
                            .any(|profile| profile.service_class() == service)
                })
                .cloned()
                .collect();
            let first = &members[0];
            let Some(shared_default) = first.default_model_id() else {
                continue;
            };
            if members.iter().all(|candidate| {
                candidate.default_model_id() == Some(shared_default)
                    && service_rows_equal(&candidate.profiles, &first.profiles, service)
                    && candidate
                        .profiles
                        .iter()
                        .filter(|profile| profile.service_class() == service)
                        .all(|profile| !profile.requires_owner())
            }) {
                result.push(HomogeneousMediaCohort {
                    trust_domain: record.trust_domain.clone(),
                    service,
                    records: members,
                });
            }
        }
    }
    result
}

fn generation_rows_equal(left: &[ServiceProfile], right: &[ServiceProfile]) -> bool {
    let generation_row =
        |profile: &&ServiceProfile| profile.service_class() == ServiceClass::GenerationHttp;
    left.iter().filter(generation_row).count() == right.iter().filter(generation_row).count()
        && left.iter().filter(generation_row).all(|profile| {
            right
                .iter()
                .filter(generation_row)
                .any(|other| profile.semantically_eq(other))
        })
}

fn service_rows_equal(
    left: &[ServiceProfile],
    right: &[ServiceProfile],
    service: ServiceClass,
) -> bool {
    let rows = |profile: &&ServiceProfile| profile.service_class() == service;
    left.iter().filter(rows).count() == right.iter().filter(rows).count()
        && left.iter().filter(rows).all(|profile| {
            right
                .iter()
                .filter(rows)
                .any(|other| profile.semantically_eq(other))
        })
}

fn compute_realtime_defaults_unambiguous(records: &[Arc<WorkerRecord>]) -> bool {
    let mut scopes: Vec<(&TrustDomain, &str)> = Vec::new();
    for record in records {
        if !record
            .profiles
            .iter()
            .any(|profile| profile.service_class() == ServiceClass::RealtimeWebsocket)
        {
            continue;
        }
        let Some(default) = record.default_model_id() else {
            return false;
        };
        if let Some((_, expected)) = scopes
            .iter()
            .find(|(trust_domain, _)| *trust_domain == &record.trust_domain)
        {
            if *expected != default {
                return false;
            }
        } else {
            scopes.push((&record.trust_domain, default));
        }
    }
    true
}

fn build_policy_candidates(
    records: &[Arc<WorkerRecord>],
    requirement: &RouteRequirement,
    voice_owner: Option<&WorkerId>,
) -> (bool, Vec<PolicyCandidate>) {
    let mut profile_found = false;
    let mut candidates = Vec::with_capacity(records.len());
    for record in records {
        if !record.has_profile(requirement) {
            continue;
        }
        profile_found = true;
        if record.immutable_eligible(requirement, voice_owner) && record.available_for_dispatch() {
            candidates.push(PolicyCandidate::new(Arc::clone(record)));
        }
    }
    (profile_found, candidates)
}

fn build_capacities(
    config: &WorkerCapacityConfig,
) -> Result<[Option<CapacitySlot>; 7], crate::error::RouterError> {
    let values = [
        config.generation_http,
        config.speech_http,
        config.transcription_http,
        config.speech_batch,
        config.speech_websocket,
        config.realtime_websocket,
        config.control,
    ];
    let mut result: [Option<CapacitySlot>; 7] = std::array::from_fn(|_| None);
    for (index, value) in values.into_iter().enumerate() {
        if let Some(limit) = value {
            let limit = usize::try_from(limit)
                .map_err(|_| crate::error::RouterError::WorkerPoolInvariant)?;
            result[index] = Some(CapacitySlot {
                limit,
                semaphore: Arc::new(Semaphore::new(limit)),
            });
        }
    }
    Ok(result)
}

#[allow(dead_code)] // Used by the Criterion target that includes this module.
pub(crate) mod benchmark {
    use std::sync::Arc;
    use std::sync::RwLock;
    use std::sync::atomic::AtomicU8;
    use std::time::Duration;

    use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};

    use super::profile::{
        InputModality, MessageContentForm, ModelSelection, OutputModality, ProfileRequirement,
        RegistrationId, RouteRequirement, ServiceClass, ServiceProfile, StreamMode, TrustDomain,
        WorkerId,
    };
    use super::{
        AdmissionController, CapacityClass, CapacitySlot, Disposition, Gate, HealthCell,
        HealthState, PolicyCandidate, RequestLease, ResolvedTarget, Selector, StaticResolver,
        WorkerPool, WorkerRecord, build_health_client, build_homogeneous_generation_cohorts,
        build_homogeneous_media_cohorts, build_policy_candidates,
        compute_realtime_defaults_unambiguous,
    };
    use crate::config::RoutingStrategy;

    pub(crate) struct Fixture {
        records: Vec<Arc<WorkerRecord>>,
        requirement: RouteRequirement,
        trust_domain: TrustDomain,
        selector: Selector,
        dispatch_pool: WorkerPool,
        _saturated: Vec<OwnedSemaphorePermit>,
    }

    pub(crate) struct CandidateBatch(Vec<PolicyCandidate>);

    pub(crate) struct ExactBatch(Vec<Arc<WorkerRecord>>);

    impl Fixture {
        pub(crate) fn new(size: usize, strategy: RoutingStrategy) -> Result<Self, &'static str> {
            let target = ResolvedTarget::from_parts("http://127.0.0.1:30000/", "/health", None)
                .ok_or("benchmark target failed to build")?;
            let mut records = Vec::with_capacity(size);
            let mut saturated = Vec::with_capacity(size.saturating_sub(1));
            for registration_ordinal in 0..size {
                let semaphore = Arc::new(Semaphore::new(1));
                if registration_ordinal + 1 < size
                    && let Ok(permit) = Arc::clone(&semaphore).try_acquire_owned()
                {
                    saturated.push(permit);
                }
                let mut capacities = std::array::from_fn(|_| None);
                capacities[0] = Some(CapacitySlot {
                    limit: 1,
                    semaphore,
                });
                let health = HealthCell::unknown();
                health.store(HealthState::Healthy);
                records.push(Arc::new(WorkerRecord {
                    worker_id: WorkerId::new(format!("worker-{registration_ordinal}")),
                    default_model_id: Some(String::from("omni")),
                    registration_id: RegistrationId::from_startup_ordinal(registration_ordinal),
                    target: target.clone(),
                    trust_domain: TrustDomain::new(String::from("local")),
                    profiles: vec![ServiceProfile::GenerationHttp {
                        model_ids: vec![String::from("omni")],
                        message_content_forms: vec![MessageContentForm::String],
                        media_placements: Vec::new(),
                        input_modalities: vec![InputModality::Text],
                        output_modalities: vec![OutputModality::Text],
                        chat_audio_formats: Vec::new(),
                        stream_modes: vec![StreamMode::NonStreaming],
                    }],
                    capacities,
                    health,
                    disposition: AtomicU8::new(Disposition::Serving as u8),
                    immediate_probe: Notify::new(),
                }));
            }
            let trust_domain = TrustDomain::new(String::from("local"));
            let requirement = RouteRequirement::new(
                ProfileRequirement::GenerationHttp {
                    model: ModelSelection::Explicit(String::from("omni")),
                    message_content_forms: vec![MessageContentForm::String],
                    media_placements: Vec::new(),
                    input_modalities: vec![InputModality::Text],
                    output_modalities: vec![OutputModality::Text],
                    audio_format: None,
                    stream_mode: StreamMode::NonStreaming,
                },
                trust_domain.clone(),
                false,
            );
            let gate = Arc::new(RwLock::new(Gate::open()));
            let targets: Vec<_> = records.iter().map(|record| record.target.clone()).collect();
            let resolver = StaticResolver::from_targets(&targets).ok_or("resolver must build")?;
            let health_client = build_health_client(
                Arc::new(resolver),
                Duration::from_secs(1),
                Duration::from_secs(1),
            )
            .map_err(|_| "health client must build")?;
            let dispatch_pool = WorkerPool {
                homogeneous_generation_http: build_homogeneous_generation_cohorts(&records),
                homogeneous_media_http: build_homogeneous_media_cohorts(&records),
                realtime_defaults_unambiguous: compute_realtime_defaults_unambiguous(&records),
                records: records.clone(),
                required_services: vec![ServiceClass::GenerationHttp],
                voice_owner: None,
                gate: Arc::clone(&gate),
                admission: AdmissionController::new(gate, 2, [2; 7]),
                selector: Selector::new(strategy),
                generation_client: Some(health_client.clone()),
                media_client: None,
                health_client,
            };
            Ok(Self {
                records,
                requirement,
                trust_domain,
                selector: Selector::new(strategy),
                dispatch_pool,
                _saturated: saturated,
            })
        }

        pub(crate) fn candidate_construction(&self) -> CandidateBatch {
            CandidateBatch(build_policy_candidates(&self.records, &self.requirement, None).1)
        }

        pub(crate) fn policy_input(&self) -> CandidateBatch {
            self.candidate_construction()
        }

        pub(crate) fn policy_order(&self, candidates: &mut CandidateBatch) -> usize {
            self.selector.order_in_place(
                &mut candidates.0,
                CapacityClass::GenerationHttp,
                self.records.len(),
            );
            candidates.0.len()
        }

        pub(crate) fn exact_input(&self) -> ExactBatch {
            ExactBatch(self.records.clone())
        }

        pub(crate) fn exact_permit_scan(&self, candidates: &ExactBatch) -> bool {
            for record in &candidates.0 {
                let Some(slot) = record.slot(CapacityClass::GenerationHttp) else {
                    continue;
                };
                if let Ok(permit) = Arc::clone(&slot.semaphore).try_acquire_owned() {
                    drop(permit);
                    return true;
                }
            }
            false
        }

        pub(crate) fn normal_dispatch(&self) -> bool {
            let Ok(admission) = self.dispatch_pool.try_admit(CapacityClass::GenerationHttp) else {
                return false;
            };
            self.dispatch_pool
                .dispatch(admission, &self.requirement)
                .is_ok()
        }

        pub(crate) fn homogeneous_dispatch(&self) -> bool {
            let Some(proof) = self
                .dispatch_pool
                .homogeneous_generation_http(&self.trust_domain)
            else {
                return false;
            };
            let Ok(admission) = self.dispatch_pool.try_admit(CapacityClass::GenerationHttp) else {
                return false;
            };
            proof.dispatch(admission).is_ok()
        }

        pub(crate) fn pool(&self) -> &WorkerPool {
            &self.dispatch_pool
        }

        pub(crate) fn trust_domain(&self) -> &TrustDomain {
            &self.trust_domain
        }

        pub(crate) fn request_lease(&self) -> Option<RequestLease> {
            let admission = self
                .dispatch_pool
                .try_admit(CapacityClass::GenerationHttp)
                .ok()?;
            self.dispatch_pool
                .dispatch(admission, &self.requirement)
                .ok()
        }

        pub(crate) fn probe_requested(&self) -> bool {
            use std::future::Future;
            use std::task::{Context, Poll, Waker};

            let Some(record) = self.records.first() else {
                return false;
            };
            let mut notified = std::pin::pin!(record.immediate_probe.notified());
            let waker = Waker::noop();
            let mut context = Context::from_waker(waker);
            matches!(
                Future::poll(notified.as_mut(), &mut context),
                Poll::Ready(())
            )
        }

        pub(crate) fn exact_semaphore(&self) -> Option<Arc<Semaphore>> {
            self.records
                .first()?
                .slot(CapacityClass::GenerationHttp)
                .map(|slot| Arc::clone(&slot.semaphore))
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use std::panic::AssertUnwindSafe;
    use std::sync::atomic::AtomicU8;
    use std::sync::{Arc, Barrier, Mutex, RwLock};
    use std::thread;
    use std::time::Duration;

    use crate::config::RoutingStrategy;
    use tokio::sync::oneshot;

    use super::health::{HealthCell, HealthState};
    use super::permit::{AdmissionController, Gate, class_index};
    use super::profile::{
        ChatAudioFormat, InputModality, MediaPlacement, MessageContentForm, ModelSelection,
        OutputModality, ProfileRequirement, ReferenceForm, RegistrationId, RouteRequirement,
        ServiceClass, ServiceProfile, SpeechResponseFormat, SpeechTask, SpeechWebsocketInput,
        StreamMode, TrustDomain, WorkerId,
    };
    use super::resolver::{ResolvedTarget, StaticResolver, build_health_client};
    use super::{
        AdmissionClass, AdmissionError, CapacityClass, CapacitySlot, DispatchError, Disposition,
        PolicyCandidate, Selector, WorkerPool, WorkerRecord, build_homogeneous_generation_cohorts,
        build_homogeneous_media_cohorts, compute_realtime_defaults_unambiguous,
    };

    type ProfileMutation = fn(&mut ServiceProfile);

    fn generation_profile() -> ServiceProfile {
        ServiceProfile::GenerationHttp {
            model_ids: vec![String::from("omni")],
            message_content_forms: vec![MessageContentForm::String],
            media_placements: Vec::new(),
            input_modalities: vec![InputModality::Text],
            output_modalities: vec![OutputModality::Text],
            chat_audio_formats: Vec::new(),
            stream_modes: vec![StreamMode::NonStreaming],
        }
    }

    fn rich_generation_profile() -> ServiceProfile {
        ServiceProfile::GenerationHttp {
            model_ids: vec![String::from("omni")],
            message_content_forms: vec![MessageContentForm::String, MessageContentForm::TypedParts],
            media_placements: vec![MediaPlacement::TopLevel, MediaPlacement::TypedParts],
            input_modalities: vec![InputModality::Text, InputModality::Image],
            output_modalities: vec![OutputModality::Text, OutputModality::Audio],
            chat_audio_formats: vec![ChatAudioFormat::Wav],
            stream_modes: vec![StreamMode::NonStreaming],
        }
    }

    fn generation_requirement() -> RouteRequirement {
        RouteRequirement::new(
            ProfileRequirement::GenerationHttp {
                model: ModelSelection::Explicit(String::from("omni")),
                message_content_forms: vec![MessageContentForm::String],
                media_placements: Vec::new(),
                input_modalities: vec![InputModality::Text],
                output_modalities: vec![OutputModality::Text],
                audio_format: None,
                stream_mode: StreamMode::NonStreaming,
            },
            TrustDomain::new(String::from("local")),
            false,
        )
    }

    #[derive(Clone, Copy)]
    enum SpeechSurface {
        Http,
        Websocket,
    }

    impl SpeechSurface {
        const fn capacity_class(self) -> CapacityClass {
            match self {
                Self::Http => CapacityClass::SpeechHttp,
                Self::Websocket => CapacityClass::SpeechWebsocket,
            }
        }

        const fn service_class(self) -> ServiceClass {
            match self {
                Self::Http => ServiceClass::SpeechHttp,
                Self::Websocket => ServiceClass::SpeechWebsocket,
            }
        }

        fn profile(
            self,
            response_format: SpeechResponseFormat,
            stream_modes: &[StreamMode],
        ) -> ServiceProfile {
            match self {
                Self::Http => ServiceProfile::SpeechHttp {
                    model_ids: vec![String::from("tts")],
                    response_formats: vec![response_format],
                    stream_modes: stream_modes.to_vec(),
                    tasks: vec![SpeechTask::TextToSpeech],
                    reference_forms: vec![ReferenceForm::None],
                    managed_voice: false,
                },
                Self::Websocket => ServiceProfile::SpeechWebsocket {
                    model_ids: vec![String::from("tts")],
                    input_profiles: vec![SpeechWebsocketInput::Text],
                    response_formats: vec![response_format],
                    stream_modes: stream_modes.to_vec(),
                    tasks: vec![SpeechTask::TextToSpeech],
                    reference_forms: vec![ReferenceForm::None],
                    managed_voice: false,
                },
            }
        }

        fn requirement(
            self,
            response_format: SpeechResponseFormat,
            stream_mode: StreamMode,
        ) -> RouteRequirement {
            let profile = match self {
                Self::Http => ProfileRequirement::SpeechHttp {
                    model: ModelSelection::Explicit(String::from("tts")),
                    response_format,
                    stream_mode,
                    task: SpeechTask::TextToSpeech,
                    reference_forms: vec![ReferenceForm::None],
                    managed_voice: false,
                },
                Self::Websocket => ProfileRequirement::SpeechWebsocket {
                    model: ModelSelection::Explicit(String::from("tts")),
                    input_profile: SpeechWebsocketInput::Text,
                    response_format,
                    stream_mode,
                    task: SpeechTask::TextToSpeech,
                    reference_forms: vec![ReferenceForm::None],
                    managed_voice: false,
                },
            };
            RouteRequirement::new(profile, TrustDomain::new(String::from("local")), false)
        }
    }

    fn record_with_profiles(
        id: usize,
        class: CapacityClass,
        exact_limit: usize,
        profiles: Vec<ServiceProfile>,
    ) -> Arc<WorkerRecord> {
        let mut capacities = std::array::from_fn(|_| None);
        capacities[class_index(class)] = Some(CapacitySlot {
            limit: exact_limit,
            semaphore: Arc::new(tokio::sync::Semaphore::new(exact_limit)),
        });
        Arc::new(WorkerRecord {
            worker_id: WorkerId::new(format!("worker-{id}")),
            default_model_id: Some(String::from("omni")),
            registration_id: RegistrationId::from_startup_ordinal(id),
            target: ResolvedTarget::from_parts(
                &format!("http://127.0.0.1:{}/", 10_000 + id),
                "/health",
                None,
            )
            .expect("test target must build"),
            trust_domain: TrustDomain::new(String::from("local")),
            profiles,
            capacities,
            health: HealthCell::unknown(),
            disposition: AtomicU8::new(super::Disposition::Serving as u8),
            immediate_probe: tokio::sync::Notify::new(),
        })
    }

    fn record(id: usize, exact_limit: usize) -> Arc<WorkerRecord> {
        record_with_profiles(
            id,
            CapacityClass::GenerationHttp,
            exact_limit,
            vec![generation_profile()],
        )
    }

    fn generation_profile_for(model_id: &str) -> ServiceProfile {
        let mut profile = generation_profile();
        if let ServiceProfile::GenerationHttp { model_ids, .. } = &mut profile {
            *model_ids = vec![model_id.to_owned()];
        }
        profile
    }

    fn speech_batch_profile_for(model_id: &str) -> ServiceProfile {
        ServiceProfile::SpeechBatch {
            model_ids: vec![model_id.to_owned()],
            response_formats: vec![SpeechResponseFormat::Wav],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::None],
            managed_voice: false,
            max_batch_size: 4,
            effective_features: vec![super::profile::BatchFeature::Model],
        }
    }

    fn media_speech_profile(
        reference_forms: Vec<ReferenceForm>,
        managed_voice: bool,
    ) -> ServiceProfile {
        ServiceProfile::SpeechHttp {
            model_ids: vec![String::from("tts")],
            response_formats: vec![SpeechResponseFormat::Wav],
            stream_modes: vec![StreamMode::NonStreaming],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms,
            managed_voice,
        }
    }

    fn transcription_profile(
        response_formats: Vec<super::profile::TranscriptionResponseFormat>,
    ) -> ServiceProfile {
        ServiceProfile::TranscriptionHttp {
            model_ids: vec![String::from("asr")],
            response_formats,
            media_profiles: vec![super::profile::MediaProfile::Audio],
            stream_modes: vec![StreamMode::NonStreaming],
        }
    }

    fn add_capacity(record: &mut Arc<WorkerRecord>, class: CapacityClass, limit: usize) {
        Arc::get_mut(record)
            .expect("new test record must be unique")
            .capacities[class_index(class)] = Some(CapacitySlot {
            limit,
            semaphore: Arc::new(tokio::sync::Semaphore::new(limit)),
        });
    }

    fn record_with_default(
        id: usize,
        exact_limit: usize,
        trust_domain: &str,
        default_model_id: &str,
        profiles: Vec<ServiceProfile>,
    ) -> Arc<WorkerRecord> {
        record_with_default_class(
            id,
            CapacityClass::GenerationHttp,
            exact_limit,
            trust_domain,
            default_model_id,
            profiles,
        )
    }

    fn record_with_default_class(
        id: usize,
        class: CapacityClass,
        exact_limit: usize,
        trust_domain: &str,
        default_model_id: &str,
        profiles: Vec<ServiceProfile>,
    ) -> Arc<WorkerRecord> {
        let mut record = record_with_profiles(id, class, exact_limit, profiles);
        let mutable = Arc::get_mut(&mut record).expect("new test record must be unique");
        mutable.default_model_id = Some(default_model_id.to_owned());
        mutable.trust_domain = TrustDomain::new(trust_domain.to_owned());
        record
    }

    fn pool_with_records(
        strategy: RoutingStrategy,
        required_service: ServiceClass,
        records: Vec<Arc<WorkerRecord>>,
    ) -> WorkerPool {
        for record in &records {
            record.health.store(HealthState::Healthy);
        }
        let gate = Arc::new(RwLock::new(Gate::open()));
        let targets: Vec<_> = records.iter().map(|record| record.target.clone()).collect();
        let resolver = StaticResolver::from_targets(&targets).expect("test resolver must build");
        let health_client = build_health_client(
            Arc::new(resolver),
            Duration::from_secs(1),
            Duration::from_secs(1),
        )
        .expect("test health client must build");
        let homogeneous_generation_http = build_homogeneous_generation_cohorts(&records);
        let homogeneous_media_http = build_homogeneous_media_cohorts(&records);
        let realtime_defaults_unambiguous = compute_realtime_defaults_unambiguous(&records);
        WorkerPool {
            records,
            required_services: vec![required_service],
            voice_owner: None,
            gate: Arc::clone(&gate),
            admission: AdmissionController::new(gate, 512, [512; 7]),
            selector: Selector::new(strategy),
            homogeneous_generation_http,
            homogeneous_media_http,
            realtime_defaults_unambiguous,
            generation_client: Some(health_client.clone()),
            media_client: None,
            health_client,
        }
    }

    fn fixture_pool(strategy: RoutingStrategy, size: usize, exact_limit: usize) -> WorkerPool {
        let records: Vec<_> = (0..size).map(|id| record(id, exact_limit)).collect();
        pool_with_records(strategy, ServiceClass::GenerationHttp, records)
    }

    #[test]
    fn voice_control_dispatch_is_exact_and_policy_independent() {
        for strategy in [RoutingStrategy::RoundRobin, RoutingStrategy::LeastRequests] {
            let owner = record_with_profiles(
                0,
                CapacityClass::Control,
                1,
                vec![ServiceProfile::VoiceControl],
            );
            let other = record(1, 8);
            let mut pool =
                pool_with_records(strategy, ServiceClass::VoiceControl, vec![owner, other]);
            pool.voice_owner = Some(WorkerId::new(String::from("worker-0")));
            assert!(pool.voice_state_enabled());
            assert!(pool.voice_owner_ready());

            let lease = pool
                .dispatch_voice_control(
                    pool.try_admit(CapacityClass::Control)
                        .expect("control admission"),
                )
                .expect("exact owner dispatch");
            assert_eq!(lease.worker_id().as_str(), "worker-0");
            assert_eq!(
                pool.dispatch_voice_control(
                    pool.try_admit(CapacityClass::Control)
                        .expect("second control admission"),
                )
                .err(),
                Some(DispatchError::Overloaded)
            );
            drop(lease);
            pool.records[0].health.store(HealthState::Unhealthy);
            assert!(!pool.voice_owner_ready());
            assert_eq!(
                pool.dispatch_voice_control(
                    pool.try_admit(CapacityClass::Control)
                        .expect("unavailable control admission"),
                )
                .err(),
                Some(DispatchError::Unavailable)
            );
        }
    }

    #[test]
    fn admission_partial_failure_rolls_back_global_and_drain_is_fail_fast() {
        let gate = Arc::new(RwLock::new(Gate::open()));
        let admission = AdmissionController::new(Arc::clone(&gate), 2, [1; 7]);
        let first = admission
            .try_admit(CapacityClass::GenerationHttp)
            .expect("first admission should fit");
        assert_eq!(
            admission.try_admit(CapacityClass::GenerationHttp).err(),
            Some(AdmissionError::Overloaded)
        );
        assert_eq!(admission.available(CapacityClass::GenerationHttp), (1, 0));
        let speech = admission
            .try_admit(CapacityClass::SpeechHttp)
            .expect("speech class remains isolated");
        assert_eq!(
            admission.try_admit(CapacityClass::TranscriptionHttp).err(),
            Some(AdmissionError::Overloaded)
        );
        drop(speech);
        drop(first);
        assert_eq!(admission.available(CapacityClass::GenerationHttp), (2, 1));
        gate.write().expect("gate lock should be healthy").open = false;
        assert_eq!(
            admission.try_admit(CapacityClass::GenerationHttp).err(),
            Some(AdmissionError::Draining)
        );
    }

    #[test]
    fn round_robin_is_canonical_and_stable_as_health_changes() {
        let pool = fixture_pool(RoutingStrategy::RoundRobin, 3, 1);
        let requirement = generation_requirement();
        let mut sequence = Vec::new();
        for _ in 0..4 {
            let admission = pool
                .try_admit(CapacityClass::GenerationHttp)
                .expect("admission should fit");
            let lease = pool
                .dispatch(admission, &requirement)
                .expect("dispatch should fit");
            sequence.push(lease.worker_id().as_str().to_owned());
        }
        assert_eq!(sequence, ["worker-0", "worker-1", "worker-2", "worker-0"]);

        pool.records[1].health.store(HealthState::Unhealthy);
        let admission = pool
            .try_admit(CapacityClass::GenerationHttp)
            .expect("admission should fit");
        let lease = pool
            .dispatch(admission, &requirement)
            .expect("healthy worker should remain");
        assert_eq!(lease.worker_id().as_str(), "worker-2");
    }

    #[test]
    fn correlated_profile_rows_never_form_a_cartesian_match() {
        let mut record = record(0, 1);
        let worker = Arc::get_mut(&mut record).expect("fixture record must be unique");
        worker.profiles = vec![
            ServiceProfile::GenerationHttp {
                model_ids: vec![String::from("omni")],
                message_content_forms: vec![MessageContentForm::String],
                media_placements: Vec::new(),
                input_modalities: vec![InputModality::Text],
                output_modalities: vec![OutputModality::Text],
                chat_audio_formats: Vec::new(),
                stream_modes: vec![StreamMode::NonStreaming],
            },
            ServiceProfile::GenerationHttp {
                model_ids: vec![String::from("omni")],
                message_content_forms: vec![MessageContentForm::String],
                media_placements: vec![MediaPlacement::TopLevel],
                input_modalities: vec![InputModality::Image],
                output_modalities: vec![OutputModality::Audio],
                chat_audio_formats: vec![ChatAudioFormat::Wav],
                stream_modes: vec![StreamMode::Streaming],
            },
        ];
        let pool = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::GenerationHttp,
            vec![record],
        );
        let cross_row_requirement = RouteRequirement::new(
            ProfileRequirement::GenerationHttp {
                model: ModelSelection::Explicit(String::from("omni")),
                message_content_forms: vec![MessageContentForm::String],
                media_placements: Vec::new(),
                input_modalities: vec![InputModality::Text],
                output_modalities: vec![OutputModality::Audio],
                audio_format: Some(ChatAudioFormat::Wav),
                stream_mode: StreamMode::Streaming,
            },
            TrustDomain::new(String::from("local")),
            false,
        );
        let admission = pool
            .try_admit(CapacityClass::GenerationHttp)
            .expect("cross-row proof admission");
        assert_eq!(
            pool.dispatch(admission, &cross_row_requirement).err(),
            Some(DispatchError::NoEligibleProfile)
        );
    }

    #[test]
    fn speech_stream_modes_drive_exact_dispatch_without_cross_row_merging() {
        for surface in [SpeechSurface::Http, SpeechSurface::Websocket] {
            let class = surface.capacity_class();
            let pool = pool_with_records(
                RoutingStrategy::RoundRobin,
                surface.service_class(),
                vec![
                    record_with_profiles(
                        0,
                        class,
                        1,
                        vec![
                            surface.profile(SpeechResponseFormat::Pcm, &[StreamMode::NonStreaming]),
                        ],
                    ),
                    record_with_profiles(
                        1,
                        class,
                        1,
                        vec![surface.profile(SpeechResponseFormat::Pcm, &[StreamMode::Streaming])],
                    ),
                ],
            );

            let streaming = pool
                .dispatch(
                    pool.try_admit(class).expect("streaming speech admission"),
                    &surface.requirement(SpeechResponseFormat::Pcm, StreamMode::Streaming),
                )
                .expect("streaming speech dispatch");
            assert_eq!(streaming.registration_id().startup_ordinal(), 1);
            assert_eq!(streaming.capacity_class(), class);
            drop(streaming);

            let non_streaming = pool
                .dispatch(
                    pool.try_admit(class)
                        .expect("non-streaming speech admission"),
                    &surface.requirement(SpeechResponseFormat::Pcm, StreamMode::NonStreaming),
                )
                .expect("non-streaming speech dispatch");
            assert_eq!(non_streaming.registration_id().startup_ordinal(), 0);
            assert_eq!(non_streaming.capacity_class(), class);
            drop(non_streaming);

            let correlated = pool_with_records(
                RoutingStrategy::RoundRobin,
                surface.service_class(),
                vec![record_with_profiles(
                    0,
                    class,
                    1,
                    vec![
                        surface.profile(SpeechResponseFormat::Mp3, &[StreamMode::NonStreaming]),
                        surface.profile(SpeechResponseFormat::Pcm, &[StreamMode::Streaming]),
                    ],
                )],
            );
            assert_eq!(
                correlated
                    .dispatch(
                        correlated
                            .try_admit(class)
                            .expect("cross-row speech admission"),
                        &surface.requirement(SpeechResponseFormat::Mp3, StreamMode::Streaming),
                    )
                    .err(),
                Some(DispatchError::NoEligibleProfile)
            );
        }
    }

    #[test]
    fn immutable_and_mutable_filters_preserve_exact_error_classes() {
        let wrong_trust = fixture_pool(RoutingStrategy::RoundRobin, 1, 1);
        let untrusted_requirement = RouteRequirement::new(
            generation_requirement().profile,
            TrustDomain::new(String::from("remote")),
            false,
        );
        assert_eq!(
            wrong_trust
                .dispatch(
                    wrong_trust
                        .try_admit(CapacityClass::GenerationHttp)
                        .expect("wrong-trust admission"),
                    &untrusted_requirement,
                )
                .err(),
            Some(DispatchError::Unavailable)
        );

        let mut owner_pool = fixture_pool(RoutingStrategy::RoundRobin, 2, 1);
        owner_pool.voice_owner = Some(WorkerId::new(String::from("worker-1")));
        let owner_requirement = RouteRequirement::new(
            generation_requirement().profile,
            TrustDomain::new(String::from("local")),
            true,
        );
        let owner_lease = owner_pool
            .dispatch(
                owner_pool
                    .try_admit(CapacityClass::GenerationHttp)
                    .expect("owner admission"),
                &owner_requirement,
            )
            .expect("configured owner must be selected");
        assert_eq!(owner_lease.worker_id().as_str(), "worker-1");
        assert_eq!(owner_lease.registration_id().startup_ordinal(), 1);
        drop(owner_lease);

        for state in [HealthState::Unknown, HealthState::Unhealthy] {
            let unavailable = fixture_pool(RoutingStrategy::RoundRobin, 1, 1);
            unavailable.records[0].health.store(state);
            assert_eq!(
                unavailable
                    .dispatch(
                        unavailable
                            .try_admit(CapacityClass::GenerationHttp)
                            .expect("health-filter admission"),
                        &generation_requirement(),
                    )
                    .err(),
                Some(DispatchError::Unavailable)
            );
        }

        let draining_record = fixture_pool(RoutingStrategy::RoundRobin, 1, 1);
        draining_record.records[0].mark_draining();
        assert_eq!(
            draining_record
                .dispatch(
                    draining_record
                        .try_admit(CapacityClass::GenerationHttp)
                        .expect("disposition-filter admission"),
                    &generation_requirement(),
                )
                .err(),
            Some(DispatchError::Unavailable)
        );
    }

    #[test]
    fn admission_mismatch_and_exact_failure_roll_back_outer_permits() {
        let pool = fixture_pool(RoutingStrategy::RoundRobin, 1, 1);
        let wrong_class = pool
            .try_admit(CapacityClass::SpeechHttp)
            .expect("cross-class admission");
        assert_eq!(
            pool.dispatch(wrong_class, &generation_requirement()).err(),
            Some(DispatchError::AdmissionClassMismatch)
        );
        assert_eq!(
            pool.admission.available(CapacityClass::SpeechHttp),
            (512, 512)
        );

        let slot = pool.records[0]
            .slot(CapacityClass::GenerationHttp)
            .expect("generation slot");
        let exact = Arc::clone(&slot.semaphore)
            .try_acquire_owned()
            .expect("saturate exact capacity");
        assert_eq!(
            pool.dispatch(
                pool.try_admit(CapacityClass::GenerationHttp)
                    .expect("exact-overload admission"),
                &generation_requirement(),
            )
            .err(),
            Some(DispatchError::Overloaded)
        );
        assert_eq!(
            pool.admission.available(CapacityClass::GenerationHttp),
            (512, 512)
        );
        assert_eq!(
            pool.admission.available(CapacityClass::SpeechHttp),
            (512, 512)
        );
        drop(exact);
    }

    #[test]
    fn least_requests_uses_exact_semaphore_occupancy_and_rr_ties() {
        let pool = fixture_pool(RoutingStrategy::LeastRequests, 2, 1);
        let requirement = generation_requirement();
        let first = pool
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp)
                    .expect("first admission"),
                &requirement,
            )
            .expect("first dispatch");
        assert_eq!(first.worker_id().as_str(), "worker-0");
        let second = pool
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp)
                    .expect("second admission"),
                &requirement,
            )
            .expect("second dispatch");
        assert_eq!(second.worker_id().as_str(), "worker-1");
        assert_eq!(
            pool.dispatch(
                pool.try_admit(CapacityClass::GenerationHttp)
                    .expect("third admission"),
                &requirement,
            )
            .err(),
            Some(DispatchError::Overloaded)
        );
        drop(first);
        let replacement = pool
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp)
                    .expect("replacement admission"),
                &requirement,
            )
            .expect("released exact capacity should be reusable");
        assert_eq!(replacement.worker_id().as_str(), "worker-0");
        drop(second);
    }

    #[test]
    fn drain_linearizes_against_admitted_and_reserved_work() {
        let pool = fixture_pool(RoutingStrategy::RoundRobin, 1, 1);
        let requirement = generation_requirement();
        let admitted = pool
            .try_admit(CapacityClass::GenerationHttp)
            .expect("admission before drain");
        let admitted_without_profile = pool
            .try_admit(CapacityClass::GenerationHttp)
            .expect("second admission before drain");
        let no_profile_requirement = RouteRequirement::new(
            ProfileRequirement::GenerationHttp {
                model: ModelSelection::Explicit(String::from("not-configured")),
                message_content_forms: vec![MessageContentForm::String],
                media_placements: Vec::new(),
                input_modalities: vec![InputModality::Text],
                output_modalities: vec![OutputModality::Text],
                audio_format: None,
                stream_mode: StreamMode::NonStreaming,
            },
            TrustDomain::new(String::from("local")),
            false,
        );
        pool.drain().expect("drain should close once");
        pool.drain().expect("drain should be idempotent");
        assert_eq!(
            pool.dispatch(admitted, &requirement).err(),
            Some(DispatchError::Draining)
        );
        assert_eq!(
            pool.dispatch(admitted_without_profile, &no_profile_requirement)
                .err(),
            Some(DispatchError::Draining)
        );
        assert_eq!(
            pool.try_admit(CapacityClass::GenerationHttp).err(),
            Some(AdmissionError::Draining)
        );

        let late_pool = fixture_pool(RoutingStrategy::RoundRobin, 1, 1);
        let lease = late_pool
            .dispatch(
                late_pool
                    .try_admit(CapacityClass::GenerationHttp)
                    .expect("admit before reserve"),
                &requirement,
            )
            .expect("reserve before drain");
        late_pool.drain().expect("drain with active lease");
        assert_eq!(
            late_pool.records[0]
                .slot(CapacityClass::GenerationHttp)
                .map(|slot| slot.semaphore.available_permits()),
            Some(0)
        );
        drop(lease);
        assert_eq!(
            late_pool.records[0]
                .slot(CapacityClass::GenerationHttp)
                .map(|slot| slot.semaphore.available_permits()),
            Some(1)
        );
    }

    #[test]
    fn lease_raii_conserves_permits_across_panic_unwind() {
        let pool = fixture_pool(RoutingStrategy::RoundRobin, 1, 1);
        let requirement = generation_requirement();
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _lease = pool
                .dispatch(
                    pool.try_admit(CapacityClass::GenerationHttp)
                        .expect("admission before unwind"),
                    &requirement,
                )
                .expect("dispatch before unwind");
            panic!("test unwind");
        }));
        assert!(outcome.is_err());
        assert_eq!(
            pool.admission.available(CapacityClass::GenerationHttp),
            (512, 512)
        );
        assert_eq!(
            pool.records[0]
                .slot(CapacityClass::GenerationHttp)
                .map(|slot| slot.semaphore.available_permits()),
            Some(1)
        );
    }

    #[tokio::test]
    async fn task_cancellation_drops_request_ownership_and_returns_every_permit() {
        let pool = Arc::new(fixture_pool(RoutingStrategy::RoundRobin, 1, 1));
        let task_pool = Arc::clone(&pool);
        let requirement = generation_requirement();
        let (ready_sender, ready_receiver) = oneshot::channel();
        let task = tokio::spawn(async move {
            let lease = task_pool
                .dispatch(
                    task_pool
                        .try_admit(CapacityClass::GenerationHttp)
                        .expect("cancellation admission"),
                    &requirement,
                )
                .expect("cancellation dispatch");
            let _send_result = ready_sender.send(());
            std::future::pending::<()>().await;
            drop(lease);
        });
        ready_receiver.await.expect("request lease must be owned");
        assert_eq!(
            pool.admission.available(CapacityClass::GenerationHttp),
            (511, 511)
        );
        assert_eq!(
            pool.records[0]
                .slot(CapacityClass::GenerationHttp)
                .map(|slot| slot.semaphore.available_permits()),
            Some(0)
        );
        task.abort();
        assert!(
            task.await
                .expect_err("task must be cancelled")
                .is_cancelled()
        );
        assert_eq!(
            pool.admission.available(CapacityClass::GenerationHttp),
            (512, 512)
        );
        assert_eq!(
            pool.records[0]
                .slot(CapacityClass::GenerationHttp)
                .map(|slot| slot.semaphore.available_permits()),
            Some(1)
        );
    }

    #[test]
    fn concurrent_dispatch_and_drain_conserve_permits_and_close_linearization() {
        const ROUNDS: usize = 8;
        const CONTENDERS: usize = 16;

        for _ in 0..ROUNDS {
            let pool = Arc::new(fixture_pool(RoutingStrategy::RoundRobin, 1, CONTENDERS));
            let post_drain_admission = pool
                .try_admit(CapacityClass::GenerationHttp)
                .expect("reserve admitted-but-unreserved proof");
            let admissions: Vec<_> = (0..CONTENDERS)
                .map(|_| {
                    pool.try_admit(CapacityClass::GenerationHttp)
                        .expect("stress admission")
                })
                .collect();
            let barrier = Arc::new(Barrier::new(CONTENDERS + 2));
            let leases = Arc::new(Mutex::new(Vec::with_capacity(CONTENDERS)));
            let requirement = Arc::new(generation_requirement());
            let mut dispatchers = Vec::with_capacity(CONTENDERS);
            for admission in admissions {
                let dispatch_pool = Arc::clone(&pool);
                let dispatch_barrier = Arc::clone(&barrier);
                let dispatch_leases = Arc::clone(&leases);
                let dispatch_requirement = Arc::clone(&requirement);
                dispatchers.push(thread::spawn(move || {
                    dispatch_barrier.wait();
                    match dispatch_pool.dispatch(admission, &dispatch_requirement) {
                        Ok(lease) => dispatch_leases
                            .lock()
                            .expect("stress lease lock")
                            .push(lease),
                        Err(DispatchError::Draining) => {}
                        Err(error) => panic!("unexpected stress dispatch result: {error}"),
                    }
                }));
            }
            let drain_pool = Arc::clone(&pool);
            let drain_barrier = Arc::clone(&barrier);
            let drainer = thread::spawn(move || {
                drain_barrier.wait();
                drain_pool.drain().expect("stress drain");
            });
            barrier.wait();
            drainer.join().expect("join stress drainer");
            assert_eq!(
                pool.dispatch(post_drain_admission, &requirement).err(),
                Some(DispatchError::Draining)
            );
            for dispatcher in dispatchers {
                dispatcher.join().expect("join stress dispatcher");
            }

            let mut owned = leases.lock().expect("inspect stress leases");
            let successes = owned.len();
            assert_eq!(
                pool.records[0]
                    .slot(CapacityClass::GenerationHttp)
                    .map(|slot| slot.semaphore.available_permits()),
                Some(CONTENDERS - successes)
            );
            assert_eq!(
                pool.admission.available(CapacityClass::GenerationHttp),
                (512 - successes, 512 - successes)
            );
            owned.clear();
            drop(owned);
            assert_eq!(
                pool.records[0]
                    .slot(CapacityClass::GenerationHttp)
                    .map(|slot| slot.semaphore.available_permits()),
                Some(CONTENDERS)
            );
            assert_eq!(
                pool.admission.available(CapacityClass::GenerationHttp),
                (512, 512)
            );
        }
    }

    #[test]
    fn selector_handles_contract_pool_sizes_without_panicking() {
        for size in [0, 1, 2, 8, 64, 256] {
            let records: Vec<_> = (0..size).map(|id| record(id, 1)).collect();
            let ordered = Selector::new(RoutingStrategy::RoundRobin).order(
                records.into_iter().map(PolicyCandidate::new).collect(),
                CapacityClass::GenerationHttp,
                size,
            );
            assert_eq!(ordered.len(), size);
        }
    }

    #[test]
    fn readiness_uses_required_health_and_disposition_not_permit_saturation() {
        let pool = fixture_pool(RoutingStrategy::RoundRobin, 1, 1);
        let requirement = generation_requirement();
        assert!(pool.is_ready());
        let lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp)
                    .expect("readiness fixture admission"),
                &requirement,
            )
            .expect("readiness fixture dispatch");
        assert!(pool.is_ready());
        pool.records[0].health.store(HealthState::Unhealthy);
        assert!(!pool.is_ready());
        pool.records[0].health.store(HealthState::Healthy);
        assert!(pool.is_ready());
        pool.drain().expect("readiness fixture drain");
        assert!(!pool.is_ready());
        drop(lease);
    }

    #[test]
    fn operations_snapshot_reads_exact_permits_and_releases_with_the_lease() {
        let pool = fixture_pool(RoutingStrategy::RoundRobin, 1, 3);
        let initial = pool
            .operations_snapshot()
            .expect("initial operations snapshot");
        assert_eq!(initial.admission[0].class, AdmissionClass::Global);
        assert_eq!(
            initial.admission[1].class,
            AdmissionClass::Capacity(CapacityClass::GenerationHttp)
        );
        assert_eq!(initial.admission[0].in_flight, 0);
        assert_eq!(initial.admission[1].in_flight, 0);
        assert_eq!(initial.workers[0].worker_id, "worker-0");
        assert_eq!(initial.workers[0].registration_ordinal, 0);
        assert_eq!(initial.workers[0].health, HealthState::Healthy);
        assert_eq!(initial.workers[0].disposition, Disposition::Serving);
        assert_eq!(initial.workers[0].capacity[0].limit, 3);
        assert_eq!(initial.workers[0].capacity[0].in_flight, 0);

        let lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp)
                    .expect("snapshot admission"),
                &generation_requirement(),
            )
            .expect("snapshot dispatch");
        let occupied = pool
            .operations_snapshot()
            .expect("occupied operations snapshot");
        assert_eq!(occupied.admission[0].in_flight, 1);
        assert_eq!(occupied.admission[1].in_flight, 1);
        assert_eq!(occupied.workers[0].capacity[0].in_flight, 1);

        drop(lease);
        let released = pool
            .operations_snapshot()
            .expect("released operations snapshot");
        assert_eq!(released.admission[0].in_flight, 0);
        assert_eq!(released.admission[1].in_flight, 0);
        assert_eq!(released.workers[0].capacity[0].in_flight, 0);

        pool.records[0].health.store(HealthState::Unhealthy);
        pool.drain().expect("drain snapshot fixture");
        let drained = pool
            .operations_snapshot()
            .expect("drained operations snapshot");
        assert_eq!(drained.workers[0].health, HealthState::Unhealthy);
        assert_eq!(drained.workers[0].disposition, Disposition::Draining);
    }

    #[test]
    fn generation_readiness_is_scoped_to_the_configured_route_trust() {
        let local = record_with_default(0, 1, "local", "omni", vec![generation_profile()]);
        let remote = record_with_default(1, 1, "remote", "omni", vec![generation_profile()]);
        let pool = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::GenerationHttp,
            vec![Arc::clone(&local), Arc::clone(&remote)],
        );
        let local_trust = TrustDomain::new(String::from("local"));
        let remote_trust = TrustDomain::new(String::from("remote"));

        local.health.store(HealthState::Unhealthy);
        assert!(
            pool.is_ready(),
            "generic required-service readiness may be satisfied by the remote trust"
        );
        assert!(!pool.generation_http_ready(&local_trust));
        assert!(pool.generation_http_ready(&remote_trust));

        local.health.store(HealthState::Healthy);
        assert!(pool.generation_http_ready(&local_trust));
        local.mark_draining();
        assert!(!pool.generation_http_ready(&local_trust));
    }

    #[test]
    fn empty_known_input_requirement_dispatches_against_a_nonempty_worker_profile() {
        let pool = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::GenerationHttp,
            vec![record_with_default(
                0,
                1,
                "local",
                "omni",
                vec![rich_generation_profile()],
            )],
        );
        let requirement = RouteRequirement::new(
            ProfileRequirement::GenerationHttp {
                model: ModelSelection::Explicit(String::from("omni")),
                message_content_forms: vec![MessageContentForm::TypedParts],
                media_placements: Vec::new(),
                input_modalities: Vec::new(),
                output_modalities: vec![OutputModality::Text],
                audio_format: None,
                stream_mode: StreamMode::NonStreaming,
            },
            TrustDomain::new(String::from("local")),
            false,
        );
        assert!(requirement.profile.is_well_formed());
        let lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp)
                    .expect("empty-input requirement admission"),
                &requirement,
            )
            .expect("empty input is a request constraint, not an empty worker profile");
        assert_eq!(lease.worker_id().as_str(), "worker-0");
    }

    #[test]
    fn default_resolution_is_service_scoped_and_availability_independent() {
        let first = record_with_default(
            0,
            1,
            "local",
            "A",
            vec![
                generation_profile_for("A"),
                ServiceProfile::SpeechHttp {
                    model_ids: vec![String::from("A")],
                    response_formats: vec![SpeechResponseFormat::Wav],
                    stream_modes: vec![StreamMode::NonStreaming],
                    tasks: vec![SpeechTask::TextToSpeech],
                    reference_forms: vec![ReferenceForm::None],
                    managed_voice: false,
                },
            ],
        );
        let second = record_with_default(1, 1, "local", "B", vec![generation_profile_for("B")]);
        second.health.store(HealthState::Unhealthy);
        second.mark_draining();
        let remote = record_with_default(2, 1, "remote", "R", vec![generation_profile_for("R")]);
        let pool = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::GenerationHttp,
            vec![first, second, remote],
        );
        let local = TrustDomain::new(String::from("local"));
        let remote = TrustDomain::new(String::from("remote"));
        assert_eq!(
            pool.resolve_default_model_id(&local, ServiceClass::GenerationHttp),
            super::DefaultModelResolution::Ambiguous
        );
        assert_eq!(
            pool.resolve_default_model_id(&local, ServiceClass::SpeechHttp),
            super::DefaultModelResolution::Unique("A")
        );
        assert_eq!(
            pool.resolve_default_model_id(&remote, ServiceClass::GenerationHttp),
            super::DefaultModelResolution::Unique("R")
        );
        assert_eq!(
            pool.resolve_default_model_id(&local, ServiceClass::TranscriptionHttp),
            super::DefaultModelResolution::NoService
        );
        assert_eq!(
            pool.resolve_default_model_id(&local, ServiceClass::VoiceControl),
            super::DefaultModelResolution::NoService
        );

        let explicit = RouteRequirement::new(
            ProfileRequirement::GenerationHttp {
                model: ModelSelection::Explicit(String::from("A")),
                message_content_forms: vec![MessageContentForm::String],
                media_placements: Vec::new(),
                input_modalities: vec![InputModality::Text],
                output_modalities: vec![OutputModality::Text],
                audio_format: None,
                stream_mode: StreamMode::NonStreaming,
            },
            local,
            false,
        );
        let lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp)
                    .expect("explicit admission"),
                &explicit,
            )
            .expect("explicit model remains routable under heterogeneous defaults");
        assert_eq!(lease.worker_id().as_str(), "worker-0");
    }

    #[test]
    fn heterogeneous_defaults_keep_explicit_generation_and_batch_routable() {
        let mut first = record_with_default(
            0,
            1,
            "local",
            "A",
            vec![generation_profile_for("A"), speech_batch_profile_for("A")],
        );
        let mut second = record_with_default(
            1,
            1,
            "local",
            "B",
            vec![generation_profile_for("B"), speech_batch_profile_for("B")],
        );
        add_capacity(&mut first, CapacityClass::SpeechBatch, 2);
        add_capacity(&mut second, CapacityClass::SpeechBatch, 2);
        let pool = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::GenerationHttp,
            vec![first, second],
        );
        let trust = TrustDomain::new(String::from("local"));
        let generation = RouteRequirement::new(
            ProfileRequirement::GenerationHttp {
                model: ModelSelection::Explicit(String::from("B")),
                message_content_forms: vec![MessageContentForm::String],
                media_placements: Vec::new(),
                input_modalities: vec![InputModality::Text],
                output_modalities: vec![OutputModality::Text],
                audio_format: None,
                stream_mode: StreamMode::NonStreaming,
            },
            trust.clone(),
            false,
        );
        let generation_lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp)
                    .expect("generation admission"),
                &generation,
            )
            .expect("explicit generation dispatch");
        assert_eq!(generation_lease.worker_id().as_str(), "worker-1");
        drop(generation_lease);

        let batch = RouteRequirement::new(
            ProfileRequirement::SpeechBatch {
                models: vec![ModelSelection::Explicit(String::from("A"))],
                response_formats: vec![SpeechResponseFormat::Wav],
                tasks: vec![SpeechTask::TextToSpeech],
                reference_forms: vec![ReferenceForm::None],
                managed_voice: false,
                batch_size: 1,
                effective_features: vec![super::profile::BatchFeature::Model],
            },
            trust,
            false,
        );
        let batch_lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::SpeechBatch)
                    .expect("batch admission"),
                &batch,
            )
            .expect("fully explicit batch dispatch");
        assert_eq!(batch_lease.worker_id().as_str(), "worker-0");
    }

    #[test]
    fn malformed_speech_batches_fail_before_candidate_scan() {
        #[derive(Clone, Copy, Debug)]
        enum InvalidSetShape {
            Empty,
            Duplicate,
            Oversized,
        }

        fn invalid_set<T: Clone>(value: T, shape: InvalidSetShape) -> Vec<T> {
            match shape {
                InvalidSetShape::Empty => Vec::new(),
                InvalidSetShape::Duplicate => vec![value; 2],
                InvalidSetShape::Oversized => vec![value; 65],
            }
        }

        type BatchRequirementMutation = fn(&mut ProfileRequirement, InvalidSetShape);

        let mut record =
            record_with_default(0, 1, "local", "D", vec![speech_batch_profile_for("D")]);
        add_capacity(&mut record, CapacityClass::SpeechBatch, 2);
        let pool = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::SpeechBatch,
            vec![record],
        );
        let make_requirement = |models, batch_size| {
            RouteRequirement::new(
                ProfileRequirement::SpeechBatch {
                    models,
                    response_formats: vec![SpeechResponseFormat::Wav],
                    tasks: vec![SpeechTask::TextToSpeech],
                    reference_forms: vec![ReferenceForm::None],
                    managed_voice: false,
                    batch_size,
                    effective_features: vec![super::profile::BatchFeature::Model],
                },
                TrustDomain::new(String::from("local")),
                false,
            )
        };
        let ordered_models = make_requirement(
            vec![
                ModelSelection::Explicit(String::from("D")),
                ModelSelection::Explicit(String::from("D")),
            ],
            2,
        );
        assert!(ordered_models.profile.is_well_formed());

        let mut invalid = vec![
            (
                String::from("zero batch size"),
                make_requirement(Vec::new(), 0),
            ),
            (
                String::from("model cardinality"),
                make_requirement(vec![ModelSelection::Explicit(String::from("D"))], 2),
            ),
            (
                String::from("invalid model"),
                make_requirement(vec![ModelSelection::Explicit(String::new())], 1),
            ),
        ];
        let required_sets: [(&str, BatchRequirementMutation); 3] = [
            ("response_formats", |profile, shape| {
                let ProfileRequirement::SpeechBatch {
                    response_formats, ..
                } = profile
                else {
                    unreachable!()
                };
                *response_formats = invalid_set(SpeechResponseFormat::Wav, shape);
            }),
            ("tasks", |profile, shape| {
                let ProfileRequirement::SpeechBatch { tasks, .. } = profile else {
                    unreachable!()
                };
                *tasks = invalid_set(SpeechTask::TextToSpeech, shape);
            }),
            ("reference_forms", |profile, shape| {
                let ProfileRequirement::SpeechBatch {
                    reference_forms, ..
                } = profile
                else {
                    unreachable!()
                };
                *reference_forms = invalid_set(ReferenceForm::None, shape);
            }),
        ];
        for (field, mutate) in required_sets {
            for shape in [
                InvalidSetShape::Empty,
                InvalidSetShape::Duplicate,
                InvalidSetShape::Oversized,
            ] {
                let mut requirement =
                    make_requirement(vec![ModelSelection::Explicit(String::from("D"))], 1);
                mutate(&mut requirement.profile, shape);
                invalid.push((format!("{field} {shape:?}"), requirement));
            }
        }
        for shape in [InvalidSetShape::Duplicate, InvalidSetShape::Oversized] {
            let mut requirement =
                make_requirement(vec![ModelSelection::Explicit(String::from("D"))], 1);
            let ProfileRequirement::SpeechBatch {
                effective_features, ..
            } = &mut requirement.profile
            else {
                unreachable!()
            };
            *effective_features = invalid_set(super::profile::BatchFeature::Model, shape);
            invalid.push((format!("effective_features {shape:?}"), requirement));
        }

        for (case, requirement) in &invalid {
            assert!(
                !requirement.profile.is_well_formed(),
                "{case} must be rejected"
            );
            assert_eq!(
                pool.dispatch(
                    pool.try_admit(CapacityClass::SpeechBatch)
                        .expect("invalid batch admission"),
                    requirement,
                )
                .err(),
                Some(DispatchError::NoEligibleProfile),
                "{case} must fail before candidate scanning"
            );
        }

        let mut defaults_only =
            make_requirement(vec![ModelSelection::Explicit(String::from("D"))], 1);
        let ProfileRequirement::SpeechBatch {
            effective_features, ..
        } = &mut defaults_only.profile
        else {
            unreachable!()
        };
        effective_features.clear();
        assert!(defaults_only.profile.is_well_formed());
        let lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::SpeechBatch)
                    .expect("defaults-only batch admission"),
                &defaults_only,
            )
            .expect("defaults-only batch dispatch");
        assert_eq!(lease.worker_id().as_str(), "worker-0");
        drop(lease);

        assert_eq!(
            pool.admission.available(CapacityClass::SpeechBatch),
            (512, 512)
        );
        assert_eq!(
            pool.records[0]
                .slot(CapacityClass::SpeechBatch)
                .expect("batch slot")
                .semaphore
                .available_permits(),
            2
        );

        let admitted = pool
            .try_admit(CapacityClass::SpeechBatch)
            .expect("pre-drain invalid admission");
        pool.drain().expect("drain invalid batch fixture");
        assert_eq!(
            pool.dispatch(admitted, &invalid[0].1).err(),
            Some(DispatchError::Draining)
        );
    }

    #[test]
    fn invalid_generation_requirements_fail_once_before_candidate_scan() {
        let pool = fixture_pool(RoutingStrategy::RoundRobin, 2, 1);
        let mut empty_forms = generation_requirement();
        let mut duplicate_forms = generation_requirement();
        let mut misplaced_media = generation_requirement();
        let mut missing_audio_format = generation_requirement();
        let mut empty_model = generation_requirement();
        if let ProfileRequirement::GenerationHttp {
            message_content_forms,
            ..
        } = &mut empty_forms.profile
        {
            message_content_forms.clear();
        }
        if let ProfileRequirement::GenerationHttp {
            message_content_forms,
            ..
        } = &mut duplicate_forms.profile
        {
            *message_content_forms = vec![MessageContentForm::String; 2];
        }
        if let ProfileRequirement::GenerationHttp {
            media_placements, ..
        } = &mut misplaced_media.profile
        {
            *media_placements = vec![MediaPlacement::TopLevel];
        }
        if let ProfileRequirement::GenerationHttp {
            output_modalities,
            audio_format,
            ..
        } = &mut missing_audio_format.profile
        {
            *output_modalities = vec![OutputModality::Audio];
            *audio_format = None;
        }
        if let ProfileRequirement::GenerationHttp { model, .. } = &mut empty_model.profile {
            *model = ModelSelection::Explicit(String::new());
        }
        for requirement in [
            empty_forms,
            duplicate_forms,
            misplaced_media,
            missing_audio_format,
            empty_model,
        ] {
            assert!(!requirement.profile.is_well_formed());
            assert_eq!(
                pool.dispatch(
                    pool.try_admit(CapacityClass::GenerationHttp)
                        .expect("invalid generation admission"),
                    &requirement,
                )
                .err(),
                Some(DispatchError::NoEligibleProfile)
            );
        }
        assert_eq!(
            pool.records
                .iter()
                .map(|record| {
                    record
                        .slot(CapacityClass::GenerationHttp)
                        .expect("generation slot")
                        .semaphore
                        .available_permits()
                })
                .collect::<Vec<_>>(),
            [1, 1]
        );
    }

    #[test]
    fn worker_default_matching_and_realtime_readiness_fail_closed() {
        let alias = record_with_default(0, 1, "local", "other", vec![generation_profile_for("D")]);
        let pool = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::GenerationHttp,
            vec![alias],
        );
        let requirement = RouteRequirement::new(
            ProfileRequirement::GenerationHttp {
                model: ModelSelection::WorkerDefault {
                    expected_model_id: String::from("D"),
                },
                message_content_forms: vec![MessageContentForm::String],
                media_placements: Vec::new(),
                input_modalities: vec![InputModality::Text],
                output_modalities: vec![OutputModality::Text],
                audio_format: None,
                stream_mode: StreamMode::NonStreaming,
            },
            TrustDomain::new(String::from("local")),
            false,
        );
        assert_eq!(
            pool.dispatch(
                pool.try_admit(CapacityClass::GenerationHttp)
                    .expect("defaulted admission"),
                &requirement,
            )
            .err(),
            Some(DispatchError::NoEligibleProfile)
        );

        let realtime_profile = ServiceProfile::RealtimeWebsocket {
            protocols: vec![super::profile::RealtimeProtocol::OpenaiRealtimeV1],
        };
        let realtime_record = |id, trust: &str, default: &str| {
            record_with_default_class(
                id,
                CapacityClass::RealtimeWebsocket,
                1,
                trust,
                default,
                vec![realtime_profile.clone()],
            )
        };
        let ambiguous = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::RealtimeWebsocket,
            vec![
                realtime_record(0, "local", "A"),
                realtime_record(1, "local", "B"),
                realtime_record(2, "remote", "R"),
                realtime_record(3, "remote", "R"),
            ],
        );
        assert!(!ambiguous.realtime_defaults_unambiguous);
        assert!(!ambiguous.is_ready());
        assert_eq!(
            ambiguous.resolve_default_model_id(
                &TrustDomain::new(String::from("local")),
                ServiceClass::RealtimeWebsocket,
            ),
            super::DefaultModelResolution::Ambiguous
        );
        assert_eq!(
            ambiguous.resolve_default_model_id(
                &TrustDomain::new(String::from("remote")),
                ServiceClass::RealtimeWebsocket,
            ),
            super::DefaultModelResolution::Unique("R")
        );

        let unique = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::RealtimeWebsocket,
            vec![
                realtime_record(0, "local", "A"),
                realtime_record(1, "local", "A"),
                realtime_record(2, "remote", "R"),
                realtime_record(3, "remote", "R"),
            ],
        );
        assert!(unique.realtime_defaults_unambiguous);
        assert!(unique.is_ready());
        assert_eq!(
            unique.resolve_default_model_id(
                &TrustDomain::new(String::from("local")),
                ServiceClass::RealtimeWebsocket,
            ),
            super::DefaultModelResolution::Unique("A")
        );
        let realtime = RouteRequirement::new(
            ProfileRequirement::RealtimeWebsocket {
                protocol: super::profile::RealtimeProtocol::OpenaiRealtimeV1,
                expected_default_model_id: String::from("A"),
            },
            TrustDomain::new(String::from("local")),
            false,
        );
        let lease = unique
            .dispatch(
                unique
                    .try_admit(CapacityClass::RealtimeWebsocket)
                    .expect("realtime admission"),
                &realtime,
            )
            .expect("unique realtime default dispatches");
        assert_eq!(lease.worker_id().as_str(), "worker-0");
    }

    #[test]
    fn homogeneous_generation_proof_is_exact_and_dispatches_without_a_requirement() {
        for size in [0_usize, 1, 2, 64, 256] {
            let records = (0..size)
                .map(|id| {
                    record_with_default(
                        id,
                        1 + (id % 2),
                        "local",
                        "omni",
                        vec![generation_profile()],
                    )
                })
                .collect();
            let pool = pool_with_records(
                RoutingStrategy::RoundRobin,
                ServiceClass::GenerationHttp,
                records,
            );
            let trust = TrustDomain::new(String::from("local"));
            assert_eq!(pool.homogeneous_generation_http(&trust).is_some(), size > 0);
            if size > 0 {
                let lease = pool
                    .homogeneous_generation_http(&trust)
                    .expect("homogeneous proof")
                    .dispatch(
                        pool.try_admit(CapacityClass::GenerationHttp)
                            .expect("homogeneous admission"),
                    )
                    .expect("requirement-free homogeneous dispatch");
                assert!(lease.worker_id().as_str().starts_with("worker-"));
            }
        }

        let mut reversed = generation_profile();
        if let ServiceProfile::GenerationHttp { stream_modes, .. } = &mut reversed {
            *stream_modes = vec![StreamMode::Streaming, StreamMode::NonStreaming];
        }
        let mut matching = generation_profile();
        if let ServiceProfile::GenerationHttp { stream_modes, .. } = &mut matching {
            *stream_modes = vec![StreamMode::NonStreaming, StreamMode::Streaming];
        }
        let text = generation_profile();
        let homogeneous = pool_with_records(
            RoutingStrategy::LeastRequests,
            ServiceClass::GenerationHttp,
            vec![
                record_with_default(0, 1, "local", "omni", vec![text.clone(), reversed]),
                record_with_default(1, 3, "local", "omni", vec![matching, text]),
                record_with_default(2, 1, "local", "ignored", vec![ServiceProfile::VoiceControl]),
            ],
        );
        let local = TrustDomain::new(String::from("local"));
        assert!(homogeneous.homogeneous_generation_http(&local).is_some());

        let different_default = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::GenerationHttp,
            vec![
                record_with_default(0, 1, "local", "A", vec![generation_profile_for("A")]),
                record_with_default(1, 1, "local", "B", vec![generation_profile_for("B")]),
            ],
        );
        assert!(
            different_default
                .homogeneous_generation_http(&local)
                .is_none()
        );

        let differences: [(&str, ProfileMutation); 7] = [
            ("model_ids", |profile| {
                let ServiceProfile::GenerationHttp { model_ids, .. } = profile else {
                    unreachable!()
                };
                model_ids.push(String::from("alias"));
            }),
            ("message_content_forms", |profile| {
                let ServiceProfile::GenerationHttp {
                    message_content_forms,
                    ..
                } = profile
                else {
                    unreachable!()
                };
                *message_content_forms = vec![MessageContentForm::TypedParts];
            }),
            ("media_placements", |profile| {
                let ServiceProfile::GenerationHttp {
                    media_placements, ..
                } = profile
                else {
                    unreachable!()
                };
                *media_placements = vec![MediaPlacement::TypedParts];
            }),
            ("input_modalities", |profile| {
                let ServiceProfile::GenerationHttp {
                    input_modalities, ..
                } = profile
                else {
                    unreachable!()
                };
                *input_modalities = vec![InputModality::Text, InputModality::Audio];
            }),
            ("output_modalities", |profile| {
                let ServiceProfile::GenerationHttp {
                    output_modalities, ..
                } = profile
                else {
                    unreachable!()
                };
                *output_modalities = vec![OutputModality::Audio];
            }),
            ("chat_audio_formats", |profile| {
                let ServiceProfile::GenerationHttp {
                    chat_audio_formats, ..
                } = profile
                else {
                    unreachable!()
                };
                *chat_audio_formats = vec![ChatAudioFormat::Mp3];
            }),
            ("stream_modes", |profile| {
                let ServiceProfile::GenerationHttp { stream_modes, .. } = profile else {
                    unreachable!()
                };
                *stream_modes = vec![StreamMode::Streaming];
            }),
        ];
        for (field, mutate) in differences {
            let baseline = rich_generation_profile();
            let mut different = baseline.clone();
            mutate(&mut different);
            let heterogeneous = pool_with_records(
                RoutingStrategy::RoundRobin,
                ServiceClass::GenerationHttp,
                vec![
                    record_with_default(0, 1, "local", "omni", vec![baseline]),
                    record_with_default(1, 1, "local", "omni", vec![different]),
                ],
            );
            assert!(
                heterogeneous.homogeneous_generation_http(&local).is_none(),
                "changing {field} must disable homogeneity"
            );
        }

        let independent = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::GenerationHttp,
            vec![
                record_with_default(0, 1, "local", "L", vec![generation_profile_for("L")]),
                record_with_default(1, 1, "local", "L", vec![generation_profile_for("L")]),
                record_with_default(2, 1, "remote", "R", vec![generation_profile_for("R")]),
                record_with_default(3, 1, "remote", "R", vec![generation_profile_for("R")]),
            ],
        );
        let remote = TrustDomain::new(String::from("remote"));
        let local_lease = independent
            .homogeneous_generation_http(&local)
            .expect("local homogeneous proof")
            .dispatch(
                independent
                    .try_admit(CapacityClass::GenerationHttp)
                    .expect("local homogeneous admission"),
            )
            .expect("local homogeneous dispatch");
        let remote_lease = independent
            .homogeneous_generation_http(&remote)
            .expect("remote homogeneous proof")
            .dispatch(
                independent
                    .try_admit(CapacityClass::GenerationHttp)
                    .expect("remote homogeneous admission"),
            )
            .expect("remote homogeneous dispatch");
        assert_eq!(local_lease.worker_id().as_str(), "worker-0");
        assert_eq!(remote_lease.worker_id().as_str(), "worker-2");
    }

    #[test]
    fn heterogeneous_generation_scope_does_not_block_a_later_independent_scope() {
        let pool = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::GenerationHttp,
            vec![
                record_with_default(0, 1, "local", "A", vec![generation_profile_for("A")]),
                record_with_default(1, 1, "local", "B", vec![generation_profile_for("B")]),
                record_with_default(2, 1, "remote", "R", vec![generation_profile_for("R")]),
                record_with_default(3, 1, "remote", "R", vec![generation_profile_for("R")]),
            ],
        );

        assert!(
            pool.homogeneous_generation_http(&TrustDomain::new(String::from("local")))
                .is_none()
        );
        assert!(
            pool.homogeneous_generation_http(&TrustDomain::new(String::from("remote")))
                .is_some()
        );
    }

    #[test]
    fn homogeneous_dispatch_uses_deterministic_existing_policies() {
        let trust = TrustDomain::new(String::from("local"));
        let round_robin = fixture_pool(RoutingStrategy::RoundRobin, 3, 1);
        let mut sequence = Vec::new();
        for _ in 0..4 {
            let lease = round_robin
                .homogeneous_generation_http(&trust)
                .expect("round-robin homogeneous proof")
                .dispatch(
                    round_robin
                        .try_admit(CapacityClass::GenerationHttp)
                        .expect("round-robin admission"),
                )
                .expect("round-robin homogeneous dispatch");
            sequence.push(lease.worker_id().as_str().to_owned());
        }
        assert_eq!(sequence, ["worker-0", "worker-1", "worker-2", "worker-0"]);

        let least = pool_with_records(
            RoutingStrategy::LeastRequests,
            ServiceClass::GenerationHttp,
            vec![
                record_with_default(0, 1, "local", "omni", vec![generation_profile()]),
                record_with_default(1, 3, "local", "omni", vec![generation_profile()]),
            ],
        );
        let first = least
            .homogeneous_generation_http(&trust)
            .expect("least-requests homogeneous proof")
            .dispatch(
                least
                    .try_admit(CapacityClass::GenerationHttp)
                    .expect("least-requests admission"),
            )
            .expect("first least-requests dispatch");
        assert_eq!(first.worker_id().as_str(), "worker-0");
        drop(first);
        let second = least
            .homogeneous_generation_http(&trust)
            .expect("least-requests homogeneous proof")
            .dispatch(
                least
                    .try_admit(CapacityClass::GenerationHttp)
                    .expect("least-requests tie admission"),
            )
            .expect("second least-requests dispatch");
        assert_eq!(second.worker_id().as_str(), "worker-1");
        drop(second);
        let saturated = Arc::clone(
            &least.records[0]
                .slot(CapacityClass::GenerationHttp)
                .expect("generation slot")
                .semaphore,
        )
        .try_acquire_owned()
        .expect("saturate smaller worker");
        let remaining = least
            .homogeneous_generation_http(&trust)
            .expect("least-requests homogeneous proof")
            .dispatch(
                least
                    .try_admit(CapacityClass::GenerationHttp)
                    .expect("saturated least-requests admission"),
            )
            .expect("remaining worker dispatch");
        assert_eq!(remaining.worker_id().as_str(), "worker-1");
        drop(remaining);
        drop(saturated);
        assert_eq!(
            least
                .records
                .iter()
                .map(|record| {
                    record
                        .slot(CapacityClass::GenerationHttp)
                        .expect("generation slot")
                        .semaphore
                        .available_permits()
                })
                .collect::<Vec<_>>(),
            [1, 3]
        );
    }

    #[test]
    fn homogeneous_dispatch_preserves_normal_error_and_permit_semantics() {
        let trust = TrustDomain::new(String::from("local"));
        let success = fixture_pool(RoutingStrategy::RoundRobin, 1, 1);
        let lease = success
            .homogeneous_generation_http(&trust)
            .expect("homogeneous proof")
            .dispatch(
                success
                    .try_admit(CapacityClass::GenerationHttp)
                    .expect("success admission"),
            )
            .expect("homogeneous success");
        assert_eq!(
            success.admission.available(CapacityClass::GenerationHttp),
            (511, 511)
        );
        assert_eq!(
            success.records[0]
                .slot(CapacityClass::GenerationHttp)
                .expect("generation slot")
                .semaphore
                .available_permits(),
            0
        );
        drop(lease);
        assert_eq!(
            success.admission.available(CapacityClass::GenerationHttp),
            (512, 512)
        );
        assert_eq!(
            success.records[0]
                .slot(CapacityClass::GenerationHttp)
                .expect("generation slot")
                .semaphore
                .available_permits(),
            1
        );

        let unavailable = fixture_pool(RoutingStrategy::RoundRobin, 1, 1);
        unavailable.records[0].health.store(HealthState::Unhealthy);
        assert_eq!(
            unavailable
                .homogeneous_generation_http(&trust)
                .expect("homogeneous proof")
                .dispatch(
                    unavailable
                        .try_admit(CapacityClass::GenerationHttp)
                        .expect("unavailable admission"),
                )
                .err(),
            Some(DispatchError::Unavailable)
        );
        assert_eq!(
            unavailable
                .admission
                .available(CapacityClass::GenerationHttp),
            (512, 512)
        );
        assert_eq!(
            unavailable.records[0]
                .slot(CapacityClass::GenerationHttp)
                .expect("generation slot")
                .semaphore
                .available_permits(),
            1
        );

        let overloaded = fixture_pool(RoutingStrategy::LeastRequests, 1, 1);
        let exact = Arc::clone(
            &overloaded.records[0]
                .slot(CapacityClass::GenerationHttp)
                .expect("generation slot")
                .semaphore,
        )
        .try_acquire_owned()
        .expect("saturate exact permit");
        assert_eq!(
            overloaded
                .homogeneous_generation_http(&trust)
                .expect("homogeneous proof")
                .dispatch(
                    overloaded
                        .try_admit(CapacityClass::GenerationHttp)
                        .expect("overloaded admission"),
                )
                .err(),
            Some(DispatchError::Overloaded)
        );
        assert_eq!(
            overloaded
                .admission
                .available(CapacityClass::GenerationHttp),
            (512, 512)
        );
        assert_eq!(
            overloaded.records[0]
                .slot(CapacityClass::GenerationHttp)
                .expect("generation slot")
                .semaphore
                .available_permits(),
            0
        );
        drop(exact);
        assert_eq!(
            overloaded.records[0]
                .slot(CapacityClass::GenerationHttp)
                .expect("generation slot")
                .semaphore
                .available_permits(),
            1
        );

        let draining = fixture_pool(RoutingStrategy::RoundRobin, 1, 1);
        let admitted = draining
            .try_admit(CapacityClass::GenerationHttp)
            .expect("pre-drain admission");
        draining.drain().expect("drain homogeneous fixture");
        assert_eq!(
            draining
                .homogeneous_generation_http(&trust)
                .expect("homogeneous proof remains immutable")
                .dispatch(admitted)
                .err(),
            Some(DispatchError::Draining)
        );
        assert_eq!(
            draining.admission.available(CapacityClass::GenerationHttp),
            (512, 512)
        );
        assert_eq!(
            draining.records[0]
                .slot(CapacityClass::GenerationHttp)
                .expect("generation slot")
                .semaphore
                .available_permits(),
            1
        );

        let mismatch = fixture_pool(RoutingStrategy::RoundRobin, 1, 1);
        assert_eq!(
            mismatch
                .homogeneous_generation_http(&trust)
                .expect("homogeneous proof")
                .dispatch(
                    mismatch
                        .try_admit(CapacityClass::SpeechHttp)
                        .expect("wrong-class admission"),
                )
                .err(),
            Some(DispatchError::AdmissionClassMismatch)
        );
        assert_eq!(
            mismatch.admission.available(CapacityClass::SpeechHttp),
            (512, 512)
        );
        assert_eq!(
            mismatch.records[0]
                .slot(CapacityClass::GenerationHttp)
                .expect("generation slot")
                .semaphore
                .available_permits(),
            1
        );
    }

    #[test]
    fn media_cohorts_are_exactly_precomputed_by_service_and_trust() {
        let local = TrustDomain::new(String::from("local"));
        let remote = TrustDomain::new(String::from("remote"));
        let baseline = media_speech_profile(vec![ReferenceForm::None], false);

        let different_default = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::SpeechHttp,
            vec![
                record_with_default_class(
                    0,
                    CapacityClass::SpeechHttp,
                    1,
                    "local",
                    "tts",
                    vec![baseline.clone()],
                ),
                record_with_default_class(
                    1,
                    CapacityClass::SpeechHttp,
                    1,
                    "local",
                    "other",
                    vec![baseline.clone()],
                ),
            ],
        );
        assert!(
            different_default
                .homogeneous_media_http(&local, ServiceClass::SpeechHttp)
                .is_none()
        );

        let different_profile = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::SpeechHttp,
            vec![
                record_with_default_class(
                    0,
                    CapacityClass::SpeechHttp,
                    1,
                    "local",
                    "tts",
                    vec![baseline.clone()],
                ),
                record_with_default_class(
                    1,
                    CapacityClass::SpeechHttp,
                    1,
                    "local",
                    "tts",
                    vec![media_speech_profile(
                        vec![ReferenceForm::None, ReferenceForm::Direct],
                        false,
                    )],
                ),
            ],
        );
        assert!(
            different_profile
                .homogeneous_media_http(&local, ServiceClass::SpeechHttp)
                .is_none()
        );

        let mut no_default = record_with_default_class(
            0,
            CapacityClass::SpeechHttp,
            1,
            "local",
            "tts",
            vec![baseline.clone()],
        );
        Arc::get_mut(&mut no_default)
            .expect("unique no-default record")
            .default_model_id = None;
        let no_default = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::SpeechHttp,
            vec![no_default],
        );
        assert!(
            no_default
                .homogeneous_media_http(&local, ServiceClass::SpeechHttp)
                .is_none()
        );

        let managed = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::SpeechHttp,
            vec![record_with_default_class(
                0,
                CapacityClass::SpeechHttp,
                1,
                "local",
                "tts",
                vec![media_speech_profile(vec![ReferenceForm::None], true)],
            )],
        );
        assert!(
            managed
                .homogeneous_media_http(&local, ServiceClass::SpeechHttp)
                .is_none()
        );

        let scoped = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::SpeechHttp,
            vec![
                record_with_default_class(
                    0,
                    CapacityClass::SpeechHttp,
                    1,
                    "local",
                    "tts",
                    vec![
                        baseline.clone(),
                        transcription_profile(vec![
                            super::profile::TranscriptionResponseFormat::Json,
                        ]),
                    ],
                ),
                record_with_default_class(
                    1,
                    CapacityClass::SpeechHttp,
                    1,
                    "local",
                    "tts",
                    vec![
                        baseline,
                        transcription_profile(vec![
                            super::profile::TranscriptionResponseFormat::Text,
                        ]),
                    ],
                ),
                record_with_default_class(
                    2,
                    CapacityClass::SpeechHttp,
                    1,
                    "remote",
                    "tts",
                    vec![media_speech_profile(vec![ReferenceForm::None], true)],
                ),
            ],
        );
        assert!(
            scoped
                .homogeneous_media_http(&local, ServiceClass::SpeechHttp)
                .is_some()
        );
        assert!(
            scoped
                .homogeneous_media_http(&local, ServiceClass::TranscriptionHttp)
                .is_none()
        );
        assert!(
            scoped
                .homogeneous_media_http(&remote, ServiceClass::SpeechHttp)
                .is_none()
        );
        assert!(
            scoped
                .homogeneous_media_http(&local, ServiceClass::SpeechBatch)
                .is_none()
        );
    }

    #[test]
    fn media_cohort_dispatch_uses_route_capacity_and_existing_policy() {
        let pool = pool_with_records(
            RoutingStrategy::LeastRequests,
            ServiceClass::SpeechHttp,
            vec![
                record_with_default_class(
                    0,
                    CapacityClass::SpeechHttp,
                    1,
                    "local",
                    "tts",
                    vec![media_speech_profile(vec![ReferenceForm::None], false)],
                ),
                record_with_default_class(
                    1,
                    CapacityClass::SpeechHttp,
                    3,
                    "local",
                    "tts",
                    vec![media_speech_profile(vec![ReferenceForm::None], false)],
                ),
            ],
        );
        let trust = TrustDomain::new(String::from("local"));
        let first = pool
            .homogeneous_media_http(&trust, ServiceClass::SpeechHttp)
            .expect("precomputed speech cohort")
            .dispatch(
                pool.try_admit(CapacityClass::SpeechHttp)
                    .expect("first speech admission"),
            )
            .expect("first media cohort dispatch");
        assert_eq!(first.worker_id().as_str(), "worker-0");
        assert_eq!(first.capacity_class(), CapacityClass::SpeechHttp);
        let second = pool
            .homogeneous_media_http(&trust, ServiceClass::SpeechHttp)
            .expect("reused speech cohort")
            .dispatch(
                pool.try_admit(CapacityClass::SpeechHttp)
                    .expect("second speech admission"),
            )
            .expect("least occupied worker dispatch");
        assert_eq!(second.worker_id().as_str(), "worker-1");
        assert_eq!(second.capacity_class(), CapacityClass::SpeechHttp);
        assert_eq!(
            pool.records
                .iter()
                .map(|record| {
                    record
                        .slot(CapacityClass::SpeechHttp)
                        .expect("speech capacity")
                        .semaphore
                        .available_permits()
                })
                .collect::<Vec<_>>(),
            [0, 2]
        );
        drop((first, second));
    }

    #[test]
    fn mixed_batch_reference_union_excludes_workers_missing_any_member() {
        let batch_profile = |reference_forms| ServiceProfile::SpeechBatch {
            model_ids: vec![String::from("tts")],
            response_formats: vec![SpeechResponseFormat::Wav],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms,
            managed_voice: false,
            max_batch_size: 4,
            effective_features: vec![super::profile::BatchFeature::Reference],
        };
        let pool = pool_with_records(
            RoutingStrategy::RoundRobin,
            ServiceClass::SpeechBatch,
            vec![
                record_with_default_class(
                    0,
                    CapacityClass::SpeechBatch,
                    1,
                    "local",
                    "tts",
                    vec![batch_profile(vec![ReferenceForm::Direct])],
                ),
                record_with_default_class(
                    1,
                    CapacityClass::SpeechBatch,
                    1,
                    "local",
                    "tts",
                    vec![batch_profile(vec![
                        ReferenceForm::None,
                        ReferenceForm::Direct,
                    ])],
                ),
            ],
        );
        let requirement = RouteRequirement::new(
            ProfileRequirement::SpeechBatch {
                models: vec![
                    ModelSelection::WorkerDefault {
                        expected_model_id: String::from("tts"),
                    },
                    ModelSelection::WorkerDefault {
                        expected_model_id: String::from("tts"),
                    },
                ],
                response_formats: vec![SpeechResponseFormat::Wav],
                tasks: vec![SpeechTask::TextToSpeech],
                reference_forms: vec![ReferenceForm::None, ReferenceForm::Direct],
                managed_voice: false,
                batch_size: 2,
                effective_features: vec![super::profile::BatchFeature::Reference],
            },
            TrustDomain::new(String::from("local")),
            false,
        );
        let lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::SpeechBatch)
                    .expect("mixed-reference batch admission"),
                &requirement,
            )
            .expect("complete mixed-reference worker dispatch");
        assert_eq!(lease.worker_id().as_str(), "worker-1");
    }
}
