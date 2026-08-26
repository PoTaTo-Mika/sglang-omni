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
pub(crate) use permit::{AdmissionError, AdmissionLease, DispatchError, RequestLease};
pub(crate) use profile::{
    ChatAudioFormat, MediaPlacement, MessageContentForm, ModelSelection, ProfileRequirement,
    RouteRequirement, ServiceClass, TrustDomain,
};
pub(crate) use resolver::ResolvedTarget;

use health::HealthCell;
use permit::{AdmissionController, Gate};
use profile::{MAX_WORKERS, RegistrationId, ServiceProfile, WorkerCapacityConfig, WorkerId};
use resolver::{StaticResolver, build_generation_client, build_health_client};
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
/// and exact generation-capacity owners.
pub(super) struct WorkerRecord {
    worker_id: WorkerId,
    default_model_id: String,
    registration_id: RegistrationId,
    target: ResolvedTarget,
    trust_domain: TrustDomain,
    profiles: Vec<ServiceProfile>,
    capacity: CapacitySlot,
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

    fn occupancy_snapshot(&self) -> usize {
        self.capacity.limit - self.capacity.semaphore.available_permits()
    }

    fn has_profile(&self, requirement: &RouteRequirement) -> bool {
        self.profiles
            .iter()
            .any(|profile| profile.matches(&requirement.profile, &self.default_model_id))
    }

    fn available_for_dispatch(&self) -> bool {
        self.health.load() == HealthState::Healthy && self.disposition() == Disposition::Serving
    }
}

/// Immutable generation worker pool with bounded admission, exact capacity,
/// deterministic policy state, and independently owned health.
pub(crate) struct WorkerPool {
    records: Vec<Arc<WorkerRecord>>,
    gate: Arc<RwLock<Gate>>,
    admission: AdmissionController,
    selector: Selector,
    homogeneous_generation_http: Vec<HomogeneousGenerationCohort>,
    health_client: reqwest::Client,
    generation_client: reqwest::Client,
}

struct HomogeneousGenerationCohort {
    trust_domain: TrustDomain,
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
        let generation_client = build_generation_client(
            resolver,
            config.http_generation.connect_timeout(),
            config.http_generation.pool_idle_timeout(),
            config.http_generation.pool_max_idle_per_host,
        )
        .map_err(crate::error::RouterError::GenerationClient)?;
        let gate = Arc::new(RwLock::new(Gate::open()));
        let admission = AdmissionController::new(
            Arc::clone(&gate),
            usize::try_from(config.admission.global)
                .map_err(|_| crate::error::RouterError::WorkerPoolInvariant)?,
            usize::try_from(config.admission.generation_http)
                .map_err(|_| crate::error::RouterError::WorkerPoolInvariant)?,
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
        Ok(Self {
            records,
            gate,
            admission,
            selector: Selector::new(config.router.strategy),
            homogeneous_generation_http,
            health_client,
            generation_client,
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

    pub(crate) fn generation_client(&self) -> reqwest::Client {
        self.generation_client.clone()
    }

    pub(crate) fn try_admit(&self) -> Result<AdmissionLease, AdmissionError> {
        self.admission.try_admit()
    }

    pub(crate) fn dispatch(
        &self,
        admission: AdmissionLease,
        requirement: &RouteRequirement,
    ) -> Result<RequestLease, DispatchError> {
        if !requirement.profile.is_well_formed() {
            return Err(DispatchError::NoEligibleProfile);
        }
        let profile_found = self
            .records
            .iter()
            .any(|record| record.has_profile(requirement));
        self.dispatch_matching(admission, profile_found, |record| {
            &record.trust_domain == requirement.trust_domain() && record.has_profile(requirement)
        })
    }

    fn dispatch_matching(
        &self,
        admission: AdmissionLease,
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
                self.reserve_round_robin(start, &matches)
            }
            RoutingStrategy::LeastRequests => {
                let start = self.selector.start(self.records.len());
                self.reserve_least_requests(start, &matches)
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
                    && let Ok(exact) = Arc::clone(&record.capacity.semaphore).try_acquire_owned()
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
        matches: &impl Fn(&WorkerRecord) -> bool,
    ) -> Option<(Arc<WorkerRecord>, tokio::sync::OwnedSemaphorePermit)> {
        let mut snapshots = [usize::MAX; MAX_WORKERS];
        let mut attempted = [false; MAX_WORKERS];
        for (index, record) in self.records.iter().enumerate() {
            if matches(record) && record.available_for_dispatch() {
                *snapshots.get_mut(index)? = record.occupancy_snapshot();
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
            if let Ok(exact) = Arc::clone(&record.capacity.semaphore).try_acquire_owned() {
                return Some((Arc::clone(record), exact));
            }
        }
        None
    }

    pub(crate) fn resolve_default_model_id(
        &self,
        trust: &TrustDomain,
        _service: ServiceClass,
    ) -> DefaultModelResolution<'_> {
        let mut resolved = None;
        for record in &self.records {
            if &record.trust_domain != trust {
                continue;
            }
            match resolved {
                None => resolved = Some(record.default_model_id.as_str()),
                Some(current) if current == record.default_model_id => {}
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

    pub(crate) fn generation_http_ready(&self, trust: &TrustDomain) -> bool {
        self.gate.read().is_ok_and(|gate| {
            gate.open
                && self
                    .records
                    .iter()
                    .any(|record| &record.trust_domain == trust && record.available_for_dispatch())
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
            record.capacity.semaphore.close();
        }
        Ok(())
    }
}

impl ContentBlindGenerationHttp<'_> {
    pub(crate) fn dispatch(self, admission: AdmissionLease) -> Result<RequestLease, DispatchError> {
        self.pool
            .dispatch_matching(admission, true, |record| &record.trust_domain == self.trust)
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
        let mut members = records
            .iter()
            .filter(|candidate| candidate.trust_domain == record.trust_domain);
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

fn generation_rows_equal(left: &[ServiceProfile], right: &[ServiceProfile]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .all(|profile| right.iter().any(|other| profile.semantically_eq(other)))
}

fn build_capacity(
    config: &WorkerCapacityConfig,
) -> Result<CapacitySlot, crate::error::RouterError> {
    let limit = usize::try_from(config.generation_http)
        .map_err(|_| crate::error::RouterError::WorkerPoolInvariant)?;
    Ok(CapacitySlot {
        limit,
        semaphore: Arc::new(Semaphore::new(limit)),
    })
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
            default_model_id: model.to_owned(),
            registration_id: RegistrationId::from_startup_ordinal(ordinal),
            target: ResolvedTarget::from_parts(
                &format!("http://127.0.0.1:{}/", 10_000 + ordinal),
                "/health",
                None,
            )
            .expect("test target"),
            trust_domain: TrustDomain::new(trust.to_owned()),
            profiles: vec![service_profile],
            capacity: CapacitySlot {
                limit,
                semaphore: Arc::new(Semaphore::new(limit)),
            },
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
            records,
            gate: Arc::clone(&gate),
            admission: AdmissionController::new(gate, admission, admission),
            selector: Selector::new(strategy),
            health_client: client.clone(),
            generation_client: client,
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
            |ServiceProfile::GenerationHttp { model_ids, .. }| {
                model_ids.push(String::from("other"));
            },
            |ServiceProfile::GenerationHttp {
                 message_content_forms,
                 ..
             }| {
                message_content_forms.push(MessageContentForm::TypedParts);
            },
            |ServiceProfile::GenerationHttp {
                 media_placements, ..
             }| {
                media_placements.push(MediaPlacement::TypedParts);
            },
            |ServiceProfile::GenerationHttp {
                 input_modalities, ..
             }| {
                input_modalities.push(InputModality::Image);
            },
            |ServiceProfile::GenerationHttp {
                 output_modalities,
                 chat_audio_formats,
                 ..
             }| {
                output_modalities.push(OutputModality::Audio);
                chat_audio_formats.push(ChatAudioFormat::Wav);
            },
            |ServiceProfile::GenerationHttp { stream_modes, .. }| {
                stream_modes.push(StreamMode::Streaming);
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
                pool.try_admit().expect("admit first"),
                &requirement("omni", "local"),
            )
            .expect("first dispatch");
        let second = pool
            .dispatch(
                pool.try_admit().expect("admit second"),
                &requirement("omni", "local"),
            )
            .expect("second dispatch");
        assert_ne!(first.registration_ordinal(), second.registration_ordinal());
        drop(first);
        drop(second);
        records[0].health.store(HealthState::Unhealthy);
        records[1].mark_draining();
        let unavailable = pool.dispatch(
            pool.try_admit().expect("admit unavailable"),
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
                    pool.try_admit().expect("admit sparse round robin"),
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
                let admission = pool.try_admit().expect("concurrent admission");
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
        let ServiceProfile::GenerationHttp {
            message_content_forms,
            media_placements,
            input_modalities,
            ..
        } = &mut multimodal;
        message_content_forms.push(MessageContentForm::TypedParts);
        media_placements.push(MediaPlacement::TypedParts);
        input_modalities.push(InputModality::Image);
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
                pool.try_admit().expect("admit heterogeneous request"),
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
                pool.try_admit().expect("admit"),
                &requirement("omni", "local"),
            )
            .expect("dispatch");
        assert_eq!(pool.admission.available(), (0, 0));
        assert_eq!(pool.records[0].capacity.semaphore.available_permits(), 0);
        drop(lease);
        assert_eq!(pool.admission.available(), (1, 1));
        assert_eq!(pool.records[0].capacity.semaphore.available_permits(), 1);
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
}
