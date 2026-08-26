use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use thiserror::Error;
use tokio::sync::{Semaphore, watch};
use tokio::task::{JoinError, JoinSet};

use super::WorkerRecord;

/// Probe health independent of serving/draining disposition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum HealthState {
    Unknown = 0,
    Healthy = 1,
    Unhealthy = 2,
}

impl HealthState {
    fn from_atomic(value: u8) -> Self {
        match value {
            1 => Self::Healthy,
            2 => Self::Unhealthy,
            _ => Self::Unknown,
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Healthy => "healthy",
            Self::Unhealthy => "unhealthy",
        }
    }
}

/// Atomic health cell. Release/Acquire publishes only the state transition;
/// capability and capacity remain immutable/semaphore-owned respectively.
pub(super) struct HealthCell(AtomicU8);

impl HealthCell {
    pub(super) const fn unknown() -> Self {
        Self(AtomicU8::new(HealthState::Unknown as u8))
    }

    pub(super) fn load(&self) -> HealthState {
        HealthState::from_atomic(self.0.load(Ordering::Acquire))
    }

    pub(super) fn store(&self, state: HealthState) {
        self.0.store(state as u8, Ordering::Release);
    }
}

#[derive(Debug, Error)]
pub(crate) enum HealthTaskError {
    #[error("health cancellation owner disappeared")]
    CancellationOwner,
    #[error("health probe semaphore closed unexpectedly")]
    ProbeBudgetClosed,
}

pub(crate) struct HealthSupervisor {
    shutdown: watch::Sender<bool>,
    tasks: JoinSet<Result<(), HealthTaskError>>,
}

impl HealthSupervisor {
    pub(super) fn start(
        records: &[Arc<WorkerRecord>],
        client: reqwest::Client,
        interval: Duration,
        success_threshold: u8,
        failure_threshold: u8,
        max_concurrent_probes: usize,
    ) -> Self {
        let (shutdown, receiver) = watch::channel(false);
        let budget = Arc::new(Semaphore::new(max_concurrent_probes));
        let mut tasks = JoinSet::new();
        for record in records {
            tasks.spawn(run_worker_health(
                Arc::clone(record),
                client.clone(),
                Arc::clone(&budget),
                receiver.clone(),
                interval,
                success_threshold,
                failure_threshold,
            ));
        }
        Self { shutdown, tasks }
    }

    pub(crate) fn cancel(&self) {
        let _receivers = self.shutdown.send(true);
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    pub(crate) async fn join_next(
        &mut self,
    ) -> Option<Result<Result<(), HealthTaskError>, JoinError>> {
        self.tasks.join_next().await
    }

    pub(crate) async fn abort_and_join_all(&mut self) {
        self.tasks.abort_all();
        while self.tasks.join_next().await.is_some() {}
    }
}

async fn run_worker_health(
    record: Arc<WorkerRecord>,
    client: reqwest::Client,
    budget: Arc<Semaphore>,
    mut shutdown: watch::Receiver<bool>,
    interval: Duration,
    success_threshold: u8,
    failure_threshold: u8,
) -> Result<(), HealthTaskError> {
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut tracker = HealthTracker::new(success_threshold, failure_threshold);

    loop {
        tokio::select! {
            biased;
            changed = shutdown.changed() => {
                match changed {
                    Ok(()) if *shutdown.borrow() => return Ok(()),
                    Ok(()) => continue,
                    Err(_) => return Err(HealthTaskError::CancellationOwner),
                }
            }
            () = record.immediate_probe.notified() => {}
            _ = ticker.tick() => {}
        }

        let permit = tokio::select! {
            biased;
            changed = shutdown.changed() => {
                match changed {
                    Ok(()) if *shutdown.borrow() => return Ok(()),
                    Ok(()) => continue,
                    Err(_) => return Err(HealthTaskError::CancellationOwner),
                }
            }
            result = Arc::clone(&budget).acquire_owned() => {
                result.map_err(|_| HealthTaskError::ProbeBudgetClosed)?
            }
        };

        let success = tokio::select! {
            biased;
            changed = shutdown.changed() => {
                drop(permit);
                match changed {
                    Ok(()) if *shutdown.borrow() => return Ok(()),
                    Ok(()) => continue,
                    Err(_) => return Err(HealthTaskError::CancellationOwner),
                }
            }
            result = client.get(record.target.health_url().clone()).send() => {
                result.is_ok_and(|response| response.status().is_success())
            }
        };
        drop(permit);
        let previous = record.health.load();
        let next = tracker.observe(success);
        record.health.store(next);
        if previous != next {
            tracing::info!(
                worker_id = record.worker_id.as_str(),
                health = ?next,
                "worker health changed"
            );
        }
    }
}

struct HealthTracker {
    successes: u8,
    failures: u8,
    success_threshold: u8,
    failure_threshold: u8,
    state: HealthState,
}

impl HealthTracker {
    fn new(success_threshold: u8, failure_threshold: u8) -> Self {
        Self {
            successes: 0,
            failures: 0,
            success_threshold,
            failure_threshold,
            state: HealthState::Unknown,
        }
    }

    fn observe(&mut self, success: bool) -> HealthState {
        if success {
            self.failures = 0;
            self.successes = self.successes.saturating_add(1);
            if self.successes >= self.success_threshold {
                self.state = HealthState::Healthy;
            }
        } else {
            self.successes = 0;
            self.failures = self.failures.saturating_add(1);
            if self.failures >= self.failure_threshold {
                self.state = HealthState::Unhealthy;
            }
        }
        self.state
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    use tokio::sync::{Notify, Semaphore};

    use super::{HealthCell, HealthState, HealthSupervisor, HealthTracker};
    use crate::worker_pool::profile::{
        InputModality, MessageContentForm, OutputModality, RegistrationId, ServiceProfile,
        StreamMode, TrustDomain, WorkerId,
    };
    use crate::worker_pool::resolver::{ResolvedTarget, StaticResolver, build_health_client};
    use crate::worker_pool::{CapacitySlot, Disposition, WorkerRecord};

    fn test_target(address: SocketAddr) -> ResolvedTarget {
        ResolvedTarget::from_parts(&format!("http://{address}/"), "/health", None)
            .expect("test target must build")
    }

    fn test_record(ordinal: usize, address: SocketAddr) -> Arc<WorkerRecord> {
        Arc::new(WorkerRecord {
            worker_id: WorkerId::new(format!("worker-{ordinal}")),
            default_model_id: Some(String::from("omni")),
            registration_id: RegistrationId::from_startup_ordinal(ordinal),
            target: test_target(address),
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
            capacity: [
                Some(CapacitySlot {
                    limit: 1,
                    semaphore: Arc::new(Semaphore::new(1)),
                }),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            health: HealthCell::unknown(),
            disposition: AtomicU8::new(Disposition::Serving as u8),
            immediate_probe: Notify::new(),
        })
    }

    fn test_client(records: &[Arc<WorkerRecord>], timeout: Duration) -> reqwest::Client {
        let targets: Vec<_> = records.iter().map(|record| record.target.clone()).collect();
        let resolver = StaticResolver::from_targets(&targets).expect("test resolver must build");
        build_health_client(Arc::new(resolver), timeout, Duration::from_secs(60))
            .expect("test health client must build")
    }

    #[test]
    fn hysteresis_preserves_unknown_and_requires_consecutive_observations() {
        let mut tracker = HealthTracker::new(2, 3);
        assert_eq!(tracker.observe(true), HealthState::Unknown);
        assert_eq!(tracker.observe(false), HealthState::Unknown);
        assert_eq!(tracker.observe(false), HealthState::Unknown);
        assert_eq!(tracker.observe(false), HealthState::Unhealthy);
        assert_eq!(tracker.observe(true), HealthState::Unhealthy);
        assert_eq!(tracker.observe(true), HealthState::Healthy);
        assert_eq!(tracker.observe(false), HealthState::Healthy);
    }

    #[tokio::test]
    async fn status_only_probe_does_not_wait_for_or_buffer_a_large_body() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind health fixture");
        let address = listener.local_addr().expect("read health fixture address");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept health probe");
            stream
                .set_read_timeout(Some(Duration::from_secs(1)))
                .expect("bound health request read");
            let mut request = [0_u8; 1024];
            let _bytes = stream.read(&mut request).expect("read health request");
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 100000000\r\nConnection: close\r\n\r\n",
                )
                .expect("write health headers");
            thread::sleep(Duration::from_millis(300));
        });
        let record = test_record(0, address);
        let client = test_client(std::slice::from_ref(&record), Duration::from_secs(1));
        let mut supervisor = HealthSupervisor::start(
            &[Arc::clone(&record)],
            client,
            Duration::from_secs(60),
            1,
            1,
            1,
        );
        let deadline = tokio::time::Instant::now() + Duration::from_millis(200);
        while record.health.load() != HealthState::Healthy && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(record.health.load(), HealthState::Healthy);
        supervisor.cancel();
        assert!(matches!(supervisor.join_next().await, Some(Ok(Ok(())))));
        server.join().expect("join health fixture server");
    }

    #[tokio::test]
    async fn probe_timeout_is_observed_as_a_failed_health_result() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind timeout fixture");
        let address = listener.local_addr().expect("read timeout fixture address");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept timed health probe");
            let mut request = [0_u8; 1024];
            let _bytes = stream.read(&mut request).expect("read timed health probe");
            thread::sleep(Duration::from_millis(200));
        });
        let record = test_record(0, address);
        let client = test_client(std::slice::from_ref(&record), Duration::from_millis(50));
        let mut supervisor = HealthSupervisor::start(
            &[Arc::clone(&record)],
            client,
            Duration::from_secs(60),
            1,
            1,
            1,
        );
        let deadline = tokio::time::Instant::now() + Duration::from_millis(150);
        while record.health.load() != HealthState::Unhealthy
            && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(record.health.load(), HealthState::Unhealthy);
        supervisor.cancel();
        assert!(matches!(supervisor.join_next().await, Some(Ok(Ok(())))));
        server.join().expect("join timeout fixture server");
    }

    #[tokio::test]
    async fn non_success_status_is_a_failed_health_result() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind non-2xx fixture");
        let address = listener.local_addr().expect("read non-2xx fixture address");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept non-2xx probe");
            let mut request = [0_u8; 1024];
            let _bytes = stream.read(&mut request).expect("read non-2xx probe");
            stream
                .write_all(
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .expect("write non-2xx response");
        });
        let record = test_record(0, address);
        let client = test_client(std::slice::from_ref(&record), Duration::from_secs(1));
        let mut supervisor = HealthSupervisor::start(
            &[Arc::clone(&record)],
            client,
            Duration::from_secs(60),
            1,
            1,
            1,
        );
        tokio::time::timeout(Duration::from_secs(1), async {
            while record.health.load() != HealthState::Unhealthy {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("non-2xx observation must complete");
        supervisor.cancel();
        assert!(matches!(supervisor.join_next().await, Some(Ok(Ok(())))));
        server.join().expect("join non-2xx fixture server");
    }

    #[tokio::test]
    async fn one_worker_never_overlaps_and_stalled_probe_cancels_and_joins() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind stalled fixture");
        let address = listener.local_addr().expect("read stalled fixture address");
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let overlap = Arc::new(AtomicBool::new(false));
        let server_started = Arc::clone(&started);
        let server_release = Arc::clone(&release);
        let server_overlap = Arc::clone(&overlap);
        let server = thread::spawn(move || {
            let (mut stalled, _) = listener.accept().expect("accept stalled probe");
            let mut request = [0_u8; 1024];
            let _bytes = stalled.read(&mut request).expect("read stalled probe");
            listener
                .set_nonblocking(true)
                .expect("set stalled listener nonblocking");
            server_started.store(true, Ordering::Release);
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while !server_release.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
                match listener.accept() {
                    Ok((_second, _)) => server_overlap.store(true, Ordering::Release),
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(error) => panic!("stalled fixture accept failed: {error}"),
                }
                thread::yield_now();
            }
            assert!(server_release.load(Ordering::Acquire));
            drop(stalled);
        });
        let record = test_record(0, address);
        let client = test_client(std::slice::from_ref(&record), Duration::from_secs(5));
        let mut supervisor = HealthSupervisor::start(
            &[Arc::clone(&record)],
            client,
            Duration::from_secs(60),
            1,
            1,
            1,
        );
        tokio::time::timeout(Duration::from_secs(1), async {
            while !started.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("stalled probe must start");
        for _ in 0..32 {
            record.immediate_probe.notify_one();
            tokio::task::yield_now().await;
        }
        assert!(!overlap.load(Ordering::Acquire));

        supervisor.cancel();
        let joined = tokio::time::timeout(Duration::from_millis(500), supervisor.join_next())
            .await
            .expect("stalled probe cancellation must be bounded");
        assert!(matches!(joined, Some(Ok(Ok(())))));
        release.store(true, Ordering::Release);
        server.join().expect("join stalled fixture server");
        assert!(!overlap.load(Ordering::Acquire));
        assert!(supervisor.is_empty());
    }

    #[tokio::test]
    async fn immediate_requests_coalesce_to_one_pending_probe() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind coalescing fixture");
        let address = listener
            .local_addr()
            .expect("read coalescing fixture address");
        let count = Arc::new(AtomicUsize::new(0));
        let first_probe_active = Arc::new(AtomicU8::new(0));
        let server_count = Arc::clone(&count);
        let server_first_probe_active = Arc::clone(&first_probe_active);
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept first health probe");
            server_count.fetch_add(1, Ordering::AcqRel);
            let mut request = [0_u8; 1024];
            let _bytes = stream
                .read(&mut request)
                .expect("read first coalesced probe");
            server_first_probe_active.store(1, Ordering::Release);
            thread::sleep(Duration::from_millis(250));
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .expect("write first coalesced health response");
            server_first_probe_active.store(0, Ordering::Release);
            listener
                .set_nonblocking(true)
                .expect("set coalescing fixture nonblocking");
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while std::time::Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        server_count.fetch_add(1, Ordering::AcqRel);
                        let _bytes = stream
                            .read(&mut request)
                            .expect("read second coalesced probe");
                        stream
                            .write_all(
                                b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                            )
                            .expect("write second coalesced health response");
                        break;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(error) => panic!("coalescing fixture accept failed: {error}"),
                }
            }
            let extra_deadline = std::time::Instant::now() + Duration::from_millis(150);
            while std::time::Instant::now() < extra_deadline {
                match listener.accept() {
                    Ok((_stream, _)) => {
                        server_count.fetch_add(1, Ordering::AcqRel);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(error) => panic!("coalescing fixture extra accept failed: {error}"),
                }
            }
            assert_eq!(server_count.load(Ordering::Acquire), 2);
        });
        let record = test_record(0, address);
        let client = test_client(std::slice::from_ref(&record), Duration::from_secs(1));
        let mut supervisor = HealthSupervisor::start(
            &[Arc::clone(&record)],
            client,
            Duration::from_secs(60),
            1,
            1,
            1,
        );
        let deadline = tokio::time::Instant::now() + Duration::from_millis(200);
        while first_probe_active.load(Ordering::Acquire) == 0
            && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        assert_eq!(first_probe_active.load(Ordering::Acquire), 1);
        assert_eq!(count.load(Ordering::Acquire), 1);
        for _ in 0..32 {
            record.immediate_probe.notify_one();
        }
        let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
        while count.load(Ordering::Acquire) < 2 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        server.join().expect("join coalescing health fixture");
        assert_eq!(count.load(Ordering::Acquire), 2);
        supervisor.cancel();
        assert!(matches!(supervisor.join_next().await, Some(Ok(Ok(())))));
    }

    #[tokio::test]
    async fn shared_probe_budget_bounds_concurrency() {
        const WORKERS: usize = 8;
        const MAX_CONCURRENT: usize = 2;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind concurrency fixture");
        let address = listener
            .local_addr()
            .expect("read concurrency fixture address");
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let server_active = Arc::clone(&active);
        let server_maximum = Arc::clone(&maximum);
        let server = thread::spawn(move || {
            let mut handlers = Vec::with_capacity(WORKERS);
            for _ in 0..WORKERS {
                let (mut stream, _) = listener.accept().expect("accept bounded health probe");
                let handler_active = Arc::clone(&server_active);
                let handler_maximum = Arc::clone(&server_maximum);
                handlers.push(thread::spawn(move || {
                    let current = handler_active.fetch_add(1, Ordering::AcqRel) + 1;
                    handler_maximum.fetch_max(current, Ordering::AcqRel);
                    let mut request = [0_u8; 1024];
                    let _bytes = stream.read(&mut request).expect("read bounded probe");
                    thread::sleep(Duration::from_millis(50));
                    stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        )
                        .expect("write bounded health response");
                    handler_active.fetch_sub(1, Ordering::AcqRel);
                }));
            }
            for handler in handlers {
                handler.join().expect("join bounded probe handler");
            }
        });
        let records: Vec<_> = (0..WORKERS)
            .map(|ordinal| test_record(ordinal, address))
            .collect();
        let client = test_client(&records, Duration::from_secs(1));
        let mut supervisor = HealthSupervisor::start(
            &records,
            client,
            Duration::from_secs(60),
            1,
            1,
            MAX_CONCURRENT,
        );
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while records
            .iter()
            .any(|record| record.health.load() != HealthState::Healthy)
            && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(
            records
                .iter()
                .all(|record| record.health.load() == HealthState::Healthy)
        );
        server.join().expect("join concurrency fixture");
        assert!(maximum.load(Ordering::Acquire) <= MAX_CONCURRENT);
        supervisor.cancel();
        while !supervisor.is_empty() {
            assert!(matches!(supervisor.join_next().await, Some(Ok(Ok(())))));
        }
    }
}
