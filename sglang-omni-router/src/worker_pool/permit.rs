use std::sync::{Arc, RwLock};

use thiserror::Error;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::telemetry::DurationGuard;

use super::profile::{CapacityClass, RegistrationId, WorkerId};
use super::{ResolvedTarget, WorkerRecord};

/// One read/write gate linearizes permit acquisition against drain.
///
/// Admission and dispatch take shared ownership only while checking `open` and
/// acquiring bounded semaphore permits. Drain takes exclusive ownership before
/// closing every semaphore and publishing draining disposition. Lock order is
/// gate, then atomics/semaphore `try_acquire`; no other synchronous lock may be
/// acquired while a gate guard is held, and no guard crosses an await point.
pub(super) struct Gate {
    pub(super) open: bool,
}

impl Gate {
    pub(super) const fn open() -> Self {
        Self { open: true }
    }
}

/// Stable fail-fast ingress outcomes, without worker topology details.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub(crate) enum AdmissionError {
    #[error("router is draining")]
    Draining,
    #[error("router admission is full")]
    Overloaded,
    #[error("router admission invariant failed")]
    Internal,
}

/// Stable dispatch outcomes, without worker topology details.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub(crate) enum DispatchError {
    #[error("no configured profile matches the request")]
    NoEligibleProfile,
    #[error("matching workers are unavailable")]
    Unavailable,
    #[error("matching worker capacity is full")]
    Overloaded,
    #[error("router is draining")]
    Draining,
    #[error("admission class does not match the request")]
    AdmissionClassMismatch,
    #[error("router dispatch invariant failed")]
    Internal,
}

/// Global and transport-class ingress ownership acquired before worker matching.
///
/// Both permits remain owned until this lease is dropped directly or as part
/// of a [`RequestLease`]. Drop is nonblocking and performs no lookup.
pub(crate) struct AdmissionLease {
    class: CapacityClass,
    _class: OwnedSemaphorePermit,
    _global: OwnedSemaphorePermit,
}

impl AdmissionLease {
    pub(super) fn class(&self) -> CapacityClass {
        self.class
    }
}

/// Complete request/session capacity ownership for one exact registration.
///
/// The retained registration `Arc` prevents identity reuse. Declaration order
/// makes ordinary RAII release exact, then class, then global permits exactly
/// once on every drop, including cancellation and panic unwind. The
/// registration `Arc` outlives all three releases; no task or map participates.
pub(crate) struct RequestLease {
    exact: OwnedSemaphorePermit,
    admission: AdmissionLease,
    pub(super) registration: Arc<WorkerRecord>,
    class: CapacityClass,
    started: std::time::Instant,
}

impl RequestLease {
    pub(super) fn new(
        admission: AdmissionLease,
        exact: OwnedSemaphorePermit,
        registration: Arc<WorkerRecord>,
        class: CapacityClass,
    ) -> Self {
        let lease = Self {
            exact,
            admission,
            registration,
            class,
            started: std::time::Instant::now(),
        };
        lease
            .registration
            .telemetry
            .record_dispatch(lease.registration.registration_id.startup_ordinal(), class);
        lease
    }

    pub(crate) fn worker_id(&self) -> &WorkerId {
        &self.registration.worker_id
    }

    /// Router-local canonical startup ordinal for this exact retained record.
    pub(crate) fn registration_id(&self) -> RegistrationId {
        self.registration.registration_id
    }

    pub(crate) fn target(&self) -> &ResolvedTarget {
        &self.registration.target
    }

    pub(crate) fn capacity_class(&self) -> CapacityClass {
        self.class
    }

    /// Requests one health observation after an exact upstream protocol fault.
    pub(crate) fn request_immediate_probe(&self) {
        self.registration.immediate_probe.notify_one();
    }

    pub(super) fn exact_permit(&self) -> &OwnedSemaphorePermit {
        &self.exact
    }

    pub(super) fn admission(&self) -> &AdmissionLease {
        &self.admission
    }

    pub(crate) fn upstream_headers_timer(&self) -> DurationGuard<'_> {
        self.registration
            .telemetry
            .upstream_headers_timer(self.class)
    }
}

impl Drop for RequestLease {
    fn drop(&mut self) {
        self.registration.telemetry.record_worker_request(
            self.registration.registration_id.startup_ordinal(),
            self.class,
            self.started.elapsed(),
        );
    }
}

pub(super) struct AdmissionController {
    gate: Arc<RwLock<Gate>>,
    global_limit: usize,
    class_limits: [usize; 7],
    global: Arc<Semaphore>,
    classes: [Arc<Semaphore>; 7],
}

impl AdmissionController {
    pub(super) fn new(gate: Arc<RwLock<Gate>>, global: usize, classes: [usize; 7]) -> Self {
        Self {
            gate,
            global_limit: global,
            class_limits: classes,
            global: Arc::new(Semaphore::new(global)),
            classes: classes.map(|limit| Arc::new(Semaphore::new(limit))),
        }
    }

    pub(super) fn try_admit(&self, class: CapacityClass) -> Result<AdmissionLease, AdmissionError> {
        let gate = self.gate.read().map_err(|_| AdmissionError::Internal)?;
        if !gate.open {
            return Err(AdmissionError::Draining);
        }
        let global = Arc::clone(&self.global)
            .try_acquire_owned()
            .map_err(|_| AdmissionError::Overloaded)?;
        let class_permit = Arc::clone(&self.classes[class_index(class)])
            .try_acquire_owned()
            .map_err(|_| AdmissionError::Overloaded)?;
        drop(gate);
        Ok(AdmissionLease {
            class,
            _class: class_permit,
            _global: global,
        })
    }

    pub(super) fn close_semaphores(&self) {
        self.global.close();
        for class in &self.classes {
            class.close();
        }
    }

    pub(super) fn is_open(&self) -> bool {
        self.gate.read().is_ok_and(|gate| gate.open)
    }

    pub(super) fn snapshot(&self) -> [(usize, usize); 8] {
        let mut snapshot = [(0, 0); 8];
        snapshot[0] = (
            self.global_limit,
            self.global_limit - self.global.available_permits(),
        );
        for (index, (limit, semaphore)) in self.class_limits.iter().zip(&self.classes).enumerate() {
            snapshot[index + 1] = (*limit, *limit - semaphore.available_permits());
        }
        snapshot
    }

    #[cfg(test)]
    pub(super) fn available(&self, class: CapacityClass) -> (usize, usize) {
        (
            self.global.available_permits(),
            self.classes[class_index(class)].available_permits(),
        )
    }
}

pub(super) const fn class_index(class: CapacityClass) -> usize {
    match class {
        CapacityClass::GenerationHttp => 0,
        CapacityClass::SpeechHttp => 1,
        CapacityClass::TranscriptionHttp => 2,
        CapacityClass::SpeechBatch => 3,
        CapacityClass::SpeechWebsocket => 4,
        CapacityClass::RealtimeWebsocket => 5,
        CapacityClass::Control => 6,
    }
}
