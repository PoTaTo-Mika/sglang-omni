use std::sync::{Arc, RwLock};

use thiserror::Error;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::{ResolvedTarget, WorkerRecord};

/// One read/write gate linearizes fail-fast admission and exact reservation
/// against process drain. No guard crosses an await point.
pub(super) struct Gate {
    pub(super) open: bool,
}

impl Gate {
    pub(super) const fn open() -> Self {
        Self { open: true }
    }
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub(crate) enum AdmissionError {
    #[error("router is draining")]
    Draining,
    #[error("router admission is full")]
    Overloaded,
    #[error("router admission invariant failed")]
    Internal,
}

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
    #[error("router dispatch invariant failed")]
    Internal,
}

/// Global and generation-class ingress ownership, released exactly once.
pub(crate) struct AdmissionLease {
    _generation: OwnedSemaphorePermit,
    _global: OwnedSemaphorePermit,
}

/// Exact generation-worker ownership retained through response termination.
pub(crate) struct RequestLease {
    _exact: OwnedSemaphorePermit,
    _admission: AdmissionLease,
    pub(super) registration: Arc<WorkerRecord>,
}

impl RequestLease {
    pub(super) fn new(
        admission: AdmissionLease,
        exact: OwnedSemaphorePermit,
        registration: Arc<WorkerRecord>,
    ) -> Self {
        Self {
            _exact: exact,
            _admission: admission,
            registration,
        }
    }

    pub(crate) fn target(&self) -> &ResolvedTarget {
        &self.registration.target
    }

    pub(crate) fn request_immediate_probe(&self) {
        self.registration.immediate_probe.notify_one();
    }

    #[cfg(test)]
    pub(super) fn registration_ordinal(&self) -> usize {
        self.registration.registration_id.startup_ordinal()
    }
}

pub(super) struct AdmissionController {
    gate: Arc<RwLock<Gate>>,
    global: Arc<Semaphore>,
    generation: Arc<Semaphore>,
}

impl AdmissionController {
    pub(super) fn new(gate: Arc<RwLock<Gate>>, global: usize, generation: usize) -> Self {
        Self {
            gate,
            global: Arc::new(Semaphore::new(global)),
            generation: Arc::new(Semaphore::new(generation)),
        }
    }

    pub(super) fn try_admit(&self) -> Result<AdmissionLease, AdmissionError> {
        let gate = self.gate.read().map_err(|_| AdmissionError::Internal)?;
        if !gate.open {
            return Err(AdmissionError::Draining);
        }
        let global = Arc::clone(&self.global)
            .try_acquire_owned()
            .map_err(|_| AdmissionError::Overloaded)?;
        let generation = Arc::clone(&self.generation)
            .try_acquire_owned()
            .map_err(|_| AdmissionError::Overloaded)?;
        drop(gate);
        Ok(AdmissionLease {
            _generation: generation,
            _global: global,
        })
    }

    pub(super) fn close(&self) {
        self.global.close();
        self.generation.close();
    }

    #[cfg(test)]
    pub(super) fn available(&self) -> (usize, usize) {
        (
            self.global.available_permits(),
            self.generation.available_permits(),
        )
    }
}
