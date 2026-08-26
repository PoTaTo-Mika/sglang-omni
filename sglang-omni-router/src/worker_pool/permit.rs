use std::sync::{Arc, RwLock};

use thiserror::Error;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::{CapacityClass, ResolvedTarget, WorkerRecord};

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

/// Global-envelope and route-class ingress ownership, released exactly once.
pub(crate) struct AdmissionLease {
    class: CapacityClass,
    credits: u32,
    _class: OwnedSemaphorePermit,
    _envelope: EnvelopeLease,
}

pub(crate) struct EnvelopeLease {
    _global: OwnedSemaphorePermit,
}

impl AdmissionLease {
    pub(super) const fn class(&self) -> CapacityClass {
        self.class
    }

    pub(super) const fn credits(&self) -> u32 {
        self.credits
    }
}

/// Exact worker-class ownership retained through response termination.
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
    classes: [Option<Arc<Semaphore>>; 4],
}

impl AdmissionController {
    pub(super) fn new(gate: Arc<RwLock<Gate>>, global: usize, limits: [Option<usize>; 4]) -> Self {
        Self {
            gate,
            global: Arc::new(Semaphore::new(global)),
            classes: limits.map(|limit| limit.map(|value| Arc::new(Semaphore::new(value)))),
        }
    }

    pub(super) fn try_admit(
        &self,
        class: CapacityClass,
        credits: u32,
    ) -> Result<AdmissionLease, AdmissionError> {
        let envelope = self.try_admit_envelope()?;
        self.try_admit_class(envelope, class, credits)
    }

    pub(super) fn try_admit_envelope(&self) -> Result<EnvelopeLease, AdmissionError> {
        let gate = self.gate.read().map_err(|_| AdmissionError::Internal)?;
        if !gate.open {
            return Err(AdmissionError::Draining);
        }
        let global = Arc::clone(&self.global)
            .try_acquire_owned()
            .map_err(|_| AdmissionError::Overloaded)?;
        drop(gate);
        Ok(EnvelopeLease { _global: global })
    }

    pub(super) fn try_admit_class(
        &self,
        envelope: EnvelopeLease,
        class: CapacityClass,
        credits: u32,
    ) -> Result<AdmissionLease, AdmissionError> {
        let gate = self.gate.read().map_err(|_| AdmissionError::Internal)?;
        if !gate.open {
            return Err(AdmissionError::Draining);
        }
        let class_semaphore = self
            .classes
            .get(class.index())
            .and_then(Option::as_ref)
            .ok_or(AdmissionError::Internal)?;
        let class_permit = Arc::clone(class_semaphore)
            .try_acquire_many_owned(credits)
            .map_err(|_| AdmissionError::Overloaded)?;
        drop(gate);
        Ok(AdmissionLease {
            class,
            credits,
            _class: class_permit,
            _envelope: envelope,
        })
    }

    pub(super) fn close(&self) {
        self.global.close();
        for semaphore in self.classes.iter().flatten() {
            semaphore.close();
        }
    }

    #[cfg(test)]
    pub(super) fn available(&self) -> (usize, [Option<usize>; 4]) {
        let classes = std::array::from_fn(|index| {
            self.classes[index]
                .as_ref()
                .map(|semaphore| semaphore.available_permits())
        });
        (self.global.available_permits(), classes)
    }
}
