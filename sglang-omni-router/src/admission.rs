use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::error::HttpFault;
use crate::lifecycle::Lifecycle;

/// Fail-fast global request-capacity owner.
pub(crate) struct Admission {
    lifecycle: Arc<Lifecycle>,
    capacity: Arc<Semaphore>,
}

impl Admission {
    pub(crate) fn new(lifecycle: Arc<Lifecycle>, limit: usize) -> Self {
        Self {
            lifecycle,
            capacity: Arc::new(Semaphore::new(limit)),
        }
    }

    pub(crate) fn try_acquire(&self) -> Result<AdmissionLease, HttpFault> {
        if !self.lifecycle.is_serving() {
            return Err(HttpFault::RouterUnavailable);
        }
        let permit = Arc::clone(&self.capacity)
            .try_acquire_owned()
            .map_err(|_source| HttpFault::RouterOverloaded)?;
        if !self.lifecycle.is_serving() {
            return Err(HttpFault::RouterUnavailable);
        }
        Ok(AdmissionLease { _permit: permit })
    }

    pub(crate) fn close(&self) {
        self.capacity.close();
    }
}

/// One exact admitted request, released synchronously on every terminal path.
pub(crate) struct AdmissionLease {
    _permit: OwnedSemaphorePermit,
}
