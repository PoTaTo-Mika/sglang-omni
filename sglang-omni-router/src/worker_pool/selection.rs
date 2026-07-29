use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::config::RoutingStrategy;

use super::WorkerRecord;
use super::permit::class_index;
use super::profile::CapacityClass;

/// One already hard-eligible exact registration in canonical manifest order.
#[derive(Clone)]
pub(super) struct PolicyCandidate {
    pub(super) record: Arc<WorkerRecord>,
    occupancy_snapshot: usize,
}

impl PolicyCandidate {
    pub(super) fn new(record: Arc<WorkerRecord>) -> Self {
        Self {
            record,
            occupancy_snapshot: 0,
        }
    }

    fn snapshot_occupancy(&mut self, class: CapacityClass) {
        self.occupancy_snapshot = self.record.occupancy_snapshot(class);
    }
}

/// Private synchronous policy state with one cursor and guard per data-plane class.
pub(super) struct Selector {
    strategy: RoutingStrategy,
    cursors: [AtomicU64; 6],
    least_requests: [Mutex<()>; 6],
}

impl Selector {
    pub(super) fn new(strategy: RoutingStrategy) -> Self {
        Self {
            strategy,
            cursors: std::array::from_fn(|_| AtomicU64::new(0)),
            least_requests: std::array::from_fn(|_| Mutex::new(())),
        }
    }

    /// Serializes load observation and exact reservation for one data-plane class.
    ///
    /// The protected value is intentionally unit state: exact semaphores remain
    /// authoritative. Recovering a poisoned guard is therefore safe.
    pub(super) fn least_requests_guard(
        &self,
        class: CapacityClass,
        candidate_count: usize,
    ) -> Option<MutexGuard<'_, ()>> {
        if self.strategy != RoutingStrategy::LeastRequests
            || class == CapacityClass::Control
            || candidate_count < 2
        {
            return None;
        }
        Some(
            self.least_requests[class_index(class)]
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    }

    pub(super) fn order(
        &self,
        mut eligible: Vec<PolicyCandidate>,
        class: CapacityClass,
        pool_size: usize,
    ) -> Vec<PolicyCandidate> {
        self.order_in_place(&mut eligible, class, pool_size);
        eligible
    }

    pub(super) fn order_in_place(
        &self,
        eligible: &mut [PolicyCandidate],
        class: CapacityClass,
        pool_size: usize,
    ) {
        // Control is exact-owner traffic and must not observe any data-plane cursor.
        if class == CapacityClass::Control || eligible.len() < 2 || pool_size == 0 {
            return;
        }

        let cursor = &self.cursors[class_index(class)];
        // The cursor only selects advisory order and publishes no correctness
        // state, so Relaxed ordering is sufficient. Semaphore acquisition and
        // the drain gate remain the authorities.
        let sequence = cursor.fetch_add(1, Ordering::Relaxed);
        let Ok(pool_size_u64) = u64::try_from(pool_size) else {
            return;
        };
        let Ok(start) = usize::try_from(sequence % pool_size_u64) else {
            return;
        };
        let rank = |candidate: &PolicyCandidate| {
            let ordinal = candidate.record.registration_id.startup_ordinal();
            if ordinal >= start {
                ordinal - start
            } else {
                pool_size - start + ordinal
            }
        };

        match self.strategy {
            RoutingStrategy::RoundRobin => {
                // Candidate construction preserves canonical manifest order. A
                // single partition plus in-place rotation yields RR order in O(n).
                let split = eligible
                    .iter()
                    .position(|candidate| {
                        candidate.record.registration_id.startup_ordinal() >= start
                    })
                    .unwrap_or(0);
                eligible.rotate_left(split);
            }
            RoutingStrategy::LeastRequests => {
                for candidate in eligible.iter_mut() {
                    candidate.snapshot_occupancy(class);
                }
                // Each occupancy is observed exactly once. RR rank is unique
                // because registration IDs are unique startup ordinals.
                eligible.sort_unstable_by_key(|candidate| {
                    (candidate.occupancy_snapshot, rank(candidate))
                });
            }
        }
    }
}
