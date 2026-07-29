use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::telemetry::Telemetry;
use crate::worker_pool::CapacityClass;

/// Failure to schedule or join finite CPU-bound request classification.
#[derive(Debug)]
pub(crate) enum ClassificationError {
    Unavailable,
    Join(tokio::task::JoinError),
}

/// Runs one finite classifier off the async runtime under the process-wide limit.
pub(crate) async fn run<T>(
    slots: &Arc<Semaphore>,
    telemetry: &Arc<Telemetry>,
    class: CapacityClass,
    operation: impl FnOnce() -> T + Send + 'static,
) -> Result<T, ClassificationError>
where
    T: Send + 'static,
{
    let slot = acquire(slots, telemetry, class).await?;
    run_with_slot(slot, telemetry, class, operation).await
}

/// Waits asynchronously for one process-wide classification slot.
pub(crate) async fn acquire(
    slots: &Arc<Semaphore>,
    telemetry: &Arc<Telemetry>,
    class: CapacityClass,
) -> Result<OwnedSemaphorePermit, ClassificationError> {
    let wait = telemetry.classification_wait(class);
    let result = Arc::clone(slots)
        .acquire_owned()
        .await
        .map_err(|_| ClassificationError::Unavailable);
    drop(wait);
    result
}

/// Runs one finite classifier off the async runtime with an acquired slot.
pub(crate) async fn run_with_slot<T>(
    slot: OwnedSemaphorePermit,
    telemetry: &Arc<Telemetry>,
    class: CapacityClass,
    operation: impl FnOnce() -> T + Send + 'static,
) -> Result<T, ClassificationError>
where
    T: Send + 'static,
{
    let telemetry = Arc::clone(telemetry);
    tokio::task::spawn_blocking(move || {
        let _slot = slot;
        let _run = telemetry.classification_run(class);
        operation()
    })
    .await
    .map_err(ClassificationError::Join)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use tokio::sync::Semaphore;

    #[tokio::test]
    async fn cancelled_waiter_does_not_release_owned_resources_while_work_runs() {
        let slots = Arc::new(Semaphore::new(1));
        let telemetry = Arc::new(crate::telemetry::Telemetry::new(std::iter::empty()));
        let payload_budget = Arc::new(Semaphore::new(1));
        let payload = Arc::clone(&payload_budget)
            .try_acquire_owned()
            .expect("reserve payload ownership");
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let task_slots = Arc::clone(&slots);
        let task_telemetry = Arc::clone(&telemetry);
        let waiter = tokio::spawn(async move {
            super::run(
                &task_slots,
                &task_telemetry,
                crate::worker_pool::CapacityClass::GenerationHttp,
                move || {
                    let _payload = payload;
                    entered_tx.send(()).expect("announce classifier entry");
                    release_rx.recv().expect("release blocking classifier");
                },
            )
            .await
        });
        entered_rx.await.expect("classifier entered");
        assert_eq!(telemetry.classification_counts(), (0, 1));

        waiter.abort();
        assert!(
            waiter
                .await
                .expect_err("waiter is cancelled")
                .is_cancelled()
        );
        assert_eq!(slots.available_permits(), 0);
        assert_eq!(payload_budget.available_permits(), 0);
        assert_eq!(telemetry.classification_counts(), (0, 1));

        release_tx.send(()).expect("release classifier");
        let slot = tokio::time::timeout(Duration::from_secs(1), Arc::clone(&slots).acquire_owned())
            .await
            .expect("blocking closure eventually releases the slot")
            .expect("classification semaphore remains open");
        drop(slot);
        assert_eq!(payload_budget.available_permits(), 1);
        assert_eq!(telemetry.classification_counts(), (0, 0));
        assert_eq!(
            telemetry
                .classification_observations(crate::worker_pool::CapacityClass::GenerationHttp),
            (1, 1)
        );
    }

    #[tokio::test]
    async fn wait_and_run_accounting_closes_on_success_and_closed_semaphore() {
        let slots = Arc::new(Semaphore::new(1));
        let telemetry = Arc::new(crate::telemetry::Telemetry::new(std::iter::empty()));
        assert_eq!(
            super::run(
                &slots,
                &telemetry,
                crate::worker_pool::CapacityClass::SpeechHttp,
                || 7_u8,
            )
            .await
            .expect("classification succeeds"),
            7
        );
        assert_eq!(telemetry.classification_counts(), (0, 0));
        assert_eq!(
            telemetry.classification_observations(crate::worker_pool::CapacityClass::SpeechHttp),
            (1, 1)
        );

        slots.close();
        assert!(matches!(
            super::acquire(
                &slots,
                &telemetry,
                crate::worker_pool::CapacityClass::TranscriptionHttp,
            )
            .await,
            Err(super::ClassificationError::Unavailable)
        ));
        assert_eq!(telemetry.classification_counts(), (0, 0));
        assert_eq!(
            telemetry
                .classification_observations(crate::worker_pool::CapacityClass::TranscriptionHttp),
            (1, 0)
        );
    }

    #[tokio::test]
    async fn cancelled_slot_waiter_returns_the_waiter_gauge_to_zero() {
        let slots = Arc::new(Semaphore::new(1));
        let _held = Arc::clone(&slots)
            .acquire_owned()
            .await
            .expect("hold classification slot");
        let telemetry = Arc::new(crate::telemetry::Telemetry::new(std::iter::empty()));
        let task_slots = Arc::clone(&slots);
        let task_telemetry = Arc::clone(&telemetry);
        let waiter = tokio::spawn(async move {
            super::acquire(
                &task_slots,
                &task_telemetry,
                crate::worker_pool::CapacityClass::SpeechBatch,
            )
            .await
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            while telemetry.classification_counts().0 == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("waiter gauge becomes visible");
        waiter.abort();
        assert!(
            waiter
                .await
                .expect_err("waiter is cancelled")
                .is_cancelled()
        );
        assert_eq!(telemetry.classification_counts(), (0, 0));
        assert_eq!(
            telemetry.classification_observations(crate::worker_pool::CapacityClass::SpeechBatch),
            (1, 0)
        );
    }

    #[tokio::test]
    #[allow(clippy::panic)]
    async fn panicking_classifier_releases_run_accounting() {
        let slots = Arc::new(Semaphore::new(1));
        let telemetry = Arc::new(crate::telemetry::Telemetry::new(std::iter::empty()));
        let result = super::run(
            &slots,
            &telemetry,
            crate::worker_pool::CapacityClass::RealtimeWebsocket,
            || panic!("classification fixture panic"),
        )
        .await;
        assert!(matches!(result, Err(super::ClassificationError::Join(_))));
        assert_eq!(telemetry.classification_counts(), (0, 0));
        assert_eq!(
            telemetry
                .classification_observations(crate::worker_pool::CapacityClass::RealtimeWebsocket),
            (1, 1)
        );
    }
}
