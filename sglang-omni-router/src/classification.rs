use std::sync::Arc;

use tokio::sync::Semaphore;

/// Failure to schedule or join finite CPU-bound request classification.
#[derive(Debug)]
pub(crate) enum ClassificationError {
    Unavailable,
    Join(tokio::task::JoinError),
}

/// Runs one finite classifier off the async runtime under the process-wide limit.
pub(crate) async fn run<T>(
    slots: &Arc<Semaphore>,
    operation: impl FnOnce() -> T + Send + 'static,
) -> Result<T, ClassificationError>
where
    T: Send + 'static,
{
    let slot = Arc::clone(slots)
        .acquire_owned()
        .await
        .map_err(|_| ClassificationError::Unavailable)?;
    tokio::task::spawn_blocking(move || {
        let _slot = slot;
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
        let payload_budget = Arc::new(Semaphore::new(1));
        let payload = Arc::clone(&payload_budget)
            .try_acquire_owned()
            .expect("reserve payload ownership");
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let task_slots = Arc::clone(&slots);
        let waiter = tokio::spawn(async move {
            super::run(&task_slots, move || {
                let _payload = payload;
                entered_tx.send(()).expect("announce classifier entry");
                release_rx.recv().expect("release blocking classifier");
            })
            .await
        });
        entered_rx.await.expect("classifier entered");

        waiter.abort();
        assert!(
            waiter
                .await
                .expect_err("waiter is cancelled")
                .is_cancelled()
        );
        assert_eq!(slots.available_permits(), 0);
        assert_eq!(payload_budget.available_permits(), 0);

        release_tx.send(()).expect("release classifier");
        let slot = tokio::time::timeout(Duration::from_secs(1), Arc::clone(&slots).acquire_owned())
            .await
            .expect("blocking closure eventually releases the slot")
            .expect("classification semaphore remains open");
        drop(slot);
        assert_eq!(payload_budget.available_permits(), 1);
    }
}
