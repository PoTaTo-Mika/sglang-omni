use std::sync::Arc;

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::get;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tracing::{error, info};

use crate::config::Config;
use crate::error::RouterError;
use crate::lifecycle::Lifecycle;
use crate::shutdown;

#[path = "bounded_listener.rs"]
mod bounded_listener;

use bounded_listener::BoundedTcpListener;

const LIVE_BODY: &str = "live\n";

pub(crate) async fn serve(config: Config) -> Result<(), RouterError> {
    let lifecycle = Arc::new(Lifecycle::starting());
    let mut signal_observer = shutdown::SignalObserver::install().map_err(RouterError::Signal)?;
    let app = route_table(Arc::clone(&lifecycle));
    let listener = tokio::net::TcpListener::bind(config.server.listen)
        .await
        .map_err(RouterError::Bind)?;
    let max_connections = config.server.max_connections_usize()?;
    let listener = BoundedTcpListener::new(listener, max_connections);
    let (shutdown_sender, shutdown_receiver) = oneshot::channel::<()>();

    lifecycle.enter_serving()?;
    info!(state = "serving", "local service started");

    let mut server_task = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _result = shutdown_receiver.await;
            })
            .await
    });

    let first_signal = tokio::select! {
        biased;
        task_result = &mut server_task => {
            lifecycle.enter_failed()?;
            return unexpected_server_exit(task_result);
        }
        signal_result = signal_observer.next() => match signal_result {
            Ok(signal) => signal,
            Err(source) => {
                abort_and_join(&mut server_task).await?;
                lifecycle.enter_failed()?;
                return Err(RouterError::Signal(source));
            }
        },
    };

    if let Err(error) = lifecycle.enter_draining() {
        abort_and_join(&mut server_task).await?;
        return Err(error);
    }
    info!(state = "draining", reason = ?first_signal, "graceful shutdown started");
    if shutdown_sender.send(()).is_err() {
        abort_and_join(&mut server_task).await?;
        lifecycle.enter_failed()?;
        return Err(RouterError::ShutdownNotify);
    }

    let drain_timeout = config.shutdown.drain_timeout();
    let drain_result = tokio::select! {
        biased;
        task_result = &mut server_task => finish_graceful(task_result, &lifecycle),
        second_signal = signal_observer.next() => {
            match second_signal {
                Ok(signal) => {
                    error!(state = "draining", reason = ?signal, "second signal forced shutdown");
                    abort_and_join(&mut server_task).await?;
                    lifecycle.enter_failed()?;
                    Err(RouterError::ForcedShutdown)
                }
                Err(source) => {
                    abort_and_join(&mut server_task).await?;
                    lifecycle.enter_failed()?;
                    Err(RouterError::Signal(source))
                }
            }
        }
        () = tokio::time::sleep(drain_timeout) => {
            error!(state = "draining", "graceful shutdown deadline elapsed");
            abort_and_join(&mut server_task).await?;
            lifecycle.enter_failed()?;
            Err(RouterError::DrainTimeout)
        }
    };

    if drain_result.is_ok() {
        info!(
            state = "stopped",
            remaining_tasks = 0_u8,
            "shutdown complete"
        );
    }
    drain_result
}

fn route_table(lifecycle: Arc<Lifecycle>) -> Router {
    Router::new()
        .route("/live", get(live).head(reject_head))
        .with_state(lifecycle)
}

async fn live(State(lifecycle): State<Arc<Lifecycle>>) -> (StatusCode, &'static str) {
    if lifecycle.is_live() {
        (StatusCode::OK, LIVE_BODY)
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "not live\n")
    }
}

async fn reject_head() -> StatusCode {
    StatusCode::METHOD_NOT_ALLOWED
}

fn unexpected_server_exit(
    task_result: Result<Result<(), std::io::Error>, tokio::task::JoinError>,
) -> Result<(), RouterError> {
    match task_result {
        Ok(Ok(())) => Err(RouterError::Lifecycle),
        Ok(Err(source)) => Err(RouterError::Server(source)),
        Err(source) => Err(RouterError::ServerTask(source)),
    }
}

fn finish_graceful(
    task_result: Result<Result<(), std::io::Error>, tokio::task::JoinError>,
    lifecycle: &Lifecycle,
) -> Result<(), RouterError> {
    match task_result {
        Ok(Ok(())) => lifecycle.enter_stopped(),
        Ok(Err(source)) => {
            lifecycle.enter_failed()?;
            Err(RouterError::Server(source))
        }
        Err(source) => {
            lifecycle.enter_failed()?;
            Err(RouterError::ServerTask(source))
        }
    }
}

async fn abort_and_join(
    server_task: &mut JoinHandle<std::io::Result<()>>,
) -> Result<(), RouterError> {
    server_task.abort();
    match server_task.await {
        Err(join_error) if join_error.is_cancelled() => Ok(()),
        Err(join_error) => Err(RouterError::ServerTask(join_error)),
        Ok(Ok(())) => Ok(()),
        Ok(Err(source)) => Err(RouterError::Server(source)),
    }
}
