use std::sync::Arc;

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use tokio::sync::{Semaphore, oneshot};
use tokio::task::JoinHandle;
use tracing::{error, info};

use crate::config::Config;
use crate::error::RouterError;
use crate::http_generation::{self, HttpGeneration};
use crate::lifecycle::Lifecycle;
use crate::shutdown;
use crate::worker_pool::{HealthSupervisor, HealthTaskError, WorkerPool};

#[path = "bounded_listener.rs"]
mod bounded_listener;

use bounded_listener::BoundedTcpListener;

const LIVE_BODY: &str = "live\n";
const NOT_READY_BODY: &str = "not ready\n";
const READY_BODY: &str = "ready\n";

#[derive(Clone)]
struct AppState {
    lifecycle: Arc<Lifecycle>,
    pool: Arc<WorkerPool>,
    generation: Option<Arc<HttpGeneration>>,
}

pub(crate) async fn serve(config: Config) -> Result<(), RouterError> {
    let lifecycle = Arc::new(Lifecycle::starting());
    let pool = Arc::new(WorkerPool::build(&config)?);
    let classification_slots = Arc::new(Semaphore::new(
        config.router.max_concurrent_classifications(),
    ));
    let generation = HttpGeneration::build(
        &config,
        Arc::clone(&pool),
        Arc::clone(&classification_slots),
    )?;
    let mut signal_observer = shutdown::SignalObserver::install().map_err(RouterError::Signal)?;
    let app = route_table(
        AppState {
            lifecycle: Arc::clone(&lifecycle),
            pool: Arc::clone(&pool),
            generation: generation.clone(),
        },
        generation,
    );
    let listener = tokio::net::TcpListener::bind(config.server.listen)
        .await
        .map_err(RouterError::Bind)?;
    let max_connections = config.server.max_connections_usize()?;
    let listener = BoundedTcpListener::new(listener, max_connections);
    let (shutdown_sender, shutdown_receiver) = oneshot::channel::<()>();
    lifecycle.enter_serving()?;
    let mut health = pool.start_health(&config);
    info!(state = "serving", ready = false, "local service started");

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
            health.cancel();
            health.abort_and_join_all().await;
            lifecycle.enter_failed()?;
            return unexpected_server_exit(task_result);
        }
        health_result = health.join_next(), if !health.is_empty() => {
            abort_all(&mut server_task, &mut health).await?;
            lifecycle.enter_failed()?;
            return unexpected_health_exit(health_result);
        }
        signal_result = signal_observer.next() => match signal_result {
            Ok(signal) => signal,
            Err(source) => {
                abort_all(&mut server_task, &mut health).await?;
                lifecycle.enter_failed()?;
                return Err(RouterError::Signal(source));
            }
        },
    };

    if lifecycle.enter_draining().is_err() || pool.drain().is_err() {
        abort_all(&mut server_task, &mut health).await?;
        lifecycle.enter_failed()?;
        return Err(RouterError::Lifecycle);
    }
    health.cancel();
    info!(state = "draining", reason = ?first_signal, "graceful shutdown started");
    if shutdown_sender.send(()).is_err() {
        abort_all(&mut server_task, &mut health).await?;
        lifecycle.enter_failed()?;
        return Err(RouterError::ShutdownNotify);
    }

    let deadline = tokio::time::Instant::now() + config.shutdown.drain_timeout();
    let mut server_done = false;
    while !server_done || !health.is_empty() {
        tokio::select! {
            biased;
            task_result = &mut server_task, if !server_done => {
                match task_result {
                    Ok(Ok(())) => server_done = true,
                    Ok(Err(source)) => {
                        health.abort_and_join_all().await;
                        lifecycle.enter_failed()?;
                        return Err(RouterError::Server(source));
                    }
                    Err(source) => {
                        health.abort_and_join_all().await;
                        lifecycle.enter_failed()?;
                        return Err(RouterError::ServerTask(source));
                    }
                }
            }
            health_result = health.join_next(), if !health.is_empty() => {
                if !expected_health_shutdown(health_result) {
                    if !server_done {
                        let server_result = abort_and_join_server(&mut server_task).await;
                        health.abort_and_join_all().await;
                        server_result?;
                    } else {
                        health.abort_and_join_all().await;
                    }
                    lifecycle.enter_failed()?;
                    return Err(RouterError::HealthTask);
                }
            }
            second_signal = signal_observer.next() => {
                let signal = match second_signal {
                    Ok(signal) => signal,
                    Err(source) => {
                        if !server_done {
                            let server_result = abort_and_join_server(&mut server_task).await;
                            health.abort_and_join_all().await;
                            server_result?;
                        } else {
                            health.abort_and_join_all().await;
                        }
                        lifecycle.enter_failed()?;
                        return Err(RouterError::Signal(source));
                    }
                };
                error!(state = "draining", reason = ?signal, "second signal forced shutdown");
                if !server_done {
                    abort_and_join_server(&mut server_task).await?;
                }
                health.abort_and_join_all().await;
                lifecycle.enter_failed()?;
                return Err(RouterError::ForcedShutdown);
            }
            () = tokio::time::sleep_until(deadline) => {
                error!(state = "draining", "graceful shutdown deadline elapsed");
                if !server_done {
                    abort_and_join_server(&mut server_task).await?;
                }
                health.abort_and_join_all().await;
                lifecycle.enter_failed()?;
                return Err(RouterError::DrainTimeout);
            }
        }
    }

    lifecycle.enter_stopped()?;
    info!(
        state = "stopped",
        remaining_tasks = 0_u8,
        "shutdown complete"
    );
    Ok(())
}

fn route_table(state: AppState, generation: Option<Arc<HttpGeneration>>) -> Router {
    let app = Router::new()
        .route("/live", get(live).head(reject_head))
        .route("/ready", get(ready).head(reject_head))
        .with_state(state);
    if let Some(generation) = generation {
        app.route(
            "/v1/chat/completions",
            post(http_generation::chat).with_state(generation),
        )
    } else {
        app
    }
}

async fn live(State(state): State<AppState>) -> (StatusCode, &'static str) {
    if state.lifecycle.is_live() {
        (StatusCode::OK, LIVE_BODY)
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "not live\n")
    }
}

async fn ready(State(state): State<AppState>) -> (StatusCode, &'static str) {
    if state.lifecycle.is_serving()
        && state.pool.is_ready()
        && state
            .generation
            .as_ref()
            .is_none_or(|generation| generation.is_ready())
    {
        (StatusCode::OK, READY_BODY)
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, NOT_READY_BODY)
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

fn unexpected_health_exit(
    result: Option<Result<Result<(), HealthTaskError>, tokio::task::JoinError>>,
) -> Result<(), RouterError> {
    let _result = result;
    Err(RouterError::HealthTask)
}

fn expected_health_shutdown(
    result: Option<Result<Result<(), HealthTaskError>, tokio::task::JoinError>>,
) -> bool {
    matches!(result, Some(Ok(Ok(()))))
}

async fn abort_all(
    server_task: &mut JoinHandle<std::io::Result<()>>,
    health: &mut HealthSupervisor,
) -> Result<(), RouterError> {
    health.cancel();
    let server_result = abort_and_join_server(server_task).await;
    health.abort_and_join_all().await;
    server_result
}

async fn abort_and_join_server(
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
