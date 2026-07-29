use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, Response, StatusCode};
use axum::middleware;
use axum::routing::{delete, get, post};
use tokio::sync::{Semaphore, oneshot};
use tokio::task::JoinHandle;
use tracing::{error, info};

use crate::config::{Config, HttpMediaRoute};
use crate::error::RouterError;
use crate::http_generation::{self, HttpGeneration};
use crate::http_media::{self, HttpMedia};
use crate::lifecycle::Lifecycle;
use crate::operations::{self, Operations, RequestContext};
use crate::shutdown;
use crate::websocket::{self, SessionTracker, WebsocketGateway};
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
    media: Option<Arc<HttpMedia>>,
    websocket: Option<Arc<WebsocketGateway>>,
    operations: Arc<Operations>,
}

impl AppState {
    fn is_ready(&self) -> bool {
        self.lifecycle.is_serving()
            && self.pool.is_ready()
            && self
                .generation
                .as_ref()
                .is_none_or(|generation| generation.is_ready())
            && self.media.as_ref().is_none_or(|media| media.is_ready())
            && self
                .websocket
                .as_ref()
                .is_none_or(|websocket| websocket.is_ready())
    }
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
    let media = HttpMedia::build(
        &config,
        Arc::clone(&pool),
        Arc::clone(&classification_slots),
    )?;
    let sessions = SessionTracker::new();
    let websocket = WebsocketGateway::build(
        &config,
        Arc::clone(&pool),
        sessions.clone(),
        Arc::clone(&classification_slots),
    );
    let operations = Arc::new(Operations::build(&config)?);
    let request_context = RequestContext::new()?;
    let local_operations = config.server.listen.ip().is_loopback();
    let mut signal_observer = shutdown::SignalObserver::install().map_err(RouterError::Signal)?;
    let app = route_table(
        AppState {
            lifecycle: Arc::clone(&lifecycle),
            pool: Arc::clone(&pool),
            generation: generation.clone(),
            media: media.clone(),
            websocket: websocket.clone(),
            operations,
        },
        generation,
        media,
        websocket,
        local_operations,
        request_context,
    );
    let listener = tokio::net::TcpListener::bind(config.server.listen)
        .await
        .map_err(RouterError::Bind)?;
    let max_connections = config.server.max_connections_usize()?;
    let listener = BoundedTcpListener::new(listener, max_connections, Arc::clone(pool.telemetry()));
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
            sessions.force();
            health.cancel();
            health.abort_and_join_all().await;
            sessions.wait_empty().await;
            lifecycle.enter_failed()?;
            return unexpected_server_exit(task_result);
        }
        health_result = health.join_next(), if !health.is_empty() => {
            sessions.force();
            let abort_result = abort_all(&mut server_task, &mut health).await;
            sessions.wait_empty().await;
            abort_result?;
            lifecycle.enter_failed()?;
            return unexpected_health_exit(health_result);
        }
        signal_result = signal_observer.next() => match signal_result {
            Ok(signal) => signal,
            Err(source) => {
                sessions.force();
                let abort_result = abort_all(&mut server_task, &mut health).await;
                sessions.wait_empty().await;
                abort_result?;
                lifecycle.enter_failed()?;
                return Err(RouterError::Signal(source));
            }
        },
    };

    if lifecycle.enter_draining().is_err() || pool.drain().is_err() {
        sessions.force();
        let abort_result = abort_all(&mut server_task, &mut health).await;
        sessions.wait_empty().await;
        abort_result?;
        lifecycle.enter_failed()?;
        return Err(RouterError::Lifecycle);
    }
    sessions.drain();
    health.cancel();
    info!(state = "draining", reason = ?first_signal, "graceful shutdown started");
    if shutdown_sender.send(()).is_err() {
        sessions.force();
        let abort_result = abort_all(&mut server_task, &mut health).await;
        sessions.wait_empty().await;
        abort_result?;
        lifecycle.enter_failed()?;
        return Err(RouterError::ShutdownNotify);
    }

    let deadline = tokio::time::Instant::now() + config.shutdown.drain_timeout();
    let mut server_done = false;
    let mut sessions_done = false;
    while !server_done || !health.is_empty() || !sessions_done {
        tokio::select! {
            biased;
            task_result = &mut server_task, if !server_done => {
                match task_result {
                    Ok(Ok(())) => server_done = true,
                    Ok(Err(source)) => {
                        sessions.force();
                        health.abort_and_join_all().await;
                        sessions.wait_empty().await;
                        lifecycle.enter_failed()?;
                        return Err(RouterError::Server(source));
                    }
                    Err(source) => {
                        sessions.force();
                        health.abort_and_join_all().await;
                        sessions.wait_empty().await;
                        lifecycle.enter_failed()?;
                        return Err(RouterError::ServerTask(source));
                    }
                }
            }
            health_result = health.join_next(), if !health.is_empty() => {
                if !expected_health_shutdown(health_result) {
                    sessions.force();
                    if !server_done {
                        let server_result = abort_and_join_server(&mut server_task).await;
                        health.abort_and_join_all().await;
                        sessions.wait_empty().await;
                        server_result?;
                    } else {
                        health.abort_and_join_all().await;
                        sessions.wait_empty().await;
                    }
                    lifecycle.enter_failed()?;
                    return Err(RouterError::HealthTask);
                }
            }
            () = sessions.wait_empty(), if !sessions_done => {
                sessions_done = true;
            }
            second_signal = signal_observer.next() => {
                let signal = match second_signal {
                    Ok(signal) => signal,
                    Err(source) => {
                        sessions.force();
                        if !server_done {
                            let server_result = abort_and_join_server(&mut server_task).await;
                            health.abort_and_join_all().await;
                            sessions.wait_empty().await;
                            server_result?;
                        } else {
                            health.abort_and_join_all().await;
                            sessions.wait_empty().await;
                        }
                        lifecycle.enter_failed()?;
                        return Err(RouterError::Signal(source));
                    }
                };
                error!(state = "draining", reason = ?signal, "second signal forced shutdown");
                sessions.force();
                let server_result = if server_done {
                    Ok(())
                } else {
                    abort_and_join_server(&mut server_task).await
                };
                health.abort_and_join_all().await;
                sessions.wait_empty().await;
                server_result?;
                lifecycle.enter_failed()?;
                return Err(RouterError::ForcedShutdown);
            }
            () = tokio::time::sleep_until(deadline) => {
                error!(state = "draining", "graceful shutdown deadline elapsed");
                sessions.force();
                let server_result = if server_done {
                    Ok(())
                } else {
                    abort_and_join_server(&mut server_task).await
                };
                health.abort_and_join_all().await;
                sessions.wait_empty().await;
                server_result?;
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

fn route_table(
    state: AppState,
    generation: Option<Arc<HttpGeneration>>,
    media: Option<Arc<HttpMedia>>,
    websocket: Option<Arc<WebsocketGateway>>,
    local_operations: bool,
    request_context: Arc<RequestContext>,
) -> Router {
    let mut app = Router::new()
        .route("/live", get(live).head(reject_head))
        .route("/ready", get(ready).head(reject_head))
        .with_state(state.clone());
    if let Some(generation) = generation {
        app = app.route(
            "/v1/chat/completions",
            post(http_generation::chat).with_state(generation),
        );
    }
    if let Some(media) = media {
        if media.enables(HttpMediaRoute::Speech) {
            app = app.route(
                "/v1/audio/speech",
                post(http_media::speech).with_state(Arc::clone(&media)),
            );
        }
        if media.enables(HttpMediaRoute::SpeechBatch) {
            app = app.route(
                "/v1/audio/speech/batch",
                post(http_media::batch).with_state(Arc::clone(&media)),
            );
        }
        if media.enables(HttpMediaRoute::Transcription) {
            app = app.route(
                "/v1/audio/transcriptions",
                post(http_media::transcription).with_state(Arc::clone(&media)),
            );
        }
        if media.voice_routes_enabled() {
            app = app
                .route(
                    "/v1/audio/voices",
                    get(http_media::voice::list)
                        .post(http_media::voice::upload)
                        .head(reject_head)
                        .with_state(Arc::clone(&media)),
                )
                .route(
                    "/v1/audio/voices/{name}",
                    delete(http_media::voice::delete).with_state(media),
                );
        }
    }
    if let Some(websocket) = websocket {
        if websocket.speech_enabled() {
            app = app.route(
                "/v1/audio/speech/stream",
                get(websocket::speech).with_state(Arc::clone(&websocket)),
            );
        }
        if websocket.realtime_enabled() {
            app = app.route(
                "/v1/realtime",
                get(websocket::realtime).with_state(websocket),
            );
        }
    }
    if local_operations {
        app = app
            .route(
                "/v1/models",
                get(models).head(reject_head).with_state(state.clone()),
            )
            .route(
                "/metrics",
                get(metrics).head(reject_head).with_state(state.clone()),
            )
            .route(
                "/diagnostics",
                get(diagnostics).head(reject_head).with_state(state),
            );
    }
    app.layer(middleware::from_fn_with_state(
        request_context,
        operations::request_context,
    ))
}

async fn live(State(state): State<AppState>) -> (StatusCode, &'static str) {
    if state.lifecycle.is_live() {
        (StatusCode::OK, LIVE_BODY)
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "not live\n")
    }
}

async fn ready(State(state): State<AppState>) -> (StatusCode, &'static str) {
    if state.is_ready() {
        (StatusCode::OK, READY_BODY)
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, NOT_READY_BODY)
    }
}

async fn models(State(state): State<AppState>, request: Request<Body>) -> Response<Body> {
    match operations::validate_get(&request, "/v1/models") {
        Ok(()) => state.operations.models_response(),
        Err(fault) => fault.into_response(),
    }
}

async fn metrics(State(state): State<AppState>, request: Request<Body>) -> Response<Body> {
    if let Err(fault) = operations::validate_get(&request, "/metrics") {
        return fault.into_response();
    }
    let lifecycle = match state.lifecycle.snapshot() {
        Ok(lifecycle) => lifecycle,
        Err(_) => return crate::error::HttpFault::InternalError.into_response(),
    };
    let snapshot = match state.pool.operations_snapshot() {
        Ok(snapshot) => snapshot,
        Err(_) => return crate::error::HttpFault::InternalError.into_response(),
    };
    operations::metrics_response(
        lifecycle,
        state.is_ready(),
        &snapshot,
        state.pool.telemetry(),
    )
}

async fn diagnostics(State(state): State<AppState>, request: Request<Body>) -> Response<Body> {
    if let Err(fault) = operations::validate_get(&request, "/diagnostics") {
        return fault.into_response();
    }
    let lifecycle = match state.lifecycle.snapshot() {
        Ok(lifecycle) => lifecycle,
        Err(_) => return crate::error::HttpFault::InternalError.into_response(),
    };
    let snapshot = match state.pool.operations_snapshot() {
        Ok(snapshot) => snapshot,
        Err(_) => return crate::error::HttpFault::InternalError.into_response(),
    };
    operations::diagnostics_response(lifecycle, state.is_ready(), &snapshot)
        .unwrap_or_else(crate::error::HttpFault::into_response)
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
