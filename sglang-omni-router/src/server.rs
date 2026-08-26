use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderValue, Method, Request, Response, StatusCode, Version};
use axum::middleware;
use axum::routing::{any, get};
use tokio::sync::{Semaphore, oneshot};
use tokio::task::JoinHandle;
use tracing::{error, info};

use crate::config::{Config, HttpMediaRoute};
use crate::error::{HttpFault, RouterError};
use crate::http_generation::{self, HttpGeneration};
use crate::http_media::{self, HttpMedia};
use crate::lifecycle::Lifecycle;
use crate::operations::Operations;
use crate::request_id::{self, RequestIds};
use crate::shutdown;
use crate::websocket::{self, SessionTracker, WebsocketGateway};
use crate::worker_pool::{HealthSupervisor, HealthTaskError, WorkerPool};

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
    let request_ids = RequestIds::new();
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
        request_ids,
    );
    let listener = tokio::net::TcpListener::bind(config.server.listen)
        .await
        .map_err(RouterError::Bind)?;
    let listener = BoundedTcpListener::new(listener, config.server.max_connections);
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
    request_ids: Arc<RequestIds>,
) -> Router {
    let mut app = Router::new()
        .route("/live", get(live).head(reject_head))
        .route("/ready", get(ready).head(reject_head))
        .with_state(state.clone());
    if let Some(generation) = generation {
        app = app.route(
            http_generation::CHAT_PATH,
            any(http_generation::chat).with_state(generation),
        );
    }
    if let Some(media) = media {
        for route in [
            HttpMediaRoute::Speech,
            HttpMediaRoute::SpeechBatch,
            HttpMediaRoute::Transcription,
            HttpMediaRoute::Translation,
        ] {
            if !media.enables(route) {
                continue;
            }
            let path = route.path();
            app = match route {
                HttpMediaRoute::Speech => {
                    app.route(path, any(http_media::speech).with_state(Arc::clone(&media)))
                }
                HttpMediaRoute::SpeechBatch => {
                    app.route(path, any(http_media::batch).with_state(Arc::clone(&media)))
                }
                HttpMediaRoute::Transcription => app.route(
                    path,
                    any(http_media::transcription).with_state(Arc::clone(&media)),
                ),
                HttpMediaRoute::Translation => app.route(
                    path,
                    any(http_media::translation).with_state(Arc::clone(&media)),
                ),
            };
        }
        if media.voice_routes_enabled() {
            app = app
                .route(
                    "/v1/audio/voices",
                    any(http_media::voice::collection).with_state(Arc::clone(&media)),
                )
                .route(
                    "/v1/audio/voices/{name}",
                    any(http_media::voice::item).with_state(Arc::clone(&media)),
                );
        }
    }
    if let Some(websocket) = websocket {
        if websocket.speech_enabled() {
            app = app.route(
                websocket::SPEECH_PATH,
                get(websocket::speech).with_state(Arc::clone(&websocket)),
            );
        }
        if websocket.realtime_enabled() {
            app = app.route(
                websocket::REALTIME_PATH,
                get(websocket::realtime).with_state(websocket),
            );
        }
    }
    app = app
        .route("/v1/models", any(models).with_state(state.clone()))
        .route("/metrics", any(metrics).with_state(state.clone()))
        .route("/diagnostics", any(diagnostics).with_state(state));
    app.layer(middleware::from_fn_with_state(
        request_ids,
        request_id::canonicalize,
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
    if let Err(fault) = validate_operation(&request) {
        return operation_fault(fault);
    }
    state.operations.models_response()
}

async fn metrics(State(state): State<AppState>, request: Request<Body>) -> Response<Body> {
    if let Err(fault) = validate_operation(&request) {
        return operation_fault(fault);
    }
    let lifecycle = match state.lifecycle.snapshot() {
        Ok(lifecycle) => lifecycle,
        Err(_) => return HttpFault::InternalError.into_response(),
    };
    let snapshot = match state.pool.operations_snapshot() {
        Ok(snapshot) => snapshot,
        Err(_) => return HttpFault::InternalError.into_response(),
    };
    state
        .operations
        .metrics_response(lifecycle, state.is_ready(), &snapshot)
}

async fn diagnostics(State(state): State<AppState>, request: Request<Body>) -> Response<Body> {
    if let Err(fault) = validate_operation(&request) {
        return operation_fault(fault);
    }
    let lifecycle = match state.lifecycle.snapshot() {
        Ok(lifecycle) => lifecycle,
        Err(_) => return HttpFault::InternalError.into_response(),
    };
    let snapshot = match state.pool.operations_snapshot() {
        Ok(snapshot) => snapshot,
        Err(_) => return HttpFault::InternalError.into_response(),
    };
    state
        .operations
        .diagnostics_response(lifecycle, state.is_ready(), &snapshot)
        .unwrap_or_else(HttpFault::into_response)
}

fn validate_operation(request: &Request<Body>) -> Result<(), HttpFault> {
    if request.method() != Method::GET {
        return Err(HttpFault::MethodNotAllowed);
    }
    if request.version() != Version::HTTP_11 {
        return Err(HttpFault::HttpVersionNotSupported);
    }
    if request.uri().query().is_some() {
        return Err(HttpFault::MalformedRequest);
    }
    http_media::validate_bodyless_request(request.headers())
}

fn operation_fault(fault: HttpFault) -> Response<Body> {
    if fault == HttpFault::MethodNotAllowed {
        fault.into_response_with_allow(HeaderValue::from_static("GET"))
    } else {
        fault.into_response()
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
