#![allow(clippy::expect_used, clippy::panic)]

//! Real-socket ordering and exact-replay tests for terminating WebSockets.

use std::fs;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocketUpgrade};
use axum::http::{StatusCode, Uri};
use axum::routing::get;
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, Notify};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message as ClientMessage;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "sgl-omni-router-websocket-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create websocket test directory");
        Self(path)
    }

    fn config(&self, contents: &str) -> PathBuf {
        let path = self.0.join("router.toml");
        fs::write(&path, contents).expect("write websocket router config");
        path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _removed = fs::remove_dir_all(&self.0);
    }
}

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _killed = self.0.kill();
        let _waited = self.0.wait();
    }
}

#[derive(Clone)]
struct WorkerState {
    speech_config: Arc<Mutex<Option<String>>>,
    realtime_path: Arc<Mutex<Option<String>>>,
    realtime_release: Arc<Notify>,
    realtime_control: Arc<Notify>,
}

const REALTIME_FLOOD: &str = r#"{"type":"test.flood"}"#;
const REALTIME_CONTROL: &str = r#"{"type":"response.cancel"}"#;

async fn health() -> StatusCode {
    StatusCode::OK
}

async fn speech_worker(
    State(state): State<WorkerState>,
    upgrade: WebSocketUpgrade,
) -> impl axum::response::IntoResponse {
    upgrade.on_upgrade(move |mut socket| async move {
        if let Some(Ok(Message::Text(text))) = socket.next().await {
            *state.speech_config.lock().await = Some(text.to_string());
            let _sent = socket
                .send(Message::Text(
                    r#"{"type":"session.configured","worker":"pinned"}"#.into(),
                ))
                .await;
            while let Some(message) = socket.next().await {
                match message {
                    Ok(Message::Text(text)) => {
                        if socket.send(Message::Text(text)).await.is_err() {
                            break;
                        }
                    }
                    Ok(Message::Binary(_)) => {
                        let _sent = socket
                            .send(Message::Text(
                                r#"{"type":"error","message":"speech WebSocket client messages must be text frames"}"#.into(),
                            ))
                            .await;
                    }
                    Ok(Message::Close(frame)) => {
                        let _closed = socket.send(Message::Close(frame)).await;
                        break;
                    }
                    Ok(Message::Ping(_) | Message::Pong(_)) => {}
                    Err(_) => break,
                }
            }
        }
    })
}

async fn realtime_worker(
    State(state): State<WorkerState>,
    uri: Uri,
    upgrade: WebSocketUpgrade,
) -> impl axum::response::IntoResponse {
    *state.realtime_path.lock().await = Some(uri.to_string());
    state.realtime_release.notified().await;
    upgrade.on_upgrade(move |socket| async move {
        let (mut sink, mut stream) = socket.split();
        let _sent = sink
            .send(Message::Text(
                r#"{"type":"session.created","session":{"model":"omni"}}"#.into(),
            ))
            .await;
        while let Some(message) = stream.next().await {
            match message {
                Ok(Message::Text(text)) if text.as_str() == REALTIME_FLOOD => {
                    let payload = axum::extract::ws::Utf8Bytes::from("x".repeat(64 * 1024));
                    let flood = async {
                        loop {
                            if sink.send(Message::Text(payload.clone())).await.is_err() {
                                return;
                            }
                        }
                    };
                    let control = async {
                        while let Some(message) = stream.next().await {
                            match message {
                                Ok(Message::Text(text)) if text.as_str() == REALTIME_CONTROL => {
                                    state.realtime_control.notify_one();
                                    return;
                                }
                                Ok(Message::Close(_)) | Err(_) => return,
                                _ => {}
                            }
                        }
                    };
                    tokio::pin!(flood, control);
                    tokio::select! {
                        () = &mut flood => {}
                        () = &mut control => {}
                    }
                    return;
                }
                Ok(Message::Text(text)) => {
                    if sink.send(Message::Text(text)).await.is_err() {
                        break;
                    }
                }
                Ok(Message::Close(_)) => break,
                Ok(Message::Binary(_) | Message::Ping(_) | Message::Pong(_)) => {}
                Err(_) => break,
            }
        }
    })
}

fn router_config(router: SocketAddr, worker: SocketAddr) -> String {
    format!(
        r#"schema_version = 1

[server]
listen = "{router}"

[shutdown]
drain_timeout_ms = 5000

[logging]
format = "json"
filter = "error"

[router]
required_services = ["speech_websocket", "realtime_websocket"]

[admission]
global = 8
generation_http = 1
speech_http = 1
transcription_http = 1
speech_batch = 1
speech_websocket = 1
realtime_websocket = 1
control = 1

[health]
interval_ms = 100
timeout_ms = 50
success_threshold = 1
failure_threshold = 1
max_concurrent_probes = 1

[websocket.speech]
trust_domain = "local"

[websocket.realtime]
trust_domain = "local"

[[workers]]
worker_id = "pinned-worker"
base_url = "http://branch-five-worker.invalid:{}"
resolved_ip = "127.0.0.1"
trust_domain = "local"
default_model_id = "omni"

[workers.capacity]
speech_websocket = 1
realtime_websocket = 1

[[workers.service_profiles]]
service = "speech_websocket"
model_ids = ["omni"]
input_profiles = ["text"]
response_formats = ["pcm"]
stream_modes = ["non_streaming", "streaming"]
tasks = ["text_to_speech"]
reference_forms = ["none"]
managed_voice = false

[[workers.service_profiles]]
service = "realtime_websocket"
protocols = ["openai_realtime_v1"]
"#,
        worker.port()
    )
}

async fn connect_with_retry(
    url: &str,
) -> tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        match connect_async(url).await {
            Ok((socket, response)) => {
                assert!(response.headers().contains_key("x-request-id"));
                return socket;
            }
            Err(_) if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            Err(error) => panic!("router websocket did not become available: {error}"),
        }
    }
}

async fn wait_ready(address: SocketAddr) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("build readiness client");
    loop {
        if client
            .get(format!("http://{address}/ready"))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
        {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "router did not become ready"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

fn websocket_status(address: SocketAddr, path: &str) -> u16 {
    let mut stream = std::net::TcpStream::connect(address).expect("connect raw websocket client");
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("set raw websocket read timeout");
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {address}\r\nConnection: Upgrade, close\r\nUpgrade: websocket\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"
    )
    .expect("write raw websocket request");
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("read raw websocket response");
    response
        .split_whitespace()
        .nth(1)
        .and_then(|status| status.parse().ok())
        .expect("parse websocket response status")
}

async fn wait_for_healthy_workers(address: SocketAddr, expected: usize) {
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("build metrics client");
    let expected_metric =
        format!("sglang_omni_router_workers_by_health{{health=\"healthy\"}} {expected}\n");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        if let Ok(response) = client.get(format!("http://{address}/metrics")).send().await
            && let Ok(metrics) = response.text().await
            && metrics.contains(&expected_metric)
        {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "workers did not become healthy"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn wait_for_realtime_ownership_zero(address: SocketAddr) {
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("build diagnostics client");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        if let Ok(response) = client
            .get(format!("http://{address}/diagnostics"))
            .send()
            .await
            && let Ok(body) = response.text().await
            && let Ok(diagnostics) = serde_json::from_str::<serde_json::Value>(&body)
        {
            let admission_zero = diagnostics["admission"].as_array().is_some_and(|entries| {
                entries.iter().all(|entry| {
                    !matches!(
                        entry["class"].as_str(),
                        Some("global" | "realtime_websocket")
                    ) || entry["in_flight"].as_u64() == Some(0)
                })
            });
            let exact_zero = diagnostics["workers"].as_array().is_some_and(|workers| {
                workers.iter().all(|worker| {
                    worker["capacity"].as_array().is_some_and(|entries| {
                        entries.iter().all(|entry| {
                            entry["class"].as_str() != Some("realtime_websocket")
                                || entry["in_flight"].as_u64() == Some(0)
                        })
                    })
                })
            });
            if admission_zero && exact_zero {
                return;
            }
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "realtime ownership did not return to zero"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

#[tokio::test]
async fn speech_exact_replay_and_realtime_precommit_and_server_first_ordering() {
    let worker_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind worker fixture");
    let worker_address = worker_listener.local_addr().expect("worker address");
    let router_probe = std::net::TcpListener::bind("127.0.0.1:0").expect("reserve router port");
    let router_address = router_probe.local_addr().expect("router address");
    drop(router_probe);
    let state = WorkerState {
        speech_config: Arc::new(Mutex::new(None)),
        realtime_path: Arc::new(Mutex::new(None)),
        realtime_release: Arc::new(Notify::new()),
        realtime_control: Arc::new(Notify::new()),
    };
    let worker_app = Router::new()
        .route("/health", get(health))
        .route("/v1/audio/speech/stream", get(speech_worker))
        .route("/v1/realtime", get(realtime_worker))
        .with_state(state.clone());
    let worker_task = tokio::spawn(async move {
        axum::serve(worker_listener, worker_app)
            .await
            .expect("serve worker fixture");
    });
    let directory = TestDir::new();
    let config = directory.config(&router_config(router_address, worker_address));
    let child = Command::new(env!("CARGO_BIN_EXE_sgl-omni-router"))
        .arg("--config")
        .arg(config)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start router process");
    let _child = ChildGuard(child);

    wait_ready(router_address).await;

    let speech_url = format!("ws://{router_address}/v1/audio/speech/stream");
    let mut speech = connect_with_retry(&speech_url).await;
    let exact =
        r#"{"type":"session.config","model":"omni","response_format":"pcm","stream_audio":true}"#;
    speech
        .send(ClientMessage::Text(exact.into()))
        .await
        .expect("send speech configuration after downstream 101");
    let configured = speech
        .next()
        .await
        .expect("configured event")
        .expect("valid event");
    assert_eq!(
        configured.into_text().expect("configured text"),
        r#"{"type":"session.configured","worker":"pinned"}"#
    );
    assert_eq!(state.speech_config.lock().await.as_deref(), Some(exact));
    speech
        .send(ClientMessage::Binary(vec![1, 2, 3].into()))
        .await
        .expect("send recoverable binary input");
    let recoverable = speech
        .next()
        .await
        .expect("recoverable response")
        .expect("valid response");
    assert!(
        recoverable
            .into_text()
            .expect("error text")
            .contains("text frames")
    );
    let _closed = speech.close(None).await;
    drop(speech);

    let mut next_speech = connect_with_retry(&speech_url).await;
    next_speech
        .send(ClientMessage::Text(exact.into()))
        .await
        .expect("send configuration after prior permit release");
    assert!(matches!(
        next_speech.next().await,
        Some(Ok(ClientMessage::Text(_)))
    ));
    let _closed = next_speech.close(None).await;
    drop(next_speech);

    let exact_realtime_path =
        "/v1/realtime?unknown=first&model=%6F%6D%6E%69&unknown=second%2fvalue";
    let realtime_url = format!("ws://{router_address}{exact_realtime_path}");
    let mut realtime_request = realtime_url
        .into_client_request()
        .expect("build realtime request");
    realtime_request.headers_mut().insert(
        "x-request-id",
        "client-websocket-1"
            .parse()
            .expect("valid request ID header"),
    );
    realtime_request.headers_mut().insert(
        "authorization",
        "Bearer downstream-secret"
            .parse()
            .expect("valid authorization header"),
    );
    realtime_request
        .headers_mut()
        .insert("cookie", "private=1".parse().expect("valid cookie header"));
    let connect_task = tokio::spawn(async move { connect_async(realtime_request).await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !connect_task.is_finished(),
        "downstream 101 must await upstream handshake"
    );
    state.realtime_release.notify_one();
    let (mut realtime, response) = connect_task
        .await
        .expect("join realtime connect")
        .expect("complete realtime downstream handshake");
    assert_eq!(
        state.realtime_path.lock().await.as_deref(),
        Some(exact_realtime_path)
    );
    assert_eq!(
        response
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok()),
        Some("client-websocket-1")
    );
    let created = realtime
        .next()
        .await
        .expect("session.created")
        .expect("valid event");
    assert_eq!(
        created.into_text().expect("session.created text"),
        r#"{"type":"session.created","session":{"model":"omni"}}"#
    );
    let update = r#"{"type":"session.update","session":{"model":"reflected"}}"#;
    let cancel = r#"{"type":"response.cancel","event_id":"ordered"}"#;
    realtime
        .send(ClientMessage::Text(update.into()))
        .await
        .expect("send realtime model-bearing update");
    realtime
        .send(ClientMessage::Text(cancel.into()))
        .await
        .expect("send ordered realtime control");
    assert_eq!(
        realtime
            .next()
            .await
            .expect("echoed update")
            .expect("valid echoed update")
            .into_text()
            .expect("text update"),
        update
    );
    assert_eq!(
        realtime
            .next()
            .await
            .expect("echoed control")
            .expect("valid echoed control")
            .into_text()
            .expect("text control"),
        cancel
    );
    let _closed = realtime.close(None).await;
    drop(realtime);

    let flood_url = format!("ws://{router_address}/v1/realtime");
    let flood_connect = tokio::spawn(async move { connect_with_retry(&flood_url).await });
    state.realtime_release.notify_one();
    let mut flood_client = flood_connect.await.expect("join flood connection");
    assert!(matches!(
        flood_client.next().await,
        Some(Ok(ClientMessage::Text(_)))
    ));
    flood_client
        .send(ClientMessage::Text(REALTIME_FLOOD.into()))
        .await
        .expect("start sustained worker output");
    tokio::time::sleep(Duration::from_millis(50)).await;
    flood_client
        .send(ClientMessage::Text(REALTIME_CONTROL.into()))
        .await
        .expect("send control while downstream output is unread");
    tokio::time::timeout(Duration::from_secs(2), state.realtime_control.notified())
        .await
        .expect("client-to-worker direction remains live under downstream backpressure");
    drop(flood_client);

    worker_task.abort();
    let _joined = worker_task.await;
}

#[derive(Clone)]
struct HeterogeneousWorkerState {
    model: &'static str,
    handshakes: Arc<AtomicUsize>,
    paths: Arc<Mutex<Vec<String>>>,
}

async fn heterogeneous_realtime_worker(
    State(state): State<HeterogeneousWorkerState>,
    uri: Uri,
    upgrade: WebSocketUpgrade,
) -> impl axum::response::IntoResponse {
    state.handshakes.fetch_add(1, Ordering::Relaxed);
    state.paths.lock().await.push(uri.to_string());
    upgrade.on_upgrade(move |mut socket| async move {
        let created = format!(
            r#"{{"type":"session.created","session":{{"model":"{}"}}}}"#,
            state.model
        );
        if socket.send(Message::Text(created.into())).await.is_err() {
            return;
        }
        while let Some(message) = socket.next().await {
            match message {
                Ok(Message::Text(_)) => {
                    let selected = format!(r#"{{"type":"test.worker","model":"{}"}}"#, state.model);
                    if socket.send(Message::Text(selected.into())).await.is_err() {
                        return;
                    }
                }
                Ok(Message::Close(frame)) => {
                    let _closed = socket.send(Message::Close(frame)).await;
                    return;
                }
                Ok(Message::Binary(_) | Message::Ping(_) | Message::Pong(_)) => {}
                Err(_) => return,
            }
        }
    })
}

fn heterogeneous_router_config(router: SocketAddr, alpha: SocketAddr, beta: SocketAddr) -> String {
    let worker = |id: &str, model: &str, address: SocketAddr| {
        format!(
            r#"
[[workers]]
worker_id = "{id}"
base_url = "http://{id}.invalid:{}"
resolved_ip = "127.0.0.1"
trust_domain = "local"
default_model_id = "{model}"

[workers.capacity]
realtime_websocket = 1

[[workers.service_profiles]]
service = "realtime_websocket"
protocols = ["openai_realtime_v1"]
"#,
            address.port()
        )
    };
    format!(
        r#"schema_version = 1

[server]
listen = "{router}"

[shutdown]
drain_timeout_ms = 5000

[logging]
format = "json"
filter = "error"

[router]
required_services = ["realtime_websocket"]

[admission]
global = 1
generation_http = 1
speech_http = 1
transcription_http = 1
speech_batch = 1
speech_websocket = 1
realtime_websocket = 1
control = 1

[health]
interval_ms = 100
timeout_ms = 50
success_threshold = 1
failure_threshold = 1
max_concurrent_probes = 2

[websocket.realtime]
trust_domain = "local"
{}{}
"#,
        worker("alpha", "omni-alpha", alpha),
        worker("beta", "omni-beta", beta)
    )
}

#[tokio::test]
async fn heterogeneous_realtime_query_selection_is_pinned_and_rejection_is_pre_admission() {
    let alpha_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind alpha worker");
    let beta_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind beta worker");
    let alpha_address = alpha_listener.local_addr().expect("alpha address");
    let beta_address = beta_listener.local_addr().expect("beta address");
    let handshakes = Arc::new(AtomicUsize::new(0));
    let alpha_paths = Arc::new(Mutex::new(Vec::new()));
    let beta_paths = Arc::new(Mutex::new(Vec::new()));
    let alpha_state = HeterogeneousWorkerState {
        model: "omni-alpha",
        handshakes: Arc::clone(&handshakes),
        paths: Arc::clone(&alpha_paths),
    };
    let beta_state = HeterogeneousWorkerState {
        model: "omni-beta",
        handshakes: Arc::clone(&handshakes),
        paths: Arc::clone(&beta_paths),
    };
    let alpha_task = tokio::spawn(async move {
        axum::serve(
            alpha_listener,
            Router::new()
                .route("/health", get(health))
                .route("/v1/realtime", get(heterogeneous_realtime_worker))
                .with_state(alpha_state),
        )
        .await
        .expect("serve alpha worker");
    });
    let beta_task = tokio::spawn(async move {
        axum::serve(
            beta_listener,
            Router::new()
                .route("/health", get(health))
                .route("/v1/realtime", get(heterogeneous_realtime_worker))
                .with_state(beta_state),
        )
        .await
        .expect("serve beta worker");
    });

    let router_probe = std::net::TcpListener::bind("127.0.0.1:0").expect("reserve router port");
    let router_address = router_probe.local_addr().expect("router address");
    drop(router_probe);
    let directory = TestDir::new();
    let config = directory.config(&heterogeneous_router_config(
        router_address,
        alpha_address,
        beta_address,
    ));
    let child = Command::new(env!("CARGO_BIN_EXE_sgl-omni-router"))
        .arg("--config")
        .arg(config)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start heterogeneous router");
    let _child = ChildGuard(child);

    wait_for_healthy_workers(router_address, 2).await;
    wait_ready(router_address).await;
    assert_eq!(
        websocket_status(router_address, "/v1/realtime"),
        400,
        "absent model is ambiguous"
    );
    assert_eq!(
        websocket_status(router_address, "/v1/realtime?model=unknown"),
        422,
        "unknown model is incompatible"
    );
    for path in [
        "/v1/realtime?model=a&model=b",
        "/v1/realtime?model=",
        "/v1/realtime?model=%",
        "/v1/realtime?model=%FF",
    ] {
        assert_eq!(websocket_status(router_address, path), 400, "{path}");
    }
    assert_eq!(handshakes.load(Ordering::Relaxed), 0);
    wait_for_realtime_ownership_zero(router_address).await;

    let beta_path = "/v1/realtime?trace=first&model=omni%2Dbeta&trace=second%2fvalue&flag";
    let (mut beta, _) = connect_async(format!("ws://{router_address}{beta_path}"))
        .await
        .expect("select beta worker");
    let created = beta
        .next()
        .await
        .expect("beta session.created")
        .expect("valid beta event")
        .into_text()
        .expect("beta event text");
    assert!(created.contains(r#""model":"omni-beta""#));
    let later_model = r#"{"type":"session.update","session":{"model":"omni-alpha"}}"#;
    beta.send(ClientMessage::Text(later_model.into()))
        .await
        .expect("send later model-bearing event");
    let pinned = beta
        .next()
        .await
        .expect("pinned worker response")
        .expect("valid pinned response")
        .into_text()
        .expect("pinned response text");
    assert!(pinned.contains(r#""model":"omni-beta""#));
    assert_eq!(alpha_paths.lock().await.len(), 0);
    assert_eq!(beta_paths.lock().await.as_slice(), [beta_path]);
    beta.close(None).await.expect("close beta session");
    drop(beta);
    wait_for_realtime_ownership_zero(router_address).await;

    let (mut alpha, _) = connect_async(format!(
        "ws://{router_address}/v1/realtime?model=omni-alpha"
    ))
    .await
    .expect("select alpha worker");
    let created = alpha
        .next()
        .await
        .expect("alpha session.created")
        .expect("valid alpha event")
        .into_text()
        .expect("alpha event text");
    assert!(created.contains(r#""model":"omni-alpha""#));
    alpha.close(None).await.expect("close alpha session");
    drop(alpha);
    wait_for_realtime_ownership_zero(router_address).await;
    assert_eq!(handshakes.load(Ordering::Relaxed), 2);

    alpha_task.abort();
    beta_task.abort();
    let _alpha = alpha_task.await;
    let _beta = beta_task.await;
}
