#![allow(clippy::expect_used, clippy::panic)]

//! Real-socket ordering and exact-replay tests for terminating WebSockets.

use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::routing::get;
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, Notify};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message as ClientMessage;

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
    upgrade: WebSocketUpgrade,
) -> impl axum::response::IntoResponse {
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
            Ok((socket, _)) => return socket,
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

    let realtime_url = format!("ws://{router_address}/v1/realtime?model=ignored");
    let connect_task = tokio::spawn(async move { connect_async(realtime_url).await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !connect_task.is_finished(),
        "downstream 101 must await upstream handshake"
    );
    state.realtime_release.notify_one();
    let (mut realtime, _) = connect_task
        .await
        .expect("join realtime connect")
        .expect("complete realtime downstream handshake");
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
