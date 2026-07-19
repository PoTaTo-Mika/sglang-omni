#![cfg(unix)]
#![allow(clippy::expect_used, clippy::panic)]

//! Real-socket exact-owner voice CRUD and speech-affinity proof.

use std::convert::Infallible;
use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener as StdTcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes, to_bytes};
use axum::extract::State;
use axum::extract::ws::{Message, WebSocketUpgrade};
use axum::http::header::CONTENT_TYPE;
use axum::http::{Method, Request, Response, StatusCode, Uri};
use axum::routing::{any, get};
use futures_util::{SinkExt, StreamExt, stream};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message as ClientMessage;

const VOICE_MAX: usize = 10_551_296;
const DEADLINE: Duration = Duration::from_secs(10);
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
struct Capture {
    method: Method,
    uri: String,
    content_type: Option<String>,
    body: Bytes,
}

#[derive(Clone)]
struct WorkerState {
    id: &'static str,
    healthy: Arc<AtomicBool>,
    captures: Arc<Mutex<Vec<Capture>>>,
}

async fn health(State(state): State<WorkerState>) -> StatusCode {
    if state.healthy.load(Ordering::Acquire) {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn http_worker(
    State(state): State<WorkerState>,
    uri: Uri,
    request: Request<Body>,
) -> Response<Body> {
    let (parts, body) = request.into_parts();
    let bytes = to_bytes(body, VOICE_MAX + 1024)
        .await
        .expect("bounded worker fixture body");
    state.captures.lock().await.push(Capture {
        method: parts.method,
        uri: uri.to_string(),
        content_type: parts
            .headers
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned),
        body: bytes,
    });
    if uri.path() == "/v1/audio/voices" && uri.query() == Some("hold=1") {
        let body =
            stream::once(async { Ok::<_, Infallible>(Bytes::from_static(br#"{"open":true"#)) })
                .chain(stream::pending());
        return Response::builder()
            .status(StatusCode::OK)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from_stream(body))
            .expect("held response");
    }
    let body = if uri.path().starts_with("/v1/audio/voices") {
        format!(r#"{{"worker":"{}","uri":"{}"}}"#, state.id, uri)
    } else if uri.path() == "/v1/audio/speech/batch" {
        format!(r#"{{"worker":"{}","results":[]}}"#, state.id)
    } else {
        String::from("WAVE")
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(
            CONTENT_TYPE,
            if uri.path().starts_with("/v1/audio/voices") || uri.path() == "/v1/audio/speech/batch"
            {
                "application/json"
            } else {
                "audio/wav"
            },
        )
        .body(Body::from(body))
        .expect("worker response")
}

async fn speech_ws(
    State(state): State<WorkerState>,
    upgrade: WebSocketUpgrade,
) -> impl axum::response::IntoResponse {
    upgrade.on_upgrade(move |mut socket| async move {
        if let Some(Ok(Message::Text(text))) = socket.next().await {
            state.captures.lock().await.push(Capture {
                method: Method::GET,
                uri: String::from("/v1/audio/speech/stream"),
                content_type: None,
                body: Bytes::copy_from_slice(text.as_bytes()),
            });
            let response = format!(r#"{{"type":"session.configured","worker":"{}"}}"#, state.id);
            let _sent = socket.send(Message::Text(response.into())).await;
        }
    })
}

struct Worker {
    address: SocketAddr,
    state: WorkerState,
    task: tokio::task::JoinHandle<()>,
}

impl Worker {
    async fn start(id: &'static str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind voice worker fixture");
        let address = listener.local_addr().expect("voice worker address");
        let state = WorkerState {
            id,
            healthy: Arc::new(AtomicBool::new(true)),
            captures: Arc::new(Mutex::new(Vec::new())),
        };
        let app = Router::new()
            .route("/health", get(health))
            .route("/v1/audio/speech/stream", get(speech_ws))
            .fallback(any(http_worker))
            .with_state(state.clone());
        let task = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("serve voice worker fixture");
        });
        Self {
            address,
            state,
            task,
        }
    }

    async fn captures(&self) -> Vec<Capture> {
        self.state.captures.lock().await.clone()
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.task.abort();
    }
}

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "sgl-omni-router-voice-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create voice test directory");
        Self(path)
    }

    fn config(&self, contents: &str) -> PathBuf {
        let path = self.0.join("router.toml");
        fs::write(&path, contents).expect("write voice router config");
        path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _removed = fs::remove_dir_all(&self.0);
    }
}

struct RouterProcess {
    child: Child,
    address: SocketAddr,
    _directory: TestDir,
}

impl RouterProcess {
    async fn start(config: impl FnOnce(SocketAddr) -> String) -> Self {
        let probe = StdTcpListener::bind("127.0.0.1:0").expect("reserve router port");
        let address = probe.local_addr().expect("router address");
        drop(probe);
        let directory = TestDir::new();
        let path = directory.config(&config(address));
        let child = Command::new(env!("CARGO_BIN_EXE_sgl-omni-router"))
            .arg("--config")
            .arg(path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("start voice router");
        let process = Self {
            child,
            address,
            _directory: directory,
        };
        wait_status(address, "/ready", StatusCode::OK).await;
        process
    }
}

impl Drop for RouterProcess {
    fn drop(&mut self) {
        let _killed = self.child.kill();
        let _waited = self.child.wait();
    }
}

fn config(router: SocketAddr, owner: &Worker, other: &Worker, voice_enabled: bool) -> String {
    let owner_config = if voice_enabled {
        "voice_owner_worker_id = \"owner\""
    } else {
        ""
    };
    let required = if voice_enabled {
        "\"speech_http\", \"speech_batch\", \"transcription_http\", \"speech_websocket\", \"voice_control\""
    } else {
        "\"speech_http\", \"speech_batch\", \"transcription_http\", \"speech_websocket\""
    };
    let mut output = format!(
        r#"schema_version = 1
[server]
listen = "{router}"
[shutdown]
drain_timeout_ms = 5000
[logging]
format = "json"
filter = "error"
[router]
strategy = "least_requests"
required_services = [{required}]
{owner_config}
[admission]
global = 32
generation_http = 1
speech_http = 4
transcription_http = 2
speech_batch = 2
speech_websocket = 2
realtime_websocket = 1
control = 2
[health]
interval_ms = 100
timeout_ms = 50
success_threshold = 1
failure_threshold = 1
max_concurrent_probes = 2
[http_media]
routes = ["speech", "speech_batch", "transcription"]
trust_domain = "local"
buffered_request_max_bytes = 1048576
buffered_request_total_bytes = 12582912
streamed_request_max_bytes = 1048576
connect_timeout_ms = 1000
request_timeout_ms = 10000
pool_idle_timeout_ms = 1000
pool_max_idle_per_host = 2
[websocket.speech]
trust_domain = "local"
"#
    );
    if voice_enabled {
        output.push_str(&worker_config("owner", owner.address, true, true));
    }
    output.push_str(&worker_config("other", other.address, false, false));
    output
}

fn worker_config(id: &str, address: SocketAddr, managed: bool, control: bool) -> String {
    let control_capacity = if control { "control = 1\n" } else { "" };
    let control_profile = if control {
        "\n[[workers.service_profiles]]\nservice = \"voice_control\"\n"
    } else {
        ""
    };
    format!(
        r#"
[[workers]]
worker_id = "{id}"
base_url = "http://{address}"
trust_domain = "local"
default_model_id = "tts"
[workers.capacity]
speech_http = 2
speech_batch = 1
transcription_http = 1
speech_websocket = 1
{control_capacity}[[workers.service_profiles]]
service = "speech_http"
model_ids = ["tts"]
response_formats = ["wav"]
stream_modes = ["non_streaming"]
tasks = ["text_to_speech"]
reference_forms = ["none", "direct"]
managed_voice = {managed}
[[workers.service_profiles]]
service = "speech_batch"
model_ids = ["tts"]
response_formats = ["wav"]
tasks = ["text_to_speech"]
reference_forms = ["none", "direct"]
managed_voice = {managed}
max_batch_size = 4
effective_features = ["voice", "reference"]
[[workers.service_profiles]]
service = "transcription_http"
model_ids = ["tts"]
response_formats = ["json"]
media_profiles = ["audio"]
stream_modes = ["non_streaming"]
[[workers.service_profiles]]
service = "speech_websocket"
model_ids = ["tts"]
input_profiles = ["text"]
response_formats = ["pcm"]
stream_modes = ["non_streaming"]
tasks = ["text_to_speech"]
reference_forms = ["none", "direct"]
managed_voice = {managed}
{control_profile}"#
    )
}

async fn wait_status(address: SocketAddr, path: &str, expected: StatusCode) {
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("build fixture client");
    let deadline = tokio::time::Instant::now() + DEADLINE;
    loop {
        if client
            .get(format!("http://{address}{path}"))
            .send()
            .await
            .is_ok_and(|response| response.status() == expected)
        {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "status did not converge"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

fn raw_request(address: SocketAddr, head: &str, body: &[u8]) -> Vec<u8> {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(1))
        .expect("connect raw fixture request");
    stream.set_read_timeout(Some(DEADLINE)).expect("bound read");
    stream
        .set_write_timeout(Some(DEADLINE))
        .expect("bound write");
    stream
        .write_all(head.as_bytes())
        .expect("write request head");
    stream.write_all(body).expect("write request body");
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .expect("read raw response");
    response
}

#[tokio::test]
async fn exact_owner_voice_state_contract_over_real_sockets() {
    let owner = Worker::start("owner").await;
    let other = Worker::start("other").await;
    let router = RouterProcess::start(|address| config(address, &owner, &other, true)).await;
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("build voice client");
    let base = format!("http://{}", router.address);

    assert_eq!(
        client
            .head(format!("{base}/v1/audio/voices"))
            .send()
            .await
            .expect("rejected voice HEAD")
            .status(),
        StatusCode::METHOD_NOT_ALLOWED
    );

    let multipart = Bytes::from_static(b"--voice-boundary\r\nContent-Disposition: form-data; name=\"name\"\r\n\r\nAlice\r\n--voice-boundary--\r\n");
    let response = client
        .post(format!("{base}/v1/audio/voices?replace=1"))
        .header(CONTENT_TYPE, "multipart/form-data; boundary=voice-boundary")
        .body(multipart.clone())
        .send()
        .await
        .expect("voice upload");
    assert_eq!(response.status(), StatusCode::OK);
    let response = client
        .get(format!("{base}/v1/audio/voices?detail=1"))
        .send()
        .await
        .expect("voice list");
    assert!(response.text().await.expect("list body").contains("owner"));
    let response = client
        .delete(format!("{base}/v1/audio/voices/Alice%20One?purge=1"))
        .send()
        .await
        .expect("voice delete");
    assert!(
        response
            .text()
            .await
            .expect("delete body")
            .contains("owner")
    );

    let owner_captures = owner.captures().await;
    let upload = owner_captures
        .iter()
        .find(|capture| capture.method == Method::POST)
        .expect("owner upload capture");
    assert_eq!(upload.body, multipart);
    assert_eq!(
        owner_captures
            .iter()
            .filter(|capture| capture.method == Method::POST
                && capture.uri.starts_with("/v1/audio/voices"))
            .count(),
        1
    );
    assert_eq!(
        upload.content_type.as_deref(),
        Some("multipart/form-data; boundary=voice-boundary")
    );
    assert!(owner_captures.iter().any(|capture| {
        capture.method == Method::GET && capture.uri == "/v1/audio/voices?detail=1"
    }));
    assert!(owner_captures.iter().any(|capture| {
        capture.method == Method::DELETE && capture.uri == "/v1/audio/voices/Alice%20One?purge=1"
    }));
    assert!(
        other
            .captures()
            .await
            .iter()
            .all(|capture| !capture.uri.starts_with("/v1/audio/voices"))
    );

    let explicit = br#"{"model":"tts","input":"x","voice":"Alice","ref_audio":"sample"}"#;
    client
        .post(format!("{base}/v1/audio/speech"))
        .header(CONTENT_TYPE, "application/json")
        .body(explicit.as_slice())
        .send()
        .await
        .expect("explicit-reference speech");
    assert!(
        other
            .captures()
            .await
            .iter()
            .any(|capture| capture.body == explicit.as_slice())
    );

    let before = owner.captures().await.len();
    let declared = format!(
        "POST /v1/audio/voices HTTP/1.1\r\nHost: {}\r\nContent-Type: multipart/form-data; boundary=x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        router.address,
        VOICE_MAX + 1
    );
    let response = raw_request(router.address, &declared, b"");
    assert!(response.starts_with(b"HTTP/1.1 413"));
    let mut chunked = Vec::with_capacity(VOICE_MAX + 128);
    for size in [VOICE_MAX, 1] {
        chunked.extend_from_slice(format!("{size:x}\r\n").as_bytes());
        chunked.extend(std::iter::repeat_n(b'x', size));
        chunked.extend_from_slice(b"\r\n");
    }
    chunked.extend_from_slice(b"0\r\n\r\n");
    let head = format!(
        "POST /v1/audio/voices HTTP/1.1\r\nHost: {}\r\nContent-Type: multipart/form-data; boundary=x\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
        router.address
    );
    let response = raw_request(router.address, &head, &chunked);
    assert!(response.starts_with(b"HTTP/1.1 413"));
    assert_eq!(owner.captures().await.len(), before);

    let named = br#"{"model":"tts","input":"x","voice":"Alice"}"#;
    let response = client
        .post(format!("{base}/v1/audio/speech"))
        .header(CONTENT_TYPE, "application/json")
        .body(named.as_slice())
        .send()
        .await
        .expect("named speech");
    assert_eq!(response.bytes().await.expect("named speech body"), "WAVE");
    assert!(
        owner
            .captures()
            .await
            .iter()
            .any(|capture| capture.body == named.as_slice())
    );

    let batch = br#"{"model":"tts","items":[{"input":"x","voice":"Alice"}]}"#;
    client
        .post(format!("{base}/v1/audio/speech/batch"))
        .header(CONTENT_TYPE, "application/json")
        .body(batch.as_slice())
        .send()
        .await
        .expect("named batch");
    assert!(
        owner
            .captures()
            .await
            .iter()
            .any(|capture| capture.body == batch.as_slice())
    );

    let default = br#"{"model":"tts","input":"x","voice":"default"}"#;
    client
        .post(format!("{base}/v1/audio/speech"))
        .header(CONTENT_TYPE, "application/json")
        .body(default.as_slice())
        .send()
        .await
        .expect("stateless speech");
    assert!(
        other
            .captures()
            .await
            .iter()
            .any(|capture| capture.body == default.as_slice())
    );

    let ws_url = format!("ws://{}/v1/audio/speech/stream", router.address);
    let (mut socket, _) = connect_async(ws_url)
        .await
        .expect("connect named speech websocket");
    let ws_config = r#"{"type":"session.config","model":"tts","voice":"Alice"}"#;
    socket
        .send(ClientMessage::Text(ws_config.into()))
        .await
        .expect("send named websocket config");
    let configured = socket
        .next()
        .await
        .expect("configured event")
        .expect("valid configured event")
        .into_text()
        .expect("configured text");
    assert!(configured.contains("owner"));
    drop(socket);

    let held = client
        .get(format!("{base}/v1/audio/voices?hold=1"))
        .send()
        .await
        .expect("held list headers");
    assert_eq!(
        client
            .get(format!("{base}/v1/audio/voices?busy=1"))
            .send()
            .await
            .expect("saturated exact owner")
            .status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    drop(held);
    wait_status(router.address, "/v1/audio/voices?reuse=1", StatusCode::OK).await;

    owner.state.healthy.store(false, Ordering::Release);
    wait_status(router.address, "/ready", StatusCode::SERVICE_UNAVAILABLE).await;
    assert_eq!(
        client
            .get(format!("{base}/v1/audio/voices"))
            .send()
            .await
            .expect("unavailable CRUD")
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        client
            .post(format!("{base}/v1/audio/speech"))
            .header(CONTENT_TYPE, "application/json")
            .body(named.as_slice())
            .send()
            .await
            .expect("unavailable named speech")
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        client
            .post(format!("{base}/v1/audio/speech"))
            .header(CONTENT_TYPE, "application/json")
            .body(default.as_slice())
            .send()
            .await
            .expect("available stateless speech")
            .status(),
        StatusCode::OK
    );
    drop(router);

    let stateless = RouterProcess::start(|address| config(address, &owner, &other, false)).await;
    let stateless_base = format!("http://{}", stateless.address);
    assert_eq!(
        client
            .get(format!("{stateless_base}/v1/audio/voices"))
            .send()
            .await
            .expect("absent CRUD route")
            .status(),
        StatusCode::NOT_FOUND
    );
    let builtin = br#"{"model":"tts","input":"x","voice":"BuiltinName"}"#;
    assert_eq!(
        client
            .post(format!("{stateless_base}/v1/audio/speech"))
            .header(CONTENT_TYPE, "application/json")
            .body(builtin.as_slice())
            .send()
            .await
            .expect("stateless built-in name")
            .status(),
        StatusCode::OK
    );
    assert!(
        other
            .captures()
            .await
            .iter()
            .any(|capture| capture.body == builtin.as_slice())
    );
}
