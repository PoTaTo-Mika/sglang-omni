#![cfg(unix)]
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Real-socket chat request classification, forwarding, and fault proof.

use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
static SOCKET_TEST_LOCK: Mutex<()> = Mutex::new(());
const DEADLINE: Duration = Duration::from_secs(10);
const LARGE_AUDIO_BYTES: usize = 1_048_576;
const LARGE_AUDIO_PREFIX: &[u8] = b"{\"audio\":{\"data\":\"";
const LARGE_AUDIO_SUFFIX: &[u8] = b"\"}}";

#[derive(Clone, Debug)]
struct Captured {
    connection_id: u64,
    mode: u8,
    head: String,
    body: Vec<u8>,
}

struct Worker {
    address: SocketAddr,
    stop: Arc<AtomicBool>,
    mode: Arc<AtomicU8>,
    health_probes: Arc<AtomicU64>,
    captured: Arc<Mutex<Vec<Captured>>>,
    thread: Option<JoinHandle<()>>,
    _socket_test_guard: MutexGuard<'static, ()>,
}

impl Worker {
    fn start() -> Self {
        let socket_test_guard = SOCKET_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind worker fixture");
        listener
            .set_nonblocking(true)
            .expect("set worker listener nonblocking");
        let address = listener.local_addr().expect("read worker address");
        let stop = Arc::new(AtomicBool::new(false));
        let mode = Arc::new(AtomicU8::new(0));
        let health_probes = Arc::new(AtomicU64::new(0));
        let captured = Arc::new(Mutex::new(Vec::new()));
        let thread_stop = Arc::clone(&stop);
        let thread_mode = Arc::clone(&mode);
        let thread_health_probes = Arc::clone(&health_probes);
        let thread_captured = Arc::clone(&captured);
        let thread = thread::spawn(move || {
            let mut connections = Vec::new();
            let mut next_connection_id = 0_u64;
            while !thread_stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _peer)) => {
                        if !thread_stop.load(Ordering::Acquire) {
                            let connection_id = next_connection_id;
                            next_connection_id = next_connection_id.saturating_add(1);
                            let mode = Arc::clone(&thread_mode);
                            let health_probes = Arc::clone(&thread_health_probes);
                            let captured = Arc::clone(&thread_captured);
                            connections.push(thread::spawn(move || {
                                let mut stream = stream;
                                stream
                                    .set_nonblocking(false)
                                    .expect("restore blocking worker socket");
                                stream
                                    .set_read_timeout(Some(DEADLINE))
                                    .expect("bound worker read");
                                stream
                                    .set_write_timeout(Some(DEADLINE))
                                    .expect("bound worker write");
                                while handle_worker(
                                    &mut stream,
                                    connection_id,
                                    mode.load(Ordering::Acquire),
                                    &health_probes,
                                    &captured,
                                ) {}
                            }));
                        }
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                    }
                    Err(error) => panic!("accept worker request: {error}"),
                }
                let mut index = 0;
                while index < connections.len() {
                    if connections[index].is_finished() {
                        let finished = connections.swap_remove(index);
                        let _joined = finished.join();
                    } else {
                        index += 1;
                    }
                }
            }
            for connection in connections {
                let _joined = connection.join();
            }
        });
        Self {
            address,
            stop,
            mode,
            health_probes,
            captured,
            thread: Some(thread),
            _socket_test_guard: socket_test_guard,
        }
    }

    fn set_mode(&self, mode: u8) {
        self.mode.store(mode, Ordering::Release);
    }

    fn chats(&self) -> Vec<Captured> {
        self.captured.lock().expect("read worker captures").clone()
    }

    fn health_probes(&self) -> u64 {
        self.health_probes.load(Ordering::Acquire)
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _wake = TcpStream::connect(self.address);
        if let Some(thread) = self.thread.take() {
            let _joined = thread.join();
        }
    }
}

fn handle_worker(
    stream: &mut TcpStream,
    connection_id: u64,
    mode: u8,
    health_probes: &AtomicU64,
    captured: &Arc<Mutex<Vec<Captured>>>,
) -> bool {
    let Some((head, mut body)) = read_head(stream) else {
        return false;
    };
    let request_line = head.lines().next().unwrap_or_default();
    if request_line.starts_with("GET /health ") {
        health_probes.fetch_add(1, Ordering::AcqRel);
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .expect("write health response");
        return false;
    }
    assert!(request_line.starts_with("POST /v1/chat/completions HTTP/1.1"));
    if mode == 1 {
        captured
            .lock()
            .expect("record early-response request")
            .push(Captured {
                connection_id,
                mode,
                head,
                body,
            });
        stream
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
            )
            .expect("write early response");
        return false;
    }
    if mode == 17 {
        captured
            .lock()
            .expect("record stalled direct request")
            .push(Captured {
                connection_id,
                mode,
                head,
                body,
            });
        thread::sleep(Duration::from_secs(2));
        return false;
    }
    let expected = header_value(&head, "content-length")
        .and_then(|value| value.parse::<usize>().ok())
        .expect("router sends canonical content length");
    while body.len() < expected {
        let mut chunk = [0_u8; 4096];
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return false,
            Ok(count) => body.extend_from_slice(&chunk[..count]),
        }
    }
    body.truncate(expected);
    captured
        .lock()
        .expect("record worker request")
        .push(Captured {
            connection_id,
            mode,
            head: head.clone(),
            body,
        });
    if mode == 5 {
        stream
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n4\r\n{\"a\"\r\n",
            )
            .expect("write first slow response chunk");
        thread::sleep(Duration::from_secs(2));
        let _late_write = stream.write_all(b"3\r\n:1}\r\n0\r\n\r\n");
        return false;
    }
    if mode == 13 {
        thread::sleep(Duration::from_secs(2));
        return false;
    }
    if mode == 18 {
        let response_length =
            LARGE_AUDIO_PREFIX.len() + LARGE_AUDIO_BYTES + LARGE_AUDIO_SUFFIX.len();
        let response_head = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {response_length}\r\nConnection: close\r\n\r\n"
        );
        stream
            .write_all(response_head.as_bytes())
            .expect("write large audio response headers");
        stream
            .write_all(LARGE_AUDIO_PREFIX)
            .expect("write large audio response prefix");
        let chunk = [b'a'; 8_192];
        for _ in 0..(LARGE_AUDIO_BYTES / chunk.len()) {
            stream
                .write_all(&chunk)
                .expect("write large audio response data");
        }
        stream
            .write_all(LARGE_AUDIO_SUFFIX)
            .expect("write large audio response suffix");
        return false;
    }
    let response: &[u8] = match mode {
        2 => b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\nB\r\ndata: one\n\n\r\nE\r\ndata: [DONE]\n\n\r\n0\r\n\r\n",
        3 => b"HTTP/1.1 422 Unprocessable Entity\r\nTransfer-Encoding: chunked\r\nSet-Cookie: private=1\r\nConnection: close\r\n\r\n5\r\nerror\r\n0\r\n\r\n",
        4 => b"HTTP/1.1 302 Found\r\nLocation: http://private.invalid/\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        6 => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Encoding: gzip\r\nContent-Length: 8\r\nConnection: close\r\n\r\nraw-gzip",
        7 => b"",
        8 => b"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/problem+json; note=\"worker;error\"\r\nContent-Length: 5\r\nConnection: close\r\n\r\nerror",
        9 => b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
        10 => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
        11 => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n3\r\nabc\r\n0\r\nX-Worker-Trailer: forbidden\r\n\r\n",
        12 => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 10\r\nConnection: close\r\n\r\nabc",
        14 => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 8\r\nConnection: keep-alive\r\n\r\n{\"ok\":1}",
        15 => b"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json; malformed\r\nContent-Length: 5\r\nConnection: close\r\n\r\nerror",
        16 => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n1\r\n{\r\n1\r\n\"\r\n1\r\no\r\n1\r\nk\r\n1\r\n\"\r\n1\r\n:\r\n1\r\n1\r\n1\r\n}\r\n0\r\n\r\n",
        19 => b"HTTP/1.1 101 Switching Protocols\r\nConnection: upgrade\r\nUpgrade: fixture\r\n\r\n",
        20 => b"HTTP/1.1 600 Fixture Status\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
        21 => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Encoding: gzip\r\nContent-Length: 8\r\nConnection: content-encoding, close\r\n\r\nraw-gzip",
        22 => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"ok\":1}",
        23 => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 8\r\nConnection: content-length, close\r\n\r\n{\"ok\":1}",
        _ => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nCache-Control: private\r\nCache-Control: max-age=0\r\nSet-Cookie: private=1\r\nServer: worker-a\r\nConnection: close\r\n\r\n3\r\n{\"o\r\n5\r\nk\":1}\r\n0\r\n\r\n",
    };
    stream.write_all(response).expect("write worker response");
    mode == 14
}

fn read_head(stream: &mut TcpStream) -> Option<(String, Vec<u8>)> {
    let mut bytes = Vec::with_capacity(4096);
    let mut chunk = [0_u8; 1024];
    while !bytes.windows(4).any(|part| part == b"\r\n\r\n") {
        let count = match stream.read(&mut chunk) {
            Ok(count) => count,
            Err(_) => return None,
        };
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    let split = bytes
        .windows(4)
        .position(|part| part == b"\r\n\r\n")
        .map(|index| index + 4)?;
    let body = bytes.split_off(split);
    Some((String::from_utf8(bytes).ok()?, body))
}

fn header_value<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    head.lines().skip(1).find_map(|line| {
        let (candidate, value) = line.split_once(':')?;
        candidate.eq_ignore_ascii_case(name).then_some(value.trim())
    })
}

struct RouterProcess {
    child: Child,
    address: SocketAddr,
    directory: PathBuf,
}

#[derive(Clone, Copy)]
struct RouterOptions {
    buffered_max: u64,
    buffered_total: u64,
    streamed_max: u64,
    connect_timeout_ms: u64,
    request_timeout_ms: u64,
    health_interval_ms: u64,
    hostname: bool,
}

impl RouterOptions {
    fn standard(buffered_max: u64) -> Self {
        Self {
            buffered_max,
            buffered_total: buffered_max.saturating_mul(2).max(buffered_max),
            streamed_max: 4096,
            connect_timeout_ms: 1000,
            request_timeout_ms: 5000,
            health_interval_ms: 100,
            hostname: false,
        }
    }
}

impl RouterProcess {
    fn start(worker: SocketAddr, buffered_max: u64) -> Self {
        Self::start_with_options(worker, RouterOptions::standard(buffered_max))
    }

    fn start_hostname(worker: SocketAddr, buffered_max: u64) -> Self {
        Self::start_with_options(
            worker,
            RouterOptions {
                hostname: true,
                ..RouterOptions::standard(buffered_max)
            },
        )
    }

    fn start_with_options(worker: SocketAddr, options: RouterOptions) -> Self {
        let reservation = TcpListener::bind("127.0.0.1:0").expect("reserve router port");
        let address = reservation.local_addr().expect("read router port");
        drop(reservation);
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "sgl-omni-chat-http-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&directory).expect("create chat fixture directory");
        let config = directory.join("router.toml");
        let (base_url, resolved_ip) = if options.hostname {
            (
                format!("http://chat-worker.invalid:{}", worker.port()),
                "resolved_ip = \"127.0.0.1\"\n",
            )
        } else {
            (format!("http://{worker}"), "")
        };
        let text = format!(
            r#"schema_version = 1
[server]
listen = "{address}"
max_connections = 32
[shutdown]
drain_timeout_ms = 2000
[logging]
format = "json"
filter = "error"
[router]
required_services = ["generation_http"]
[admission]
global = 8
generation_http = 2
speech_http = 1
transcription_http = 1
speech_batch = 1
speech_websocket = 1
realtime_websocket = 1
control = 1
[health]
interval_ms = {health_interval_ms}
timeout_ms = 50
success_threshold = 1
failure_threshold = 1
max_concurrent_probes = 1
[http_generation]
trust_domain = "local"
buffered_request_max_bytes = {buffered_max}
buffered_request_total_bytes = {buffered_total}
streamed_request_max_bytes = {streamed_max}
connect_timeout_ms = {connect_timeout_ms}
request_timeout_ms = {request_timeout_ms}
pool_idle_timeout_ms = 1000
pool_max_idle_per_host = 2
[[workers]]
worker_id = "worker-a"
base_url = "{base_url}"
{resolved_ip}trust_domain = "local"
default_model_id = "omni"
[workers.capacity]
generation_http = 1
[[workers.service_profiles]]
service = "generation_http"
model_ids = ["omni"]
message_content_forms = ["string", "typed_parts"]
media_placements = []
input_modalities = ["text"]
output_modalities = ["text"]
chat_audio_formats = []
stream_modes = ["non_streaming", "streaming"]
"#,
            buffered_max = options.buffered_max,
            buffered_total = options.buffered_total,
            streamed_max = options.streamed_max,
            connect_timeout_ms = options.connect_timeout_ms,
            request_timeout_ms = options.request_timeout_ms,
            health_interval_ms = options.health_interval_ms,
        );
        fs::write(&config, text).expect("write chat router config");
        let process = Self::launch(address, directory, config);
        process.wait_ready();
        process
    }

    fn start_with_unhealthy_local_and_healthy_remote(
        unavailable_local: SocketAddr,
        remote: SocketAddr,
    ) -> Self {
        let reservation = TcpListener::bind("127.0.0.1:0").expect("reserve trust router port");
        let address = reservation.local_addr().expect("read trust router port");
        drop(reservation);
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "sgl-omni-chat-http-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&directory).expect("create trust fixture directory");
        let config = directory.join("router.toml");
        let text = format!(
            r#"schema_version = 1
[server]
listen = "{address}"
max_connections = 32
[shutdown]
drain_timeout_ms = 2000
[logging]
format = "json"
filter = "error"
[router]
required_services = ["generation_http"]
[admission]
global = 8
generation_http = 2
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
[http_generation]
trust_domain = "local"
buffered_request_max_bytes = 1024
buffered_request_total_bytes = 2048
streamed_request_max_bytes = 4096
connect_timeout_ms = 1000
request_timeout_ms = 5000
pool_idle_timeout_ms = 1000
pool_max_idle_per_host = 2
[[workers]]
worker_id = "local-unavailable"
base_url = "http://{unavailable_local}"
trust_domain = "local"
default_model_id = "omni"
[workers.capacity]
generation_http = 1
[[workers.service_profiles]]
service = "generation_http"
model_ids = ["omni"]
message_content_forms = ["string"]
media_placements = []
input_modalities = ["text"]
output_modalities = ["text"]
chat_audio_formats = []
stream_modes = ["non_streaming"]
[[workers]]
worker_id = "remote-healthy"
base_url = "http://{remote}"
trust_domain = "remote"
default_model_id = "omni"
[workers.capacity]
generation_http = 1
[[workers.service_profiles]]
service = "generation_http"
model_ids = ["omni"]
message_content_forms = ["string"]
media_placements = []
input_modalities = ["text"]
output_modalities = ["text"]
chat_audio_formats = []
stream_modes = ["non_streaming"]
"#,
        );
        fs::write(&config, text).expect("write trust router config");
        let process = Self::launch(address, directory, config);
        process.wait_live();
        process
    }

    fn launch(address: SocketAddr, directory: PathBuf, config: PathBuf) -> Self {
        let child = Command::new(env!("CARGO_BIN_EXE_sgl-omni-router"))
            .arg("--config")
            .arg(config)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .env("HTTP_PROXY", "http://127.0.0.1:1")
            .env("HTTPS_PROXY", "http://127.0.0.1:1")
            .env("ALL_PROXY", "http://127.0.0.1:1")
            .env("http_proxy", "http://127.0.0.1:1")
            .env("https_proxy", "http://127.0.0.1:1")
            .env("all_proxy", "http://127.0.0.1:1")
            .env("NO_PROXY", "")
            .env("no_proxy", "")
            .spawn()
            .expect("spawn chat router");
        Self {
            child,
            address,
            directory,
        }
    }

    fn wait_live(&self) {
        let deadline = Instant::now() + DEADLINE;
        loop {
            if let Ok(response) = exchange(
                self.address,
                b"GET /live HTTP/1.1\r\nHost: router\r\nConnection: close\r\n\r\n",
            ) && response.starts_with(b"HTTP/1.1 200")
            {
                return;
            }
            assert!(Instant::now() < deadline, "router did not become live");
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn wait_ready(&self) {
        let deadline = Instant::now() + DEADLINE;
        loop {
            if let Ok(response) = exchange(
                self.address,
                b"GET /ready HTTP/1.1\r\nHost: router\r\nConnection: close\r\n\r\n",
            ) && response.starts_with(b"HTTP/1.1 200")
            {
                return;
            }
            assert!(Instant::now() < deadline, "router did not become ready");
            thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for RouterProcess {
    fn drop(&mut self) {
        let _killed = self.child.kill();
        let _waited = self.child.wait();
        let _removed = fs::remove_dir_all(&self.directory);
    }
}

fn exchange(address: SocketAddr, request: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(200))?;
    stream.set_read_timeout(Some(DEADLINE))?;
    stream.set_write_timeout(Some(DEADLINE))?;
    stream.write_all(request)?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    Ok(response)
}

fn status(response: &[u8]) -> u16 {
    let line_end = response
        .windows(2)
        .position(|part| part == b"\r\n")
        .expect("response status line");
    std::str::from_utf8(&response[..line_end])
        .expect("response status is ASCII")
        .split_whitespace()
        .nth(1)
        .expect("response status code")
        .parse()
        .expect("numeric response status")
}

fn chat_request(body: &[u8], extra_headers: &str) -> Vec<u8> {
    let mut request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: downstream\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{extra_headers}Connection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    request.extend_from_slice(body);
    request
}

fn poll_capacity_convergence(
    address: SocketAddr,
    request: &[u8],
    expected: u16,
    transitional: &[u16],
) -> Vec<u8> {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let response = exchange(address, request).expect("poll bounded router state");
        let observed = status(&response);
        if observed == expected {
            return response;
        }
        assert!(
            transitional.contains(&observed),
            "unexpected transitional status {observed}: {}",
            String::from_utf8_lossy(&response)
        );
        assert!(
            Instant::now() < deadline,
            "router state did not converge to {expected}; last status {observed}: {}",
            String::from_utf8_lossy(&response)
        );
        thread::sleep(Duration::from_millis(5));
    }
}

fn chunked_chat_request(body: &[u8], extra_headers: &str, trailers: &str) -> Vec<u8> {
    let mut request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: downstream\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n{extra_headers}Connection: close\r\n\r\n"
    )
    .into_bytes();
    for byte in body {
        request.extend_from_slice(b"1\r\n");
        request.push(*byte);
        request.extend_from_slice(b"\r\n");
    }
    request.extend_from_slice(b"0\r\n");
    request.extend_from_slice(trailers.as_bytes());
    request.extend_from_slice(b"\r\n");
    request
}

fn response_body(response: &[u8]) -> Vec<u8> {
    let split = response
        .windows(4)
        .position(|part| part == b"\r\n\r\n")
        .map(|index| index + 4)
        .expect("response has complete headers");
    let head = String::from_utf8(response[..split].to_vec()).expect("response head is ASCII");
    let body = &response[split..];
    if header_value(&head, "transfer-encoding")
        .is_some_and(|value| value.eq_ignore_ascii_case("chunked"))
    {
        decode_chunked(body)
    } else {
        body.to_vec()
    }
}

fn wire_body(response: &[u8]) -> &[u8] {
    let split = response
        .windows(4)
        .position(|part| part == b"\r\n\r\n")
        .map(|index| index + 4)
        .expect("response has complete headers");
    &response[split..]
}

fn decode_chunked(mut bytes: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    loop {
        let line = bytes
            .windows(2)
            .position(|part| part == b"\r\n")
            .expect("chunk size line");
        let size = usize::from_str_radix(
            std::str::from_utf8(&bytes[..line]).expect("chunk size ASCII"),
            16,
        )
        .expect("chunk size hex");
        bytes = &bytes[line + 2..];
        if size == 0 {
            break;
        }
        output.extend_from_slice(&bytes[..size]);
        bytes = &bytes[size + 2..];
    }
    output
}

#[test]
fn production_relay_is_structurally_task_channel_free_and_one_send() {
    let source_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/http_generation");
    let mut combined = String::new();
    for file in [
        "mod.rs",
        "classify.rs",
        "headers.rs",
        "request_body.rs",
        "response_body.rs",
    ] {
        let source = fs::read_to_string(source_root.join(file))
            .unwrap_or_else(|error| panic!("read {file}: {error}"));
        combined.push_str(source.split("#[cfg(test)]").next().unwrap_or(&source));
    }
    for forbidden in [
        "tokio::spawn(",
        "tokio::task::spawn(",
        "JoinSet",
        "mpsc::",
        "oneshot::",
        "broadcast::",
        "watch::channel",
    ] {
        assert!(
            !combined.contains(forbidden),
            "production relay contains forbidden task/channel mechanism {forbidden}"
        );
    }
    assert_eq!(
        combined.matches("tokio::task::spawn_blocking(").count(),
        1,
        "classification must have one bounded blocking task site"
    );
    assert_eq!(
        combined.matches("request.send()").count(),
        1,
        "the production relay must have one syntactic upstream send site"
    );
}

#[test]
fn pinned_hostname_transport_bypasses_dns_proxy_retry_and_decompression() {
    let worker = Worker::start();
    worker.set_mode(6);
    let router = RouterProcess::start_hostname(worker.address, 1024);
    let request = chat_request(br#"{"messages":[{"content":"hello"}]}"#, "");

    let response = exchange(router.address, &request).expect("one hostname relay exchange");
    assert_eq!(status(&response), 200);
    assert_eq!(response_body(&response), b"raw-gzip");
    let head = String::from_utf8_lossy(&response).to_ascii_lowercase();
    assert!(head.contains("content-encoding: gzip"));
    let mut hostname_captures = worker.chats();
    assert_eq!(
        hostname_captures.len(),
        1,
        "hostname success must produce one accepted upstream request"
    );
    let capture = hostname_captures
        .pop()
        .expect("hostname request reached worker");
    assert_eq!(capture.body, br#"{"messages":[{"content":"hello"}]}"#);
    assert!(capture.head.to_ascii_lowercase().contains(&format!(
        "host: chat-worker.invalid:{}",
        worker.address.port()
    )));

    drop(router);
    worker.set_mode(7);
    let router = RouterProcess::start_hostname(worker.address, 1024);
    let before = worker.chats().len();
    let response = exchange(router.address, &request).expect("one failed hostname exchange");
    assert_eq!(status(&response), 502);
    assert_eq!(
        worker.chats().len(),
        before + 1,
        "one failed transport exchange must produce exactly one upstream attempt"
    );

    drop(router);
    worker.set_mode(14);
    let router = RouterProcess::start_hostname(worker.address, 1024);
    let before_reuse = worker.chats().len();
    for _ in 0..2 {
        let response = exchange(router.address, &request).expect("one pooled hostname exchange");
        assert_eq!(status(&response), 200);
        assert_eq!(response_body(&response), br#"{"ok":1}"#);
    }
    let reuse_captures = worker.chats();
    assert_eq!(reuse_captures.len(), before_reuse + 2);
    let reused = &reuse_captures[before_reuse..];
    assert!(reused.iter().all(|capture| capture.mode == 14));
    assert_eq!(
        reused[0].connection_id, reused[1].connection_id,
        "the long-lived generation client did not reuse its idle upstream connection"
    );
}

#[test]
fn readiness_requires_a_healthy_generation_worker_in_the_route_trust() {
    let unavailable = TcpListener::bind("127.0.0.1:0").expect("reserve unavailable local target");
    let unavailable_local = unavailable
        .local_addr()
        .expect("read unavailable local target");
    drop(unavailable);

    let remote = Worker::start();
    let router = RouterProcess::start_with_unhealthy_local_and_healthy_remote(
        unavailable_local,
        remote.address,
    );
    let health_deadline = Instant::now() + DEADLINE;
    while remote.health_probes() < 2 {
        assert!(
            Instant::now() < health_deadline,
            "remote-trust worker did not complete repeated successful health probes"
        );
        thread::sleep(Duration::from_millis(5));
    }

    let response = exchange(
        router.address,
        b"GET /ready HTTP/1.1\r\nHost: router\r\nConnection: close\r\n\r\n",
    )
    .expect("one trust-scoped readiness exchange");
    assert_eq!(
        status(&response),
        503,
        "a healthy required-service worker in another trust must not make chat ready"
    );
    assert!(remote.chats().is_empty());
}

#[test]
fn exact_route_method_version_and_ingress_header_matrix() {
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 1024);
    let body = br#"{"messages":[{"content":"hello"}]}"#;
    let length = body.len();
    let cases = [
        (
            "GET exact chat route",
            b"GET /v1/chat/completions HTTP/1.1\r\nHost: downstream\r\nConnection: close\r\n\r\n".to_vec(),
            405,
            false,
        ),
        (
            "POST trailing slash",
            format!("POST /v1/chat/completions/ HTTP/1.1\r\nHost: downstream\r\nContent-Type: application/json\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n{}", String::from_utf8_lossy(body)).into_bytes(),
            404,
            false,
        ),
        (
            "forbidden generate route",
            b"POST /generate HTTP/1.1\r\nHost: downstream\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}".to_vec(),
            404,
            false,
        ),
        (
            "query rejected by exact handler",
            format!("POST /v1/chat/completions?key=private HTTP/1.1\r\nHost: downstream\r\nContent-Type: application/json\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n{}", String::from_utf8_lossy(body)).into_bytes(),
            400,
            false,
        ),
        (
            "HTTP/1.0 rejected",
            format!("POST /v1/chat/completions HTTP/1.0\r\nHost: downstream\r\nContent-Type: application/json\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n{}", String::from_utf8_lossy(body)).into_bytes(),
            505,
            false,
        ),
        (
            "missing content type",
            format!("POST /v1/chat/completions HTTP/1.1\r\nHost: downstream\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n{}", String::from_utf8_lossy(body)).into_bytes(),
            415,
            false,
        ),
        (
            "wrong content type",
            format!("POST /v1/chat/completions HTTP/1.1\r\nHost: downstream\r\nContent-Type: text/plain\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n{}", String::from_utf8_lossy(body)).into_bytes(),
            415,
            false,
        ),
        (
            "repeated content type",
            format!("POST /v1/chat/completions HTTP/1.1\r\nHost: downstream\r\nContent-Type: application/json\r\nContent-Type: application/json\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n{}", String::from_utf8_lossy(body)).into_bytes(),
            415,
            false,
        ),
        (
            "identity content encoding",
            chat_request(body, "Content-Encoding: identity\r\n"),
            200,
            true,
        ),
        (
            "unsupported content encoding",
            chat_request(body, "Content-Encoding: gzip\r\n"),
            415,
            false,
        ),
        (
            "expect rejected before body ownership",
            chat_request(body, "Expect: 100-continue\r\n"),
            417,
            false,
        ),
        (
            "trailer declaration rejected",
            chunked_chat_request(body, "Trailer: X-Checksum\r\n", ""),
            400,
            false,
        ),
    ];

    let mut expected_captures = 0;
    for (name, request, expected_status, reaches_worker) in cases {
        let response = exchange(router.address, &request).unwrap_or_else(|error| {
            panic!("{name}: exchange failed: {error}");
        });
        assert_eq!(
            status(&response),
            expected_status,
            "{name}: {}",
            String::from_utf8_lossy(&response)
        );
        if reaches_worker {
            expected_captures += 1;
        }
        assert_eq!(worker.chats().len(), expected_captures, "{name}");
    }
}

#[test]
fn hyper_framing_boundary_trailers_and_one_byte_chunks_preserve_decoded_bytes() {
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 1024);
    let body = br#"{"messages":[{"content":"one-byte"}]}"#;

    let one_byte = chunked_chat_request(body, "", "");
    let response = exchange(router.address, &one_byte).expect("one-byte chunk exchange");
    assert_eq!(status(&response), 200);
    assert_eq!(worker.chats().last().expect("one-byte capture").body, body);

    let actual_trailer = chunked_chat_request(body, "", "X-Checksum: forbidden\r\n");
    let before = worker.chats().len();
    let response = exchange(router.address, &actual_trailer).expect("send actual trailer");
    assert_eq!(status(&response), 400);
    assert_eq!(worker.chats().len(), before);

    // Hyper applies RFC framing before Axum creates HeaderMap: when both raw
    // Content-Length and final chunked Transfer-Encoding are present, the
    // handler observes transfer framing only. The decoded body remains
    // bounded and is forwarded with a fresh canonical Content-Length.
    let normalized = chunked_chat_request(body, &format!("Content-Length: {}\r\n", body.len()), "");
    let response = exchange(router.address, &normalized).expect("normalized framing exchange");
    let normalized_captures = worker.chats();
    assert_eq!(
        status(&response),
        200,
        "Hyper boundary changed: {}; captures={normalized_captures:?}",
        String::from_utf8_lossy(&response),
    );
    let capture = normalized_captures.last().expect("normalized capture");
    assert_eq!(capture.body, body);
    assert_eq!(
        header_value(&capture.head, "content-length").and_then(|value| value.parse().ok()),
        Some(body.len())
    );
    assert!(header_value(&capture.head, "transfer-encoding").is_none());
}

#[test]
fn ingress_and_precommit_deadlines_release_capacity_with_408_and_504() {
    let options = RouterOptions {
        buffered_total: 32,
        connect_timeout_ms: 250,
        request_timeout_ms: 750,
        health_interval_ms: 100,
        ..RouterOptions::standard(16)
    };
    let buffered_options = RouterOptions {
        buffered_total: 256,
        connect_timeout_ms: 250,
        request_timeout_ms: 750,
        health_interval_ms: 100,
        ..RouterOptions::standard(128)
    };
    let body = br#"{"messages":[{"content":"hello"}]}"#;
    let request = chat_request(body, "");
    let incomplete = |address, declared| {
        let mut stream = TcpStream::connect(address).expect("connect incomplete ingress");
        stream
            .set_read_timeout(Some(DEADLINE))
            .expect("bound incomplete read");
        write!(
            stream,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: downstream\r\nContent-Type: application/json\r\nContent-Length: {declared}\r\nConnection: close\r\n\r\n{{"
        )
        .expect("write incomplete request");
        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .expect("read incomplete-ingress timeout");
        response
    };

    {
        let worker = Worker::start();
        let router = RouterProcess::start_with_options(worker.address, options);
        let response = incomplete(router.address, 8_u64);
        assert_eq!(
            status(&response),
            408,
            "buffered ingress deadline: {}",
            String::from_utf8_lossy(&response)
        );
        let released = exchange(router.address, &request).expect("request after buffered 408");
        assert_eq!(status(&released), 200);
        assert_eq!(response_body(&released), br#"{"ok":1}"#);
    }

    {
        let worker = Worker::start();
        worker.set_mode(17);
        let router = RouterProcess::start_with_options(worker.address, options);
        let response = incomplete(router.address, 512_u64);
        assert_eq!(
            status(&response),
            408,
            "direct ingress deadline: {}",
            String::from_utf8_lossy(&response)
        );
        assert_eq!(
            worker.chats().last().map(|capture| capture.mode),
            Some(17),
            "direct 408 must accompany an accepted stalled upstream request"
        );
        worker.set_mode(0);
        let released = exchange(router.address, &request).expect("request after direct 408");
        assert_eq!(status(&released), 200);
        assert_eq!(response_body(&released), br#"{"ok":1}"#);
    }

    {
        let worker = Worker::start();
        worker.set_mode(13);
        let router = RouterProcess::start_with_options(worker.address, buffered_options);
        let response = exchange(router.address, &request).expect("one no-header exchange");
        assert_eq!(
            status(&response),
            504,
            "completed ingress must use upstream timeout: {}",
            String::from_utf8_lossy(&response),
        );
        let no_header_captures = worker.chats();
        assert_eq!(
            no_header_captures.last().map(|capture| capture.mode),
            Some(13),
            "no-header request did not reach the selected fixture mode"
        );
        assert_eq!(
            no_header_captures.len(),
            1,
            "the completed-ingress timeout proof must record one accepted upstream request"
        );
        worker.set_mode(0);
        let released = exchange(router.address, &request).expect("request after 504");
        assert_eq!(status(&released), 200);
        assert_eq!(response_body(&released), br#"{"ok":1}"#);
    }

    {
        let worker = Worker::start();
        worker.set_mode(5);
        let router = RouterProcess::start_with_options(worker.address, buffered_options);
        let response = exchange(router.address, &request).expect("delayed postcommit body");
        assert_eq!(status(&response), 200);
        assert_eq!(
            response_body(&response),
            br#"{"a":1}"#,
            "the precommit request deadline must not truncate a committed body"
        );
        worker.set_mode(0);
        let released = exchange(router.address, &request).expect("request after delayed EOF");
        assert_eq!(status(&released), 200);
        assert_eq!(response_body(&released), br#"{"ok":1}"#);
    }
}

#[test]
fn aggregate_unknown_length_budget_overloads_and_is_exactly_conserved() {
    let worker = Worker::start();
    let router = RouterProcess::start_with_options(
        worker.address,
        RouterOptions {
            buffered_total: 128,
            request_timeout_ms: 2000,
            ..RouterOptions::standard(128)
        },
    );
    let mut partial = TcpStream::connect(router.address).expect("connect partial ingress");
    partial
        .set_nodelay(true)
        .expect("send partial ingress promptly");
    partial
        .set_write_timeout(Some(DEADLINE))
        .expect("bound partial write");
    partial
        .write_all(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: downstream\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n1\r\n{\r\n",
        )
        .expect("hold one full unknown-length reservation");
    thread::sleep(Duration::from_millis(25));

    let incompatible = br#"{"messages":[{"content":[{"type":"image"}]}]}"#;
    let contender = chunked_chat_request(incompatible, "", "");
    let overload_deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let response = exchange(router.address, &contender).expect("probe byte budget");
        match status(&response) {
            429 => break,
            422 => assert!(
                Instant::now() < overload_deadline,
                "partial ingress did not acquire its full logical reservation"
            ),
            observed => panic!(
                "unexpected budget probe status {observed}: {}",
                String::from_utf8_lossy(&response)
            ),
        }
        thread::sleep(Duration::from_millis(5));
    }
    assert!(worker.chats().is_empty(), "budget probes must not dispatch");

    drop(partial);
    let released = poll_capacity_convergence(router.address, &contender, 422, &[429]);
    assert_eq!(status(&released), 422);
    let ordinary = chat_request(br#"{"messages":[{"content":"hello"}]}"#, "");
    let response =
        exchange(router.address, &ordinary).expect("ordinary request after budget release");
    assert_eq!(status(&response), 200);
    assert_eq!(response_body(&response), br#"{"ok":1}"#);
}

#[test]
fn buffered_and_chunked_ingress_preserve_bytes_and_sanitize_both_directions() {
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 1024);
    let body = br#"{"model":"omni","messages":[{"content":"hello"}],"temperature":0.25}"#;
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: downstream\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAuthorization: secret\r\nCookie: private=1\r\nX-Smg-Target-Worker: forbidden\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let mut wire = request.into_bytes();
    wire.extend_from_slice(body);
    let response = exchange(router.address, &wire).expect("one buffered relay exchange");
    assert!(
        response.starts_with(b"HTTP/1.1 200"),
        "unexpected buffered response: {}",
        String::from_utf8_lossy(&response)
    );
    assert_eq!(response_body(&response), br#"{"ok":1}"#);
    let response_head = String::from_utf8_lossy(&response);
    assert!(!response_head.to_ascii_lowercase().contains("set-cookie"));
    assert!(
        !response_head
            .to_ascii_lowercase()
            .contains("server: worker")
    );
    let first = worker.chats().pop().expect("worker observed buffered chat");
    assert_eq!(first.body, body);
    assert!(
        first
            .head
            .to_ascii_lowercase()
            .contains(&format!("host: {}", worker.address))
    );
    for forbidden in ["authorization", "cookie", "x-smg-target-worker"] {
        assert!(!first.head.to_ascii_lowercase().contains(forbidden));
    }

    let chunked_body = br#"{"messages":[{"content":"chunked"}]}"#;
    let split = 7;
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: downstream\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:X}\r\n{}\r\n{:X}\r\n{}\r\n0\r\n\r\n",
        split,
        std::str::from_utf8(&chunked_body[..split]).expect("ASCII JSON"),
        chunked_body.len() - split,
        std::str::from_utf8(&chunked_body[split..]).expect("ASCII JSON"),
    );
    let response = exchange(router.address, request.as_bytes()).expect("one chunked exchange");
    assert!(
        response.starts_with(b"HTTP/1.1 200"),
        "unexpected chunked response: {}",
        String::from_utf8_lossy(&response)
    );
    let captures = worker.chats();
    let second = captures.last().expect("worker observed chunked chat");
    assert_eq!(second.body, chunked_body);
    assert_eq!(
        header_value(&second.head, "content-length").and_then(|value| value.parse().ok()),
        Some(chunked_body.len())
    );
    assert!(header_value(&second.head, "transfer-encoding").is_none());

    const EMPTY_TYPED: &[u8] = b"{\"messages\":[{\"content\":[]}]}";
    const UNKNOWN_TYPED: &[u8] =
        b"{\"messages\":[{\"content\":[{\"type\":\"vendor_future_part\"}]}]}";
    let before_empty_constraints = worker.chats().len();
    for typed_body in [EMPTY_TYPED, UNKNOWN_TYPED] {
        let request = chat_request(typed_body, "");
        let response = exchange(router.address, &request).expect("one empty-input exchange");
        assert_eq!(status(&response), 200);
    }
    let captures = worker.chats();
    assert_eq!(captures.len(), before_empty_constraints + 2);
    assert_eq!(captures[captures.len() - 2].body, EMPTY_TYPED);
    assert_eq!(captures[captures.len() - 1].body, UNKNOWN_TYPED);
}

#[test]
fn homogeneous_large_upload_uses_direct_body_and_rejects_early_response() {
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 64);
    let padding = "x".repeat(256);
    let body =
        format!(r#"{{"model":"omni","messages":[{{"content":"hello"}}],"ignored":"{padding}"}}"#);
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: downstream\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let response = exchange(router.address, request.as_bytes()).expect("one direct exchange");
    assert!(
        response.starts_with(b"HTTP/1.1 200"),
        "unexpected direct response: {}",
        String::from_utf8_lossy(&response)
    );
    assert_eq!(
        worker.chats().last().expect("large capture").body,
        body.as_bytes()
    );

    let captures_before_identity = worker.chats().len();
    let identity = chat_request(body.as_bytes(), "Content-Encoding: identity\r\n");
    let response = exchange(router.address, &identity).expect("one identity-encoded exchange");
    assert_eq!(
        status(&response),
        413,
        "a present identity Content-Encoding must disqualify the direct path"
    );
    assert_eq!(worker.chats().len(), captures_before_identity);

    drop(router);
    worker.set_mode(1);
    let router = RouterProcess::start(worker.address, 64);
    let captures_before_early = worker.chats().len();
    let mut stream = TcpStream::connect(router.address).expect("connect early-response client");
    stream
        .set_read_timeout(Some(DEADLINE))
        .expect("bound early read");
    stream
        .write_all(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: downstream\r\nContent-Type: application/json\r\nContent-Length: 512\r\nConnection: close\r\n\r\n{",
        )
        .expect("write partial direct upload");
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .expect("read early rejection");
    assert!(response.starts_with(b"HTTP/1.1 502"));
    assert!(
        response_body(&response)
            .windows(b"upstream_protocol_error".len())
            .any(|part| part == b"upstream_protocol_error")
    );
    let early_captures = worker.chats();
    assert_eq!(early_captures.len(), captures_before_early + 1);
    assert_eq!(early_captures.last().map(|capture| capture.mode), Some(1));
}

#[test]
fn response_status_content_type_and_postcommit_terminal_matrix() {
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 1024);
    let request = chat_request(br#"{"messages":[{"content":"hello"}]}"#, "");

    worker.set_mode(2);
    let streaming_body = br#"{"messages":[{"content":"hello"}],"stream":true}"#;
    let streaming_head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: downstream\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        streaming_body.len()
    );
    let mut streaming_request = streaming_head.into_bytes();
    streaming_request.extend_from_slice(streaming_body);
    let response = exchange(router.address, &streaming_request).expect("one SSE exchange");
    assert_eq!(status(&response), 200);
    assert_eq!(response_body(&response), b"data: one\n\ndata: [DONE]\n\n");

    worker.set_mode(16);
    let response = exchange(router.address, &request).expect("one one-byte response exchange");
    assert_eq!(status(&response), 200);
    assert_eq!(response_body(&response), br#"{"ok":1}"#);

    worker.set_mode(18);
    let response = exchange(router.address, &request).expect("one large response exchange");
    let large_audio = response_body(&response);
    assert_eq!(
        large_audio.len(),
        LARGE_AUDIO_PREFIX.len() + LARGE_AUDIO_BYTES + LARGE_AUDIO_SUFFIX.len()
    );
    assert!(large_audio.starts_with(LARGE_AUDIO_PREFIX));
    assert!(large_audio.ends_with(LARGE_AUDIO_SUFFIX));
    assert!(
        large_audio[LARGE_AUDIO_PREFIX.len()..LARGE_AUDIO_PREFIX.len() + LARGE_AUDIO_BYTES]
            .iter()
            .all(|byte| *byte == b'a')
    );

    for (mode, name) in [
        (22, "absent upstream Content-Length"),
        (23, "Connection-nominated upstream Content-Length"),
    ] {
        worker.set_mode(mode);
        let response = exchange(router.address, &request).expect("one unknown-size relay exchange");
        assert_eq!(status(&response), 200, "{name}");
        assert_eq!(response_body(&response), br#"{"ok":1}"#, "{name}");
        let head = String::from_utf8_lossy(&response).to_ascii_lowercase();
        let head = head.split_once("\r\n\r\n").expect("response head").0;
        assert!(
            !head.contains("content-length:"),
            "{name} was re-synthesized downstream: {head}"
        );
    }

    for (mode, expected_status, name) in [
        (3, 422, "trusted 4xx"),
        (8, 500, "trusted 5xx with quoted Content-Type parameter"),
    ] {
        worker.set_mode(mode);
        let response = exchange(router.address, &request).expect("one trusted-error exchange");
        assert_eq!(status(&response), expected_status, "{name}");
        assert_eq!(response_body(&response), b"error", "{name}");
        assert!(
            !String::from_utf8_lossy(&response)
                .to_ascii_lowercase()
                .contains("set-cookie"),
            "{name} leaked a denied header"
        );
    }

    for (mode, name) in [
        (4, "redirect"),
        (9, "wrong success Content-Type"),
        (10, "repeated success Content-Type"),
        (15, "malformed trusted-error Content-Type"),
        (19, "101 informational/upgrade response"),
        (20, "status above 599"),
        (21, "Connection-nominated Content-Encoding"),
    ] {
        worker.set_mode(mode);
        let response = exchange(router.address, &request).expect("one rejected-response exchange");
        assert_eq!(
            status(&response),
            502,
            "{name}: {}",
            String::from_utf8_lossy(&response)
        );
        assert!(response_body(&response).len() <= 256, "{name}");
        assert!(
            !String::from_utf8_lossy(&response)
                .to_ascii_lowercase()
                .contains("location:"),
            "{name} leaked Location"
        );
        worker.set_mode(0);
        let released = exchange(router.address, &request).expect("one post-rejection exchange");
        assert_eq!(status(&released), 200);
        assert_eq!(response_body(&released), br#"{"ok":1}"#);
    }

    for (mode, expected_fragment, name) in [
        (11, &b"abc"[..], "response trailer"),
        (12, &b"abc"[..], "mid-body upstream failure"),
    ] {
        worker.set_mode(mode);
        let response =
            exchange(router.address, &request).expect("one postcommit-terminal exchange");
        assert_eq!(status(&response), 200, "{name}");
        assert!(
            wire_body(&response)
                .windows(expected_fragment.len())
                .any(|window| window == expected_fragment),
            "{name} lost the bytes committed before termination: {:?}",
            wire_body(&response)
        );
        assert!(
            !String::from_utf8_lossy(&response)
                .to_ascii_lowercase()
                .contains("x-worker-trailer"),
            "{name} forwarded a trailer"
        );
        worker.set_mode(0);
        let released = exchange(router.address, &request).expect("one post-terminal exchange");
        assert_eq!(status(&released), 200);
        assert_eq!(response_body(&released), br#"{"ok":1}"#);
    }
}

#[test]
fn downstream_disconnect_drops_response_body_and_releases_exact_worker_lease() {
    let worker = Worker::start();
    worker.set_mode(5);
    let router = RouterProcess::start(worker.address, 1024);
    let request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: downstream\r\nContent-Type: application/json\r\nContent-Length: 34\r\nConnection: close\r\n\r\n{\"messages\":[{\"content\":\"hello\"}]}";
    let mut stream = TcpStream::connect(router.address).expect("connect disconnect client");
    stream
        .set_read_timeout(Some(DEADLINE))
        .expect("bound disconnect read");
    stream.write_all(request).expect("write disconnect request");
    let mut first = Vec::with_capacity(512);
    while !first.windows(4).any(|part| part == b"\r\n\r\n") {
        let mut chunk = [0_u8; 512];
        let count = stream
            .read(&mut chunk)
            .expect("read first relayed response headers");
        assert!(
            count > 0,
            "disconnect fixture closed before response headers"
        );
        first.extend_from_slice(&chunk[..count]);
    }
    assert_eq!(status(&first), 200);

    worker.set_mode(0);
    let saturated = exchange(router.address, request).expect("probe exact capacity while held");
    assert_eq!(
        status(&saturated),
        429,
        "observable 200 commitment plus exact 429 proves capacity-one lease retention"
    );
    drop(stream);

    let released = poll_capacity_convergence(router.address, request, 200, &[429]);
    assert_eq!(response_body(&released), br#"{"ok":1}"#);
}
