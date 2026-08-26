#![cfg(unix)]
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Real-socket proof for the one-worker direct chat relay.

use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
static SOCKET_TEST_LOCK: Mutex<()> = Mutex::new(());
const DEADLINE: Duration = Duration::from_secs(10);

#[derive(Clone, Debug)]
struct Captured {
    head: String,
    body: Vec<u8>,
}

struct Worker {
    address: SocketAddr,
    stop: Arc<AtomicBool>,
    captured: Arc<Mutex<Vec<Captured>>>,
    thread: Option<JoinHandle<()>>,
    _guard: MutexGuard<'static, ()>,
}

impl Worker {
    fn start() -> Self {
        let guard = SOCKET_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind worker fixture");
        listener
            .set_nonblocking(true)
            .expect("set worker nonblocking");
        let address = listener.local_addr().expect("read worker address");
        let stop = Arc::new(AtomicBool::new(false));
        let captured = Arc::new(Mutex::new(Vec::new()));
        let thread_stop = Arc::clone(&stop);
        let thread_captured = Arc::clone(&captured);
        let thread = thread::spawn(move || {
            let mut connections = Vec::new();
            while !thread_stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _peer)) if !thread_stop.load(Ordering::Acquire) => {
                        let captures = Arc::clone(&thread_captured);
                        connections.push(thread::spawn(move || serve_connection(stream, captures)));
                    }
                    Ok((_stream, _peer)) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                    }
                    Err(error) => panic!("accept worker fixture: {error}"),
                }
                let mut index = 0;
                while index < connections.len() {
                    if connections[index].is_finished() {
                        let joined = connections.swap_remove(index);
                        let _result = joined.join();
                    } else {
                        index += 1;
                    }
                }
            }
            for connection in connections {
                let _result = connection.join();
            }
        });
        Self {
            address,
            stop,
            captured,
            thread: Some(thread),
            _guard: guard,
        }
    }

    fn captures(&self) -> Vec<Captured> {
        self.captured.lock().expect("read captures").clone()
    }

    fn wait_for_requests(&self, count: usize) {
        let deadline = Instant::now() + DEADLINE;
        while self.captures().len() < count {
            assert!(Instant::now() < deadline, "worker did not receive request");
            thread::sleep(Duration::from_millis(2));
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _wake = TcpStream::connect(self.address);
        if let Some(thread) = self.thread.take() {
            let _result = thread.join();
        }
    }
}

fn serve_connection(mut stream: TcpStream, captured: Arc<Mutex<Vec<Captured>>>) {
    stream
        .set_nonblocking(false)
        .expect("set worker connection blocking");
    stream
        .set_read_timeout(Some(DEADLINE))
        .expect("bound worker read");
    stream
        .set_write_timeout(Some(DEADLINE))
        .expect("bound worker write");
    while let Some((head, body)) = read_request(&mut stream) {
        captured.lock().expect("record request").push(Captured {
            head,
            body: body.clone(),
        });
        match body.as_slice() {
            b"not-json" => write_response(
                &mut stream,
                b"HTTP/1.1 422 Unprocessable Entity\r\nContent-Type: application/json\r\nContent-Length: 22\r\nConnection: close\r\n\r\n{\"worker\":\"malformed\"}",
            ),
            b"unsupported-model" => write_response(
                &mut stream,
                b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: 24\r\nConnection: close\r\n\r\n{\"worker\":\"unsupported\"}",
            ),
            b"slow" => {
                write_response(
                    &mut stream,
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\nB\r\ndata: one\n\n\r\n",
                );
                thread::sleep(Duration::from_millis(750));
                write_response(&mut stream, b"E\r\ndata: [DONE]\n\n\r\n0\r\n\r\n");
            }
            b"timeout" => {
                thread::sleep(Duration::from_millis(750));
                write_response(
                    &mut stream,
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}",
                );
            }
            b"reset" => return,
            _ => write_response(
                &mut stream,
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 8\r\nCache-Control: private\r\nCache-Control: max-age=0\r\nSet-Cookie: hidden=1\r\nConnection: keep-alive\r\n\r\n{\"ok\":1}",
            ),
        }
    }
}

fn read_request(stream: &mut TcpStream) -> Option<(String, Vec<u8>)> {
    let mut bytes = Vec::with_capacity(4096);
    let mut chunk = [0_u8; 1024];
    while !bytes.windows(4).any(|window| window == b"\r\n\r\n") {
        let count = stream.read(&mut chunk).ok()?;
        if count == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    let split = bytes.windows(4).position(|window| window == b"\r\n\r\n")? + 4;
    let mut body = bytes.split_off(split);
    let head = String::from_utf8(bytes).ok()?;
    if let Some(length) = header(&head, "content-length").and_then(|value| value.parse().ok()) {
        while body.len() < length {
            let count = stream.read(&mut chunk).ok()?;
            if count == 0 {
                return None;
            }
            body.extend_from_slice(&chunk[..count]);
        }
        body.truncate(length);
    } else if header(&head, "transfer-encoding")
        .is_some_and(|value| value.eq_ignore_ascii_case("chunked"))
    {
        while !body.windows(5).any(|window| window == b"0\r\n\r\n") {
            let count = stream.read(&mut chunk).ok()?;
            if count == 0 {
                return None;
            }
            body.extend_from_slice(&chunk[..count]);
        }
        body = decode_chunks(&body)?;
    } else {
        return None;
    }
    Some((head, body))
}

fn decode_chunks(encoded: &[u8]) -> Option<Vec<u8>> {
    let mut cursor = 0;
    let mut decoded = Vec::new();
    loop {
        let line_end = encoded[cursor..]
            .windows(2)
            .position(|window| window == b"\r\n")?
            + cursor;
        let length =
            usize::from_str_radix(std::str::from_utf8(&encoded[cursor..line_end]).ok()?, 16)
                .ok()?;
        cursor = line_end + 2;
        if length == 0 {
            return Some(decoded);
        }
        let end = cursor.checked_add(length)?;
        decoded.extend_from_slice(encoded.get(cursor..end)?);
        if encoded.get(end..end + 2)? != b"\r\n" {
            return None;
        }
        cursor = end + 2;
    }
}

fn write_response(stream: &mut TcpStream, bytes: &[u8]) {
    let _result = stream.write_all(bytes);
    let _result = stream.flush();
}

fn header<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    head.split("\r\n").skip(1).find_map(|line| {
        let (candidate, value) = line.split_once(':')?;
        candidate.eq_ignore_ascii_case(name).then_some(value.trim())
    })
}

struct RouterProcess {
    child: Child,
    address: SocketAddr,
    directory: PathBuf,
}

impl RouterProcess {
    fn start(worker: SocketAddr, global: u32, timeout_ms: u64, hostname: bool) -> Self {
        let reservation = TcpListener::bind("127.0.0.1:0").expect("reserve router address");
        let address = reservation.local_addr().expect("read router address");
        drop(reservation);
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "sgl-omni-core-http-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&directory).expect("create router test directory");
        let config = directory.join("router.toml");
        let (base_url, resolved_ip) = if hostname {
            (
                format!("http://worker.invalid:{}/", worker.port()),
                String::from("resolved_ip = \"127.0.0.1\"\n"),
            )
        } else {
            (format!("http://{worker}/"), String::new())
        };
        fs::write(
            &config,
            format!(
                "schema_version = 1\n\n[server]\nlisten = \"{address}\"\nmax_connections = 128\n\n[shutdown]\ndrain_timeout_ms = 2000\n\n[logging]\nformat = \"json\"\nfilter = \"info\"\n\n[admission]\nglobal = {global}\n\n[http_generation]\nstreamed_request_max_bytes = 1048576\nconnect_timeout_ms = 2000\nrequest_timeout_ms = {timeout_ms}\npool_idle_timeout_ms = 30000\npool_max_idle_per_host = 8\n\n[[workers]]\nworker_id = \"worker-a\"\nbase_url = \"{base_url}\"\n{resolved_ip}"
            ),
        )
        .expect("write router config");
        let child = Command::new(env!("CARGO_BIN_EXE_sgl-omni-router"))
            .arg("--config")
            .arg(config)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn router");
        let mut process = Self {
            child,
            address,
            directory,
        };
        process.wait_live();
        process
    }

    fn wait_live(&mut self) {
        let deadline = Instant::now() + DEADLINE;
        loop {
            if let Ok(response) = raw_request(
                self.address,
                b"GET /live HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            ) && response.starts_with(b"HTTP/1.1 200")
            {
                return;
            }
            if let Some(status) = self.child.try_wait().expect("poll router") {
                panic!("router exited before liveness: {status}");
            }
            assert!(Instant::now() < deadline, "router did not become live");
            thread::sleep(Duration::from_millis(5));
        }
    }
}

impl Drop for RouterProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _result = Command::new("kill")
                .arg("-TERM")
                .arg(self.child.id().to_string())
                .status();
            let _result = self.child.wait();
        }
        let _result = fs::remove_dir_all(&self.directory);
    }
}

fn raw_request(address: SocketAddr, request: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(500))?;
    stream.set_read_timeout(Some(DEADLINE))?;
    stream.set_write_timeout(Some(DEADLINE))?;
    stream.write_all(request)?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    Ok(response)
}

fn post(address: SocketAddr, body: &[u8], request_id: Option<&str>) -> Vec<u8> {
    let request_id =
        request_id.map_or_else(String::new, |value| format!("X-Request-ID: {value}\r\n"));
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{request_id}Connection: close\r\n\r\n",
        body.len()
    );
    let mut request = head.into_bytes();
    request.extend_from_slice(body);
    raw_request(address, &request).expect("complete routed request")
}

fn status(response: &[u8]) -> u16 {
    let line_end = response
        .windows(2)
        .position(|window| window == b"\r\n")
        .expect("response status line");
    std::str::from_utf8(&response[..line_end])
        .expect("ASCII response status")
        .split_whitespace()
        .nth(1)
        .expect("response status code")
        .parse()
        .expect("numeric response status")
}

fn response_head(response: &[u8]) -> &str {
    let end = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("complete response head");
    std::str::from_utf8(&response[..end]).expect("ASCII response head")
}

#[test]
fn worker_owns_body_semantics_and_receives_exact_bytes_and_request_id() {
    for (body, expected_status, marker) in [
        (&b"not-json"[..], 422, &b"malformed"[..]),
        (&b"unsupported-model"[..], 400, &b"unsupported"[..]),
    ] {
        let worker = Worker::start();
        let router = RouterProcess::start(worker.address, 8, 2_000, true);
        let response = post(router.address, body, Some("caller-id"));
        assert_eq!(
            status(&response),
            expected_status,
            "response={} captures={:?}",
            String::from_utf8_lossy(&response),
            worker.captures()
        );
        assert!(response.windows(marker.len()).any(|part| part == marker));
        assert!(
            response_head(&response)
                .to_ascii_lowercase()
                .contains("x-request-id: caller-id")
        );
        let captures = worker.captures();
        assert_eq!(captures.len(), 1);
        assert_eq!(captures[0].body, body);
        assert_eq!(header(&captures[0].head, "x-request-id"), Some("caller-id"));
        drop(router);
        drop(worker);
    }
}

#[test]
fn strict_envelopes_fail_before_dispatch_and_missing_ids_are_generated() {
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 8, 2_000, false);
    let valid = post(router.address, b"{}", None);
    assert_eq!(status(&valid), 200);
    let generated =
        header(response_head(&valid), "x-request-id").expect("generated downstream request ID");
    worker.wait_for_requests(1);
    assert_eq!(
        header(&worker.captures()[0].head, "x-request-id"),
        Some(generated)
    );

    for (request, expected_status, expected_id) in [
        (
            b"GET /live HTTP/1.1\r\nHost: localhost\r\nX-Request-ID: live-id\r\nConnection: close\r\n\r\n".as_slice(),
            200,
            Some("live-id"),
        ),
        (
            b"GET /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nX-Request-ID: method-id\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".as_slice(),
            405,
            Some("method-id"),
        ),
        (
            b"GET /missing HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".as_slice(),
            404,
            None,
        ),
    ] {
        let response = raw_request(router.address, request).expect("canonical route response");
        assert_eq!(status(&response), expected_status);
        let response_id =
            header(response_head(&response), "x-request-id").expect("canonical response ID");
        if let Some(expected_id) = expected_id {
            assert_eq!(response_id, expected_id);
        } else {
            assert!(response_id.starts_with("sglang-omni-"));
        }
    }

    for (request, expected) in [
        (b"POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}".as_slice(), 400),
        (b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n2\r\n{}\r\n0\r\n\r\n".as_slice(), 400),
    ] {
        let response = raw_request(router.address, request).expect("router envelope response");
        assert_eq!(status(&response), expected);
    }

    for request in [
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\nX-Request-ID: one\r\nX-Request-ID: two\r\nConnection: close\r\n\r\n{}".as_slice(),
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\nX-Request-ID: has space\r\nConnection: close\r\n\r\n{}".as_slice(),
    ] {
        let response = raw_request(router.address, request).expect("invalid request-ID response");
        assert_eq!(status(&response), 400);
        let head = response_head(&response);
        let replacement = header(head, "x-request-id").expect("generated canonical replacement");
        assert!(replacement.starts_with("sglang-omni-"));
        assert!(replacement.len() <= 128);
        assert_eq!(
            head.to_ascii_lowercase().matches("x-request-id:").count(),
            1
        );
    }
    thread::sleep(Duration::from_millis(20));
    assert_eq!(worker.captures().len(), 1);
}

#[test]
fn relay_holds_admission_and_is_not_cut_off_after_commitment() {
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 1, 500, false);

    let address = router.address;
    let slow = thread::spawn(move || post(address, b"slow", Some("slow-id")));
    worker.wait_for_requests(1);
    let overloaded = post(router.address, b"{}", None);
    assert_eq!(status(&overloaded), 429);

    let slow_response = slow.join().expect("join slow client");
    assert_eq!(status(&slow_response), 200);
    assert!(slow_response.windows(6).any(|part| part == b"[DONE]"));

    drop(router);
    drop(worker);
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 1, 500, false);
    let first = post(router.address, b"{}", None);
    assert_eq!(status(&first), 200);
    let head = response_head(&first).to_ascii_lowercase();
    assert_eq!(head.matches("cache-control:").count(), 2);
    assert!(!head.contains("set-cookie:"));
}

#[test]
fn precommit_timeout_and_upstream_reset_are_bounded_and_release_admission() {
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 1, 500, false);

    let timeout = post(router.address, b"timeout", None);
    assert_eq!(
        status(&timeout),
        504,
        "{} captures={:?}",
        String::from_utf8_lossy(&timeout),
        worker.captures()
    );
    thread::sleep(Duration::from_millis(1_000));
    let recovered = post(router.address, b"{}", None);
    assert_ne!(
        status(&recovered),
        429,
        "precommit timeout retained the sole admission permit"
    );

    drop(router);
    drop(worker);
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 1, 500, false);
    let reset = post(router.address, b"reset", None);
    assert_eq!(status(&reset), 502);
}

#[test]
fn early_upload_eof_and_downstream_disconnect_release_admission() {
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 1, 2_000, false);

    let mut short = TcpStream::connect(router.address).expect("connect short upload");
    short
        .set_read_timeout(Some(DEADLINE))
        .expect("bound short response");
    short
        .write_all(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 10\r\nConnection: close\r\n\r\n{}",
        )
        .expect("write short upload");
    short
        .shutdown(std::net::Shutdown::Write)
        .expect("finish short client write");
    let mut short_response = Vec::new();
    short
        .read_to_end(&mut short_response)
        .expect("read short-upload response");
    assert_eq!(status(&short_response), 400);

    let mut disconnect = TcpStream::connect(router.address).expect("connect disconnect client");
    disconnect
        .set_read_timeout(Some(DEADLINE))
        .expect("bound disconnect response");
    disconnect
        .write_all(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 4\r\nConnection: close\r\n\r\nslow",
        )
        .expect("write disconnect request");
    worker.wait_for_requests(1);
    let mut prefix = [0_u8; 256];
    let count = disconnect.read(&mut prefix).expect("read committed prefix");
    assert_ne!(count, 0);
    drop(disconnect);

    thread::sleep(Duration::from_millis(50));
    let released = post(router.address, b"{}", None);
    assert_ne!(
        status(&released),
        429,
        "downstream drop retained the sole admission permit"
    );
}
