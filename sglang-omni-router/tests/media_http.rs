#![cfg(unix)]
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Real-socket media request classification, exact-byte forwarding, and relay proof.

use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
static SOCKET_LOCK: Mutex<()> = Mutex::new(());
const DEADLINE: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MediaRoute {
    Speech,
    SpeechBatch,
    Transcription,
}

impl MediaRoute {
    const ALL: [Self; 3] = [Self::Speech, Self::SpeechBatch, Self::Transcription];

    const fn route_name(self) -> &'static str {
        match self {
            Self::Speech => "speech",
            Self::SpeechBatch => "speech_batch",
            Self::Transcription => "transcription",
        }
    }

    const fn service_name(self) -> &'static str {
        match self {
            Self::Speech => "speech_http",
            Self::SpeechBatch => "speech_batch",
            Self::Transcription => "transcription_http",
        }
    }

    const fn path(self) -> &'static str {
        match self {
            Self::Speech => "/v1/audio/speech",
            Self::SpeechBatch => "/v1/audio/speech/batch",
            Self::Transcription => "/v1/audio/transcriptions",
        }
    }
}

const MEDIA_ROUTE_SUBSETS: [&[MediaRoute]; 7] = [
    &[MediaRoute::Speech],
    &[MediaRoute::SpeechBatch],
    &[MediaRoute::Transcription],
    &[MediaRoute::Speech, MediaRoute::SpeechBatch],
    &[MediaRoute::Speech, MediaRoute::Transcription],
    &[MediaRoute::SpeechBatch, MediaRoute::Transcription],
    &MediaRoute::ALL,
];

#[derive(Clone, Debug)]
struct Captured {
    path: String,
    content_type: String,
    body: Vec<u8>,
}

struct Worker {
    address: SocketAddr,
    stop: Arc<AtomicBool>,
    health_completed: Arc<(Mutex<bool>, Condvar)>,
    captured: Arc<Mutex<Vec<Captured>>>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind media worker fixture");
        let address = listener.local_addr().expect("read worker address");
        let stop = Arc::new(AtomicBool::new(false));
        let health_completed = Arc::new((Mutex::new(false), Condvar::new()));
        let captured = Arc::new(Mutex::new(Vec::new()));
        let thread_stop = Arc::clone(&stop);
        let thread_health_completed = Arc::clone(&health_completed);
        let thread_captured = Arc::clone(&captured);
        let thread = thread::spawn(move || {
            let mut connections = Vec::new();
            loop {
                let (stream, _) = listener.accept().expect("accept media worker request");
                if thread_stop.load(Ordering::Acquire) {
                    break;
                }
                let captures = Arc::clone(&thread_captured);
                let health_completed = Arc::clone(&thread_health_completed);
                connections.push(thread::spawn(move || {
                    handle_connection(stream, captures, health_completed)
                }));
            }
            for connection in connections {
                connection.join().expect("join media worker connection");
            }
        });
        Self {
            address,
            stop,
            health_completed,
            captured,
            thread: Some(thread),
        }
    }

    fn captures(&self) -> Vec<Captured> {
        self.captured.lock().expect("read media captures").clone()
    }

    fn wait_health_completed(&self) {
        let deadline = Instant::now() + DEADLINE;
        let (completed, changed) = &*self.health_completed;
        let mut completed = completed.lock().expect("lock health completion");
        while !*completed {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                panic!("worker health probe did not complete");
            }
            let waited = changed
                .wait_timeout(completed, remaining)
                .expect("wait for health completion");
            completed = waited.0;
            if waited.1.timed_out() && !*completed {
                panic!("worker health probe did not complete");
            }
        }
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

fn handle_connection(
    mut stream: TcpStream,
    captured: Arc<Mutex<Vec<Captured>>>,
    health_completed: Arc<(Mutex<bool>, Condvar)>,
) {
    stream
        .set_read_timeout(Some(DEADLINE))
        .expect("bound worker read");
    stream
        .set_write_timeout(Some(DEADLINE))
        .expect("bound worker write");
    loop {
        let Some((head, mut body)) = read_head(&mut stream) else {
            return;
        };
        let request_line = head.lines().next().unwrap_or_default();
        if request_line.starts_with("GET /health ") {
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .expect("write health response");
            stream.flush().expect("flush health response");
            drop(stream);
            let (completed, changed) = &*health_completed;
            *completed.lock().expect("publish health completion") = true;
            changed.notify_all();
            return;
        }
        let path = request_line
            .split_ascii_whitespace()
            .nth(1)
            .expect("request path")
            .to_owned();
        let expected = header_value(&head, "content-length")
            .and_then(|value| value.parse::<usize>().ok())
            .expect("router sends canonical content length");
        while body.len() < expected {
            let mut chunk = [0_u8; 4096];
            let count = stream.read(&mut chunk).expect("read worker request body");
            if count == 0 {
                return;
            }
            body.extend_from_slice(&chunk[..count]);
        }
        body.truncate(expected);
        let content_type = header_value(&head, "content-type")
            .expect("router sends content type")
            .to_owned();
        captured
            .lock()
            .expect("record media request")
            .push(Captured {
                path: path.clone(),
                content_type,
                body: body.clone(),
            });
        let response: &[u8] = match path.as_str() {
        "/v1/audio/speech" if contains_bytes(&body, b"\"stream\":true") => {
            b"HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nContent-Length: 4\r\nX-Sample-Rate: 24000\r\nX-Channels: 1\r\nX-Bit-Depth: 16\r\n\r\nPCM!"
        }
        "/v1/audio/speech" if contains_bytes(&body, b"\"response_format\":\"opus\"") => {
            b"HTTP/1.1 200 OK\r\nContent-Type: audio/opus\r\nContent-Length: 4\r\n\r\nOPUS"
        }
        "/v1/audio/speech" if contains_bytes(&body, b"\"response_format\":\"pcm\"") => {
            b"HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nContent-Length: 4\r\n\r\nPCM0"
        }
        "/v1/audio/speech" => b"HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nContent-Length: 4\r\nContent-Disposition: attachment; filename=\"speech.wav\"\r\nX-Prompt-Tokens: 7\r\nX-Completion-Tokens: 9\r\nX-Engine-Time: 0.25\r\n\r\nWAVE",
        "/v1/audio/speech/batch" => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 23\r\n\r\n{\"results\":[0,1],\"n\":2}",
        "/v1/audio/transcriptions"
            if body
                .windows(4)
                .any(|window| window.eq_ignore_ascii_case(b"true")) =>
        {
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nX-Request-Id: transcription-public\r\nContent-Length: 65\r\n\r\nevent: transcript.text.delta\ndata: {\"delta\":\"hi\"}\n\ndata: [DONE]\n\n"
        }
        "/v1/audio/transcriptions" => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nX-Request-Id: transcription-public\r\nContent-Length: 13\r\n\r\n{\"text\":\"hi\"}",
        _ => b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n",
        };
        stream.write_all(response).expect("write media response");
    }
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

struct RouterProcess {
    child: Child,
    address: SocketAddr,
    directory: PathBuf,
}

impl RouterProcess {
    fn start(routes: &[MediaRoute], workers: &[(&Worker, bool)]) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("reserve router port");
        let address = listener.local_addr().expect("read router port");
        drop(listener);
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "sgl-omni-router-media-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&directory).expect("create media router directory");
        let config_path = directory.join("router.toml");
        fs::write(&config_path, config(address, routes, workers))
            .expect("write media router config");
        let binary = env!("CARGO_BIN_EXE_sgl-omni-router");
        let child = Command::new(binary)
            .arg("--config")
            .arg(&config_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("start media router");
        let process = Self {
            child,
            address,
            directory,
        };
        process.wait_ready();
        for (worker, _) in workers {
            worker.wait_health_completed();
        }
        process
    }

    fn wait_ready(&self) {
        let deadline = Instant::now() + DEADLINE;
        while Instant::now() < deadline {
            if let Ok(response) = request(self.address, "GET", "/ready", None, b"")
                && response.starts_with(b"HTTP/1.1 200")
            {
                return;
            }
            thread::sleep(Duration::from_millis(20));
        }
        panic!("media router did not become ready");
    }
}

impl Drop for RouterProcess {
    fn drop(&mut self) {
        let _killed = self.child.kill();
        let _waited = self.child.wait();
        let _cleanup = fs::remove_dir_all(&self.directory);
    }
}

fn config(address: SocketAddr, routes: &[MediaRoute], workers: &[(&Worker, bool)]) -> String {
    let enabled_routes = routes
        .iter()
        .map(|route| format!("\"{}\"", route.route_name()))
        .collect::<Vec<_>>()
        .join(", ");
    let required_services = routes
        .iter()
        .map(|route| format!("\"{}\"", route.service_name()))
        .collect::<Vec<_>>()
        .join(", ");
    let mut output = format!(
        "schema_version = 1\n\n[server]\nlisten = \"{address}\"\nmax_connections = 64\n\n[shutdown]\ndrain_timeout_ms = 5000\n\n[logging]\nformat = \"json\"\nfilter = \"error\"\n\n[router]\nrequired_services = [{required_services}]\n\n[admission]\nglobal = 32\ngeneration_http = 1\nspeech_http = 8\ntranscription_http = 8\nspeech_batch = 8\nspeech_websocket = 1\nrealtime_websocket = 1\ncontrol = 1\n\n[health]\ninterval_ms = 1000\ntimeout_ms = 500\nsuccess_threshold = 1\nfailure_threshold = 3\nmax_concurrent_probes = 8\n\n[http_media]\nroutes = [{enabled_routes}]\ntrust_domain = \"local\"\nbuffered_request_max_bytes = 1024\nbuffered_request_total_bytes = 4096\nstreamed_request_max_bytes = 8192\nconnect_timeout_ms = 1000\nrequest_timeout_ms = 5000\npool_idle_timeout_ms = 1000\npool_max_idle_per_host = 4\n"
    );
    for (index, (worker, alternate)) in workers.iter().enumerate() {
        let speech_formats = if *alternate {
            "[\"mp3\"]"
        } else {
            "[\"mp3\", \"opus\", \"aac\", \"flac\", \"wav\"]"
        };
        output.push_str(&format!(
            "\n[[workers]]\nworker_id = \"worker-{index}\"\nbase_url = \"http://{}\"\ntrust_domain = \"local\"\ndefault_model_id = \"tts\"\n\n[workers.capacity]\n",
            worker.address
        ));
        for route in routes {
            output.push_str(&format!("{} = 8\n", route.service_name()));
        }
        for route in routes {
            output.push('\n');
            output.push_str(&media_profile(*route, speech_formats));
        }
    }
    output
}

fn media_profile(route: MediaRoute, speech_formats: &str) -> String {
    match route {
        MediaRoute::Speech => format!(
            "[[workers.service_profiles]]\nservice = \"speech_http\"\nmodel_ids = [\"tts\"]\nresponse_formats = {speech_formats}\nstream_modes = [\"non_streaming\"]\ntasks = [\"text_to_speech\", \"voice_clone\", \"voice_design\"]\nreference_forms = [\"none\", \"direct\", \"list\", \"vq_codes\"]\nmanaged_voice = false\n\n[[workers.service_profiles]]\nservice = \"speech_http\"\nmodel_ids = [\"tts\"]\nresponse_formats = [\"pcm\"]\nstream_modes = [\"non_streaming\", \"streaming\"]\ntasks = [\"text_to_speech\", \"voice_clone\", \"voice_design\"]\nreference_forms = [\"none\", \"direct\", \"list\", \"vq_codes\"]\nmanaged_voice = false\n"
        ),
        MediaRoute::SpeechBatch => String::from(
            "[[workers.service_profiles]]\nservice = \"speech_batch\"\nmodel_ids = [\"tts\"]\nresponse_formats = [\"mp3\", \"opus\", \"aac\", \"flac\", \"wav\", \"pcm\"]\ntasks = [\"text_to_speech\", \"voice_clone\", \"voice_design\"]\nreference_forms = [\"none\", \"direct\", \"list\", \"vq_codes\"]\nmanaged_voice = false\nmax_batch_size = 16\neffective_features = [\"model\", \"format\", \"task\", \"reference\", \"voice\"]\n",
        ),
        MediaRoute::Transcription => String::from(
            "[[workers.service_profiles]]\nservice = \"transcription_http\"\nmodel_ids = [\"tts\", \"asr\"]\nresponse_formats = [\"json\", \"text\", \"verbose_json\", \"sse\"]\nmedia_profiles = [\"audio\", \"audio_video\"]\nstream_modes = [\"non_streaming\", \"streaming\"]\n",
        ),
    }
}

#[test]
fn every_media_subset_is_ready_and_registers_only_enabled_routes() {
    let _guard = socket_guard();
    for routes in MEDIA_ROUTE_SUBSETS {
        let worker = Worker::start();
        let router = RouterProcess::start(routes, &[(&worker, false)]);
        for route in MediaRoute::ALL {
            let (content_type, body) = match route {
                MediaRoute::Speech => (
                    "application/json",
                    br#"{"model":"tts","input":"hello","response_format":"wav"}"#.to_vec(),
                ),
                MediaRoute::SpeechBatch => (
                    "application/json",
                    br#"{"model":"tts","response_format":"wav","items":[{"input":"hello"}]}"#
                        .to_vec(),
                ),
                MediaRoute::Transcription => (
                    "multipart/form-data; boundary=media-boundary",
                    multipart_body(false, 32),
                ),
            };
            let prior = worker.captures().len();
            let response = request(
                router.address,
                "POST",
                route.path(),
                Some(content_type),
                &body,
            )
            .expect("media subset route response");
            if routes.contains(&route) {
                assert!(
                    response.starts_with(b"HTTP/1.1 200"),
                    "enabled route {route:?} failed for {routes:?}: {}",
                    String::from_utf8_lossy(&response)
                );
                let captures = worker.captures();
                assert_eq!(captures.len(), prior + 1);
                assert_eq!(
                    captures.last().map(|capture| capture.path.as_str()),
                    Some(route.path())
                );
            } else {
                assert!(
                    response.starts_with(b"HTTP/1.1 404"),
                    "disabled route {route:?} was registered for {routes:?}: {}",
                    String::from_utf8_lossy(&response)
                );
                assert_eq!(worker.captures().len(), prior);
            }
        }
    }
}

#[test]
fn relays_all_media_routes_with_exact_bytes_headers_and_large_direct_uploads() {
    let _guard = socket_guard();
    let worker = Worker::start();
    let router = RouterProcess::start(&MediaRoute::ALL, &[(&worker, false)]);
    let roundtrip =
        |path, content_type, body| roundtrip(&worker, &router, path, content_type, body);
    let speech = br#"{"model":"tts","input":"hello","response_format":"wav","task_type":"CustomVoice","ref_audio":"data:audio/wav;base64,AA==","references":[{"audio":"x"},{"vq_codes":[1,2]}]}"#;
    let (response, capture) = roundtrip("/v1/audio/speech", "application/json", speech);
    assert!(
        response.starts_with(b"HTTP/1.1 200"),
        "unexpected speech response: {}",
        String::from_utf8_lossy(&response)
    );
    assert!(header(&response, "content-disposition").is_some());
    assert_eq!(response_body(&response), b"WAVE");
    assert_eq!(capture.body, speech);

    let opus = br#"{"model":"tts","input":"hello","response_format":"opus"}"#;
    let (response, capture) = roundtrip("/v1/audio/speech", "application/json", opus);
    assert_eq!(header(&response, "content-type"), Some("audio/opus"));
    assert_eq!(response_body(&response), b"OPUS");
    assert_eq!(capture.body, opus);

    let completed_pcm = br#"{"model":"tts","input":"hello","response_format":"pcm"}"#;
    let (response, capture) = roundtrip("/v1/audio/speech", "application/json", completed_pcm);
    assert!(response.starts_with(b"HTTP/1.1 200"));
    assert_eq!(header(&response, "content-type"), Some("audio/pcm"));
    assert!(header(&response, "x-sample-rate").is_none());
    assert!(header(&response, "x-channels").is_none());
    assert!(header(&response, "x-bit-depth").is_none());
    assert_eq!(response_body(&response), b"PCM0");
    assert_eq!(capture.body, completed_pcm);

    let streaming_pcm = br#"{"model":"tts","input":"hello","response_format":"pcm","stream":true}"#;
    let (response, capture) = roundtrip("/v1/audio/speech", "application/json", streaming_pcm);
    assert!(
        response.starts_with(b"HTTP/1.1 200"),
        "unexpected PCM response: {}",
        String::from_utf8_lossy(&response)
    );
    assert_eq!(header(&response, "x-sample-rate"), Some("24000"));
    assert_eq!(response_body(&response), b"PCM!");
    assert_eq!(capture.body, streaming_pcm);

    let batch = br#"{"model":"tts","response_format":"wav","items":[{"input":"a"},{"input":"b","response_format":"mp3","task_type":"VoiceDesign","ref_audio":"x"}]}"#;
    let (response, capture) = roundtrip("/v1/audio/speech/batch", "application/json", batch);
    assert_eq!(response_body(&response), b"{\"results\":[0,1],\"n\":2}");
    assert_eq!(capture.body, batch);

    let multipart = multipart_body(false, 32);
    let (response, capture) = roundtrip(
        "/v1/audio/transcriptions",
        "multipart/form-data; boundary=\"media-boundary\"",
        &multipart,
    );
    assert_eq!(
        header(&response, "x-request-id"),
        Some("transcription-public")
    );
    assert_eq!(response_body(&response), b"{\"text\":\"hi\"}");
    assert_eq!(capture.body, multipart);
    assert_eq!(
        capture.content_type,
        "multipart/form-data; boundary=\"media-boundary\""
    );

    let streaming_multipart = multipart_body(true, 32);
    let (response, capture) = roundtrip(
        "/v1/audio/transcriptions",
        "multipart/form-data; boundary=media-boundary",
        &streaming_multipart,
    );
    assert!(
        capture
            .body
            .windows(4)
            .any(|window| window.eq_ignore_ascii_case(b"true")),
        "streaming multipart bytes reached the worker"
    );
    assert!(
        response.starts_with(b"HTTP/1.1 200"),
        "unexpected SSE response: {}",
        String::from_utf8_lossy(&response)
    );
    assert_eq!(header(&response, "content-type"), Some("text/event-stream"));
    assert!(response_body(&response).ends_with(b"data: [DONE]\n\n"));
    assert_eq!(capture.body, streaming_multipart);

    let mut large_speech = vec![b' '; 2_048];
    large_speech[..2].copy_from_slice(b"{}");
    let (response, capture) = roundtrip("/v1/audio/speech", "application/json", &large_speech);
    assert!(response.starts_with(b"HTTP/1.1 200"));
    assert_eq!(capture.body, large_speech);

    let large_batch = format!(
        r#"{{"model":"tts","items":[{{"input":"{}"}}]}}"#,
        "b".repeat(2_048)
    )
    .into_bytes();
    let (response, capture) = roundtrip("/v1/audio/speech/batch", "application/json", &large_batch);
    assert!(response.starts_with(b"HTTP/1.1 200"));
    assert_eq!(capture.body, large_batch);

    let large_multipart = multipart_body(false, 2_048);
    let (response, capture) = roundtrip(
        "/v1/audio/transcriptions",
        "multipart/form-data; boundary=media-boundary",
        &large_multipart,
    );
    assert!(response.starts_with(b"HTTP/1.1 200"));
    assert_eq!(capture.body, large_multipart);
}

fn roundtrip(
    worker: &Worker,
    router: &RouterProcess,
    path: &str,
    content_type: &str,
    body: &[u8],
) -> (Vec<u8>, Captured) {
    let prior = worker.captures().len();
    let response = request(router.address, "POST", path, Some(content_type), body)
        .expect("media roundtrip response");
    let captures = worker.captures();
    let capture = captures
        .get(prior)
        .filter(|capture| capture.path == path)
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "missing {path} capture for {} bytes; captures: {captures:?}; response: {}",
                body.len(),
                String::from_utf8_lossy(&response)
            )
        });
    (response, capture)
}

#[test]
fn oversized_heterogeneous_request_is_rejected_without_upstream_send() {
    let _guard = socket_guard();
    let first = Worker::start();
    let second = Worker::start();
    let router = RouterProcess::start(&MediaRoute::ALL, &[(&first, false), (&second, true)]);
    let body = vec![b'x'; 2_048];
    let response = request(
        router.address,
        "POST",
        "/v1/audio/speech",
        Some("application/json"),
        &body,
    )
    .expect("heterogeneous oversized request response");
    assert!(response.starts_with(b"HTTP/1.1 413"));
    assert!(first.captures().is_empty());
    assert!(second.captures().is_empty());
}

fn multipart_body(stream: bool, file_size: usize) -> Vec<u8> {
    let mut output = Vec::new();
    output.extend_from_slice(b"--media-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF");
    output.extend(std::iter::repeat_n(b'a', file_size));
    output.extend_from_slice(b"\r\n--media-boundary-not-a-delimiter\r\n");
    output.extend_from_slice(b"--media-boundary\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n");
    if stream {
        output.extend_from_slice(
            b"--media-boundary\r\nContent-Disposition: form-data; name=\"stream\"\r\n\r\ntrue\r\n",
        );
    }
    output.extend_from_slice(b"--media-boundary--\r\n");
    output
}

fn socket_guard() -> MutexGuard<'static, ()> {
    SOCKET_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn request(
    address: SocketAddr,
    method: &str,
    path: &str,
    content_type: Option<&str>,
    body: &[u8],
) -> std::io::Result<Vec<u8>> {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(200))?;
    stream.set_read_timeout(Some(DEADLINE))?;
    stream.set_write_timeout(Some(DEADLINE))?;
    let content = content_type.map_or_else(String::new, |value| {
        format!(
            "Content-Type: {value}\r\nContent-Length: {}\r\n",
            body.len()
        )
    });
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\n{content}Connection: close\r\n\r\n"
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    Ok(response)
}

fn read_head(stream: &mut TcpStream) -> Option<(String, Vec<u8>)> {
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        let count = stream.read(&mut chunk).ok()?;
        if count == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            let body = bytes.split_off(index + 4);
            bytes.truncate(index + 4);
            return String::from_utf8(bytes).ok().map(|head| (head, body));
        }
    }
}

fn header_value<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    head.lines().skip(1).find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.eq_ignore_ascii_case(name).then_some(value.trim())
    })
}

fn header<'a>(response: &'a [u8], name: &str) -> Option<&'a str> {
    let split = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")?;
    let head = std::str::from_utf8(&response[..split]).ok()?;
    header_value(head, name)
}

fn response_body(response: &[u8]) -> &[u8] {
    response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .and_then(|index| response.get(index + 4..))
        .unwrap_or_default()
}
