#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Strict configuration boundary tests.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sgl_omni_router::{Config, ConfigError, MAX_CONFIG_BYTES, MAX_DRAIN_TIMEOUT_MS};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "sgl-omni-router-config-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create isolated config test directory");
        Self(path)
    }

    fn write(&self, contents: &[u8]) -> PathBuf {
        let path = self.0.join("router.toml");
        fs::write(&path, contents).expect("write isolated config fixture");
        path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _cleanup_result = fs::remove_dir_all(&self.0);
    }
}

fn valid_config(listen: &str, drain_timeout_ms: u64, filter: &str) -> String {
    format!(
        "schema_version = 1\n\n[server]\nlisten = \"{listen}\"\n\n[shutdown]\ndrain_timeout_ms = {drain_timeout_ms}\n\n[logging]\nformat = \"json\"\nfilter = \"{filter}\"\n\n[router]\nrequired_services = [\"generation_http\"]\n\n[admission]\nglobal = 8\ngeneration_http = 4\nspeech_http = 1\ntranscription_http = 1\nspeech_batch = 1\nspeech_websocket = 1\nrealtime_websocket = 1\ncontrol = 1\n\n[health]\ninterval_ms = 100\ntimeout_ms = 50\nsuccess_threshold = 2\nfailure_threshold = 3\nmax_concurrent_probes = 2\n\n[http_generation]\ntrust_domain = \"local\"\nbuffered_request_max_bytes = 8388608\nbuffered_request_total_bytes = 268435456\nstreamed_request_max_bytes = 536870912\nconnect_timeout_ms = 5000\nrequest_timeout_ms = 1800000\npool_idle_timeout_ms = 90000\npool_max_idle_per_host = 8\n\n[[workers]]\nworker_id = \"worker-1\"\nbase_url = \"http://127.0.0.1:9\"\ntrust_domain = \"local\"\ndefault_model_id = \"omni\"\n\n[workers.capacity]\ngeneration_http = 4\n\n[[workers.service_profiles]]\nservice = \"generation_http\"\nmodel_ids = [\"omni\"]\nmessage_content_forms = [\"string\"]\nmedia_placements = []\ninput_modalities = [\"text\"]\noutput_modalities = [\"text\"]\nchat_audio_formats = []\nstream_modes = [\"non_streaming\", \"streaming\"]\n"
    )
}

fn generation_worker(worker_id: &str, base_url: &str, resolved_ip: Option<&str>) -> String {
    let resolved_ip =
        resolved_ip.map_or_else(String::new, |ip| format!("resolved_ip = \"{ip}\"\n"));
    format!(
        "[[workers]]\nworker_id = \"{worker_id}\"\nbase_url = \"{base_url}\"\n{resolved_ip}trust_domain = \"local\"\ndefault_model_id = \"omni\"\n\n[workers.capacity]\ngeneration_http = 4\n\n[[workers.service_profiles]]\nservice = \"generation_http\"\nmodel_ids = [\"omni\"]\nmessage_content_forms = [\"string\"]\nmedia_placements = []\ninput_modalities = [\"text\"]\noutput_modalities = [\"text\"]\nchat_audio_formats = []\nstream_modes = [\"non_streaming\", \"streaming\"]\n"
    )
}

fn config_with_generation_workers(workers: &[(&str, &str, Option<&str>)]) -> String {
    let base = valid_config("127.0.0.1:30000", 30_000, "info");
    let prefix = base
        .split("[[workers]]")
        .next()
        .expect("valid fixture has a worker marker");
    let worker_blocks = workers
        .iter()
        .map(|(worker_id, base_url, resolved_ip)| {
            generation_worker(worker_id, base_url, *resolved_ip)
        })
        .collect::<String>();
    format!("{prefix}{worker_blocks}")
}

#[derive(Clone, Copy)]
enum SpeechSurface {
    Http,
    Websocket,
}

impl SpeechSurface {
    const fn service(self) -> &'static str {
        match self {
            Self::Http => "speech_http",
            Self::Websocket => "speech_websocket",
        }
    }
}

fn worker_config_prefix(required_service: &str) -> String {
    let base = valid_config("127.0.0.1:30000", 30_000, "info");
    base.split("[[workers]]")
        .next()
        .expect("valid fixture has a worker marker")
        .replace(
            "required_services = [\"generation_http\"]",
            &format!("required_services = [\"{required_service}\"]"),
        )
        .replace(
            "[http_generation]\ntrust_domain = \"local\"\nbuffered_request_max_bytes = 8388608\nbuffered_request_total_bytes = 268435456\nstreamed_request_max_bytes = 536870912\nconnect_timeout_ms = 5000\nrequest_timeout_ms = 1800000\npool_idle_timeout_ms = 90000\npool_max_idle_per_host = 8\n\n",
            "",
        )
}

fn speech_config(
    surface: SpeechSurface,
    response_formats: &str,
    stream_modes: Option<&str>,
) -> String {
    let service = surface.service();
    let prefix = worker_config_prefix(service);
    let input_profiles = match surface {
        SpeechSurface::Http => "",
        SpeechSurface::Websocket => "input_profiles = [\"text\"]\n",
    };
    let stream_modes =
        stream_modes.map_or_else(String::new, |modes| format!("stream_modes = {modes}\n"));
    format!(
        "{prefix}[[workers]]\nworker_id = \"worker-1\"\nbase_url = \"http://127.0.0.1:9\"\ntrust_domain = \"local\"\ndefault_model_id = \"tts\"\n\n[workers.capacity]\n{service} = 4\n\n[[workers.service_profiles]]\nservice = \"{service}\"\nmodel_ids = [\"tts\"]\n{input_profiles}response_formats = {response_formats}\n{stream_modes}tasks = [\"text_to_speech\"]\nreference_forms = [\"none\"]\nmanaged_voice = false\n"
    )
}

fn speech_batch_config(stream_modes: Option<&str>) -> String {
    let prefix = worker_config_prefix("speech_batch");
    let stream_modes =
        stream_modes.map_or_else(String::new, |modes| format!("stream_modes = {modes}\n"));
    format!(
        "{prefix}[[workers]]\nworker_id = \"worker-1\"\nbase_url = \"http://127.0.0.1:9\"\ntrust_domain = \"local\"\ndefault_model_id = \"tts\"\n\n[workers.capacity]\nspeech_batch = 4\n\n[[workers.service_profiles]]\nservice = \"speech_batch\"\nmodel_ids = [\"tts\"]\nresponse_formats = [\"mp3\"]\n{stream_modes}tasks = [\"text_to_speech\"]\nreference_forms = [\"none\"]\nmanaged_voice = false\nmax_batch_size = 4\neffective_features = [\"model\"]\n"
    )
}

fn with_max_connections(config: String, max_connections: u32) -> String {
    config.replace(
        "listen = \"127.0.0.1:30000\"",
        &format!("listen = \"127.0.0.1:30000\"\nmax_connections = {max_connections}"),
    )
}

fn load_bytes(contents: &[u8]) -> Result<Config, ConfigError> {
    let directory = TestDir::new();
    Config::load(&directory.write(contents))
}

#[test]
fn omitted_connection_cap_defaults_to_1024() {
    let config = load_bytes(valid_config("127.0.0.1:30000", 30_000, "info").as_bytes())
        .expect("complete strict configuration should be valid");
    assert_eq!(config.server.listen.to_string(), "127.0.0.1:30000");
    assert_eq!(config.server.max_connections, 1024);
    assert_eq!(config.shutdown.drain_timeout().as_millis(), 30_000);
}

#[test]
fn classification_concurrency_defaults_and_boundaries_are_validated() {
    let base = valid_config("127.0.0.1:30000", 30_000, "info");
    load_bytes(base.as_bytes()).expect("omitted classification concurrency defaults to four");

    for limit in [1, 64] {
        let configured = base.replace(
            "required_services = [\"generation_http\"]",
            &format!(
                "required_services = [\"generation_http\"]\nmax_concurrent_classifications = {limit}"
            ),
        );
        load_bytes(configured.as_bytes()).expect("classification concurrency boundary is valid");
    }

    for limit in [0, 65] {
        let configured = base.replace(
            "required_services = [\"generation_http\"]",
            &format!(
                "required_services = [\"generation_http\"]\nmax_concurrent_classifications = {limit}"
            ),
        );
        assert!(load_bytes(configured.as_bytes()).is_err());
    }
}

#[test]
fn validates_connection_cap_boundaries() {
    for max_connections in [1, 65_535] {
        let config = with_max_connections(
            valid_config("127.0.0.1:30000", 30_000, "info"),
            max_connections,
        );
        assert!(load_bytes(config.as_bytes()).is_ok());
    }

    for max_connections in [0, 65_536] {
        let config = with_max_connections(
            valid_config("127.0.0.1:30000", 30_000, "info"),
            max_connections,
        );
        assert!(load_bytes(config.as_bytes()).is_err());
    }
}

#[test]
fn validates_chat_enablement_security_and_resource_bounds() {
    let base = valid_config("127.0.0.1:30000", 30_000, "info");
    let rejected = [
        base.replace("127.0.0.1:30000", "0.0.0.0:30000"),
        base.replace(
            "required_services = [\"generation_http\"]",
            "required_services = [\"speech_http\"]",
        ),
        base.replace(
            "trust_domain = \"local\"\nbuffered",
            "trust_domain = \"remote\"\nbuffered",
        ),
        base.replace(
            "buffered_request_max_bytes = 8388608",
            "buffered_request_max_bytes = 0",
        ),
        base.replace(
            "buffered_request_total_bytes = 268435456",
            "buffered_request_total_bytes = 1024",
        ),
        base.replace(
            "streamed_request_max_bytes = 536870912",
            "streamed_request_max_bytes = 1024",
        ),
        base.replace("connect_timeout_ms = 5000", "connect_timeout_ms = 0"),
        base.replace("request_timeout_ms = 1800000", "request_timeout_ms = 1000"),
        base.replace("pool_idle_timeout_ms = 90000", "pool_idle_timeout_ms = 999"),
        base.replace("pool_max_idle_per_host = 8", "pool_max_idle_per_host = 0"),
        base.replace(
            "pool_max_idle_per_host = 8",
            "pool_max_idle_per_host = 8\nunknown = true",
        ),
    ];
    for contents in rejected {
        assert!(load_bytes(contents.as_bytes()).is_err());
    }
    let disabled = base.replace(
        "[http_generation]\ntrust_domain = \"local\"\nbuffered_request_max_bytes = 8388608\nbuffered_request_total_bytes = 268435456\nstreamed_request_max_bytes = 536870912\nconnect_timeout_ms = 5000\nrequest_timeout_ms = 1800000\npool_idle_timeout_ms = 90000\npool_max_idle_per_host = 8\n\n",
        "",
    );
    load_bytes(disabled.as_bytes())
        .expect("omitted section disables chat without weakening other routes");
}

#[test]
fn rejects_unknown_duplicate_missing_and_unsupported_schema_fields() {
    let cases = [
        valid_config("127.0.0.1:30000", 30_000, "info").replace(
            "listen = \"127.0.0.1:30000\"",
            "listen = \"127.0.0.1:30000\"\nunknown = true",
        ),
        valid_config("127.0.0.1:30000", 30_000, "info").replace(
            "filter = \"info\"",
            "filter = \"info\"\nsecret = \"must-not-appear\"",
        ),
        valid_config("127.0.0.1:30000", 30_000, "info")
            + "\n[server]\nlisten = \"127.0.0.1:30001\"\n",
        "schema_version = 1\n".to_owned(),
        valid_config("127.0.0.1:30000", 30_000, "info")
            .replace("schema_version = 1", "schema_version = 2"),
    ];

    for contents in cases {
        assert!(load_bytes(contents.as_bytes()).is_err());
    }
}

#[test]
fn rejects_invalid_address_timeout_format_and_filter() {
    let cases = [
        valid_config("localhost:30000", 30_000, "info"),
        valid_config("127.0.0.1:30000", 0, "info"),
        valid_config(
            "127.0.0.1:30000",
            MAX_DRAIN_TIMEOUT_MS.saturating_add(1),
            "info",
        ),
        valid_config("127.0.0.1:30000", 30_000, "[invalid"),
        valid_config("127.0.0.1:30000", 30_000, &"x".repeat(257)),
        valid_config("127.0.0.1:30000", 30_000, "info")
            .replace("format = \"json\"", "format = \"yaml\""),
    ];

    for contents in cases {
        assert!(load_bytes(contents.as_bytes()).is_err());
    }
}

#[test]
fn rejects_oversized_and_non_utf8_input_without_echoing_contents() {
    let oversized = vec![b'x'; MAX_CONFIG_BYTES.saturating_add(1)];
    let oversized_error = load_bytes(&oversized).expect_err("oversized config must fail");
    assert!(matches!(oversized_error, ConfigError::TooLarge { .. }));

    let invalid_utf8 = [0xff, b's', b'e', b'c', b'r', b'e', b't'];
    let encoding_error = load_bytes(&invalid_utf8).expect_err("non-UTF-8 config must fail");
    assert!(matches!(encoding_error, ConfigError::Encoding(_)));
    assert!(!encoding_error.to_string().contains("secret"));
}

#[test]
fn read_errors_do_not_disclose_the_config_path() {
    let path = Path::new("/definitely-not-present/secret-router.toml");
    let error = Config::load(path).expect_err("missing config must fail");
    assert!(!error.to_string().contains("secret-router"));
}

#[test]
fn rejects_admission_health_and_required_service_boundaries() {
    let base = valid_config("127.0.0.1:30000", 30_000, "info");
    let cases = [
        base.replace("global = 8", "global = 0"),
        base.replace("generation_http = 4", "generation_http = 9"),
        base.replace("interval_ms = 100", "interval_ms = 99"),
        base.replace("timeout_ms = 50", "timeout_ms = 101"),
        base.replace("success_threshold = 2", "success_threshold = 33"),
        base.replace("max_concurrent_probes = 2", "max_concurrent_probes = 65"),
        base.replace(
            "required_services = [\"generation_http\"]",
            "required_services = [\"generation_http\", \"generation_http\"]",
        ),
        base.replace(
            "required_services = [\"generation_http\"]",
            "required_services = [\"speech_http\"]",
        ),
    ];
    for contents in cases {
        assert!(load_bytes(contents.as_bytes()).is_err());
    }
}

#[test]
fn rejects_worker_profile_counterexamples_without_normalizing_arrays() {
    let base = valid_config("127.0.0.1:30000", 30_000, "info");
    let cases = [
        base.replace("worker_id = \"worker-1\"", "worker_id = \"bad worker\""),
        base.replace(
            "base_url = \"http://127.0.0.1:9\"",
            "base_url = \"http://user@127.0.0.1:9/path?query=yes\"",
        ),
        base.replace(
            "generation_http = 4\n\n[[workers.service_profiles]]",
            "\n[[workers.service_profiles]]",
        ),
        base.replace("model_ids = [\"omni\"]", "model_ids = []"),
        base.replace(
            "input_modalities = [\"text\"]",
            "input_modalities = [\"text\", \"text\"]",
        ),
        base.replace(
            "service = \"generation_http\"",
            "service = \"future_service\"",
        ),
        base.replace(
            "stream_modes = [\"non_streaming\", \"streaming\"]",
            "stream_modes = [\"non_streaming\", \"streaming\"]\nfuture_claim = true",
        ),
    ];
    for contents in cases {
        assert!(load_bytes(contents.as_bytes()).is_err());
    }
}

#[test]
fn validates_worker_defaults_and_generation_cross_field_schema() {
    let base = valid_config("127.0.0.1:30000", 30_000, "info");
    let rejected = [
        base.replace("default_model_id = \"omni\"\n", ""),
        base.replace("default_model_id = \"omni\"", "default_model_id = \"\""),
        base.replace(
            "default_model_id = \"omni\"",
            &format!("default_model_id = \"{}\"", "x".repeat(257)),
        ),
        base.replace(
            "default_model_id = \"omni\"",
            "default_model_id = \"other\"",
        ),
        base.replace(
            "message_content_forms = [\"string\"]",
            "message_content_forms = [\"string\"]\ninput_profile = \"text\"",
        ),
        base.replace(
            "media_placements = []",
            "media_placements = [\"top_level\"]",
        ),
        base.replace(
            "media_placements = []\ninput_modalities = [\"text\"]",
            "media_placements = []\ninput_modalities = [\"text\", \"image\"]",
        ),
        base.replace(
            "media_placements = []\ninput_modalities = [\"text\"]",
            "media_placements = [\"typed_parts\"]\ninput_modalities = [\"text\", \"image\"]",
        ),
        base.replace(
            "output_modalities = [\"text\"]\nchat_audio_formats = []",
            "output_modalities = [\"audio\"]\nchat_audio_formats = []",
        ),
        base.replace("chat_audio_formats = []", "chat_audio_formats = [\"wav\"]"),
    ];
    for contents in rejected {
        assert!(load_bytes(contents.as_bytes()).is_err());
    }

    let mixed = base
        .replace(
            "message_content_forms = [\"string\"]",
            "message_content_forms = [\"string\", \"typed_parts\"]",
        )
        .replace(
            "media_placements = []\ninput_modalities = [\"text\"]",
            "media_placements = [\"top_level\", \"typed_parts\"]\ninput_modalities = [\"text\", \"image\", \"audio\", \"video\"]",
        )
        .replace(
            "output_modalities = [\"text\"]\nchat_audio_formats = []",
            "output_modalities = [\"text\", \"audio\"]\nchat_audio_formats = [\"wav\", \"mp3\", \"flac\", \"pcm\", \"aac\", \"opus\"]",
        );
    load_bytes(mixed.as_bytes()).expect("complete mixed generation row must validate");
    let strict_new_set_failures = [
        mixed.replace(
            "message_content_forms = [\"string\", \"typed_parts\"]",
            "message_content_forms = []",
        ),
        mixed.replace(
            "message_content_forms = [\"string\", \"typed_parts\"]",
            "message_content_forms = [\"string\", \"string\"]",
        ),
        mixed.replace(
            "message_content_forms = [\"string\", \"typed_parts\"]",
            "message_content_forms = [\"future_form\"]",
        ),
        mixed.replace(
            "media_placements = [\"top_level\", \"typed_parts\"]",
            "media_placements = [\"top_level\", \"top_level\"]",
        ),
        mixed.replace(
            "media_placements = [\"top_level\", \"typed_parts\"]",
            "media_placements = [\"future_placement\"]",
        ),
        mixed.replace(
            "chat_audio_formats = [\"wav\", \"mp3\", \"flac\", \"pcm\", \"aac\", \"opus\"]",
            "chat_audio_formats = [\"wav\", \"wav\"]",
        ),
        mixed.replace(
            "chat_audio_formats = [\"wav\", \"mp3\", \"flac\", \"pcm\", \"aac\", \"opus\"]",
            "chat_audio_formats = [\"future_format\"]",
        ),
    ];
    for contents in strict_new_set_failures {
        assert!(load_bytes(contents.as_bytes()).is_err());
    }

    let prefix = worker_config_prefix("voice_control");
    let control_only = format!(
        "{prefix}[[workers]]\nworker_id = \"voice\"\nbase_url = \"http://127.0.0.1:9\"\ntrust_domain = \"local\"\ndefault_model_id = \"forbidden\"\n\n[workers.capacity]\ncontrol = 1\n\n[[workers.service_profiles]]\nservice = \"voice_control\"\n"
    );
    assert!(load_bytes(control_only.as_bytes()).is_err());
}

#[test]
fn validates_speech_stream_modes_and_pcm_correlation() {
    for surface in [SpeechSurface::Http, SpeechSurface::Websocket] {
        for (formats, modes) in [
            ("[\"pcm\"]", "[\"non_streaming\"]"),
            ("[\"pcm\"]", "[\"streaming\"]"),
            ("[\"pcm\"]", "[\"non_streaming\", \"streaming\"]"),
            ("[\"mp3\"]", "[\"non_streaming\"]"),
        ] {
            let contents = speech_config(surface, formats, Some(modes));
            load_bytes(contents.as_bytes()).expect("valid speech stream profile must pass");
        }

        for (formats, modes) in [
            ("[\"pcm\"]", None),
            ("[\"pcm\"]", Some("[]")),
            ("[\"pcm\"]", Some("[\"non_streaming\", \"non_streaming\"]")),
            ("[\"pcm\"]", Some("[\"future_mode\"]")),
            ("[\"mp3\"]", Some("[\"streaming\"]")),
            (
                "[\"pcm\", \"mp3\"]",
                Some("[\"non_streaming\", \"streaming\"]"),
            ),
        ] {
            let contents = speech_config(surface, formats, modes);
            assert!(load_bytes(contents.as_bytes()).is_err());
        }
    }

    let valid_batch = speech_batch_config(None);
    load_bytes(valid_batch.as_bytes()).expect("mode-free speech batch must pass");
    let invalid_batch = speech_batch_config(Some("[\"non_streaming\"]"));
    assert!(load_bytes(invalid_batch.as_bytes()).is_err());
}

#[test]
fn validates_static_dns_pins_and_canonical_authority_rules() {
    let accepted = [
        config_with_generation_workers(&[(
            "worker-a",
            "http://worker-a.invalid:18080/",
            Some("192.0.2.10"),
        )]),
        config_with_generation_workers(&[
            (
                "worker-a",
                "http://EXAMPLE.invalid:18080/",
                Some("192.0.2.10"),
            ),
            (
                "worker-b",
                "http://example.invalid:18081/",
                Some("192.0.2.10"),
            ),
        ]),
        config_with_generation_workers(&[
            (
                "worker-a",
                "http://BÜCHER.invalid:18080/",
                Some("192.0.2.10"),
            ),
            (
                "worker-b",
                "http://xn--bcher-kva.invalid:18081/",
                Some("192.0.2.10"),
            ),
        ]),
        config_with_generation_workers(&[
            (
                "worker-a",
                "http://worker-a.invalid:18080/",
                Some("192.0.2.10"),
            ),
            (
                "worker-b",
                "http://worker-b.invalid:18080/",
                Some("192.0.2.10"),
            ),
        ]),
    ];
    for contents in accepted {
        load_bytes(contents.as_bytes()).expect("valid static DNS declaration must pass");
    }

    let malformed_pin = config_with_generation_workers(&[(
        "worker-a",
        "http://worker-a.invalid:18080/",
        Some("192.0.2.10"),
    )])
    .replace(
        "resolved_ip = \"192.0.2.10\"",
        "resolved_ip = \"not-an-ip\"",
    );
    let rejected = [
        config_with_generation_workers(&[("worker-a", "http://worker-a.invalid:18080/", None)]),
        config_with_generation_workers(&[(
            "worker-a",
            "http://127.0.0.1:18080/",
            Some("127.0.0.1"),
        )]),
        malformed_pin,
        config_with_generation_workers(&[("worker-a", "http://127.0.0.1:0/", None)]),
        config_with_generation_workers(&[(
            "worker-a",
            "http://worker-a.invalid.:18080/",
            Some("192.0.2.10"),
        )]),
        config_with_generation_workers(&[
            (
                "worker-a",
                "http://EXAMPLE.invalid:18080/",
                Some("192.0.2.10"),
            ),
            (
                "worker-b",
                "http://example.invalid:18081/",
                Some("192.0.2.11"),
            ),
        ]),
        config_with_generation_workers(&[
            (
                "worker-a",
                "http://EXAMPLE.invalid:18080/",
                Some("192.0.2.10"),
            ),
            (
                "worker-b",
                "HTTP://example.invalid:18080",
                Some("192.0.2.10"),
            ),
        ]),
    ];
    for contents in rejected {
        assert!(load_bytes(contents.as_bytes()).is_err());
    }
}

#[test]
fn accepts_all_seven_correlated_profile_variants() {
    let contents = r#"
schema_version = 1

[server]
listen = "127.0.0.1:30000"

[shutdown]
drain_timeout_ms = 30000

[logging]
format = "json"
filter = "info"

[router]
strategy = "least_requests"
required_services = ["generation_http", "speech_http", "transcription_http", "speech_batch", "speech_websocket", "realtime_websocket", "voice_control"]
voice_owner_worker_id = "worker-1"

[admission]
global = 64
generation_http = 8
speech_http = 8
transcription_http = 8
speech_batch = 8
speech_websocket = 8
realtime_websocket = 8
control = 8

[health]

[http_generation]
trust_domain = "production"

[[workers]]
worker_id = "worker-1"
base_url = "https://EXAMPLE.com:443/"
resolved_ip = "192.0.2.20"
trust_domain = "production"
default_model_id = "omni"
health_path = "/health"

[workers.capacity]
generation_http = 8
speech_http = 8
transcription_http = 8
speech_batch = 8
speech_websocket = 8
realtime_websocket = 8
control = 8

[[workers.service_profiles]]
service = "generation_http"
model_ids = ["omni"]
message_content_forms = ["string", "typed_parts"]
media_placements = ["top_level", "typed_parts"]
input_modalities = ["text", "image", "audio", "video"]
output_modalities = ["text", "audio"]
chat_audio_formats = ["wav", "mp3", "flac", "pcm", "aac", "opus"]
stream_modes = ["non_streaming", "streaming"]

[[workers.service_profiles]]
service = "speech_http"
model_ids = ["tts", "omni"]
response_formats = ["mp3", "pcm"]
stream_modes = ["non_streaming"]
tasks = ["text_to_speech", "voice_clone"]
reference_forms = ["none", "direct", "list", "vq_codes"]
managed_voice = false

[[workers.service_profiles]]
service = "speech_batch"
model_ids = ["tts", "omni"]
response_formats = ["wav"]
tasks = ["text_to_speech"]
reference_forms = ["none"]
managed_voice = false
max_batch_size = 32
effective_features = ["model", "format", "task", "reference", "voice"]

[[workers.service_profiles]]
service = "transcription_http"
model_ids = ["asr", "omni"]
response_formats = ["json", "text", "verbose_json", "sse"]
media_profiles = ["audio", "audio_video"]
stream_modes = ["non_streaming", "streaming"]

[[workers.service_profiles]]
service = "speech_websocket"
model_ids = ["tts", "omni"]
input_profiles = ["text", "text_and_control"]
response_formats = ["pcm"]
stream_modes = ["non_streaming", "streaming"]
tasks = ["text_to_speech"]
reference_forms = ["none"]
managed_voice = false

[[workers.service_profiles]]
service = "realtime_websocket"
protocols = ["openai_realtime_v1"]

[[workers.service_profiles]]
service = "voice_control"
"#;
    load_bytes(contents.as_bytes()).expect("all closed profile variants should validate");
}
