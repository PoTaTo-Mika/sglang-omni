# SGLang-Omni Rust router

This directory contains the standalone `sgl-omni-router` service. The
worker-pool release validates one bounded static worker manifest, runs isolated
bounded health probes, serves router-local liveness and readiness, and owns
graceful process shutdown. It has no proxy or inference route.

Run the service with:

```console
sgl-omni-router --config router.toml
```

Validate configuration without binding a listener with:

```console
sgl-omni-router --config router.toml --check-config
```

The complete initial configuration is:

```toml
schema_version = 1

[server]
listen = "127.0.0.1:30000"
max_connections = 1024

[shutdown]
drain_timeout_ms = 30000

[logging]
format = "json"
filter = "info"

[router]
strategy = "round_robin"
required_services = ["generation_http"]

[admission]
global = 1024
generation_http = 256
speech_http = 64
transcription_http = 64
speech_batch = 32
speech_websocket = 64
realtime_websocket = 64
control = 16

[health]
interval_ms = 5000
timeout_ms = 1000
success_threshold = 2
failure_threshold = 3
max_concurrent_probes = 16

[[workers]]
worker_id = "omni-a"
base_url = "http://omni-a.internal:30001"
resolved_ip = "127.0.0.1"
trust_domain = "local"
default_model_id = "omni-model"
health_path = "/health"

[workers.capacity]
generation_http = 64

[[workers.service_profiles]]
service = "generation_http"
model_ids = ["omni-model"]
message_content_forms = ["string", "typed_parts"]
media_placements = ["top_level", "typed_parts"]
input_modalities = ["text", "image", "audio", "video"]
output_modalities = ["text", "audio"]
chat_audio_formats = ["wav", "mp3", "flac", "pcm", "aac", "opus"]
stream_modes = ["non_streaming", "streaming"]
```

All fields are explicit. Unknown and duplicate fields are rejected. The
logging filter comes only from this file; environment variables such as
`RUST_LOG` are not read.

Worker transport resolution is static and fail-closed. A hostname `base_url`
requires one `resolved_ip`; an IPv4 or IPv6 literal `base_url` forbids it.
Every occurrence of the same canonical hostname, including across different
ports, must use the same IP. The router performs no startup or request-time DNS
fallback, does not use configured proxies, and preserves the original hostname
authority for HTTP `Host`, TLS certificate verification, and SNI. For an HTTPS
IP-literal URL, the worker certificate must be valid for that IP address. Each
registration names one concrete endpoint; DNS service aliases, multi-address
resolution, TTL refresh, and runtime discovery are not supported.

The following speech rows are illustrative additions for workers that also
declare the corresponding speech capacities. Speech HTTP and Speech-WebSocket
profile rows require an explicit
`stream_modes` set. For example, a worker approved for both completed and
streaming raw-PCM requests can declare:

```toml
[[workers.service_profiles]]
service = "speech_http"
model_ids = ["omni-model", "tts-model"]
response_formats = ["pcm"]
stream_modes = ["non_streaming", "streaming"]
tasks = ["text_to_speech"]
reference_forms = ["none"]
managed_voice = false

[[workers.service_profiles]]
service = "speech_websocket"
model_ids = ["omni-model", "tts-model"]
input_profiles = ["text"]
response_formats = ["pcm"]
stream_modes = ["non_streaming", "streaming"]
tasks = ["text_to_speech"]
reference_forms = ["none"]
managed_voice = false
```

For HTTP, omitted or false `stream` maps to `non_streaming`, while true maps to
`streaming`. For Speech WebSocket, omitted or false `stream_audio` maps to
`non_streaming`, while true maps to `streaming`. Any row advertising
`streaming` must contain only `pcm`; completed encoded formats such as `mp3`
belong in a separate correlated row advertising only `non_streaming`. The mode
is operator-approved eligibility for the public request behavior, not a
promise of one universal chunk cadence. This branch records and matches these
facts but still exposes no inference route, request parser, or WebSocket relay.

`GET /live` is worker-independent and remains logically live while serving or
draining. `GET /ready` starts at `503` and returns `200` only while serving and
every configured required service class has a healthy serving profile. Permit
saturation does not change readiness. Worker health is status-only exact
`GET /health` by default; response bodies are not parsed or buffered. No public
`/health` alias or inference route is registered.
