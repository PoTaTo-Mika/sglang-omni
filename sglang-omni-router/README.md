# SGLang-Omni Rust router

This directory contains the standalone `sgl-omni-router` service. The
worker-pool release validates one bounded static worker manifest, runs isolated
bounded health probes, serves router-local liveness and readiness, and owns
graceful process shutdown. Optional exact-byte relays serve chat generation,
speech, speech batch, and transcription HTTP routes described below.

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
max_concurrent_classifications = 4

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

[http_generation]
trust_domain = "local"
buffered_request_max_bytes = 8388608
buffered_request_total_bytes = 268435456
streamed_request_max_bytes = 536870912
connect_timeout_ms = 5000
request_timeout_ms = 1800000
pool_idle_timeout_ms = 90000
pool_max_idle_per_host = 8

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
message_content_forms = ["string"]
media_placements = []
input_modalities = ["text"]
output_modalities = ["text"]
chat_audio_formats = []
stream_modes = ["non_streaming", "streaming"]
```

This runnable row deliberately claims only model-proven text chat; media rows
must be added per model/topology after integration proof. All fields are
explicit. Unknown and duplicate fields are rejected. The
logging filter comes only from this file; environment variables such as
`RUST_LOG` are not read.

`router.max_concurrent_classifications` bounds buffered request classification
across the process. Admitted requests wait for a slot within their existing
request deadline. The default is `4`; valid values are `1` through `64`.

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
profile rows require an explicit `stream_modes` set. For example, a worker
approved for both completed and streaming raw-PCM requests can declare:

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
promise of one universal chunk cadence. The `speech` media HTTP route uses the
`speech_http` rows. Speech-WebSocket rows are recorded by the manifest but this
router does not yet expose a Speech WebSocket relay.

`POST /v1/chat/completions` is enabled only when `[http_generation]` is
present. It accepts HTTP/1.1 JSON, selects one compatible generation row, and
relays original request and response bytes. `request_timeout_ms` is one
absolute precommit deadline covering ingress through upstream response
headers. After response commitment, the direct body has no absolute wall-clock
deadline; it ends on upstream EOF/error, downstream drop, or process shutdown.
The focused direct-dependency decisions for this relay are recorded in
[`DEPENDENCIES.md`](DEPENDENCIES.md).

`[http_media]` keeps one shared client, body budget, classifier slot, and
transport policy while enabling an explicit subset of media HTTP routes. The
`routes` field is required, nonempty, duplicate-free, and accepts only
`speech`, `speech_batch`, and `transcription`:

```toml
[http_media]
routes = ["speech", "speech_batch", "transcription"]
trust_domain = "local"
buffered_request_max_bytes = 8388608
buffered_request_total_bytes = 268435456
streamed_request_max_bytes = 536870912
connect_timeout_ms = 5000
request_timeout_ms = 1800000
pool_idle_timeout_ms = 90000
pool_max_idle_per_host = 8
```

The route-to-service mapping is:

| Route value | Registered endpoint | Required service |
| --- | --- | --- |
| `speech` | `POST /v1/audio/speech` | `speech_http` |
| `speech_batch` | `POST /v1/audio/speech/batch` | `speech_batch` |
| `transcription` | `POST /v1/audio/transcriptions` | `transcription_http` |

All seven nonempty subsets are valid. Each enabled route's service must appear
in `router.required_services`, and at least one worker in the exact
`http_media.trust_domain` must declare its matching profile and capacity.
Disabled routes are not registered and return `404`; they need no profile or
capacity. Additional entries in `router.required_services` retain their global
meaning and must still be present somewhere in the worker manifest. Omitting
`[http_media]` disables all media HTTP routes, while omitting `routes` from a
present section is a configuration error. Route registration is fixed at
startup rather than checked for every request.

JSON and multipart requests at or below the buffered limit are completely
classified and forwarded as their original bytes. Larger fixed-length bodies
use the direct upload path only when every worker in the exact service/trust
scope has the same default and correlated profile rows; otherwise they are
rejected with `413`. Batch requests always select one worker and are never
split. Responses remain direct backpressured bodies through clean EOF or
truncation, with route-specific status, framing, content-type, and header
validation.

`GET /live` is worker-independent and remains logically live while serving or
draining. `GET /ready` starts at `503` and returns `200` only while serving and
every configured required service class has a healthy serving profile. When
chat generation is enabled, readiness additionally requires a healthy serving
generation worker in the exact `http_generation.trust_domain`. When media HTTP
is enabled, readiness checks only the enabled media services in the exact
`http_media.trust_domain`. Permit saturation does not change readiness. Worker
health is status-only exact `GET /health` by default; response bodies are not
parsed or buffered. No public `/health` alias is registered.
