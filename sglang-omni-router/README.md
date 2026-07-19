# SGLang-Omni Router

`sgl-omni-router` is the standalone Rust routing process for SGLang-Omni
workers. It presents one bounded HTTP/1.1 and WebSocket edge, selects one
healthy worker whose declared profile matches the request, and relays the
request to that worker. It does not load models, split requests across workers,
or launch and supervise worker processes.

Workers, transport addresses, trust domains, correlated capabilities, and
capacity limits come from one strict TOML manifest at startup. This makes
routing decisions explicit and keeps worker discovery and orchestration
outside the process.

## Quick start

Start at least one SGLang-Omni worker separately. For example, the repository's
[server examples](../examples/README.md) launch complete workers on port 8000.
Then build the router:

```bash
cd sglang-omni-router
cargo build --release --locked --bin sgl-omni-router
```

For toolchain installation and the local development checks, see
[DEVELOPMENT.md](DEVELOPMENT.md).

Copy the checked-in generation example, then edit the worker address, model
ID, declared capabilities, and capacities to match the worker you started:

```bash
cp examples/text-generation.toml router.toml
```

The example is a complete, validated configuration rather than a hardware
recommendation. Keep only capability rows that the worker actually supports,
and measure admission and worker capacities for the target model and workload.

Validate without creating a runtime or binding the listener, then start the
standalone process:

```bash
./target/release/sgl-omni-router --config router.toml --check-config
./target/release/sgl-omni-router --config router.toml
```

Wait for the worker to pass its health threshold:

```bash
curl -i http://127.0.0.1:30000/live
curl -i http://127.0.0.1:30000/ready
```

Send a non-streaming request:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "What can you hear?"}]
  }'
```

There is no managed one-command worker-plus-router launcher. Start, monitor,
and stop each worker independently, then run the router with its validated
static manifest.

## Architecture

```text
                                      status-only health probes
                                      +-----------------------+
                                      |                       v
client -> local TLS/auth proxy -> SGLang-Omni Router -> eligible worker A
                                     |  |             -> eligible worker B
                                     |  +-- exact worker lease held for body/session
                                     +----- /live, /ready, /v1/models,
                                            /metrics, /diagnostics
```

For each request or session, the data flow is:

1. Validate the public route, framing, bounded routing fields, and canonical
   `x-request-id`.
2. Acquire one global and one service-class admission permit.
3. Build the exact compatibility requirement from the request, then filter by
   profile, trust domain, voice ownership, health, and serving disposition.
4. Apply `round_robin` or `least_requests` only to that eligible set and acquire
   one exact worker/service permit.
5. Start one upstream request attempt to the selected worker and relay with
   backpressure. The exact worker lease remains attached to the downstream
   HTTP body or WebSocket session until completion, disconnect, cancellation,
   or forced shutdown.

The router parses only the bounded facts needed to select a worker. Buffered
HTTP payload bytes are relayed without reserialization, while accepted uploads
and response bodies are streamed with backpressure. HTTP transfer framing and
a narrow header set are validated and reconstructed, so “byte-preserving”
refers to application body bytes, not the original wire encoding.

## Public surface

All inference HTTP routes require HTTP/1.1. A route is registered only when its
own configuration is enabled. Except for `/live` and `/ready`, inference and
operations routes require an exact path and reject query strings unless noted.

| Method and path | Enabled by | Input and output | Streaming behavior |
|---|---|---|---|
| `GET /live` | Always | Plain text process liveness | Not applicable |
| `GET /ready` | Always | Plain text routing readiness | Not applicable |
| `POST /v1/chat/completions` | `[http_generation]` | JSON; text or typed multimodal input; text and/or audio output as profiled | JSON when `stream` is false/omitted; SSE when true |
| `POST /v1/audio/speech` | `[http_media]` route `speech` | JSON TTS, voice cloning, or voice design; encoded audio or PCM | `stream=true` is supported only by a correlated PCM row |
| `POST /v1/audio/speech/batch` | `[http_media]` route `speech_batch` | JSON batch; one worker handles the complete batch and returns JSON | No router-level streaming or batch splitting |
| `POST /v1/audio/transcriptions` | `[http_media]` route `transcription` | Multipart audio or video file; JSON, verbose JSON, text, or SSE | `stream=true` selects SSE |
| `GET /v1/audio/speech/stream` | `[websocket.speech]` | Terminating TTS WebSocket; text/control messages and worker binary audio | Session is pinned to one worker |
| `GET /v1/realtime` | `[websocket.realtime]` | Terminating OpenAI Realtime V1 WebSocket | Session is pinned to one worker |
| `GET`, `POST /v1/audio/voices` | `router.voice_owner_worker_id` | List or multipart upload to one exact owner; query strings pass through | Direct JSON response body |
| `DELETE /v1/audio/voices/{name}` | `router.voice_owner_worker_id` | Delete from one exact owner; query strings pass through | Direct JSON response body |
| `GET /v1/models` | Loopback listener | Immutable sorted manifest model list | Not applicable |
| `GET /metrics` | Loopback listener | Prometheus 0.0.4 gauges | Not applicable |
| `GET /diagnostics` | Loopback listener | Bounded, redacted JSON snapshot | Not applicable |

`HEAD` is deliberately rejected. There is no public `/health` alias or worker
administration API.

## Configuration

The configuration is UTF-8 TOML, at most 64 KiB, with `schema_version = 1`.
Unknown fields, duplicate fields, duplicate set members, unknown enum values,
and invalid cross-field combinations fail startup. Most limits are explicit;
the documented defaults below apply only to fields whose containing section is
present. The logging filter comes only from this file—`RUST_LOG` is not read.
Use [examples/text-generation.toml](examples/text-generation.toml) as the
complete, explicit generation example; the README focuses on fields that
change routing, resource ownership, or failure behavior.

### Process, routing, admission, and health

| Section/field | Meaning and accepted values |
|---|---|
| `server.listen` | Required socket address. Inference HTTP, voice, and WebSocket routes require a loopback address. A non-loopback listener is health-only. |
| `server.max_connections` | Accepted connection-task bound, `1..=65535`; default `1024`. This is separate from request/session admission. |
| `shutdown.drain_timeout_ms` | Required graceful-drain deadline, `1..=300000`. |
| `logging.format` | Required: `json` or `pretty`. |
| `logging.filter` | Required valid tracing filter, 1–256 bytes, for example `info` or `debug`. |
| `router.strategy` | `round_robin` (default) or `least_requests`. |
| `router.max_concurrent_classifications` | Process-wide buffered-classification concurrency, `1..=64`; default `4`. Contention waits asynchronously instead of blocking a Tokio worker thread. |
| `router.required_services` | Required, nonempty set drawn from `generation_http`, `speech_http`, `transcription_http`, `speech_batch`, `speech_websocket`, `realtime_websocket`, and `voice_control`. It must include every enabled route family. |
| `router.voice_owner_worker_id` | Optional exact worker ID that enables voice-control routes and managed-voice pinning. |
| `admission.global` | Process-wide in-flight request/session limit, `1..=1000000`. |
| `admission.*` | Required class limits for all seven classes, each `1..=65535` and no greater than `global`. Unused classes still need a value. |
| `health.interval_ms` | Probe interval, `100..=300000`; default `5000`. |
| `health.timeout_ms` | Complete probe timeout, `10..=interval_ms`; default `1000`. |
| `health.success_threshold` / `failure_threshold` | Consecutive observations required to become healthy/unhealthy, each `1..=32`; defaults `2` and `3`. |
| `health.max_concurrent_probes` | Shared health-probe concurrency, `1..=64`; default `16`. |

Workers begin `unknown`, so readiness initially returns `503`. Each probe is an
exact status-only `GET` to `health_path` (`/health` by default); any 2xx status
is success, and the body is neither parsed nor buffered. A protocol failure on
a selected worker requests an immediate, coalesced probe, but routed request
status codes do not directly change health.

### HTTP relay policy

`[http_generation]` enables chat. `[http_media]` keeps one shared transport
policy while enabling an explicit `routes` subset drawn from `speech`,
`speech_batch`, and `transcription`. The field is required, nonempty, and
duplicate-free; all seven nonempty subsets are valid. Each enabled route's
matching service must be in `router.required_services` and have a worker
profile and capacity in the exact media trust domain. Disabled endpoints are
not registered and need no fake profile or capacity. Omitting `[http_media]`
disables all media HTTP endpoints. The route table is fixed at startup; there
is no per-request configuration lookup.

After the media-only `routes` field, both HTTP sections use the same transport
fields and defaults:

| Field | Default | Validation and behavior |
|---|---:|---|
| `trust_domain` | `local` | Nonempty, at most 128 bytes; only workers in this exact partition are eligible. |
| `buffered_request_max_bytes` | `8388608` | Per-request classification limit, `1..=67108864`. |
| `buffered_request_total_bytes` | `268435456` | Aggregate reserved buffered bytes; at least the per-request limit and at most `2147483647`. |
| `streamed_request_max_bytes` | `536870912` | Maximum direct fixed-length upload; at least the buffered limit and at most `4294967296`. |
| `connect_timeout_ms` | `5000` | Upstream connect timeout, `1..=60000`. |
| `request_timeout_ms` | `1800000` | One absolute precommit deadline from ingress through upstream response headers; at least the connect timeout and at most `3600000`. |
| `pool_idle_timeout_ms` | `90000` | Upstream idle connection lifetime, `1000..=300000`. |
| `pool_max_idle_per_host` | `8` | Idle pooled connections per host, `1..=256`. |

### Defaults, safety bounds, and tuning

The manifest deliberately separates three kinds of number:

| Kind | Examples | How to use it |
|---|---|---|
| Safety/correctness bound | 64 KiB config size; validation maxima for body, URI, header, frame, message, timeout, worker-count, and profile-count fields; fixed 10,551,296-byte voice upload limit | A parser, memory, protocol, or platform boundary. Passing validation does not imply that the value is affordable for a deployment. |
| Code default | `server.max_connections=1024`; classifier concurrency `4`; health `5000/1000 ms`, thresholds `2/3`, probes `16`; HTTP body/transport defaults in the table above; WebSocket defaults below | Applied only when the field is omitted from a present section. These are starting values, not measured capacity targets. |
| Explicit capacity choice | Every `admission.*`, every `workers.capacity.*`, and `shutdown.drain_timeout_ms` | Required in the manifest. There is no built-in capacity or drain default. |

`server.max_connections` bounds accepted client sockets, not worker requests.
Admission first takes one process-wide permit and one service-class permit;
dispatch then takes one exact permit from the selected worker/service pair.
Set all three layers from expected simultaneous sockets, request/session mix,
worker queueing behavior, host memory, and measured latency under overload.

Body budgets, upstream timeouts, connection-pool sizes, health cadence,
WebSocket limits, and drain time are also workload and deployment knobs even
when they have code defaults. None of the current values is claimed to be
optimal for H100s or any other hardware. Choose them with repeated
end-to-end measurements against the exact model, topology, request mix, and
failure budget; vary one resource layer at a time without changing validation
ceilings merely to improve a benchmark.

JSON and multipart bodies at or below the buffered limit are fully classified
and relayed as their original payload bytes. A larger request takes the direct
upload path only when it has one valid `Content-Length`, no transfer framing,
route hint, or content encoding, and every worker in the exact service/trust
scope has the same default model and the same correlated rows. Otherwise it is
rejected with `413`. Unknown-length requests are buffered and reserve the full
per-request limit from the aggregate byte budget.

Buffered HTTP requests wait asynchronously for a shared classification slot.
That wait and the classification itself remain inside the route's existing
`request_timeout_ms` precommit deadline.

### WebSocket policy

`[websocket]` must contain at least `[websocket.speech]` or
`[websocket.realtime]`; each route subsection has one required `trust_domain`.
The byte defaults are also their validation ceilings:

| Field | Default | Maximum |
|---|---:|---:|
| `uri_max_bytes` | 2048 | 2048 |
| `header_max_fields` | 64 | 64 |
| `header_max_bytes` | 32768 | 32768 |
| `frame_max_bytes` | 16777216 | 16777216 |
| `worker_message_max_bytes` | 67108864 | 67108864 |
| `speech_config_max_bytes` | 15029592 | 15029592 |
| `speech_message_max_bytes` | 131072 | 131072 |
| `realtime_message_max_bytes` | 16777216 | 16777216 |
| `connect_timeout_ms` | 5000 | 60000 |
| `handshake_timeout_ms` | 5000 | 60000 |
| `speech_config_timeout_ms` | 10000 | 60000 |
| `speech_idle_timeout_ms` | 30000 | 300000 |
| `close_timeout_ms` | 5000 | 60000 |

Every value must be positive. Message limits must be large enough for their
corresponding frame or route limits. Lower byte limits reduce worst-case
per-session memory exposure but must still admit observed application
messages; timeout changes alter failure detection and close behavior rather
than GPU throughput directly.

Speech upgrades first, then requires one bounded text `session.config` within
`speech_config_timeout_ms`. The router classifies it, pins a worker, connects,
replays the exact configuration text once, and requires the worker's first
application message to be `session.configured` or `error`. It then directly
relays text/control and binary audio. `stream_audio` false/omitted maps to
`non_streaming`; true maps to `streaming` and requires `pcm`.

Speech-WebSocket classification uses the same process-wide slots. It waits
asynchronously and remains cancellable by process drain; it has no separate
classifier-local timeout after the bounded `session.config` has been received.

Realtime selects a worker with the trust domain's unique default model and
connects upstream before returning the downstream `101`. The worker's first
application message must be an exact `session.created`, which is forwarded
before client application input is polled. Realtime client messages are text;
binary client messages are rejected. WebSocket subprotocols and compression
extensions are not negotiated.

### Worker manifest

The manifest contains 1–256 workers. Each worker has:

| Field | Contract |
|---|---|
| `worker_id` | Unique 1–128 byte token matching ASCII `[A-Za-z0-9._-]`. |
| `base_url` | Unique origin-only `http` or `https` URL: no credentials, path, query, fragment, or port zero. |
| `resolved_ip` | Required for a hostname and forbidden for an IP-literal URL. |
| `trust_domain` | 1–128 byte token; must equal the enabled route's domain to be eligible. |
| `default_model_id` | Required for every model-executing worker, forbidden for a voice-control-only worker, and required to appear in every model-ID-bearing service advertised by that worker. |
| `health_path` | Optional bounded absolute path; default `/health`. |
| `workers.capacity` | Exact per-service permit limits, each `1..=65535`. Every profile needs its class capacity, and unused capacity entries are rejected. |
| `workers.service_profiles` | 1–64 correlated rows. Every set is nonempty and unique unless explicitly allowed to be empty. Rows are matched independently and are never combined. |

Capacity uses `control` for `voice_control`; all other profile service names
match their capacity field. `worker_id`, target origin, and semantic profile
rows must be unique where applicable.

#### Static resolution and trust

Transport resolution is fail-closed. A hostname `base_url` must have exactly
one configured `resolved_ip`; every occurrence of the same canonical hostname,
even on different ports, must pin the same IP. IP-literal URLs derive their
transport address directly. The router never falls back to startup or
request-time DNS, never uses environment proxies, follows no redirects, and
preserves the configured URL authority for HTTP `Host`, TLS certificate
verification, and SNI. An HTTPS IP-literal worker therefore needs a certificate
valid for that IP.

There is no TTL refresh, multi-address failover, service alias expansion,
runtime registration, or dynamic discovery. Restart with a new validated
manifest to change membership or addresses.

## Correlated profile examples

These rows are representative, not model capability claims. Add only rows
proved for the exact worker and topology. A request must fit one complete row;
capabilities from separate rows are not unioned.

### Generation: text and typed multimodal

The first row accepts ordinary string-content text requests. The second accepts
typed text/image/audio/video parts, text-plus-audio output, all supported chat
audio formats, and both response modes:

```toml
[[workers.service_profiles]]
service = "generation_http"
model_ids = ["qwen3-omni"]
message_content_forms = ["string"]
media_placements = []
input_modalities = ["text"]
output_modalities = ["text"]
chat_audio_formats = []
stream_modes = ["non_streaming", "streaming"]

[[workers.service_profiles]]
service = "generation_http"
model_ids = ["qwen3-omni"]
message_content_forms = ["typed_parts"]
media_placements = ["typed_parts"]
input_modalities = ["text", "image", "audio", "video"]
output_modalities = ["text", "audio"]
chat_audio_formats = ["wav", "mp3", "flac", "pcm", "aac", "opus"]
stream_modes = ["non_streaming", "streaming"]
```

Non-text input requires a nonempty `media_placements` set. `typed_parts`
placement requires the corresponding content form. Audio output requires a
nonempty `chat_audio_formats` set, and a format set is forbidden without audio
output. Top-level `audios`, `images`, and `videos` use `top_level` placement.
Omitted/empty `modalities` defaults to text output; omitted audio format
defaults to `wav` when audio output is requested.

### TTS: HTTP, batch, and WebSocket

Enable only the TTS routes the deployment serves, then add their matching
capacities to the worker. This example includes speech HTTP, batch, and
Speech-WebSocket:

```toml
[workers.capacity]
speech_http = 32
speech_batch = 8
speech_websocket = 16
```

Use separate encoded and streaming PCM rows so their format/mode correlation
is truthful:

```toml
[[workers.service_profiles]]
service = "speech_http"
model_ids = ["tts-model"]
response_formats = ["mp3", "opus", "aac", "flac", "wav"]
stream_modes = ["non_streaming"]
tasks = ["text_to_speech", "voice_clone", "voice_design"]
reference_forms = ["none", "direct", "list", "vq_codes"]
managed_voice = false

[[workers.service_profiles]]
service = "speech_http"
model_ids = ["tts-model"]
response_formats = ["pcm"]
stream_modes = ["non_streaming", "streaming"]
tasks = ["text_to_speech", "voice_clone", "voice_design"]
reference_forms = ["none", "direct", "list", "vq_codes"]
managed_voice = false

[[workers.service_profiles]]
service = "speech_batch"
model_ids = ["tts-model"]
response_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
tasks = ["text_to_speech", "voice_clone", "voice_design"]
reference_forms = ["none", "direct", "list", "vq_codes"]
managed_voice = false
max_batch_size = 16
effective_features = ["model", "format", "task", "reference", "voice"]

[[workers.service_profiles]]
service = "speech_websocket"
model_ids = ["tts-model"]
input_profiles = ["text"]
response_formats = ["pcm"]
stream_modes = ["non_streaming", "streaming"]
tasks = ["text_to_speech", "voice_clone", "voice_design"]
reference_forms = ["none", "direct", "list", "vq_codes"]
managed_voice = false
```

Public `task_type` values map as follows: `Base` to `text_to_speech`,
`CustomVoice` to `voice_clone`, and `VoiceDesign` to `voice_design` (case,
hyphens, and underscores are normalized). Direct `ref_audio`, reference-list
audio, and `vq_codes` map to `direct`, `list`, and `vq_codes`; no reference maps
to `none`. The HTTP `stream` and WebSocket `stream_audio` flags determine the
stream mode. Any speech HTTP or speech WebSocket row containing `streaming`
must contain only `pcm`. Batch rows do not have `stream_modes`; the router
selects one worker for the whole batch, checks `max_batch_size`, and checks
which per-item override categories are listed in `effective_features`.

### ASR/transcription

```toml
[workers.capacity]
transcription_http = 32

[[workers.service_profiles]]
service = "transcription_http"
model_ids = ["asr-model"]
response_formats = ["json", "text", "verbose_json", "sse"]
media_profiles = ["audio", "audio_video"]
stream_modes = ["non_streaming", "streaming"]
```

The router classifies the multipart `file` as `audio` or `audio_video` from its
content type and/or filename extension. For non-streaming requests,
`response_format` may be `json`, `text`, or `verbose_json`; `stream=true`
selects the correlated `sse`/`streaming` requirement.

### Realtime

Every realtime worker in one route trust domain must have the same nonempty
default model:

```toml
[workers.capacity]
realtime_websocket = 16

[[workers.service_profiles]]
service = "realtime_websocket"
protocols = ["openai_realtime_v1"]
```

### Managed voice ownership

Managed voice state is optional. Set one exact owner and require its control
service:

```toml
[router]
strategy = "round_robin"
required_services = [
  "speech_http",
  "transcription_http",
  "speech_batch",
  "speech_websocket",
  "voice_control",
]
voice_owner_worker_id = "voice-owner"
```

The owner needs `control` capacity and a control row:

```toml
[workers.capacity]
control = 8

[[workers.service_profiles]]
service = "voice_control"
```

For each enabled speech surface, the owner must also have a
`managed_voice = true` row in that surface's trust domain. For example:

```toml
[[workers.service_profiles]]
service = "speech_http"
model_ids = ["tts-model"]
response_formats = ["wav"]
stream_modes = ["non_streaming"]
tasks = ["text_to_speech"]
reference_forms = ["none"]
managed_voice = true

[[workers.service_profiles]]
service = "speech_batch"
model_ids = ["tts-model"]
response_formats = ["wav"]
tasks = ["text_to_speech"]
reference_forms = ["none"]
managed_voice = true
max_batch_size = 16
effective_features = ["voice"]

[[workers.service_profiles]]
service = "speech_websocket"
model_ids = ["tts-model"]
input_profiles = ["text"]
response_formats = ["pcm"]
stream_modes = ["non_streaming"]
tasks = ["text_to_speech"]
reference_forms = ["none"]
managed_voice = true
```

When voice state is enabled, a nonempty, non-`default` `voice`/`speaker` with
no explicit reference is pinned to the owner. Default voices and explicit
references remain stateless and need matching `managed_voice = false` rows.
The router stores no voice bytes or revisions: list, upload, delete, and
owner-pinned synthesis all go directly to the one owner. Uploads are buffered
once up to 10,551,296 bytes, and mutations are never retried. Persistence after
restart is only the worker's behavior when the same owner retains the same
`SPEAKER_SAMPLES_DIR` volume; there is no replication, owner failover, shared
namespace, `fsync` guarantee, or ephemeral-volume recovery.

## Request examples

### Streaming chat

The selected profile must include `streaming`:

```bash
curl -N http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "Describe this in one sentence."}],
    "stream": true
  }'
```

### Typed multimodal chat with text and audio output

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-omni",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe the image."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }],
    "modalities": ["text", "audio"],
    "audio": {"format": "wav"}
  }'
```

### Speech and speech batch

```bash
curl http://127.0.0.1:30000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tts-model",
    "input": "Hello from SGLang-Omni.",
    "voice": "default",
    "response_format": "wav"
  }' \
  --output speech.wav
```

```bash
curl http://127.0.0.1:30000/v1/audio/speech/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tts-model",
    "voice": "default",
    "response_format": "wav",
    "items": [
      {"input": "First utterance."},
      {"input": "Second utterance."}
    ]
  }'
```

For streaming PCM over HTTP, set `"response_format":"pcm"` and
`"stream":true`; a successful worker response must include positive
`x-sample-rate`, `x-channels`, and `x-bit-depth` headers.

### Transcription

```bash
curl http://127.0.0.1:30000/v1/audio/transcriptions \
  -F model=asr-model \
  -F file=@sample.wav \
  -F response_format=verbose_json
```

Use `-F stream=true` with `curl -N` for an SSE transcription profile.

### Speech WebSocket

The first client message must be `session.config`; after the worker's
`session.configured`, send the worker-supported `input.text` and `input.done`
events. The router relays these application messages without translating the
worker protocol:

```json
{"type":"session.config","session":{"model":"tts-model","response_format":"pcm","stream_audio":true,"task_type":"Base","voice":"default"}}
```

Connect to `ws://127.0.0.1:30000/v1/audio/speech/stream`. For a complete client
example and event vocabulary, see the repository's
[TTS documentation](../docs/basic_usage/tts.md#websocket-speech-streaming).

### Realtime WebSocket

Connect an OpenAI Realtime V1 client to
`ws://127.0.0.1:30000/v1/realtime`. The router expects the worker to send
`session.created` first and then relays text events in both directions. The
[realtime playground](../playground/qwen-omni/realtime/README.md) shows the
worker-side event protocol used by SGLang-Omni.

### Voice control

```bash
curl -X POST http://127.0.0.1:30000/v1/audio/voices \
  -F name=narrator \
  -F consent=consent-recording-id \
  -F ref_text='Transcript of the reference clip.' \
  -F 'speaker_description=Clear narration voice' \
  -F 'audio_sample=@reference.wav;type=audio/wav'

curl http://127.0.0.1:30000/v1/audio/voices
curl -X DELETE http://127.0.0.1:30000/v1/audio/voices/narrator
```

## Selection, permits, retries, and cancellation

Eligibility is always computed before policy. A worker must have one complete
matching profile and default-model relationship, be in the route's trust
domain, satisfy any voice-owner requirement, be healthy, and still be serving.
Only then does policy order candidates:

- `round_robin` advances a separate cursor for each of the six data-plane
  classes and rotates eligible workers in startup manifest order. Missing or
  unhealthy workers are skipped; control traffic is exact-owner and does not
  use a data-plane cursor.
- `least_requests` snapshots each eligible worker's in-flight exact permits for
  that service class once, orders by the smallest count, and uses that class's
  round-robin rank to break ties.

Policy ordering is advisory: the router scans the ordered candidates and
atomically takes the first available exact permit. If matching healthy workers
exist but all exact permits are full, the request fails fast with `429`; it is
not queued. No matching profile yields `422`, while matching profiles with no
healthy serving worker yield `503`.

Every admitted request/session owns exactly one global permit, one service
class permit, and one selected worker's exact service permit. HTTP permits live
through upstream response EOF/error or downstream body drop. WebSocket permits
live through the complete pinned session. All are returned together by direct
ownership; there is no background lookup or delayed accounting.

The router makes one upstream attempt. Reqwest retries, redirects, proxies, and
automatic decompression are disabled, and WebSockets do not reconnect. Once an
upstream HTTP send/connection attempt begins—or a WebSocket worker is
selected—there is no retry, reselection, or failover. This remains true before
response commitment and is mandatory after response headers or a WebSocket
upgrade are committed, where replay could duplicate model work or stateful
effects.

The HTTP `request_timeout_ms` deadline covers body ingress, bounded parsing,
dispatch, upload, connection, and receipt/validation of upstream response
headers. After valid response headers are committed, the direct response body
has no absolute wall-clock timeout. It ends on clean upstream EOF, upstream
error/truncation, downstream disconnect, or process shutdown. Dropping a
downstream body immediately drops the upstream body and exact lease. WebSocket
sessions use their explicit setup, idle, message, and close bounds and the same
structural disconnect boundary.

## Health, operations, and shutdown

`GET /live` is worker-independent and remains logically live while serving or
draining. `GET /ready` starts at `503` and returns `200` only while serving and
every
`router.required_services` class has at least one healthy serving profile. It
also checks each enabled route's exact trust domain, the voice owner when
enabled, and the unique realtime default. Permit saturation does not change
readiness. Worker health uses a status-only exact `GET /health` by default;
response bodies are not parsed or buffered. No public `/health` alias is
registered.

On a loopback listener, inspect the immutable model catalog and bounded runtime
snapshots:

```bash
curl http://127.0.0.1:30000/v1/models
curl http://127.0.0.1:30000/metrics
curl http://127.0.0.1:30000/diagnostics
```

`/v1/models` is the sorted, deduplicated union of profile model IDs and worker
defaults; it is manifest state, not a live worker query. `/metrics` exposes
fixed-label lifecycle/readiness, worker health/disposition, admission, and
aggregate exact-capacity gauges. `/diagnostics` returns the same bounded state
as JSON and includes only worker IDs, startup registration ordinals,
health/disposition, and configured/in-flight capacity—never URLs, IPs, model
payloads, or request data. Operations endpoints are health-independent,
bodyless HTTP/1.1 GETs with no query, and are not registered on non-loopback
listeners.

The first `SIGINT` or `SIGTERM` atomically closes admission and exact worker
semaphores, marks workers draining, cancels health probes, stops accepting new
connections, and asks active WebSockets to close with service-restart semantics.
Existing HTTP bodies and WebSocket sessions may finish within
`drain_timeout_ms`. A second signal or deadline expiry forces session and
server termination and exits with failure; a completed first-signal drain
exits successfully.

## Logging, request IDs, and headers

`logging.format` selects newline-delimited JSON or compact text, and
`logging.filter` is a `tracing-subscriber` filter expression. The emitted
tracing events currently cover process lifecycle transitions: start, drain,
forced shutdown/deadline failure, and clean stop. There is no per-request
access log, worker-selection log, health-transition log, or request-latency
histogram in this process. Use the local proxy and client-side measurements
when those records are required; `/metrics` and `/diagnostics` expose bounded
state and current permit occupancy.

Lifecycle logs do not include request bodies, query data, arbitrary headers,
worker URLs, or manifest contents. Configuration and client errors use stable
outer messages. Treat proxy logs separately because that component may have a
different sensitive-data policy.

Every router response has one canonical `x-request-id`. A client value is
preserved only when exactly one value is present, it is 1–128 bytes, and every
byte is visible ASCII; duplicate or malformed values receive `400` with a new
router-generated ID. Otherwise the router generates a process-local monotonic
ID. The canonical value is sent to the selected worker, and any worker value is
overridden on the downstream response.

HTTP relays accept only the expected content type and framing and reconstruct a
narrow route-specific response header set. They do not forward arbitrary
client authentication, cookies, forwarding headers, or routing hints.
WebSocket handshakes copy only a validated single `Origin` and the canonical
request ID upstream. Client credentials, cookies, custom headers,
subprotocols, extensions, and worker-routing headers are not forwarded.

## Testing and benchmarks

See [DEVELOPMENT.md](DEVELOPMENT.md) for rustup setup, build/run commands, the
MSRV check, cargo-deny, and common local failures. The usual source checks are:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-targets --all-features --locked
cargo deny --locked check
```

The integration tests exercise real process, socket, HTTP relay, media, voice,
and WebSocket boundaries. The optimized process integration test can be run
separately:

```bash
cargo test --release --workspace --all-features --locked --test process
```

Build all Criterion targets without running them, or run a focused benchmark:

```bash
cargo bench --locked --no-run
cargo bench --locked --bench worker_pool
cargo bench --locked --bench chat_relay
cargo bench --locked --bench media_relay
```

These are microbenchmarks for candidate construction, policy/permit scans,
classification, multipart scanning, and direct body mechanics. They are not
end-to-end throughput or production performance claims. Dependency rationale
and audit commands are recorded in [DEPENDENCIES.md](DEPENDENCIES.md).

## Troubleshooting

- **`--check-config` exits 2:** the file is invalid strict TOML or violates a
  cross-field rule. Check service/profile/capacity alignment, the worker
  default in every advertised model service, loopback requirements, and static
  hostname pins. Configuration errors identify a field but do not echo file
  contents.
- **`/ready` stays `503`:** workers start `unknown`. Verify the pinned address,
  `health_path`, probe thresholds, every required service, enabled-route trust
  domain, and realtime default agreement. `/diagnostics` shows health and
  disposition without exposing targets.
- **`422 no_compatible_worker`:** the request's model, modalities, content
  form, placement, format, stream mode, task, reference form, batch features,
  media profile, or managed-voice state does not fit one complete row.
- **`429 router_overloaded`:** a global, service-class, exact-worker, or
  buffered-byte permit is unavailable. Those admission and memory limits fail
  fast; classifier contention instead waits asynchronously under the timing
  rules documented above.
- **Large request receives `413`:** direct upload requires fixed length and a
  homogeneous exact service/trust cohort. Reduce the payload, raise validated
  limits, or make defaults and rows identical across that cohort.
- **`502 upstream_protocol_error`:** the selected worker failed transport or
  returned an invalid HTTP status, framing, content type, required PCM
  metadata, or truncated body. The router requests an immediate health probe
  but does not retry the request. A WebSocket setup message that is invalid
  after upgrade closes the session instead of becoming an HTTP 502.
- **Hostname connects to the wrong place or TLS fails:** `resolved_ip` controls
  only the TCP destination. The original hostname remains authoritative for
  `Host`, certificate identity, and SNI.
- **Operations endpoint is `404`:** `/v1/models`, `/metrics`, and
  `/diagnostics` are registered only when `server.listen` is loopback. A
  non-loopback configuration exposes only `/live` and `/ready` and cannot
  enable inference or WebSockets.

## Security and deployment limits

The checked-in CI workflow builds and tests on Ubuntu 24.04
`x86_64-unknown-linux-gnu`; that alone does not establish support or equivalent
behavior on every OS, libc, architecture, or target. The router has no
downstream authentication, API-key validation, authorization, rate-based
limiter, CORS policy, or TLS listener. Bind it to loopback and place it behind
a trusted local proxy or sidecar that owns TLS, authentication, authorization,
request-rate policy, and any off-host exposure. Protect the TOML manifest and
worker network as trusted deployment inputs.

Build a locked optimized binary and record its identity with its config:

```bash
cargo build --release --locked --bin sgl-omni-router
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
rustc --version --verbose
sha256sum Cargo.lock target/release/sgl-omni-router
```

Redistributed binaries must include the repository root Apache-2.0 `LICENSE`.

Current non-goals and unsupported behavior include:

- launching, supervising, or colocating SGLang-Omni workers;
- dynamic registration/discovery, DNS refresh, worker metadata discovery, or
  manifest reload;
- request retries, redirects, reconnect, worker failover, or migration of
  in-flight HTTP bodies/WebSocket sessions;
- cache-aware, random, token-aware, or payload-affinity routing;
- active-active router coordination, shared control state, or voice-state
  replication;
- downstream TLS/authentication, upstream client certificates, arbitrary
  header forwarding, HTTP/2, WebSocket compression, or subprotocols;
- Python bindings or a managed worker launcher for this Rust process,
  Kubernetes integration, container packaging, signing, or an SBOM produced
  by this crate;
- tokenizer, parser, conversation, persistence, gRPC, prefill/decode, or model
  execution functionality.

The router makes no claim of zero-copy operation, performance superiority, or
unmeasured production throughput.
