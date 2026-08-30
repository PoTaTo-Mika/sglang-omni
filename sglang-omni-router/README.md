# SGLang-Omni Rust Router

`sgl-omni-router` is a standalone Rust process that routes OpenAI-compatible
chat, speech, transcription, translation, and realtime traffic to static
SGLang-Omni workers. It validates worker capabilities at startup, selects one
compatible healthy worker per request or session, and relays the original
request and response with bounded admission.

The router does not launch model workers. Start each worker named by the
selected manifest before starting the router.

## Quick start

Install [Rustup](https://rustup.rs/), enter this directory, and build the
optimized binary:

```console
cargo build --release --locked
```

Validate and run the one-worker chat example:

```console
./target/release/sgl-omni-router \
  --config examples/chat.toml \
  --check-config

./target/release/sgl-omni-router \
  --config examples/chat.toml
```

The example expects its worker at `127.0.0.1:8000`. Once `/ready` returns
`200`, send a request through the router:

```console
curl --http1.1 http://127.0.0.1:30000/v1/chat/completions \
  --header 'content-type: application/json' \
  --data-binary \
  '{"model":"chat-model","messages":[{"role":"user","content":"hello"}]}'
```

The checked examples are:

- [chat.toml](examples/chat.toml): one text chat worker;
- [omni.toml](examples/omni.toml): two replicas serving multimodal chat and
  audio output;
- [tts.toml](examples/tts.toml): two replicas serving encoded speech and PCM
  speech streaming;
- [asr.toml](examples/asr.toml): two replicas with separate transcription and
  translation capability rows.

These manifests use generic safety limits, not performance-optimal values.
Set admission and worker capacities from the measured concurrency of the
deployment. Configuration is strict: unknown fields, duplicate identities,
invalid limits, or inconsistent capability rows fail `--check-config` and
startup.

For the pinned toolchains and complete local quality gates, see
[DEVELOPMENT.md](DEVELOPMENT.md).

## Routes

Only configured data-plane routes are installed:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | Chat, multimodal generation, and chat audio output |
| `POST` | `/v1/audio/speech` | Encoded speech or streaming PCM |
| `POST` | `/v1/audio/speech/batch` | One ordered, unsplit speech batch |
| `POST` | `/v1/audio/transcriptions` | Multipart transcription |
| `POST` | `/v1/audio/translations` | Multipart speech translation |
| `GET` | `/v1/audio/speech/stream` | Terminating speech WebSocket |
| `GET` | `/v1/realtime?model=<id>` | Terminating OpenAI-compatible realtime WebSocket |
| `GET` | `/v1/audio/voices` | List voices on the configured voice owner |
| `POST` | `/v1/audio/voices` | Upload one voice to the configured voice owner |
| `DELETE` | `/v1/audio/voices/{name}` | Delete one voice from the configured voice owner |

Voice routes exist only when `router.voice_owner_worker_id` names one worker
with control capacity and a `voice_control` profile. The router stores no voice
registry and does not replicate or persist worker voice state.

Non-streaming transcription profiles may advertise `json`, `text`,
`verbose_json`, `srt`, and `vtt`. Streaming requests use form format `json` or
`text` and receive SSE. Translation support is independently declared by a
`task = "translate"` profile row, so it cannot be inferred from transcription
support. A translation row may advertise `srt` or `vtt` only when that worker
supports segment timestamps; the checked ASR example intentionally does not.
Speech profiles keep encoded non-streaming formats separate from PCM rows that
support streaming.

The WebSocket routes terminate both handshakes and pin one worker for the
session. Speech accepts one bounded `session.config` before worker selection;
realtime selects and connects upstream before completing the downstream
upgrade. Frames retain their type and order under direct destination
backpressure.

`websocket.setup_timeout_ms` is one absolute pre-relay setup deadline: it
starts after a valid speech configuration for speech and before upstream
dispatch for realtime. It is not a relay idle timeout.

## Configuration contracts

`http_media.routes` may contain any enabled subset of `speech`,
`speech_batch`, `transcription`, and `translation`. Each enabled route needs
its admission class, matching worker capacity, and a correlated profile in the
same trust domain. Transcription and translation share
`transcription_http` capacity but use separate profile rows with exactly one
`task` value.

Speech-batch capacity is counted in items, not HTTP requests. A batch is never
split: the router atomically acquires one global envelope plus its complete
item count from `speech_batch` admission and one worker. Every profile
`max_batch_size` must fit both the selected worker's batch capacity and
`admission.speech_batch`.

Enabling `websocket.speech` requires `speech_websocket` admission and worker
capacity plus matching `speech_websocket` profiles. Enabling
`websocket.realtime` similarly requires `realtime_websocket` admission and
capacity plus a `realtime_websocket` profile declaring
`openai_realtime_v1`. Each route has its own trust domain and may be enabled
independently.

Voice state has one exact owner: `router.voice_owner_worker_id` must name a
configured worker with `control` capacity and a `voice_control` profile, and
`admission.control` must be configured. For every enabled speech HTTP, batch,
or speech-WebSocket consumer, that owner also needs a matching
`managed_voice = true` row in the route's trust domain. Managed named-voice
traffic without an explicit reference is pinned to the owner; stateless/default
voice and explicit-reference traffic retain normal replica selection.

Chat generation requires HTTP/1.1 with one valid `Content-Length` and rejects
ambiguous or transfer-framed uploads. Fixed-length non-batch media requests can
use a proven direct cohort; media without a usable fixed length uses the
bounded buffered path. On media routes,
`x-sglang-omni-route-model` and `x-sglang-omni-route-stream` are router-local
assertions and are stripped before the upstream request. Explicit body/form
facts must match their assertions. A model assertion can only confirm a
configured worker default, and an absent stream value is `false`, so a
header-only `true` stream assertion is rejected. Original body bytes are not
reconstructed.

## Worker profiles and routing

Each worker has a static URL, trust domain, capacity table, and correlated
service-profile rows. A profile row describes combinations the worker actually
supports; the router never combines a model from one row with modalities,
formats, tasks, or streaming support from another. A DNS worker authority must
also provide `resolved_ip`; the router preserves the configured authority for
HTTP `Host` and TLS SNI while dialing the pinned address.

Homogeneous means replicas of the same service contract, not merely multiple
workers and not different models. At startup, the router proves content-blind
eligibility separately for generation and each media service/task from worker
defaults and correlated rows. A fixed-length, non-batch request can then use
the direct path without body classification. Batch speech is always classified
because the body owns its item-credit requirement.

Heterogeneous cohorts and requests with body-owned routing facts use the
bounded path: reserve aggregate byte capacity, buffer once, classify once on
the shared blocking-work limit, select one compatible worker, and relay the
original bytes unchanged. Both paths make one upstream attempt and retain
admission and exact-worker leases through response EOF, error, or downstream
drop.

`round_robin` is the default strategy for equal replicas. `least_requests` is
also available and selects from current exact worker-capacity occupancy with
deterministic ties. Neither policy is universally faster; qualify the policy
with the deployment's models, topology, and workload.

The listener must be loopback because the router has no client authentication.
Use a separately managed local TLS/auth proxy when external access is needed.

## Health, readiness, and shutdown

- `GET /live` reports process liveness.
- `GET /ready` returns `200` only while serving and every configured route has
  a compatible healthy worker. Voice state also requires its exact owner.
- Workers begin `Unknown`. Periodic status-only probes apply the configured
  consecutive success and failure thresholds; transport/protocol failures can
  request an immediate probe.
- Worker application errors are relayed and do not directly change health.
- Capacity exhaustion does not make a worker unhealthy.

The first `SIGINT` or `SIGTERM` fails readiness, closes admission, stops new
dispatch, asks tracked WebSockets to close, cancels health work, and joins
owned work within `shutdown.drain_timeout_ms`. A second signal forces
a failed shutdown.

## Operations

The mandatory loopback boundary also owns these read-only routes:

- `GET /v1/models`: startup-precomputed, sorted model inventory;
- `GET /metrics`: Prometheus lifecycle, readiness, worker-state, admission,
  and capacity gauges rendered from current enforcing state;
- `GET /diagnostics`: bounded worker, lifecycle, admission, and capacity JSON.

Operations endpoints never contact workers or mutate request-path metrics.
They require HTTP/1.1 with no query or body. Unsupported methods return `405`
with `Allow: GET`. Diagnostics are redacted and fixed-order; default tracing is
limited to lifecycle, health, and exceptions, with no access logs, per-request
spans, or stage histograms.

One process-wide canonical `x-request-id` covers data, WebSocket, health,
operations, `404`, and `405` responses. A valid caller value is preserved;
otherwise the router generates one. Where a worker is contacted, that same ID
is authoritative upstream and downstream.

## Current scope

The router intentionally uses one static manifest and one multi-threaded Rust
process. It does not provide dynamic worker CRUD or discovery, queues, circuit
breakers, cache-aware or PD routing, CP/DP shared state, Python bindings,
model-worker supervision, or a Python-router compatibility API. One committed
request or session has no in-request retry, worker reselection, or failover;
later requests continue to select from workers that remain healthy.
