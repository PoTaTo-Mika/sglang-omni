# Direct dependency decisions

This record covers the direct dependency promotions owned by the first
`POST /v1/chat/completions` Rust relay. Versions are exact in `Cargo.toml` and
`Cargo.lock`; all sources are crates.io registry packages. The router itself
adds no `unsafe` block.

## `bytes` 1.12.1

Purpose and semantic owner: `http_generation` owns immutable buffered request
identity (`Bytes`), bounded ingress assembly (`BytesMut`), and request/response
`http_body::Frame` payloads.

Why the current graph/standard library is insufficient: Axum, Reqwest, and
HTTP Body already use this exact transitive crate, but Rust requires a direct
dependency to name its public types. `Vec<u8>` would add conversions/copies and
would not provide the shared immutable body type used by the transport APIs.

Exact enabled features and disabled defaults: direct defaults are disabled and
no direct feature is requested. The existing Axum/HTTP graph unifies the
crate's `default`/`std` feature, as shown by `cargo tree --locked -e features -i
bytes`.

Hot-path, build-script, proc-macro, native-code, and unsafe impact: request and
response hot path; no build script, proc macro, or native code. The crate uses
reviewed internal unsafe pointer/reference-counting and `BufMut` machinery;
this change adds no new version and no router-owned unsafe call.

License, source, maintenance, and advisory state: MIT; crates.io; maintained by
the Tokio project and already exercised by Axum/Reqwest in this lockfile.
`cargo deny --locked check` is the advisory/license/source gate.

Duplicate runtime/TLS/parser/telemetry functionality: none. This promotes the
single already-locked `bytes` version and adds no runtime, TLS stack, parser,
or telemetry implementation.

Alternatives and removal condition: a private `Vec<u8>` body would weaken the
shared immutable, byte-preserving representation and require additional
conversions or copies. Remove the direct dependency only if all owned body
APIs stop naming `Bytes`/`BytesMut` or a framework-owned stable re-export
replaces the direct type boundary.

## `http-body` 1.1.0

Purpose and semantic owner: `http_generation::{request_body,response_body}`
owns direct `poll_frame`, exact EOF/trailer/error state, and size hints for the
task/channel-free relay.

Why the current graph/standard library is insufficient: the standard library
has no asynchronous HTTP body trait. Axum and Reqwest already share
`http-body` 1.x transitively, but the custom wrappers must implement and invoke
that trait directly.

Exact enabled features and disabled defaults: direct defaults are disabled;
the crate defines no optional feature. Existing transitive users activate its
normal default dependency graph, confirmed by `cargo tree --locked -e features
-i http-body`.

Hot-path, build-script, proc-macro, native-code, and unsafe impact: every body
frame is on the hot path; no build script, proc macro, or native code. The
crate contains narrow internal pin-projection unsafe; router code uses only the
safe `Body`, `Frame`, and `SizeHint` APIs.

License, source, maintenance, and advisory state: MIT; crates.io; maintained in
the Hyper ecosystem and already exercised by Axum/Reqwest in this lockfile.
`cargo deny --locked check` is the advisory/license/source gate.

Duplicate runtime/TLS/parser/telemetry functionality: none. This promotes the
single locked 1.1.0 body-trait package; it does not add `http-body-util`, a
runtime, TLS, parser, or telemetry layer.

Alternatives and removal condition: Axum `Body` alone cannot define the custom
Reqwest-facing request and downstream-facing response wrappers. Remove only if
both frameworks expose an equivalent stable direct-body boundary that retains
request EOF/trailer/deadline state and response EOF/trailer/lease ownership
without an adapter task or queue.

## `serde_json` 1.0.150

Purpose and semantic owner: `http_generation::classify` owns bounded,
duplicate-aware deserialization of routing facts from a borrowed view of the
original request bytes. It never reserializes the forwarded request.

Why the current graph/standard library is insufficient: the standard library
has no JSON parser or Serde deserializer. The crate was already locked through
logging/benchmark dependencies, but routing code must directly name its
streaming deserializer and preserve Serde's recursion behavior.

Exact enabled features and disabled defaults: direct defaults are disabled and
only `std` is enabled. Existing Criterion/tracing users also activate the
crate's equivalent default (`std`) feature; no `raw_value`, `preserve_order`,
`arbitrary_precision`, or `unbounded_depth` feature is enabled.

Hot-path, build-script, proc-macro, native-code, and unsafe impact:
classification is a bounded buffered-request hot path. The crate has a Rust
build script for compiler capability/configuration checks, no proc macro and no
native code, and uses mature internal unsafe UTF-8/parser optimizations. The
router invokes only safe borrowed-deserializer APIs.

License, source, maintenance, and advisory state: MIT OR Apache-2.0; crates.io;
maintained by the Serde project and already present in the lockfile. `cargo
deny --locked check` is the advisory/license/source gate.

Duplicate runtime/TLS/parser/telemetry functionality: it is the one existing
JSON parser version in the host graph and adds no runtime, TLS, or telemetry
stack. TOML remains configuration-only and is not a duplicate JSON parser.

Alternatives and removal condition: a hand-written JSON parser would expand a
security-sensitive grammar and recursion boundary; parsing into `Value` would
allocate/copy ignored fields and obscure duplicate routing keys. Remove if
body-derived routing is removed or an authoritative shared parser provides the
same duplicate, depth, borrowed-view, and replay proof.

## `sync_wrapper` 1.0.2

Purpose and semantic owner: `http_generation::request_body` owns the proof that
Axum's downstream body is polled sequentially while satisfying Reqwest's
`Send + Sync` wrapped-body boundary.

Why the current graph/standard library is insufficient: the standard library
has no safe marker wrapper expressing “not concurrently accessed.” Copying,
spawning, or inserting a channel solely to change this bound violates the
direct-relay contract. Axum/Reqwest already use the same crate transitively.

Exact enabled features and disabled defaults: direct defaults are disabled and
no feature is requested. Reqwest transitively enables `futures`; router code
does not directly use that feature or add a `futures` dependency. Evidence is
`cargo tree --locked -e features -i sync_wrapper`.

Hot-path, build-script, proc-macro, native-code, and unsafe impact: the direct
upload wrapper is on the large-request hot path; no build script, proc macro,
or native code. The crate's central implementation contains an unsafe `Sync`
impl and pin projections whose safety contract is sequential access; the
router preserves that contract by owning and polling the body from one
`poll_frame` path and never exposing the inner body concurrently.

License, source, maintenance, and advisory state: Apache-2.0; crates.io;
already selected by Axum/Reqwest/Tower in this lockfile. `cargo deny --locked
check` is the advisory/license/source gate.

Duplicate runtime/TLS/parser/telemetry functionality: none. This promotes the
single locked version and adds no executor, futures facade, channel, TLS,
parser, or telemetry implementation.

Alternatives and removal condition: a task/channel relay or body copy is
contractually rejected. Remove if Axum's body becomes `Sync`, Reqwest relaxes
the wrapped-body bound, or a framework-owned safe direct adapter proves the
same sequential-poll invariant.

## Reproducible graph evidence

The media HTTP relay adds no dependency. Its bounded multipart scanner operates
only on complete buffered bytes, and its direct request/response paths reuse the
existing `bytes`, `http-body`, `serde_json`, and `sync_wrapper` decisions above.

Run from `sglang-omni-router`:

```text
cargo deny --locked check
cargo tree --locked -d
cargo tree --locked -e features -i bytes
cargo tree --locked -e features -i http-body
cargo tree --locked -e features -i serde_json
cargo tree --locked -e features -i sync_wrapper
```

At this decision point the host `cargo tree --locked -d` prints no duplicate
packages. The feature trees show each promoted package at exactly one version;
the release gate reruns both host and all-target inspection so target-only
platform packages are reported separately rather than mistaken for host
runtime duplication.
