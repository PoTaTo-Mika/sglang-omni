# Direct dependency decisions

The entries below record five packages used directly by the transport and
classification layers. Versions are exact and the lockfile remains
authoritative.

## `reqwest` 0.12.28

Purpose and semantic owner: `worker_pool::resolver` owns pooled HTTP/1.1 health
and generation clients, static name resolution, rustls TLS identity,
connection timeout, and direct bodies.
The standard library and Axum do not provide an outbound client. Defaults are
disabled; only `rustls-tls` is enabled. The client is request/response hot-path
code and introduces no second runtime or TLS stack. It is MIT OR Apache-2.0,
from crates.io, and matches the selected SGLang dependency family. Redirects,
ambient proxies, retries, and decompression are disabled explicitly. Remove it
only if a measured direct-Hyper implementation replaces the complete owner.

## `bytes` 1.12.1

Purpose and semantic owner: the direct body adapters name immutable transport
frames without copying them into a router-owned representation. The standard
library has no shared body-frame type. Defaults are disabled and no feature is
requested. It is hot-path code, has no build script, proc macro, or native
code, and is the existing Tokio/Hyper version under the MIT license. Remove it
when neither adapter names `Bytes`.

## `http-body` 1.1.0

Purpose and semantic owner: the request and response adapters implement direct
`poll_frame`, terminal error, trailer, EOF, and backpressure behavior. The
standard library has no asynchronous HTTP body trait. Defaults are disabled;
the crate has no optional feature. It is existing MIT Hyper ecosystem code and
adds no runtime, parser, TLS, build script, or native code. Remove it only when
both framework boundaries expose an equivalent direct adapter.

## `sync_wrapper` 1.0.2

Purpose and semantic owner: `request_body` expresses sequential polling of the
Axum body at Reqwest's `Send + Sync` body boundary without a relay task or
channel. Defaults are disabled and no feature is requested. The package is an
existing Apache-2.0 dependency with no build script, proc macro, native code,
or second runtime. It contains the reviewed unsafe marker implementation; the
router calls only its safe sequential-access API. Remove it if Axum's body
becomes `Sync` or Reqwest no longer requires this boundary.

## `serde_json` 1.0.150

Purpose and semantic owner: `http_generation::classify` performs one bounded
Serde pass only when the startup manifest cannot prove content-blind routing.
The standard library and TOML parser cannot classify JSON request facts.
Defaults are disabled and only `std` is enabled. Classification runs in a
bounded blocking slot and is absent from the homogeneous hot path. The crate
is MIT OR Apache-2.0, from crates.io, already present in the selected ecosystem,
and adds no runtime, TLS stack, native code, or build script. Remove it only if
the heterogeneous JSON route is removed or an equivalent bounded parser owns
the complete classification contract.

Run `cargo tree --locked --duplicates` after every dependency change. No direct
dependency is accepted for future-only use.
