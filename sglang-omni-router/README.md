# SGLang-Omni Rust router

`sgl-omni-router` is the standalone Rust router process for SGLang-Omni. This
foundation provides strict configuration loading, a bounded TCP listener,
process liveness, and owned graceful shutdown. Worker routing and inference
endpoints are added by later branches.

## Development setup

Install [Rustup](https://rustup.rs/), then enter this directory. The checked-in
`rust-toolchain.toml` selects Rust 1.97.1 with rustfmt and Clippy. Rust 1.90.0
is the minimum supported Rust version and is used only for the separate MSRV
check.

```console
rustup toolchain install 1.90.0 --profile minimal
cargo build --locked
cargo test --locked
cargo +1.90.0 check --workspace --all-targets --all-features --locked
```

Install `cargo-deny` 0.20.2 to run the dependency-policy check locally:

```console
cargo deny --locked check
```

The complete formatting, lint, build, test, documentation, dependency, and
MSRV commands are recorded in `.github/workflows/rust-router.yml`.

## Configuration and CLI

Create `router.toml`:

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
```

All fields are explicit. Unknown and duplicate fields are rejected. The
configuration is limited to 64 KiB and must be UTF-8. Diagnostics never include
the configuration contents or path. The logging filter comes only from this
file; `RUST_LOG` is not read.

Validate configuration without binding the listener:

```console
cargo run --locked -- --config router.toml --check-config
```

Run the service:

```console
cargo run --locked -- --config router.toml
```

`--help` and `--version` do not require a configuration file.

## Process behavior

While the listener is serving, exact `GET /live` returns `200` with `live\n`.
No `/ready`, `/health`, inference, worker, or metrics route is registered in
this foundation.

The listener accepts at most `server.max_connections` active connections. On
the first `SIGINT` or `SIGTERM`, it stops accepting new sockets and drains all
owned connection tasks for at most `shutdown.drain_timeout_ms`. A distinct
second signal forces shutdown. A drain deadline or forced shutdown exits with
a failure status; a completed graceful drain exits successfully.
