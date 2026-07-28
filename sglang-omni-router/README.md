# SGLang-Omni Rust router

This directory contains the standalone `sgl-omni-router` service. The
foundation release validates one bounded TOML configuration, serves only
router-local liveness and readiness, and owns graceful process shutdown. It has
no worker, proxy, or inference behavior.

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
```

All fields are explicit. Unknown and duplicate fields are rejected. The
logging filter comes only from this file; environment variables such as
`RUST_LOG` are not read.

While serving, `GET /live` returns `200`. In this foundation release,
`GET /ready` always returns `503` because no worker pool exists. No `/health`
alias or inference route is registered.
