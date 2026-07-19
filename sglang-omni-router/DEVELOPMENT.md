# Developing the SGLang-Omni Router

Run these commands from `sglang-omni-router/`.

## Prerequisites

- Git and a platform C compiler/linker (Xcode Command Line Tools on macOS, or
  the usual build-essential toolchain on Linux).
- [rustup](https://rustup.rs/) with the checked-in toolchain and components:

```bash
rustup toolchain install 1.97.1 \
  --profile minimal \
  --component clippy,rustfmt
rustup show active-toolchain
```

Inside this directory, `rust-toolchain.toml` selects Rust 1.97.1 automatically.
`Cargo.toml` separately declares Rust 1.90 as the minimum supported Rust
version; CI checks that version with `cargo +1.90.0 check`.

## Build, validate, and run

```bash
cargo build --locked --bin sgl-omni-router
./target/debug/sgl-omni-router \
  --config examples/text-generation.toml \
  --check-config
./target/debug/sgl-omni-router --config examples/text-generation.toml
```

The example expects a separately managed worker at `127.0.0.1:8000`. Stop the
router with `Ctrl-C`; the first signal begins bounded drain.

## Local checks

Format code, then run the same focused quality checks used by CI:

```bash
cargo fmt --all
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-targets --all-features --locked
```

To verify the stated MSRV after installing it:

```bash
rustup toolchain install 1.90.0 --profile minimal
cargo +1.90.0 check --workspace --all-targets --all-features --locked
```

The dependency policy is also a CI gate. Install the same cargo-deny version
and run it from this directory:

```bash
cargo install --locked cargo-deny --version 0.20.2
cargo deny --locked check
```

Compile every Criterion target, or perform a short worker-pool benchmark smoke:

```bash
cargo bench --locked --no-run
cargo bench --locked --bench worker_pool -- --test
```

Use optimized builds only for measurements:

```bash
cargo build --release --locked --bin sgl-omni-router
```

## Common failures

- **The wrong compiler is active:** enter this directory and run
  `rustup show active-toolchain`; it should report 1.97.1 selected by
  `rust-toolchain.toml`.
- **`rustfmt`, Clippy, or cargo-deny is missing:** install the component or
  exact tool version shown above. cargo-deny is not bundled with Rust.
- **A linker, C compiler, or assembler is missing:** install the platform build
  tools; the reviewed TLS/crypto dependency graph includes build scripts.
- **`--check-config` fails:** keep every enabled service, worker capacity, and
  correlated profile aligned. Unknown fields and enum spellings are rejected.
- **The listener is already in use or `/ready` stays at 503:** change
  `server.listen`, then verify the worker URL and its `/health` endpoint.

`deny.toml` is retained because it defines the repository's advisory, license,
source, duplicate, feature, and build-script policy. `rust-toolchain.toml` is
retained to make the normal contributor compiler and components reproducible;
it does not replace the separate MSRV check.
