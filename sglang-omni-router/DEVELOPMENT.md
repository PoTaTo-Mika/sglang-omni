# Development

Run all commands from `sglang-omni-router/`. CI uses the same commands and the
committed `Cargo.lock`.

## Toolchains

Install Rustup, then install the pinned implementation toolchain and the
separate minimum-supported toolchain:

```console
rustup toolchain install 1.97.1 \
  --profile minimal \
  --component clippy,rustfmt
rustup toolchain install 1.90.0 --profile minimal
```

`rust-toolchain.toml` selects Rust 1.97.1 for normal commands. Rust 1.90.0 is
used only for the compatibility check; do not build the operator binary with
the minimum-supported toolchain.

## Quality gates

Run the gates in CI order:

```console
cargo fmt --all -- --check
cargo +1.90.0 check --workspace --all-targets --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-targets --all-features --locked -- --test-threads=1
RUSTDOCFLAGS="-D warnings" \
  cargo doc --workspace --all-features --no-deps --locked
```

Build the optimized binary once and use that artifact for operator-input
checks:

```console
cargo build --release --workspace --all-features --locked

binary="./target/release/sgl-omni-router"
"$binary" --help
"$binary" --version

configs="$(git ls-files -- 'examples/*.toml' | LC_ALL=C sort)"
if [[ -n "$configs" ]]; then
  while IFS= read -r config; do
    "$binary" --config "$config" --check-config
  done <<< "$configs"
fi
```

The example loop requires Bash. Before examples are added it has no inputs;
afterward, every tracked example must pass.
