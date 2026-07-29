#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![deny(clippy::await_holding_lock)]
#![deny(clippy::dbg_macro)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::print_stderr)]
#![deny(clippy::print_stdout)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![deny(clippy::unwrap_used)]
#![deny(missing_docs, unreachable_pub, unused_must_use)]

//! Process foundation for the standalone SGLang-Omni Rust router.
//!
//! This crate owns strict startup configuration, a static correlated worker
//! pool, isolated health, router-local liveness/readiness, and joined process
//! shutdown, and bounded HTTP inference relays.

mod classification;
mod config;
mod error;
mod http_generation;
mod http_media;
mod lifecycle;
mod operations;
mod server;
mod shutdown;
mod speech_facts;
mod telemetry;
mod websocket;
mod worker_pool;

use std::path::Path;

pub use config::{Config, LogFormat, MAX_CONFIG_BYTES, MAX_DRAIN_TIMEOUT_MS};
pub use error::{ConfigError, RouterError};

/// Successful result of executing the process composition root.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunOutcome {
    /// The configuration was valid and no listener was created.
    ConfigValid,
    /// The service terminated cleanly after receiving its first signal.
    CleanShutdown,
}

/// Loads configuration and either validates it or runs the service to a
/// terminal outcome.
///
/// Configuration loading and tracing initialization occur before the Tokio
/// runtime is created. Runtime work owns one server task and joins it on every
/// clean or forced shutdown path.
pub fn execute(config_path: &Path, check_config: bool) -> Result<RunOutcome, RouterError> {
    let config = Config::load(config_path)?;
    if check_config {
        return Ok(RunOutcome::ConfigValid);
    }

    init_tracing(&config)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .thread_name("sgl-omni-router")
        .enable_io()
        .enable_time()
        .build()
        .map_err(RouterError::RuntimeBuild)?;
    runtime.block_on(server::serve(config))?;
    Ok(RunOutcome::CleanShutdown)
}

fn init_tracing(config: &Config) -> Result<(), RouterError> {
    use tracing_subscriber::prelude::*;

    let filter = tracing_subscriber::EnvFilter::try_new(config.logging.filter.as_str())
        .map_err(|source| RouterError::LoggingFilter { source })?;
    let registry = tracing_subscriber::registry().with(filter);

    match config.logging.format {
        LogFormat::Json => registry
            .with(tracing_subscriber::fmt::layer().json())
            .try_init()
            .map_err(|source| RouterError::TracingInit { source }),
        LogFormat::Pretty => registry
            .with(tracing_subscriber::fmt::layer().compact())
            .try_init()
            .map_err(|source| RouterError::TracingInit { source }),
    }
}
