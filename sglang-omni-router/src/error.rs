use std::io;

use thiserror::Error;

/// Strict configuration loading and validation failures.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The configuration file could not be opened or read.
    #[error("failed to read configuration")]
    Read(#[source] io::Error),
    /// The configured byte bound cannot be represented by this platform.
    #[error("configuration size limit is not supported on this platform")]
    InternalLimit,
    /// The configuration exceeded its fixed input bound.
    #[error("configuration exceeds the {maximum}-byte limit")]
    TooLarge {
        /// Maximum accepted bytes.
        maximum: usize,
    },
    /// The configuration was not UTF-8.
    #[error("configuration must be UTF-8")]
    Encoding(#[source] std::str::Utf8Error),
    /// TOML syntax, duplicate fields, or unknown fields were invalid.
    #[error("configuration is not valid strict TOML")]
    Parse(#[source] toml::de::Error),
    /// A parsed field violated a bounded semantic rule.
    #[error("invalid configuration field {field}: {reason}")]
    InvalidField {
        /// Stable schema field name.
        field: &'static str,
        /// Stable, non-sensitive validation reason.
        reason: &'static str,
    },
}

/// Top-level startup, runtime, and shutdown failures.
#[derive(Debug, Error)]
pub enum RouterError {
    /// Configuration failed before process infrastructure was created.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// The validated log filter could not be reconstructed.
    #[error("failed to construct the configured logging filter")]
    LoggingFilter {
        /// Internal parser source, available to structured diagnostics.
        #[source]
        source: tracing_subscriber::filter::ParseError,
    },
    /// The process-global tracing subscriber was already initialized or failed.
    #[error("failed to initialize structured diagnostics")]
    TracingInit {
        /// Internal initialization source.
        #[source]
        source: tracing_subscriber::util::TryInitError,
    },
    /// The bounded Tokio runtime could not be created.
    #[error("failed to initialize the async runtime")]
    RuntimeBuild(#[source] io::Error),
    /// The configured listener could not be bound.
    #[error("failed to bind the configured listener")]
    Bind(#[source] io::Error),
    /// The HTTP server stopped without a shutdown request.
    #[error("the local HTTP server stopped unexpectedly")]
    Server(#[source] io::Error),
    /// A server task panicked or otherwise failed to join.
    #[error("the local HTTP server task failed")]
    ServerTask(#[source] tokio::task::JoinError),
    /// Graceful drain exceeded the configured monotonic deadline.
    #[error("graceful shutdown exceeded its configured deadline")]
    DrainTimeout,
    /// A second signal forced an incomplete graceful drain.
    #[error("a second signal forced shutdown before graceful drain completed")]
    ForcedShutdown,
    /// The lifecycle owner observed an illegal or poisoned transition.
    #[error("the process lifecycle invariant failed")]
    Lifecycle,
    /// The graceful-shutdown notification owner disappeared unexpectedly.
    #[error("failed to notify the local HTTP server to drain")]
    ShutdownNotify,
    /// Signal observation could not be installed or completed.
    #[error("failed to observe process termination signals")]
    Signal(#[source] io::Error),
    /// The isolated HTTPS health client failed to build.
    #[error("failed to initialize the isolated health client")]
    HealthClient(#[source] reqwest::Error),
    /// Validated worker-pool data could not be reconstructed internally.
    #[error("the validated worker-pool invariant failed")]
    WorkerPoolInvariant,
    /// A health worker exited before cancellation or failed during shutdown.
    #[error("an owned health task failed")]
    HealthTask,
}

impl RouterError {
    /// Returns the stable process exit code for this failure.
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::Config(_) => 2,
            Self::LoggingFilter { .. }
            | Self::TracingInit { .. }
            | Self::RuntimeBuild(_)
            | Self::Bind(_)
            | Self::Server(_)
            | Self::ServerTask(_)
            | Self::DrainTimeout
            | Self::ForcedShutdown
            | Self::Lifecycle
            | Self::ShutdownNotify
            | Self::Signal(_)
            | Self::HealthClient(_)
            | Self::WorkerPoolInvariant
            | Self::HealthTask => 1,
        }
    }
}

impl ConfigError {
    pub(crate) const fn invalid(field: &'static str, reason: &'static str) -> Self {
        Self::InvalidField { field, reason }
    }
}
