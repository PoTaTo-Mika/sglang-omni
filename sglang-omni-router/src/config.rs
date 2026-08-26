use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;

use crate::error::ConfigError;

const DEFAULT_MAX_CONNECTIONS: usize = 1024;
const SCHEMA_VERSION: u32 = 1;
const MAX_GLOBAL_ADMISSION: u32 = 1_000_000;
const MAX_REQUEST_BYTES: u64 = 536_870_912;
const MAX_TIMEOUT_MS: u64 = 1_800_000;
const MAX_POOL_IDLE_PER_HOST: usize = 1_024;
const MAX_WORKER_ID_BYTES: usize = 128;
const MAX_BASE_URL_BYTES: usize = 2_048;

/// Fully parsed and validated process configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    schema_version: u32,
    /// Listener configuration for router-local endpoints.
    pub server: ServerConfig,
    /// Graceful-shutdown limits.
    pub shutdown: ShutdownConfig,
    /// Structured diagnostic output configuration.
    pub logging: LoggingConfig,
    /// Fail-fast ingress capacity shared by all inference routes.
    pub(crate) admission: AdmissionConfig,
    /// HTTP generation transport limits.
    pub(crate) http_generation: HttpGenerationConfig,
    /// Static worker manifest. This branch requires exactly one worker.
    pub(crate) workers: Vec<WorkerConfig>,
}

/// Global fail-fast request admission.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct AdmissionConfig {
    /// Maximum requests admitted concurrently.
    pub(crate) global: u32,
}

impl AdmissionConfig {
    pub(crate) fn global_usize(&self) -> Result<usize, ConfigError> {
        usize::try_from(self.global).map_err(|_source| {
            ConfigError::invalid("admission.global", "cannot be represented on this platform")
        })
    }
}

/// Bounded transport policy for generation HTTP.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct HttpGenerationConfig {
    /// Maximum fixed-length request body accepted by the direct relay.
    pub(crate) streamed_request_max_bytes: u64,
    pub(crate) connect_timeout_ms: u64,
    pub(crate) request_timeout_ms: u64,
    pub(crate) pool_idle_timeout_ms: u64,
    /// Maximum pooled idle connections retained for the worker authority.
    pub(crate) pool_max_idle_per_host: usize,
}

impl HttpGenerationConfig {
    pub(crate) const fn connect_timeout(&self) -> Duration {
        Duration::from_millis(self.connect_timeout_ms)
    }

    pub(crate) const fn request_timeout(&self) -> Duration {
        Duration::from_millis(self.request_timeout_ms)
    }

    pub(crate) const fn pool_idle_timeout(&self) -> Duration {
        Duration::from_millis(self.pool_idle_timeout_ms)
    }
}

/// Stable static-worker record extended by later routing branches.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerConfig {
    /// Operator-assigned identity.
    pub(crate) worker_id: String,
    /// Canonical HTTP or HTTPS authority and base path.
    pub(crate) base_url: String,
    /// Pinned transport address for a DNS-name authority.
    pub(crate) resolved_ip: Option<IpAddr>,
}

/// Listener configuration for router-local endpoints.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// Address on which the router-local HTTP service listens.
    pub listen: SocketAddr,
    /// Maximum number of sockets accepted into Axum connection tasks.
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
}

/// Graceful-shutdown limits.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ShutdownConfig {
    drain_timeout_ms: u64,
}

impl ShutdownConfig {
    /// Monotonic duration available for graceful server drain.
    pub fn drain_timeout(&self) -> Duration {
        Duration::from_millis(self.drain_timeout_ms)
    }
}

/// Structured diagnostic output configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LoggingConfig {
    /// Output encoding for structured diagnostics.
    pub format: LogFormat,
    /// Tracing filter expression. This value comes only from the config file.
    pub filter: String,
}

/// Supported diagnostic output encodings.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    /// One JSON object per event.
    Json,
    /// Compact human-readable events.
    Pretty,
}

impl Config {
    /// Reads and validates one TOML file.
    ///
    /// Errors identify safe schema fields but never include file contents.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let bytes = fs::read(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let text = std::str::from_utf8(&bytes).map_err(|source| ConfigError::Encoding {
            path: path.to_path_buf(),
            source,
        })?;
        let config: Self =
            toml::from_str(text).map_err(|source: toml::de::Error| ConfigError::Parse {
                path: path.to_path_buf(),
                message: source.message().to_owned(),
            })?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ConfigError::InvalidField {
                field: "schema_version",
                reason: "unsupported version",
            });
        }
        if self.server.max_connections == 0
            || self.server.max_connections > tokio::sync::Semaphore::MAX_PERMITS
        {
            return Err(ConfigError::InvalidField {
                field: "server.max_connections",
                reason: "must fit the listener semaphore and be greater than zero",
            });
        }
        if self.shutdown.drain_timeout_ms == 0 {
            return Err(ConfigError::InvalidField {
                field: "shutdown.drain_timeout_ms",
                reason: "must be greater than zero",
            });
        }
        if tokio::time::Instant::now()
            .checked_add(self.shutdown.drain_timeout())
            .is_none()
        {
            return Err(ConfigError::InvalidField {
                field: "shutdown.drain_timeout_ms",
                reason: "cannot be represented by the monotonic clock",
            });
        }
        if self.logging.filter.is_empty() {
            return Err(ConfigError::InvalidField {
                field: "logging.filter",
                reason: "must not be empty",
            });
        }
        tracing_subscriber::EnvFilter::try_new(self.logging.filter.as_str()).map_err(|_| {
            ConfigError::InvalidField {
                field: "logging.filter",
                reason: "invalid filter expression",
            }
        })?;
        self.validate_core_http()?;
        Ok(())
    }

    fn validate_core_http(&self) -> Result<(), ConfigError> {
        if self.admission.global == 0 || self.admission.global > MAX_GLOBAL_ADMISSION {
            return Err(ConfigError::invalid(
                "admission.global",
                "must be between 1 and 1000000",
            ));
        }
        let _global = self.admission.global_usize()?;
        let generation = &self.http_generation;
        if !(1..=MAX_REQUEST_BYTES).contains(&generation.streamed_request_max_bytes) {
            return Err(ConfigError::invalid(
                "http_generation.streamed_request_max_bytes",
                "must be between 1 and 536870912",
            ));
        }
        for (field, value) in [
            (
                "http_generation.connect_timeout_ms",
                generation.connect_timeout_ms,
            ),
            (
                "http_generation.request_timeout_ms",
                generation.request_timeout_ms,
            ),
            (
                "http_generation.pool_idle_timeout_ms",
                generation.pool_idle_timeout_ms,
            ),
        ] {
            if value == 0 || value > MAX_TIMEOUT_MS {
                return Err(ConfigError::invalid(field, "must be between 1 and 1800000"));
            }
        }
        if !(1..=MAX_POOL_IDLE_PER_HOST).contains(&generation.pool_max_idle_per_host) {
            return Err(ConfigError::invalid(
                "http_generation.pool_max_idle_per_host",
                "must be between 1 and 1024",
            ));
        }
        if self.workers.len() != 1 {
            return Err(ConfigError::invalid(
                "workers",
                "core HTTP requires exactly one worker",
            ));
        }
        let worker = &self.workers[0];
        if worker.worker_id.is_empty()
            || worker.worker_id.len() > MAX_WORKER_ID_BYTES
            || !worker
                .worker_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        {
            return Err(ConfigError::invalid(
                "workers.worker_id",
                "must be 1 to 128 ASCII identifier bytes",
            ));
        }
        if worker.base_url.is_empty() || worker.base_url.len() > MAX_BASE_URL_BYTES {
            return Err(ConfigError::invalid(
                "workers.base_url",
                "must contain between 1 and 2048 bytes",
            ));
        }
        crate::upstream::ResolvedTarget::validate(worker)
    }
}

const fn default_max_connections() -> usize {
    DEFAULT_MAX_CONNECTIONS
}
