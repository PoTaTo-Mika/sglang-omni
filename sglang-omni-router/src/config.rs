use std::fs::File;
use std::io::{self, Read};
use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;

use crate::error::ConfigError;
use crate::worker_pool::profile::{ServiceClass, WorkerConfig, validate_workers};

/// Maximum accepted configuration size, in bytes.
pub const MAX_CONFIG_BYTES: usize = 64 * 1024;
/// Maximum graceful-drain timeout, in milliseconds.
pub const MAX_DRAIN_TIMEOUT_MS: u64 = 300_000;
const DEFAULT_MAX_CONNECTIONS: u32 = 1024;
const MAX_CONNECTIONS: u32 = 65_535;
const MAX_LOG_FILTER_BYTES: usize = 256;
const SCHEMA_VERSION: u32 = 1;
const MAX_GLOBAL_ADMISSION: u32 = 1_000_000;
const MAX_CLASS_ADMISSION: u32 = 65_535;

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
    pub(crate) router: RouterConfig,
    pub(crate) admission: AdmissionConfig,
    pub(crate) health: HealthConfig,
    pub(crate) workers: Vec<WorkerConfig>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RouterConfig {
    #[serde(default)]
    pub(crate) strategy: RoutingStrategy,
    pub(crate) required_services: Vec<ServiceClass>,
    pub(crate) voice_owner_worker_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RoutingStrategy {
    #[default]
    RoundRobin,
    LeastRequests,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct AdmissionConfig {
    pub(crate) global: u32,
    pub(crate) generation_http: u32,
    pub(crate) speech_http: u32,
    pub(crate) transcription_http: u32,
    pub(crate) speech_batch: u32,
    pub(crate) speech_websocket: u32,
    pub(crate) realtime_websocket: u32,
    pub(crate) control: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct HealthConfig {
    interval_ms: u64,
    timeout_ms: u64,
    success_threshold: u8,
    failure_threshold: u8,
    max_concurrent_probes: u8,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            interval_ms: 5_000,
            timeout_ms: 1_000,
            success_threshold: 2,
            failure_threshold: 3,
            max_concurrent_probes: 16,
        }
    }
}

impl HealthConfig {
    pub(crate) fn interval(&self) -> Duration {
        Duration::from_millis(self.interval_ms)
    }

    pub(crate) fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms)
    }

    pub(crate) fn success_threshold(&self) -> u8 {
        self.success_threshold
    }

    pub(crate) fn failure_threshold(&self) -> u8 {
        self.failure_threshold
    }

    pub(crate) fn max_concurrent_probes(&self) -> usize {
        usize::from(self.max_concurrent_probes)
    }
}

/// Listener configuration for router-local endpoints.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// Address on which the router-local HTTP service listens.
    pub listen: SocketAddr,
    /// Maximum number of sockets accepted into Axum connection tasks.
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
}

impl ServerConfig {
    /// Returns the validated connection bound in the platform semaphore type.
    pub(crate) fn max_connections_usize(&self) -> Result<usize, ConfigError> {
        usize::try_from(self.max_connections).map_err(|_| ConfigError::InvalidField {
            field: "server.max_connections",
            reason: "cannot be represented on this platform",
        })
    }
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
    /// Reads no more than [`MAX_CONFIG_BYTES`] and validates one TOML file.
    ///
    /// Errors identify safe schema fields but never include file contents.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let file = File::open(path).map_err(ConfigError::Read)?;
        let limit = u64::try_from(MAX_CONFIG_BYTES)
            .map_err(|_| ConfigError::InternalLimit)?
            .saturating_add(1);
        let mut bytes = Vec::with_capacity(MAX_CONFIG_BYTES.saturating_add(1));
        file.take(limit)
            .read_to_end(&mut bytes)
            .map_err(ConfigError::Read)?;
        if bytes.len() > MAX_CONFIG_BYTES {
            return Err(ConfigError::TooLarge {
                maximum: MAX_CONFIG_BYTES,
            });
        }

        let text = std::str::from_utf8(&bytes).map_err(ConfigError::Encoding)?;
        let config: Self = toml::from_str(text).map_err(ConfigError::Parse)?;
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
        if self.server.max_connections == 0 || self.server.max_connections > MAX_CONNECTIONS {
            return Err(ConfigError::InvalidField {
                field: "server.max_connections",
                reason: "must be between 1 and 65535",
            });
        }
        let _max_connections = self.server.max_connections_usize()?;
        if self.shutdown.drain_timeout_ms == 0 {
            return Err(ConfigError::InvalidField {
                field: "shutdown.drain_timeout_ms",
                reason: "must be greater than zero",
            });
        }
        if self.shutdown.drain_timeout_ms > MAX_DRAIN_TIMEOUT_MS {
            return Err(ConfigError::InvalidField {
                field: "shutdown.drain_timeout_ms",
                reason: "exceeds the maximum",
            });
        }
        if self.logging.filter.is_empty() || self.logging.filter.len() > MAX_LOG_FILTER_BYTES {
            return Err(ConfigError::InvalidField {
                field: "logging.filter",
                reason: "must contain between 1 and 256 bytes",
            });
        }
        tracing_subscriber::EnvFilter::try_new(self.logging.filter.as_str()).map_err(|_| {
            ConfigError::InvalidField {
                field: "logging.filter",
                reason: "invalid filter expression",
            }
        })?;
        self.validate_router()?;
        self.validate_admission()?;
        self.validate_health()?;
        validate_workers(
            &self.workers,
            &self.router.required_services,
            self.router.voice_owner_worker_id.as_deref(),
        )?;
        Ok(())
    }

    fn validate_router(&self) -> Result<(), ConfigError> {
        if self.router.required_services.is_empty() {
            return Err(ConfigError::invalid(
                "router.required_services",
                "must be nonempty",
            ));
        }
        Ok(())
    }

    fn validate_admission(&self) -> Result<(), ConfigError> {
        if !(1..=MAX_GLOBAL_ADMISSION).contains(&self.admission.global) {
            return Err(ConfigError::invalid(
                "admission.global",
                "must be between 1 and 1000000",
            ));
        }
        let class_limits = [
            self.admission.generation_http,
            self.admission.speech_http,
            self.admission.transcription_http,
            self.admission.speech_batch,
            self.admission.speech_websocket,
            self.admission.realtime_websocket,
            self.admission.control,
        ];
        if class_limits.iter().any(|limit| {
            !(1..=MAX_CLASS_ADMISSION).contains(limit) || *limit > self.admission.global
        }) {
            return Err(ConfigError::invalid(
                "admission",
                "class limits must be between 1 and 65535 and not exceed global",
            ));
        }
        Ok(())
    }

    fn validate_health(&self) -> Result<(), ConfigError> {
        if !(100..=300_000).contains(&self.health.interval_ms) {
            return Err(ConfigError::invalid(
                "health.interval_ms",
                "must be between 100 and 300000",
            ));
        }
        if self.health.timeout_ms < 10 || self.health.timeout_ms > self.health.interval_ms {
            return Err(ConfigError::invalid(
                "health.timeout_ms",
                "must be between 10 and interval_ms",
            ));
        }
        if !(1..=32).contains(&self.health.success_threshold)
            || !(1..=32).contains(&self.health.failure_threshold)
        {
            return Err(ConfigError::invalid(
                "health",
                "thresholds must be between 1 and 32",
            ));
        }
        if !(1..=64).contains(&self.health.max_concurrent_probes) {
            return Err(ConfigError::invalid(
                "health.max_concurrent_probes",
                "must be between 1 and 64",
            ));
        }
        Ok(())
    }
}

const fn default_max_connections() -> u32 {
    DEFAULT_MAX_CONNECTIONS
}

impl From<io::Error> for ConfigError {
    fn from(source: io::Error) -> Self {
        Self::Read(source)
    }
}
