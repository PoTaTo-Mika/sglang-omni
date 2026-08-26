use std::fs::File;
use std::io::{self, Read};
use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;

use crate::error::ConfigError;

/// Maximum accepted configuration size, in bytes.
pub const MAX_CONFIG_BYTES: usize = 64 * 1024;
/// Maximum graceful-drain timeout, in milliseconds.
pub const MAX_DRAIN_TIMEOUT_MS: u64 = 300_000;
const DEFAULT_MAX_CONNECTIONS: u32 = 1024;
const MAX_CONNECTIONS: u32 = 65_535;
const MAX_LOG_FILTER_BYTES: usize = 256;
const SCHEMA_VERSION: u32 = 1;

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
