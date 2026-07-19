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
const DEFAULT_BUFFERED_REQUEST_MAX_BYTES: u64 = 8_388_608;
const DEFAULT_BUFFERED_REQUEST_TOTAL_BYTES: u64 = 268_435_456;
const DEFAULT_STREAMED_REQUEST_MAX_BYTES: u64 = 536_870_912;
const DEFAULT_CONNECT_TIMEOUT_MS: u64 = 5_000;
const DEFAULT_REQUEST_TIMEOUT_MS: u64 = 1_800_000;
const DEFAULT_POOL_IDLE_TIMEOUT_MS: u64 = 90_000;
const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 8;
const DEFAULT_MAX_CONNECTIONS: u32 = 1024;
const MAX_CONNECTIONS: u32 = 65_535;
const MAX_LOG_FILTER_BYTES: usize = 256;
const SCHEMA_VERSION: u32 = 1;
const MAX_GLOBAL_ADMISSION: u32 = 1_000_000;
const MAX_CLASS_ADMISSION: u32 = 65_535;
const DEFAULT_MAX_CONCURRENT_CLASSIFICATIONS: u8 = 4;

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
    pub(crate) http_generation: Option<HttpGenerationConfig>,
    pub(crate) http_media: Option<HttpMediaConfig>,
    pub(crate) workers: Vec<WorkerConfig>,
}

/// Bounded transport and buffering policy for chat generation HTTP.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct HttpGenerationConfig {
    pub(crate) trust_domain: String,
    pub(crate) buffered_request_max_bytes: u64,
    pub(crate) buffered_request_total_bytes: u64,
    pub(crate) streamed_request_max_bytes: u64,
    connect_timeout_ms: u64,
    request_timeout_ms: u64,
    pool_idle_timeout_ms: u64,
    pub(crate) pool_max_idle_per_host: usize,
}

impl Default for HttpGenerationConfig {
    fn default() -> Self {
        Self {
            trust_domain: String::from("local"),
            buffered_request_max_bytes: DEFAULT_BUFFERED_REQUEST_MAX_BYTES,
            buffered_request_total_bytes: DEFAULT_BUFFERED_REQUEST_TOTAL_BYTES,
            streamed_request_max_bytes: DEFAULT_STREAMED_REQUEST_MAX_BYTES,
            connect_timeout_ms: DEFAULT_CONNECT_TIMEOUT_MS,
            request_timeout_ms: DEFAULT_REQUEST_TIMEOUT_MS,
            pool_idle_timeout_ms: DEFAULT_POOL_IDLE_TIMEOUT_MS,
            pool_max_idle_per_host: DEFAULT_POOL_MAX_IDLE_PER_HOST,
        }
    }
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

    pub(crate) fn buffered_max_usize(&self) -> Result<usize, ConfigError> {
        usize::try_from(self.buffered_request_max_bytes).map_err(|_| {
            ConfigError::invalid(
                "http_generation.buffered_request_max_bytes",
                "cannot be represented on this platform",
            )
        })
    }

    pub(crate) fn buffered_total_usize(&self) -> Result<usize, ConfigError> {
        usize::try_from(self.buffered_request_total_bytes).map_err(|_| {
            ConfigError::invalid(
                "http_generation.buffered_request_total_bytes",
                "cannot be represented on this platform",
            )
        })
    }
}

/// Bounded transport and buffering policy shared by all media HTTP routes.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct HttpMediaConfig {
    pub(crate) routes: Vec<HttpMediaRoute>,
    pub(crate) trust_domain: String,
    pub(crate) buffered_request_max_bytes: u64,
    pub(crate) buffered_request_total_bytes: u64,
    pub(crate) streamed_request_max_bytes: u64,
    connect_timeout_ms: u64,
    request_timeout_ms: u64,
    pool_idle_timeout_ms: u64,
    pub(crate) pool_max_idle_per_host: usize,
}

impl Default for HttpMediaConfig {
    fn default() -> Self {
        Self {
            routes: Vec::new(),
            trust_domain: String::from("local"),
            buffered_request_max_bytes: DEFAULT_BUFFERED_REQUEST_MAX_BYTES,
            buffered_request_total_bytes: DEFAULT_BUFFERED_REQUEST_TOTAL_BYTES,
            streamed_request_max_bytes: DEFAULT_STREAMED_REQUEST_MAX_BYTES,
            connect_timeout_ms: DEFAULT_CONNECT_TIMEOUT_MS,
            request_timeout_ms: DEFAULT_REQUEST_TIMEOUT_MS,
            pool_idle_timeout_ms: DEFAULT_POOL_IDLE_TIMEOUT_MS,
            pool_max_idle_per_host: DEFAULT_POOL_MAX_IDLE_PER_HOST,
        }
    }
}

/// Media HTTP routes enabled by the shared transport policy.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum HttpMediaRoute {
    Speech,
    SpeechBatch,
    Transcription,
}

impl HttpMediaRoute {
    pub(crate) const fn service_class(self) -> ServiceClass {
        match self {
            Self::Speech => ServiceClass::SpeechHttp,
            Self::SpeechBatch => ServiceClass::SpeechBatch,
            Self::Transcription => ServiceClass::TranscriptionHttp,
        }
    }
}

impl HttpMediaConfig {
    pub(crate) const fn connect_timeout(&self) -> Duration {
        Duration::from_millis(self.connect_timeout_ms)
    }

    pub(crate) const fn request_timeout(&self) -> Duration {
        Duration::from_millis(self.request_timeout_ms)
    }

    pub(crate) const fn pool_idle_timeout(&self) -> Duration {
        Duration::from_millis(self.pool_idle_timeout_ms)
    }

    pub(crate) fn buffered_total_usize(&self) -> Result<usize, ConfigError> {
        usize::try_from(self.buffered_request_total_bytes).map_err(|_| {
            ConfigError::invalid(
                "http_media.buffered_request_total_bytes",
                "cannot be represented on this platform",
            )
        })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RouterConfig {
    #[serde(default)]
    pub(crate) strategy: RoutingStrategy,
    #[serde(default = "default_max_concurrent_classifications")]
    max_concurrent_classifications: u8,
    pub(crate) required_services: Vec<ServiceClass>,
    pub(crate) voice_owner_worker_id: Option<String>,
}

impl RouterConfig {
    pub(crate) const fn max_concurrent_classifications(&self) -> usize {
        self.max_concurrent_classifications as usize
    }
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
        self.validate_http_generation()?;
        self.validate_http_media()?;
        validate_workers(
            &self.workers,
            &self.router.required_services,
            self.router.voice_owner_worker_id.as_deref(),
        )?;
        Ok(())
    }

    fn validate_http_generation(&self) -> Result<(), ConfigError> {
        let Some(generation) = self.http_generation.as_ref() else {
            return Ok(());
        };
        if generation.trust_domain.is_empty() || generation.trust_domain.len() > 128 {
            return Err(ConfigError::invalid(
                "http_generation.trust_domain",
                "must contain between 1 and 128 bytes",
            ));
        }
        if !self.server.listen.ip().is_loopback() {
            return Err(ConfigError::invalid(
                "server.listen",
                "chat generation requires a loopback listener",
            ));
        }
        if !self
            .router
            .required_services
            .contains(&ServiceClass::GenerationHttp)
        {
            return Err(ConfigError::invalid(
                "router.required_services",
                "must contain generation_http while chat generation is enabled",
            ));
        }
        if !(1..=67_108_864).contains(&generation.buffered_request_max_bytes) {
            return Err(ConfigError::invalid(
                "http_generation.buffered_request_max_bytes",
                "must be between 1 and 67108864",
            ));
        }
        if generation.buffered_request_total_bytes < generation.buffered_request_max_bytes
            || generation.buffered_request_total_bytes > 2_147_483_647
        {
            return Err(ConfigError::invalid(
                "http_generation.buffered_request_total_bytes",
                "must be at least the per-request limit and at most 2147483647",
            ));
        }
        let _buffered_max = generation.buffered_max_usize()?;
        let buffered_total = generation.buffered_total_usize()?;
        if buffered_total > tokio::sync::Semaphore::MAX_PERMITS {
            return Err(ConfigError::invalid(
                "http_generation.buffered_request_total_bytes",
                "exceeds the platform semaphore permit limit",
            ));
        }
        if generation.streamed_request_max_bytes < generation.buffered_request_max_bytes
            || generation.streamed_request_max_bytes > 4_294_967_296
        {
            return Err(ConfigError::invalid(
                "http_generation.streamed_request_max_bytes",
                "must be at least the buffered limit and at most 4294967296",
            ));
        }
        if !(1..=60_000).contains(&generation.connect_timeout_ms) {
            return Err(ConfigError::invalid(
                "http_generation.connect_timeout_ms",
                "must be between 1 and 60000",
            ));
        }
        if generation.request_timeout_ms < generation.connect_timeout_ms
            || generation.request_timeout_ms > 3_600_000
        {
            return Err(ConfigError::invalid(
                "http_generation.request_timeout_ms",
                "must be at least connect_timeout_ms and at most 3600000",
            ));
        }
        if !(1_000..=300_000).contains(&generation.pool_idle_timeout_ms) {
            return Err(ConfigError::invalid(
                "http_generation.pool_idle_timeout_ms",
                "must be between 1000 and 300000",
            ));
        }
        if !(1..=256).contains(&generation.pool_max_idle_per_host) {
            return Err(ConfigError::invalid(
                "http_generation.pool_max_idle_per_host",
                "must be between 1 and 256",
            ));
        }
        if !self.workers.iter().any(|worker| {
            worker.trust_domain == generation.trust_domain
                && worker.service_profiles.iter().any(|profile| {
                    matches!(
                        profile,
                        crate::worker_pool::profile::ServiceProfile::GenerationHttp { .. }
                    )
                })
        }) {
            return Err(ConfigError::invalid(
                "http_generation.trust_domain",
                "must contain at least one generation worker",
            ));
        }
        Ok(())
    }

    fn validate_http_media(&self) -> Result<(), ConfigError> {
        let Some(media) = self.http_media.as_ref() else {
            return Ok(());
        };
        if media.routes.is_empty() {
            return Err(ConfigError::invalid(
                "http_media.routes",
                "must contain at least one route",
            ));
        }
        if media
            .routes
            .iter()
            .enumerate()
            .any(|(index, route)| media.routes[..index].contains(route))
        {
            return Err(ConfigError::invalid(
                "http_media.routes",
                "must not contain duplicates",
            ));
        }
        if media.trust_domain.is_empty() || media.trust_domain.len() > 128 {
            return Err(ConfigError::invalid(
                "http_media.trust_domain",
                "must contain between 1 and 128 bytes",
            ));
        }
        if !self.server.listen.ip().is_loopback() {
            return Err(ConfigError::invalid(
                "server.listen",
                "media HTTP requires a loopback listener",
            ));
        }
        if media
            .routes
            .iter()
            .map(|route| route.service_class())
            .any(|service| !self.router.required_services.contains(&service))
        {
            return Err(ConfigError::invalid(
                "router.required_services",
                "must contain every service enabled by http_media.routes",
            ));
        }
        if !(1..=67_108_864).contains(&media.buffered_request_max_bytes) {
            return Err(ConfigError::invalid(
                "http_media.buffered_request_max_bytes",
                "must be between 1 and 67108864",
            ));
        }
        if media.buffered_request_total_bytes < media.buffered_request_max_bytes
            || media.buffered_request_total_bytes > 2_147_483_647
        {
            return Err(ConfigError::invalid(
                "http_media.buffered_request_total_bytes",
                "must be at least the per-request limit and at most 2147483647",
            ));
        }
        let buffered_total = media.buffered_total_usize()?;
        if buffered_total > tokio::sync::Semaphore::MAX_PERMITS {
            return Err(ConfigError::invalid(
                "http_media.buffered_request_total_bytes",
                "exceeds the platform semaphore permit limit",
            ));
        }
        if media.streamed_request_max_bytes < media.buffered_request_max_bytes
            || media.streamed_request_max_bytes > 4_294_967_296
        {
            return Err(ConfigError::invalid(
                "http_media.streamed_request_max_bytes",
                "must be at least the buffered limit and at most 4294967296",
            ));
        }
        if !(1..=60_000).contains(&media.connect_timeout_ms) {
            return Err(ConfigError::invalid(
                "http_media.connect_timeout_ms",
                "must be between 1 and 60000",
            ));
        }
        if media.request_timeout_ms < media.connect_timeout_ms
            || media.request_timeout_ms > 3_600_000
        {
            return Err(ConfigError::invalid(
                "http_media.request_timeout_ms",
                "must be at least connect_timeout_ms and at most 3600000",
            ));
        }
        if !(1_000..=300_000).contains(&media.pool_idle_timeout_ms) {
            return Err(ConfigError::invalid(
                "http_media.pool_idle_timeout_ms",
                "must be between 1000 and 300000",
            ));
        }
        if !(1..=256).contains(&media.pool_max_idle_per_host) {
            return Err(ConfigError::invalid(
                "http_media.pool_max_idle_per_host",
                "must be between 1 and 256",
            ));
        }
        for service in media.routes.iter().map(|route| route.service_class()) {
            if !self.workers.iter().any(|worker| {
                worker.trust_domain == media.trust_domain
                    && worker
                        .service_profiles
                        .iter()
                        .any(|profile| profile.service_class() == service)
            }) {
                return Err(ConfigError::invalid(
                    "http_media.trust_domain",
                    "must contain at least one worker for every enabled media HTTP route",
                ));
            }
        }
        Ok(())
    }

    fn validate_router(&self) -> Result<(), ConfigError> {
        if self.router.required_services.is_empty() {
            return Err(ConfigError::invalid(
                "router.required_services",
                "must be nonempty",
            ));
        }
        if !(1..=64).contains(&self.router.max_concurrent_classifications) {
            return Err(ConfigError::invalid(
                "router.max_concurrent_classifications",
                "must be between 1 and 64",
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

const fn default_max_concurrent_classifications() -> u8 {
    DEFAULT_MAX_CONCURRENT_CLASSIFICATIONS
}

impl From<io::Error> for ConfigError {
    fn from(source: io::Error) -> Self {
        Self::Read(source)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::RouterConfig;

    #[test]
    fn router_classification_concurrency_defaults_to_four() {
        let router: RouterConfig = toml::from_str("required_services = [\"generation_http\"]")
            .expect("deserialize minimal router section");
        assert_eq!(router.max_concurrent_classifications(), 4);
    }
}
