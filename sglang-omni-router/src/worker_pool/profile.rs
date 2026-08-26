use std::collections::HashSet;
use std::net::IpAddr;

use serde::Deserialize;

use crate::error::ConfigError;

pub(super) const MAX_WORKERS: usize = 256;
const MAX_PROFILES_PER_WORKER: usize = 64;
const MAX_SET_ITEMS: usize = 64;
const MAX_ID_BYTES: usize = 128;
const MAX_MODEL_ID_BYTES: usize = 256;
const MAX_BASE_URL_BYTES: usize = 2_048;
const MAX_HEALTH_PATH_BYTES: usize = 128;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ServiceClass {
    GenerationHttp,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct WorkerId(String);

impl WorkerId {
    pub(super) fn new(value: String) -> Self {
        Self(value)
    }

    pub(super) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct RegistrationId(usize);

impl RegistrationId {
    pub(super) const fn from_startup_ordinal(ordinal: usize) -> Self {
        Self(ordinal)
    }

    pub(super) const fn startup_ordinal(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct TrustDomain(String);

impl TrustDomain {
    pub(crate) fn new(value: String) -> Self {
        Self(value)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerConfig {
    pub(crate) worker_id: String,
    pub(crate) base_url: String,
    pub(crate) resolved_ip: Option<IpAddr>,
    pub(crate) trust_domain: String,
    pub(crate) default_model_id: String,
    #[serde(default = "default_health_path")]
    pub(crate) health_path: String,
    pub(crate) capacity: WorkerCapacityConfig,
    pub(crate) service_profiles: Vec<ServiceProfile>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerCapacityConfig {
    pub(crate) generation_http: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MessageContentForm {
    String,
    TypedParts,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MediaPlacement {
    TopLevel,
    TypedParts,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ChatAudioFormat {
    Wav,
    Mp3,
    Flac,
    Pcm,
    Aac,
    Opus,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InputModality {
    Text,
    Image,
    Audio,
    Video,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OutputModality {
    Text,
    Audio,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StreamMode {
    NonStreaming,
    Streaming,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(tag = "service", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ServiceProfile {
    GenerationHttp {
        model_ids: Vec<String>,
        message_content_forms: Vec<MessageContentForm>,
        media_placements: Vec<MediaPlacement>,
        input_modalities: Vec<InputModality>,
        output_modalities: Vec<OutputModality>,
        chat_audio_formats: Vec<ChatAudioFormat>,
        stream_modes: Vec<StreamMode>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RouteRequirement {
    pub(super) profile: ProfileRequirement,
    trust_domain: TrustDomain,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProfileRequirement {
    GenerationHttp {
        model: ModelSelection,
        message_content_forms: Vec<MessageContentForm>,
        media_placements: Vec<MediaPlacement>,
        input_modalities: Vec<InputModality>,
        output_modalities: Vec<OutputModality>,
        audio_format: Option<ChatAudioFormat>,
        stream_mode: StreamMode,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ModelSelection {
    Explicit(String),
    WorkerDefault { expected_model_id: String },
}

impl ModelSelection {
    pub(crate) fn model_id(&self) -> &str {
        match self {
            Self::Explicit(model_id) => model_id,
            Self::WorkerDefault { expected_model_id } => expected_model_id,
        }
    }

    fn matches_worker_default(&self, worker_default: &str) -> bool {
        match self {
            Self::Explicit(_) => true,
            Self::WorkerDefault { expected_model_id } => expected_model_id == worker_default,
        }
    }
}

impl RouteRequirement {
    pub(crate) fn new(profile: ProfileRequirement, trust_domain: TrustDomain) -> Self {
        Self {
            profile,
            trust_domain,
        }
    }

    pub(super) fn trust_domain(&self) -> &TrustDomain {
        &self.trust_domain
    }
}

impl ProfileRequirement {
    pub(super) fn is_well_formed(&self) -> bool {
        match self {
            Self::GenerationHttp {
                model,
                message_content_forms,
                media_placements,
                input_modalities,
                output_modalities,
                audio_format,
                ..
            } => {
                valid_model_id(model.model_id())
                    && valid_requirement_set(message_content_forms, false)
                    && valid_requirement_set(media_placements, true)
                    && valid_requirement_set(input_modalities, false)
                    && valid_requirement_set(output_modalities, false)
                    && output_modalities.contains(&OutputModality::Audio) == audio_format.is_some()
            }
        }
    }
}

impl ServiceProfile {
    pub(super) fn validate(&self) -> Result<(), ConfigError> {
        match self {
            Self::GenerationHttp {
                model_ids,
                message_content_forms,
                media_placements,
                input_modalities,
                output_modalities,
                chat_audio_formats,
                stream_modes,
            } => {
                validate_models(model_ids)?;
                validate_set(
                    message_content_forms,
                    "workers.service_profiles.message_content_forms",
                    false,
                )?;
                validate_set(
                    media_placements,
                    "workers.service_profiles.media_placements",
                    true,
                )?;
                validate_set(
                    input_modalities,
                    "workers.service_profiles.input_modalities",
                    false,
                )?;
                validate_set(
                    output_modalities,
                    "workers.service_profiles.output_modalities",
                    false,
                )?;
                validate_set(
                    chat_audio_formats,
                    "workers.service_profiles.chat_audio_formats",
                    true,
                )?;
                validate_set(stream_modes, "workers.service_profiles.stream_modes", false)?;
                if output_modalities.contains(&OutputModality::Audio)
                    != !chat_audio_formats.is_empty()
                {
                    return Err(ConfigError::invalid(
                        "workers.service_profiles.chat_audio_formats",
                        "must be nonempty exactly when audio output is supported",
                    ));
                }
                Ok(())
            }
        }
    }

    pub(super) fn semantically_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::GenerationHttp {
                    model_ids: a_models,
                    message_content_forms: a_forms,
                    media_placements: a_placements,
                    input_modalities: a_inputs,
                    output_modalities: a_outputs,
                    chat_audio_formats: a_audio,
                    stream_modes: a_streams,
                },
                Self::GenerationHttp {
                    model_ids: b_models,
                    message_content_forms: b_forms,
                    media_placements: b_placements,
                    input_modalities: b_inputs,
                    output_modalities: b_outputs,
                    chat_audio_formats: b_audio,
                    stream_modes: b_streams,
                },
            ) => {
                set_eq(a_models, b_models)
                    && set_eq(a_forms, b_forms)
                    && set_eq(a_placements, b_placements)
                    && set_eq(a_inputs, b_inputs)
                    && set_eq(a_outputs, b_outputs)
                    && set_eq(a_audio, b_audio)
                    && set_eq(a_streams, b_streams)
            }
        }
    }

    pub(super) fn matches(&self, requirement: &ProfileRequirement, worker_default: &str) -> bool {
        match (self, requirement) {
            (
                Self::GenerationHttp {
                    model_ids,
                    message_content_forms,
                    media_placements,
                    input_modalities,
                    output_modalities,
                    chat_audio_formats,
                    stream_modes,
                },
                ProfileRequirement::GenerationHttp {
                    model,
                    message_content_forms: required_forms,
                    media_placements: required_placements,
                    input_modalities: required_inputs,
                    output_modalities: required_outputs,
                    audio_format,
                    stream_mode,
                },
            ) => {
                model.matches_worker_default(worker_default)
                    && model_ids
                        .iter()
                        .any(|candidate| candidate == model.model_id())
                    && contains_all(message_content_forms, required_forms)
                    && contains_all(media_placements, required_placements)
                    && contains_all(input_modalities, required_inputs)
                    && contains_all(output_modalities, required_outputs)
                    && audio_format.is_none_or(|format| chat_audio_formats.contains(&format))
                    && stream_modes.contains(stream_mode)
            }
        }
    }

    fn contains_model(&self, model: &str) -> bool {
        match self {
            Self::GenerationHttp { model_ids, .. } => model_ids.iter().any(|item| item == model),
        }
    }
}

pub(crate) fn validate_workers(workers: &[WorkerConfig]) -> Result<(), ConfigError> {
    if workers.is_empty() || workers.len() > MAX_WORKERS {
        return Err(ConfigError::invalid(
            "workers",
            "must contain between 1 and 256 workers",
        ));
    }
    let mut ids = HashSet::with_capacity(workers.len());
    let mut targets = HashSet::with_capacity(workers.len());
    let mut resolved_targets = Vec::with_capacity(workers.len());
    for worker in workers {
        validate_identifier(&worker.worker_id, "workers.worker_id")?;
        if !ids.insert(worker.worker_id.as_str()) {
            return Err(ConfigError::invalid("workers.worker_id", "must be unique"));
        }
        validate_identifier(&worker.trust_domain, "workers.trust_domain")?;
        if !valid_model_id(&worker.default_model_id) {
            return Err(ConfigError::invalid(
                "workers.default_model_id",
                "must be 1 to 256 bytes",
            ));
        }
        if worker.base_url.is_empty() || worker.base_url.len() > MAX_BASE_URL_BYTES {
            return Err(ConfigError::invalid(
                "workers.base_url",
                "must contain between 1 and 2048 bytes",
            ));
        }
        if worker.health_path.is_empty()
            || worker.health_path.len() > MAX_HEALTH_PATH_BYTES
            || !worker.health_path.starts_with('/')
            || worker.health_path.contains('?')
            || worker.health_path.contains('#')
        {
            return Err(ConfigError::invalid(
                "workers.health_path",
                "must be an absolute path without query or fragment",
            ));
        }
        let target = super::resolver::ResolvedTarget::from_worker(worker).ok_or_else(|| {
            ConfigError::invalid(
                "workers.base_url",
                "must be a canonical statically resolved HTTP or HTTPS target",
            )
        })?;
        let target_key = (target.base_url().as_str().to_owned(), target.socket_addr());
        if !targets.insert(target_key) {
            return Err(ConfigError::invalid(
                "workers.base_url",
                "resolved targets must be unique",
            ));
        }
        resolved_targets.push(target);
        if worker.capacity.generation_http == 0 || worker.capacity.generation_http > 65_535 {
            return Err(ConfigError::invalid(
                "workers.capacity.generation_http",
                "must be between 1 and 65535",
            ));
        }
        if worker.service_profiles.is_empty()
            || worker.service_profiles.len() > MAX_PROFILES_PER_WORKER
        {
            return Err(ConfigError::invalid(
                "workers.service_profiles",
                "must contain between 1 and 64 rows",
            ));
        }
        for (index, profile) in worker.service_profiles.iter().enumerate() {
            profile.validate()?;
            if worker
                .service_profiles
                .iter()
                .take(index)
                .any(|earlier| profile.semantically_eq(earlier))
            {
                return Err(ConfigError::invalid(
                    "workers.service_profiles",
                    "contains a duplicate correlated row",
                ));
            }
        }
        if !worker
            .service_profiles
            .iter()
            .any(|profile| profile.contains_model(&worker.default_model_id))
        {
            return Err(ConfigError::invalid(
                "workers.default_model_id",
                "must belong to a generation profile row",
            ));
        }
    }
    if super::resolver::StaticResolver::from_targets(&resolved_targets).is_none() {
        return Err(ConfigError::invalid(
            "workers.resolved_ip",
            "hostname pins must be consistent across workers",
        ));
    }
    Ok(())
}

fn validate_models(values: &[String]) -> Result<(), ConfigError> {
    if values.is_empty()
        || values.len() > MAX_SET_ITEMS
        || values.iter().any(|value| !valid_model_id(value))
    {
        return Err(ConfigError::invalid(
            "workers.service_profiles.model_ids",
            "must contain 1 to 64 unique model IDs",
        ));
    }
    let unique: HashSet<_> = values.iter().collect();
    if unique.len() != values.len() {
        return Err(ConfigError::invalid(
            "workers.service_profiles.model_ids",
            "must not contain duplicates",
        ));
    }
    Ok(())
}

fn validate_set<T: Eq + std::hash::Hash>(
    values: &[T],
    field: &'static str,
    allow_empty: bool,
) -> Result<(), ConfigError> {
    if values.len() > MAX_SET_ITEMS || (!allow_empty && values.is_empty()) {
        return Err(ConfigError::invalid(field, "has an invalid item count"));
    }
    let unique: HashSet<_> = values.iter().collect();
    if unique.len() != values.len() {
        return Err(ConfigError::invalid(field, "must not contain duplicates"));
    }
    Ok(())
}

fn valid_requirement_set<T: Eq + std::hash::Hash>(values: &[T], allow_empty: bool) -> bool {
    validate_set(values, "internal", allow_empty).is_ok()
}

fn set_eq<T: Eq>(left: &[T], right: &[T]) -> bool {
    left.len() == right.len() && left.iter().all(|item| right.contains(item))
}

fn contains_all<T: Eq>(available: &[T], required: &[T]) -> bool {
    required.iter().all(|item| available.contains(item))
}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), ConfigError> {
    if value.is_empty()
        || value.len() > MAX_ID_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(ConfigError::invalid(
            field,
            "must be 1 to 128 ASCII identifier bytes",
        ));
    }
    Ok(())
}

fn valid_model_id(value: &str) -> bool {
    !value.is_empty() && value.len() <= MAX_MODEL_ID_BYTES && !value.chars().any(char::is_control)
}

fn default_health_path() -> String {
    String::from("/health")
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn profile(model: &str) -> ServiceProfile {
        ServiceProfile::GenerationHttp {
            model_ids: vec![model.to_owned()],
            message_content_forms: vec![MessageContentForm::String],
            media_placements: Vec::new(),
            input_modalities: vec![InputModality::Text],
            output_modalities: vec![OutputModality::Text],
            chat_audio_formats: Vec::new(),
            stream_modes: vec![StreamMode::NonStreaming],
        }
    }

    fn worker() -> WorkerConfig {
        WorkerConfig {
            worker_id: String::from("worker-a"),
            base_url: String::from("http://127.0.0.1:8000/"),
            resolved_ip: None,
            trust_domain: String::from("local"),
            default_model_id: String::from("omni"),
            health_path: String::from("/health"),
            capacity: WorkerCapacityConfig { generation_http: 2 },
            service_profiles: vec![profile("omni")],
        }
    }

    #[test]
    fn stable_worker_shape_and_correlated_generation_row_validate() {
        assert!(validate_workers(&[worker()]).is_ok());
        let mut maximum = Vec::new();
        for index in 0..MAX_WORKERS {
            let mut item = worker();
            item.worker_id = format!("worker-{index}");
            item.base_url = format!("http://127.0.0.1:{}/", 10_000 + index);
            maximum.push(item);
        }
        assert!(validate_workers(&maximum).is_ok());
    }

    #[test]
    fn strict_profile_and_default_correlation_fail_closed() {
        let mut missing_default = worker();
        missing_default.default_model_id = String::from("other");
        assert!(validate_workers(&[missing_default]).is_err());

        let mut duplicate = worker();
        duplicate.service_profiles.push(profile("omni"));
        assert!(validate_workers(&[duplicate]).is_err());

        let mut invalid_audio = worker();
        invalid_audio.service_profiles = vec![ServiceProfile::GenerationHttp {
            model_ids: vec![String::from("omni")],
            message_content_forms: vec![MessageContentForm::TypedParts],
            media_placements: vec![MediaPlacement::TypedParts],
            input_modalities: vec![InputModality::Audio],
            output_modalities: vec![OutputModality::Audio],
            chat_audio_formats: Vec::new(),
            stream_modes: vec![StreamMode::Streaming],
        }];
        assert!(validate_workers(&[invalid_audio]).is_err());
    }

    #[test]
    fn matching_never_combines_correlated_rows() {
        let text = profile("omni");
        let audio = ServiceProfile::GenerationHttp {
            model_ids: vec![String::from("audio")],
            message_content_forms: vec![MessageContentForm::TypedParts],
            media_placements: vec![MediaPlacement::TypedParts],
            input_modalities: vec![InputModality::Audio],
            output_modalities: vec![OutputModality::Audio],
            chat_audio_formats: vec![ChatAudioFormat::Wav],
            stream_modes: vec![StreamMode::Streaming],
        };
        let cross_row = ProfileRequirement::GenerationHttp {
            model: ModelSelection::Explicit(String::from("omni")),
            message_content_forms: vec![MessageContentForm::TypedParts],
            media_placements: vec![MediaPlacement::TypedParts],
            input_modalities: vec![InputModality::Audio],
            output_modalities: vec![OutputModality::Audio],
            audio_format: Some(ChatAudioFormat::Wav),
            stream_mode: StreamMode::Streaming,
        };
        assert!(!text.matches(&cross_row, "omni"));
        assert!(!audio.matches(&cross_row, "audio"));
    }
}
