use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::net::IpAddr;

use reqwest::Url;
use serde::Deserialize;

use crate::error::ConfigError;

pub(super) const MAX_WORKERS: usize = 256;
const MAX_PROFILES_PER_WORKER: usize = 64;
const MAX_SET_ITEMS: usize = 64;
const MAX_ID_BYTES: usize = 128;
const MAX_MODEL_ID_BYTES: usize = 256;
const MAX_BASE_URL_BYTES: usize = 2_048;
const MAX_HEALTH_PATH_BYTES: usize = 128;

/// A configured service class used by readiness and capacity ownership.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ServiceClass {
    GenerationHttp,
    SpeechHttp,
    TranscriptionHttp,
    SpeechBatch,
    SpeechWebsocket,
    RealtimeWebsocket,
    VoiceControl,
}

/// The six data-plane capacity classes plus isolated control traffic.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CapacityClass {
    GenerationHttp,
    SpeechHttp,
    TranscriptionHttp,
    SpeechBatch,
    SpeechWebsocket,
    RealtimeWebsocket,
    Control,
}

impl CapacityClass {
    pub(crate) const ALL: [Self; 7] = [
        Self::GenerationHttp,
        Self::SpeechHttp,
        Self::TranscriptionHttp,
        Self::SpeechBatch,
        Self::SpeechWebsocket,
        Self::RealtimeWebsocket,
        Self::Control,
    ];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::GenerationHttp => "generation_http",
            Self::SpeechHttp => "speech_http",
            Self::TranscriptionHttp => "transcription_http",
            Self::SpeechBatch => "speech_batch",
            Self::SpeechWebsocket => "speech_websocket",
            Self::RealtimeWebsocket => "realtime_websocket",
            Self::Control => "control",
        }
    }
}

impl ServiceClass {
    pub(super) fn capacity(self) -> CapacityClass {
        match self {
            Self::GenerationHttp => CapacityClass::GenerationHttp,
            Self::SpeechHttp => CapacityClass::SpeechHttp,
            Self::TranscriptionHttp => CapacityClass::TranscriptionHttp,
            Self::SpeechBatch => CapacityClass::SpeechBatch,
            Self::SpeechWebsocket => CapacityClass::SpeechWebsocket,
            Self::RealtimeWebsocket => CapacityClass::RealtimeWebsocket,
            Self::VoiceControl => CapacityClass::Control,
        }
    }
}

/// Operator-assigned worker identity, validated before construction.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct WorkerId(String);

impl WorkerId {
    pub(super) fn new(value: String) -> Self {
        Self(value)
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

/// Router-local identity for one immutable startup registration.
///
/// This canonical manifest ordinal is meaningful only inside this router
/// process. It is not a worker boot ID, incarnation, or continuity token.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct RegistrationId(usize);

impl RegistrationId {
    pub(super) const fn from_startup_ordinal(ordinal: usize) -> Self {
        Self(ordinal)
    }

    pub(crate) const fn startup_ordinal(self) -> usize {
        self.0
    }
}

/// Operator-assigned trust partition, validated before construction.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct TrustDomain(String);

impl TrustDomain {
    pub(crate) fn new(value: String) -> Self {
        Self(value)
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerConfig {
    pub(crate) worker_id: String,
    pub(crate) base_url: String,
    pub(crate) resolved_ip: Option<IpAddr>,
    pub(crate) trust_domain: String,
    pub(crate) default_model_id: Option<String>,
    #[serde(default = "default_health_path")]
    pub(crate) health_path: String,
    pub(crate) capacity: WorkerCapacityConfig,
    pub(crate) service_profiles: Vec<ServiceProfile>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerCapacityConfig {
    pub(crate) generation_http: Option<u32>,
    pub(crate) speech_http: Option<u32>,
    pub(crate) transcription_http: Option<u32>,
    pub(crate) speech_batch: Option<u32>,
    pub(crate) speech_websocket: Option<u32>,
    pub(crate) realtime_websocket: Option<u32>,
    pub(crate) control: Option<u32>,
}

impl WorkerCapacityConfig {
    pub(super) fn get(&self, class: CapacityClass) -> Option<u32> {
        match class {
            CapacityClass::GenerationHttp => self.generation_http,
            CapacityClass::SpeechHttp => self.speech_http,
            CapacityClass::TranscriptionHttp => self.transcription_http,
            CapacityClass::SpeechBatch => self.speech_batch,
            CapacityClass::SpeechWebsocket => self.speech_websocket,
            CapacityClass::RealtimeWebsocket => self.realtime_websocket,
            CapacityClass::Control => self.control,
        }
    }

    fn entries(&self) -> [(CapacityClass, Option<u32>); 7] {
        [
            (CapacityClass::GenerationHttp, self.generation_http),
            (CapacityClass::SpeechHttp, self.speech_http),
            (CapacityClass::TranscriptionHttp, self.transcription_http),
            (CapacityClass::SpeechBatch, self.speech_batch),
            (CapacityClass::SpeechWebsocket, self.speech_websocket),
            (CapacityClass::RealtimeWebsocket, self.realtime_websocket),
            (CapacityClass::Control, self.control),
        ]
    }
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

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SpeechResponseFormat {
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    Pcm,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SpeechTask {
    TextToSpeech,
    VoiceClone,
    VoiceDesign,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReferenceForm {
    None,
    Direct,
    List,
    VqCodes,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BatchFeature {
    Model,
    Format,
    Task,
    Reference,
    Voice,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TranscriptionResponseFormat {
    Json,
    Text,
    VerboseJson,
    Sse,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MediaProfile {
    Audio,
    AudioVideo,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SpeechWebsocketInput {
    Text,
    TextAndControl,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RealtimeProtocol {
    OpenaiRealtimeV1,
}

/// One correlated capability row. Sets within a row never combine with another row.
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
    SpeechHttp {
        model_ids: Vec<String>,
        response_formats: Vec<SpeechResponseFormat>,
        stream_modes: Vec<StreamMode>,
        tasks: Vec<SpeechTask>,
        reference_forms: Vec<ReferenceForm>,
        managed_voice: bool,
    },
    SpeechBatch {
        model_ids: Vec<String>,
        response_formats: Vec<SpeechResponseFormat>,
        tasks: Vec<SpeechTask>,
        reference_forms: Vec<ReferenceForm>,
        managed_voice: bool,
        max_batch_size: u16,
        effective_features: Vec<BatchFeature>,
    },
    TranscriptionHttp {
        model_ids: Vec<String>,
        response_formats: Vec<TranscriptionResponseFormat>,
        media_profiles: Vec<MediaProfile>,
        stream_modes: Vec<StreamMode>,
    },
    SpeechWebsocket {
        model_ids: Vec<String>,
        input_profiles: Vec<SpeechWebsocketInput>,
        response_formats: Vec<SpeechResponseFormat>,
        stream_modes: Vec<StreamMode>,
        tasks: Vec<SpeechTask>,
        reference_forms: Vec<ReferenceForm>,
        managed_voice: bool,
    },
    RealtimeWebsocket {
        protocols: Vec<RealtimeProtocol>,
    },
    VoiceControl,
}

/// Bounded compatibility facts for one future data-plane request or session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RouteRequirement {
    pub(super) profile: ProfileRequirement,
    trust_domain: TrustDomain,
    require_voice_owner: bool,
}

/// One concrete point in a correlated service-profile row.
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
    SpeechHttp {
        model: ModelSelection,
        response_format: SpeechResponseFormat,
        stream_mode: StreamMode,
        task: SpeechTask,
        reference_forms: Vec<ReferenceForm>,
        managed_voice: bool,
    },
    SpeechBatch {
        models: Vec<ModelSelection>,
        response_formats: Vec<SpeechResponseFormat>,
        tasks: Vec<SpeechTask>,
        reference_forms: Vec<ReferenceForm>,
        managed_voice: bool,
        batch_size: u16,
        effective_features: Vec<BatchFeature>,
    },
    TranscriptionHttp {
        model: ModelSelection,
        response_format: TranscriptionResponseFormat,
        media_profile: MediaProfile,
        stream_mode: StreamMode,
    },
    SpeechWebsocket {
        model: ModelSelection,
        input_profile: SpeechWebsocketInput,
        response_format: SpeechResponseFormat,
        stream_mode: StreamMode,
        task: SpeechTask,
        reference_forms: Vec<ReferenceForm>,
        managed_voice: bool,
    },
    RealtimeWebsocket {
        protocol: RealtimeProtocol,
        expected_default_model_id: String,
    },
    VoiceControl,
}

/// Effective model selection with provenance preserved for worker defaults.
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

    fn matches_worker_default(&self, worker_default: Option<&str>) -> bool {
        match self {
            Self::Explicit(_) => true,
            Self::WorkerDefault { expected_model_id } => {
                worker_default == Some(expected_model_id.as_str())
            }
        }
    }

    fn is_valid(&self) -> bool {
        valid_model_id(self.model_id())
    }
}

impl RouteRequirement {
    pub(crate) fn new(
        profile: ProfileRequirement,
        trust_domain: TrustDomain,
        require_voice_owner: bool,
    ) -> Self {
        Self {
            profile,
            trust_domain,
            require_voice_owner,
        }
    }

    pub(crate) fn service_class(&self) -> ServiceClass {
        self.profile.service_class()
    }

    pub(crate) fn profile(&self) -> &ProfileRequirement {
        &self.profile
    }

    pub(super) fn capacity_class(&self) -> CapacityClass {
        self.service_class().capacity()
    }

    pub(super) fn trust_domain(&self) -> &TrustDomain {
        &self.trust_domain
    }

    pub(super) fn requires_voice_owner(&self) -> bool {
        self.require_voice_owner || self.profile.requires_voice_owner()
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
                model.is_valid()
                    && valid_bounded_set(message_content_forms, false)
                    && valid_bounded_set(media_placements, true)
                    && valid_bounded_set(input_modalities, true)
                    && valid_bounded_set(output_modalities, false)
                    && validate_generation_relationships(
                        message_content_forms,
                        media_placements,
                        input_modalities,
                        output_modalities,
                        audio_format.is_some(),
                    )
                    .is_ok()
            }
            Self::SpeechHttp {
                model,
                reference_forms,
                ..
            }
            | Self::SpeechWebsocket {
                model,
                reference_forms,
                ..
            } => model.is_valid() && valid_speech_reference_requirement(reference_forms),
            Self::TranscriptionHttp { model, .. } => model.is_valid(),
            Self::SpeechBatch {
                models,
                response_formats,
                tasks,
                reference_forms,
                batch_size,
                effective_features,
                ..
            } => {
                *batch_size > 0
                    && models.len() == usize::from(*batch_size)
                    && models.iter().all(ModelSelection::is_valid)
                    && valid_bounded_set(response_formats, false)
                    && valid_bounded_set(tasks, false)
                    && valid_bounded_set(reference_forms, false)
                    && valid_bounded_set(effective_features, true)
            }
            Self::RealtimeWebsocket {
                expected_default_model_id,
                ..
            } => valid_model_id(expected_default_model_id),
            Self::VoiceControl => true,
        }
    }

    fn service_class(&self) -> ServiceClass {
        match self {
            Self::GenerationHttp { .. } => ServiceClass::GenerationHttp,
            Self::SpeechHttp { .. } => ServiceClass::SpeechHttp,
            Self::SpeechBatch { .. } => ServiceClass::SpeechBatch,
            Self::TranscriptionHttp { .. } => ServiceClass::TranscriptionHttp,
            Self::SpeechWebsocket { .. } => ServiceClass::SpeechWebsocket,
            Self::RealtimeWebsocket { .. } => ServiceClass::RealtimeWebsocket,
            Self::VoiceControl => ServiceClass::VoiceControl,
        }
    }

    fn requires_voice_owner(&self) -> bool {
        match self {
            Self::SpeechHttp { managed_voice, .. }
            | Self::SpeechBatch { managed_voice, .. }
            | Self::SpeechWebsocket { managed_voice, .. } => *managed_voice,
            Self::VoiceControl => true,
            Self::GenerationHttp { .. }
            | Self::TranscriptionHttp { .. }
            | Self::RealtimeWebsocket { .. } => false,
        }
    }
}

impl ServiceProfile {
    pub(crate) fn service_class(&self) -> ServiceClass {
        match self {
            Self::GenerationHttp { .. } => ServiceClass::GenerationHttp,
            Self::SpeechHttp { .. } => ServiceClass::SpeechHttp,
            Self::SpeechBatch { .. } => ServiceClass::SpeechBatch,
            Self::TranscriptionHttp { .. } => ServiceClass::TranscriptionHttp,
            Self::SpeechWebsocket { .. } => ServiceClass::SpeechWebsocket,
            Self::RealtimeWebsocket { .. } => ServiceClass::RealtimeWebsocket,
            Self::VoiceControl => ServiceClass::VoiceControl,
        }
    }

    pub(super) fn requires_owner(&self) -> bool {
        match self {
            Self::SpeechHttp { managed_voice, .. }
            | Self::SpeechBatch { managed_voice, .. }
            | Self::SpeechWebsocket { managed_voice, .. } => *managed_voice,
            Self::VoiceControl => true,
            Self::GenerationHttp { .. }
            | Self::TranscriptionHttp { .. }
            | Self::RealtimeWebsocket { .. } => false,
        }
    }

    pub(crate) fn model_ids(&self) -> Option<&[String]> {
        match self {
            Self::GenerationHttp { model_ids, .. }
            | Self::SpeechHttp { model_ids, .. }
            | Self::SpeechBatch { model_ids, .. }
            | Self::TranscriptionHttp { model_ids, .. }
            | Self::SpeechWebsocket { model_ids, .. } => Some(model_ids),
            Self::RealtimeWebsocket { .. } | Self::VoiceControl => None,
        }
    }

    fn executes_model(&self) -> bool {
        !matches!(self, Self::VoiceControl)
    }

    fn validate(&self) -> Result<(), ConfigError> {
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
                validate_model_ids(model_ids)?;
                validate_set(
                    message_content_forms,
                    "workers.service_profiles.message_content_forms",
                )?;
                validate_optional_set(
                    media_placements,
                    "workers.service_profiles.media_placements",
                )?;
                validate_set(
                    input_modalities,
                    "workers.service_profiles.input_modalities",
                )?;
                validate_set(
                    output_modalities,
                    "workers.service_profiles.output_modalities",
                )?;
                validate_optional_set(
                    chat_audio_formats,
                    "workers.service_profiles.chat_audio_formats",
                )?;
                validate_set(stream_modes, "workers.service_profiles.stream_modes")?;
                validate_generation_relationships(
                    message_content_forms,
                    media_placements,
                    input_modalities,
                    output_modalities,
                    !chat_audio_formats.is_empty(),
                )
            }
            Self::SpeechHttp {
                model_ids,
                response_formats,
                stream_modes,
                tasks,
                reference_forms,
                ..
            } => {
                validate_model_ids(model_ids)?;
                validate_speech_formats_and_modes(response_formats, stream_modes)?;
                validate_set(tasks, "workers.service_profiles.tasks")?;
                validate_set(reference_forms, "workers.service_profiles.reference_forms")
            }
            Self::SpeechBatch {
                model_ids,
                response_formats,
                tasks,
                reference_forms,
                max_batch_size,
                effective_features,
                ..
            } => {
                validate_model_ids(model_ids)?;
                validate_set(
                    response_formats,
                    "workers.service_profiles.response_formats",
                )?;
                validate_set(tasks, "workers.service_profiles.tasks")?;
                validate_set(reference_forms, "workers.service_profiles.reference_forms")?;
                if *max_batch_size == 0 {
                    return invalid(
                        "workers.service_profiles.max_batch_size",
                        "must be positive",
                    );
                }
                validate_set(
                    effective_features,
                    "workers.service_profiles.effective_features",
                )
            }
            Self::TranscriptionHttp {
                model_ids,
                response_formats,
                media_profiles,
                stream_modes,
            } => {
                validate_model_ids(model_ids)?;
                validate_set(
                    response_formats,
                    "workers.service_profiles.response_formats",
                )?;
                validate_set(media_profiles, "workers.service_profiles.media_profiles")?;
                validate_set(stream_modes, "workers.service_profiles.stream_modes")
            }
            Self::SpeechWebsocket {
                model_ids,
                input_profiles,
                response_formats,
                stream_modes,
                tasks,
                reference_forms,
                ..
            } => {
                validate_model_ids(model_ids)?;
                validate_set(input_profiles, "workers.service_profiles.input_profiles")?;
                validate_speech_formats_and_modes(response_formats, stream_modes)?;
                validate_set(tasks, "workers.service_profiles.tasks")?;
                validate_set(reference_forms, "workers.service_profiles.reference_forms")
            }
            Self::RealtimeWebsocket { protocols } => {
                validate_set(protocols, "workers.service_profiles.protocols")
            }
            Self::VoiceControl => Ok(()),
        }
    }

    pub(super) fn semantically_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::GenerationHttp {
                    model_ids: am,
                    message_content_forms: ac,
                    media_placements: ap,
                    input_modalities: aim,
                    output_modalities: aom,
                    chat_audio_formats: af,
                    stream_modes: asm,
                },
                Self::GenerationHttp {
                    model_ids: bm,
                    message_content_forms: bc,
                    media_placements: bp,
                    input_modalities: bim,
                    output_modalities: bom,
                    chat_audio_formats: bf,
                    stream_modes: bsm,
                },
            ) => {
                same_set(am, bm)
                    && same_set(ac, bc)
                    && same_set(ap, bp)
                    && same_set(aim, bim)
                    && same_set(aom, bom)
                    && same_set(af, bf)
                    && same_set(asm, bsm)
            }
            (
                Self::SpeechHttp {
                    model_ids: am,
                    response_formats: af,
                    stream_modes: ast,
                    tasks: at,
                    reference_forms: ar,
                    managed_voice: av,
                },
                Self::SpeechHttp {
                    model_ids: bm,
                    response_formats: bf,
                    stream_modes: bst,
                    tasks: bt,
                    reference_forms: br,
                    managed_voice: bv,
                },
            ) => {
                av == bv
                    && same_set(am, bm)
                    && same_set(af, bf)
                    && same_set(ast, bst)
                    && same_set(at, bt)
                    && same_set(ar, br)
            }
            (
                Self::SpeechBatch {
                    model_ids: am,
                    response_formats: af,
                    tasks: at,
                    reference_forms: ar,
                    managed_voice: av,
                    max_batch_size: ab,
                    effective_features: ae,
                },
                Self::SpeechBatch {
                    model_ids: bm,
                    response_formats: bf,
                    tasks: bt,
                    reference_forms: br,
                    managed_voice: bv,
                    max_batch_size: bb,
                    effective_features: be,
                },
            ) => {
                av == bv
                    && ab == bb
                    && same_set(am, bm)
                    && same_set(af, bf)
                    && same_set(at, bt)
                    && same_set(ar, br)
                    && same_set(ae, be)
            }
            (
                Self::TranscriptionHttp {
                    model_ids: am,
                    response_formats: af,
                    media_profiles: ap,
                    stream_modes: ast,
                },
                Self::TranscriptionHttp {
                    model_ids: bm,
                    response_formats: bf,
                    media_profiles: bp,
                    stream_modes: bst,
                },
            ) => same_set(am, bm) && same_set(af, bf) && same_set(ap, bp) && same_set(ast, bst),
            (
                Self::SpeechWebsocket {
                    model_ids: am,
                    input_profiles: ai,
                    response_formats: af,
                    stream_modes: ast,
                    tasks: at,
                    reference_forms: ar,
                    managed_voice: av,
                },
                Self::SpeechWebsocket {
                    model_ids: bm,
                    input_profiles: bi,
                    response_formats: bf,
                    stream_modes: bst,
                    tasks: bt,
                    reference_forms: br,
                    managed_voice: bv,
                },
            ) => {
                av == bv
                    && same_set(am, bm)
                    && same_set(ai, bi)
                    && same_set(af, bf)
                    && same_set(ast, bst)
                    && same_set(at, bt)
                    && same_set(ar, br)
            }
            (
                Self::RealtimeWebsocket { protocols: a },
                Self::RealtimeWebsocket { protocols: b },
            ) => same_set(a, b),
            (Self::VoiceControl, Self::VoiceControl) => true,
            _ => false,
        }
    }

    pub(super) fn matches(
        &self,
        requirement: &ProfileRequirement,
        worker_default: Option<&str>,
    ) -> bool {
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
                    input_modalities: required_modalities,
                    output_modalities: required_outputs,
                    audio_format,
                    stream_mode,
                },
            ) => {
                model_matches(model_ids, model, worker_default)
                    && required_forms
                        .iter()
                        .all(|value| message_content_forms.contains(value))
                    && required_placements
                        .iter()
                        .all(|value| media_placements.contains(value))
                    && required_modalities
                        .iter()
                        .all(|value| input_modalities.contains(value))
                    && required_outputs
                        .iter()
                        .all(|value| output_modalities.contains(value))
                    && audio_format.is_none_or(|value| chat_audio_formats.contains(&value))
                    && stream_modes.contains(stream_mode)
            }
            (
                Self::SpeechHttp {
                    model_ids,
                    response_formats,
                    stream_modes,
                    tasks,
                    reference_forms,
                    managed_voice,
                },
                ProfileRequirement::SpeechHttp {
                    model,
                    response_format,
                    stream_mode,
                    task,
                    reference_forms: required_references,
                    managed_voice: required_voice,
                },
            ) => {
                model_matches(model_ids, model, worker_default)
                    && response_formats.contains(response_format)
                    && stream_modes.contains(stream_mode)
                    && tasks.contains(task)
                    && required_references
                        .iter()
                        .all(|value| reference_forms.contains(value))
                    && managed_voice == required_voice
            }
            (
                Self::SpeechBatch {
                    model_ids,
                    response_formats,
                    tasks,
                    reference_forms,
                    managed_voice,
                    max_batch_size,
                    effective_features,
                },
                ProfileRequirement::SpeechBatch {
                    models: required_models,
                    response_formats: required_formats,
                    tasks: required_tasks,
                    reference_forms: required_references,
                    managed_voice: required_voice,
                    batch_size,
                    effective_features: required_features,
                },
            ) => {
                batch_size <= max_batch_size
                    && managed_voice == required_voice
                    && required_models
                        .iter()
                        .all(|value| model_matches(model_ids, value, worker_default))
                    && required_formats
                        .iter()
                        .all(|value| response_formats.contains(value))
                    && required_tasks.iter().all(|value| tasks.contains(value))
                    && required_references
                        .iter()
                        .all(|value| reference_forms.contains(value))
                    && required_features
                        .iter()
                        .all(|value| effective_features.contains(value))
            }
            (
                Self::TranscriptionHttp {
                    model_ids,
                    response_formats,
                    media_profiles,
                    stream_modes,
                },
                ProfileRequirement::TranscriptionHttp {
                    model,
                    response_format,
                    media_profile,
                    stream_mode,
                },
            ) => {
                model_matches(model_ids, model, worker_default)
                    && response_formats.contains(response_format)
                    && media_profiles.contains(media_profile)
                    && stream_modes.contains(stream_mode)
            }
            (
                Self::SpeechWebsocket {
                    model_ids,
                    input_profiles,
                    response_formats,
                    stream_modes,
                    tasks,
                    reference_forms,
                    managed_voice,
                },
                ProfileRequirement::SpeechWebsocket {
                    model,
                    input_profile,
                    response_format,
                    stream_mode,
                    task,
                    reference_forms: required_references,
                    managed_voice: required_voice,
                },
            ) => {
                model_matches(model_ids, model, worker_default)
                    && input_profiles.contains(input_profile)
                    && response_formats.contains(response_format)
                    && stream_modes.contains(stream_mode)
                    && tasks.contains(task)
                    && required_references
                        .iter()
                        .all(|value| reference_forms.contains(value))
                    && managed_voice == required_voice
            }
            (
                Self::RealtimeWebsocket { protocols },
                ProfileRequirement::RealtimeWebsocket {
                    protocol,
                    expected_default_model_id,
                },
            ) => {
                valid_model_id(expected_default_model_id)
                    && worker_default == Some(expected_default_model_id.as_str())
                    && protocols.contains(protocol)
            }
            (Self::VoiceControl, ProfileRequirement::VoiceControl) => true,
            _ => false,
        }
    }
}

pub(crate) fn validate_workers(
    workers: &[WorkerConfig],
    required_services: &[ServiceClass],
    voice_owner: Option<&str>,
) -> Result<(), ConfigError> {
    if workers.is_empty() || workers.len() > MAX_WORKERS {
        return invalid("workers", "must contain between 1 and 256 workers");
    }
    validate_set(required_services, "router.required_services")?;

    let mut worker_ids = HashSet::with_capacity(workers.len());
    let mut targets = HashSet::with_capacity(workers.len());
    let mut domain_ips = HashMap::with_capacity(workers.len());
    let mut present_services = HashSet::new();
    let mut owner_has_control = false;
    for worker in workers {
        validate_token(&worker.worker_id, "workers.worker_id")?;
        validate_token(&worker.trust_domain, "workers.trust_domain")?;
        if !worker_ids.insert(worker.worker_id.as_str()) {
            return invalid("workers.worker_id", "must be unique");
        }
        let origin = validate_base_url(&worker.base_url)?;
        if !targets.insert(origin.normalized) {
            return invalid("workers.base_url", "normalized targets must be unique");
        }
        match (origin.host, worker.resolved_ip) {
            (ValidatedHost::Domain(domain), Some(ip)) => match domain_ips.entry(domain) {
                Entry::Vacant(entry) => {
                    entry.insert(ip);
                }
                Entry::Occupied(entry) if *entry.get() == ip => {}
                Entry::Occupied(_) => {
                    return invalid(
                        "workers.resolved_ip",
                        "one canonical hostname must resolve to one configured IP",
                    );
                }
            },
            (ValidatedHost::Domain(_), None) => {
                return invalid("workers.resolved_ip", "is required for a hostname base_url");
            }
            (ValidatedHost::Ip, Some(_)) => {
                return invalid(
                    "workers.resolved_ip",
                    "is forbidden for an IP-literal base_url",
                );
            }
            (ValidatedHost::Ip, None) => {}
        }
        validate_health_path(&worker.health_path)?;
        validate_worker_capacity(&worker.capacity)?;
        if worker.service_profiles.is_empty()
            || worker.service_profiles.len() > MAX_PROFILES_PER_WORKER
        {
            return invalid(
                "workers.service_profiles",
                "must contain between 1 and 64 rows",
            );
        }
        let executes_model = worker
            .service_profiles
            .iter()
            .any(ServiceProfile::executes_model);
        match (executes_model, worker.default_model_id.as_deref()) {
            (true, Some(default)) if valid_model_id(default) => {}
            (true, _) => {
                return invalid(
                    "workers.default_model_id",
                    "is required and must contain 1 to 256 bytes for model services",
                );
            }
            (false, None) => {}
            (false, Some(_)) => {
                return invalid(
                    "workers.default_model_id",
                    "is forbidden without a model-executing service",
                );
            }
        }
        for (index, profile) in worker.service_profiles.iter().enumerate() {
            profile.validate()?;
            if worker
                .capacity
                .get(profile.service_class().capacity())
                .is_none()
            {
                return invalid("workers.capacity", "missing capacity for a service profile");
            }
            if profile.requires_owner() && voice_owner != Some(worker.worker_id.as_str()) {
                return invalid(
                    "workers.service_profiles",
                    "owner-required row is not on the voice owner",
                );
            }
            if worker.service_profiles[..index]
                .iter()
                .any(|prior| prior.semantically_eq(profile))
            {
                return invalid("workers.service_profiles", "duplicate semantic profile");
            }
            present_services.insert(profile.service_class());
            if voice_owner == Some(worker.worker_id.as_str())
                && profile.service_class() == ServiceClass::VoiceControl
            {
                owner_has_control = true;
            }
        }
        if let Some(default) = worker.default_model_id.as_deref() {
            for service in [
                ServiceClass::GenerationHttp,
                ServiceClass::SpeechHttp,
                ServiceClass::SpeechBatch,
                ServiceClass::TranscriptionHttp,
                ServiceClass::SpeechWebsocket,
            ] {
                let advertises_service = worker
                    .service_profiles
                    .iter()
                    .any(|profile| profile.service_class() == service);
                if advertises_service
                    && !worker.service_profiles.iter().any(|profile| {
                        profile.service_class() == service
                            && profile
                                .model_ids()
                                .is_some_and(|model_ids| model_ids.iter().any(|id| id == default))
                    })
                {
                    return invalid(
                        "workers.default_model_id",
                        "must belong to every advertised model-ID-bearing service",
                    );
                }
            }
        }
        if worker.capacity.entries().iter().any(|(class, value)| {
            value.is_some()
                && !worker
                    .service_profiles
                    .iter()
                    .any(|profile| profile.service_class().capacity() == *class)
        }) {
            return invalid(
                "workers.capacity",
                "capacity entries without a matching service profile are not allowed",
            );
        }
    }

    if let Some(owner) = voice_owner
        && (!worker_ids.contains(owner) || !owner_has_control)
    {
        return invalid(
            "router.voice_owner_worker_id",
            "must name a worker with voice_control",
        );
    }
    if required_services
        .iter()
        .any(|service| !present_services.contains(service))
    {
        return invalid(
            "router.required_services",
            "required service is absent from all profiles",
        );
    }
    Ok(())
}

fn validate_worker_capacity(capacity: &WorkerCapacityConfig) -> Result<(), ConfigError> {
    for (_, value) in capacity.entries() {
        if value.is_some_and(|entry| !(1..=65_535).contains(&entry)) {
            return invalid("workers.capacity", "entries must be between 1 and 65535");
        }
    }
    Ok(())
}

struct ValidatedOrigin {
    normalized: String,
    host: ValidatedHost,
}

enum ValidatedHost {
    Domain(String),
    Ip,
}

fn validate_base_url(value: &str) -> Result<ValidatedOrigin, ConfigError> {
    if value.len() > MAX_BASE_URL_BYTES {
        return invalid("workers.base_url", "exceeds 2048 bytes");
    }
    let mut url = Url::parse(value)
        .map_err(|_| ConfigError::invalid("workers.base_url", "must be absolute HTTP(S)"))?;
    let Some(host) = url.host_str() else {
        return invalid(
            "workers.base_url",
            "must be an origin-only absolute HTTP(S) URL",
        );
    };
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path() != "/"
        || url.port() == Some(0)
    {
        return invalid(
            "workers.base_url",
            "must be an origin-only absolute HTTP(S) URL",
        );
    }
    let host = if parse_ip_literal(host).is_some() {
        ValidatedHost::Ip
    } else {
        if !host.is_ascii() || host.ends_with('.') {
            return invalid(
                "workers.base_url",
                "DNS host must be canonical ASCII without a trailing dot",
            );
        }
        ValidatedHost::Domain(host.to_owned())
    };
    url.set_path("");
    Ok(ValidatedOrigin {
        normalized: url.into(),
        host,
    })
}

fn parse_ip_literal(host: &str) -> Option<IpAddr> {
    host.strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host)
        .parse()
        .ok()
}

fn validate_health_path(value: &str) -> Result<(), ConfigError> {
    if value.is_empty()
        || value.len() > MAX_HEALTH_PATH_BYTES
        || !value.starts_with('/')
        || value.starts_with("//")
        || value.contains(['?', '#', '\\'])
        || value.chars().any(char::is_control)
    {
        return invalid("workers.health_path", "must be a bounded absolute path");
    }
    Ok(())
}

fn validate_speech_formats_and_modes(
    response_formats: &[SpeechResponseFormat],
    stream_modes: &[StreamMode],
) -> Result<(), ConfigError> {
    validate_set(
        response_formats,
        "workers.service_profiles.response_formats",
    )?;
    validate_set(stream_modes, "workers.service_profiles.stream_modes")?;
    if stream_modes.contains(&StreamMode::Streaming)
        && !matches!(response_formats, [SpeechResponseFormat::Pcm])
    {
        return invalid(
            "workers.service_profiles.response_formats",
            "must contain only pcm when stream_modes contains streaming",
        );
    }
    Ok(())
}

fn validate_model_ids(values: &[String]) -> Result<(), ConfigError> {
    validate_set(values, "workers.service_profiles.model_ids")?;
    if values
        .iter()
        .any(|value| value.is_empty() || value.len() > MAX_MODEL_ID_BYTES)
    {
        return invalid(
            "workers.service_profiles.model_ids",
            "IDs must contain 1 to 256 bytes",
        );
    }
    Ok(())
}

fn valid_model_id(value: &str) -> bool {
    !value.is_empty() && value.len() <= MAX_MODEL_ID_BYTES
}

fn model_matches(
    model_ids: &[String],
    selection: &ModelSelection,
    worker_default: Option<&str>,
) -> bool {
    selection.matches_worker_default(worker_default)
        && model_ids
            .iter()
            .any(|model_id| model_id == selection.model_id())
}

fn validate_generation_relationships(
    message_content_forms: &[MessageContentForm],
    media_placements: &[MediaPlacement],
    input_modalities: &[InputModality],
    output_modalities: &[OutputModality],
    has_audio_format: bool,
) -> Result<(), ConfigError> {
    let has_non_text_input = input_modalities
        .iter()
        .any(|modality| *modality != InputModality::Text);
    if has_non_text_input != !media_placements.is_empty() {
        return invalid(
            "workers.service_profiles.media_placements",
            "must be nonempty exactly for non-text input",
        );
    }
    if media_placements.contains(&MediaPlacement::TypedParts)
        && !message_content_forms.contains(&MessageContentForm::TypedParts)
    {
        return invalid(
            "workers.service_profiles.media_placements",
            "typed_parts requires typed_parts message content",
        );
    }
    if output_modalities.contains(&OutputModality::Audio) != has_audio_format {
        return invalid(
            "workers.service_profiles.chat_audio_formats",
            "must be nonempty exactly for audio output",
        );
    }
    Ok(())
}

fn validate_token(value: &str, field: &'static str) -> Result<(), ConfigError> {
    if value.is_empty()
        || value.len() > MAX_ID_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return invalid(
            field,
            "must match ASCII [A-Za-z0-9._-] and contain 1 to 128 bytes",
        );
    }
    Ok(())
}

fn validate_set<T: Eq + std::hash::Hash>(
    values: &[T],
    field: &'static str,
) -> Result<(), ConfigError> {
    if values.is_empty() || values.len() > MAX_SET_ITEMS {
        return invalid(field, "must contain between 1 and 64 values");
    }
    let mut seen = HashSet::with_capacity(values.len());
    if values.iter().any(|value| !seen.insert(value)) {
        return invalid(field, "must not contain duplicates");
    }
    Ok(())
}

fn validate_optional_set<T: Eq + std::hash::Hash>(
    values: &[T],
    field: &'static str,
) -> Result<(), ConfigError> {
    if values.len() > MAX_SET_ITEMS {
        return invalid(field, "must contain at most 64 values");
    }
    let mut seen = HashSet::with_capacity(values.len());
    if values.iter().any(|value| !seen.insert(value)) {
        return invalid(field, "must not contain duplicates");
    }
    Ok(())
}

fn valid_bounded_set<T: Eq>(values: &[T], allow_empty: bool) -> bool {
    (allow_empty || !values.is_empty())
        && values.len() <= MAX_SET_ITEMS
        && values
            .iter()
            .enumerate()
            .all(|(index, value)| !values[index + 1..].contains(value))
}

fn valid_speech_reference_requirement(reference_forms: &[ReferenceForm]) -> bool {
    valid_bounded_set(reference_forms, false)
        && (reference_forms.len() == 1 || !reference_forms.contains(&ReferenceForm::None))
}

fn same_set<T: Eq>(left: &[T], right: &[T]) -> bool {
    left.len() == right.len() && left.iter().all(|value| right.contains(value))
}

fn invalid<T>(field: &'static str, reason: &'static str) -> Result<T, ConfigError> {
    Err(ConfigError::invalid(field, reason))
}

fn default_health_path() -> String {
    String::from("/health")
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::{
        ChatAudioFormat, InputModality, MediaPlacement, MessageContentForm, ModelSelection,
        OutputModality, ProfileRequirement, ReferenceForm, ServiceClass, ServiceProfile,
        SpeechResponseFormat, SpeechTask, SpeechWebsocketInput, StreamMode, WorkerCapacityConfig,
        WorkerConfig, validate_workers,
    };

    fn profile(models: &[&str], modalities: &[InputModality]) -> ServiceProfile {
        ServiceProfile::GenerationHttp {
            model_ids: models.iter().map(|value| (*value).to_owned()).collect(),
            message_content_forms: vec![MessageContentForm::String],
            media_placements: if modalities
                .iter()
                .any(|modality| *modality != InputModality::Text)
            {
                vec![super::MediaPlacement::TopLevel]
            } else {
                Vec::new()
            },
            input_modalities: modalities.to_vec(),
            output_modalities: vec![OutputModality::Text],
            chat_audio_formats: Vec::new(),
            stream_modes: vec![StreamMode::NonStreaming],
        }
    }

    fn speech_http_profile(stream_modes: &[StreamMode]) -> ServiceProfile {
        ServiceProfile::SpeechHttp {
            model_ids: vec![String::from("tts")],
            response_formats: vec![SpeechResponseFormat::Pcm],
            stream_modes: stream_modes.to_vec(),
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::None],
            managed_voice: false,
        }
    }

    fn speech_websocket_profile(stream_modes: &[StreamMode]) -> ServiceProfile {
        ServiceProfile::SpeechWebsocket {
            model_ids: vec![String::from("tts")],
            input_profiles: vec![SpeechWebsocketInput::Text],
            response_formats: vec![SpeechResponseFormat::Pcm],
            stream_modes: stream_modes.to_vec(),
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::None],
            managed_voice: false,
        }
    }

    fn worker(id: usize) -> WorkerConfig {
        WorkerConfig {
            worker_id: format!("worker-{id}"),
            base_url: format!("http://127.0.0.1:{}/", 20_000 + id),
            resolved_ip: None,
            trust_domain: String::from("local"),
            default_model_id: Some(String::from("omni")),
            health_path: String::from("/health"),
            capacity: WorkerCapacityConfig {
                generation_http: Some(1),
                speech_http: None,
                transcription_http: None,
                speech_batch: None,
                speech_websocket: None,
                realtime_websocket: None,
                control: None,
            },
            service_profiles: vec![profile(&["omni"], &[InputModality::Text])],
        }
    }

    #[test]
    fn manifest_accepts_one_and_maximum_workers_but_rejects_empty() {
        assert!(validate_workers(&[worker(0)], &[ServiceClass::GenerationHttp], None).is_ok());
        let maximum: Vec<_> = (0..256).map(worker).collect();
        assert!(validate_workers(&maximum, &[ServiceClass::GenerationHttp], None).is_ok());
        assert!(validate_workers(&[], &[ServiceClass::GenerationHttp], None).is_err());
    }

    #[test]
    fn normalized_targets_and_semantic_profiles_are_unique() {
        let first = worker(0);
        let mut duplicate_target = worker(1);
        duplicate_target.base_url = String::from("HTTP://127.0.0.1:20000");
        assert!(
            validate_workers(
                &[first.clone(), duplicate_target],
                &[ServiceClass::GenerationHttp],
                None,
            )
            .is_err()
        );

        let mut duplicate_profile = first;
        duplicate_profile.service_profiles = vec![
            profile(
                &["omni", "other"],
                &[InputModality::Text, InputModality::Audio],
            ),
            profile(
                &["other", "omni"],
                &[InputModality::Audio, InputModality::Text],
            ),
        ];
        assert!(
            validate_workers(&[duplicate_profile], &[ServiceClass::GenerationHttp], None,).is_err()
        );
    }

    #[test]
    fn speech_stream_modes_are_order_insensitive_semantic_identity() {
        let builders: [fn(&[StreamMode]) -> ServiceProfile; 2] =
            [speech_http_profile, speech_websocket_profile];
        for build in builders {
            let non_streaming = build(&[StreamMode::NonStreaming]);
            let streaming = build(&[StreamMode::Streaming]);
            assert!(!non_streaming.semantically_eq(&streaming));

            let both = build(&[StreamMode::NonStreaming, StreamMode::Streaming]);
            let reordered = build(&[StreamMode::Streaming, StreamMode::NonStreaming]);
            assert!(both.semantically_eq(&reordered));
        }
    }

    #[test]
    fn speech_reference_requirements_match_complete_sets_and_exact_voice() {
        let http_profile =
            |reference_forms: Vec<ReferenceForm>, managed_voice: bool| ServiceProfile::SpeechHttp {
                model_ids: vec![String::from("tts")],
                response_formats: vec![SpeechResponseFormat::Pcm],
                stream_modes: vec![StreamMode::NonStreaming],
                tasks: vec![SpeechTask::TextToSpeech],
                reference_forms,
                managed_voice,
            };
        let http_requirement = |reference_forms: Vec<ReferenceForm>, managed_voice: bool| {
            ProfileRequirement::SpeechHttp {
                model: ModelSelection::Explicit(String::from("tts")),
                response_format: SpeechResponseFormat::Pcm,
                stream_mode: StreamMode::NonStreaming,
                task: SpeechTask::TextToSpeech,
                reference_forms,
                managed_voice,
            }
        };
        let websocket_profile = |reference_forms: Vec<ReferenceForm>, managed_voice: bool| {
            ServiceProfile::SpeechWebsocket {
                model_ids: vec![String::from("tts")],
                input_profiles: vec![SpeechWebsocketInput::Text],
                response_formats: vec![SpeechResponseFormat::Pcm],
                stream_modes: vec![StreamMode::NonStreaming],
                tasks: vec![SpeechTask::TextToSpeech],
                reference_forms,
                managed_voice,
            }
        };
        let websocket_requirement = |reference_forms: Vec<ReferenceForm>, managed_voice: bool| {
            ProfileRequirement::SpeechWebsocket {
                model: ModelSelection::Explicit(String::from("tts")),
                input_profile: SpeechWebsocketInput::Text,
                response_format: SpeechResponseFormat::Pcm,
                stream_mode: StreamMode::NonStreaming,
                task: SpeechTask::TextToSpeech,
                reference_forms,
                managed_voice,
            }
        };

        let cases = [
            (
                http_profile(
                    vec![
                        ReferenceForm::Direct,
                        ReferenceForm::List,
                        ReferenceForm::VqCodes,
                    ],
                    false,
                ),
                [
                    http_profile(vec![ReferenceForm::Direct, ReferenceForm::List], false),
                    http_profile(vec![ReferenceForm::Direct, ReferenceForm::VqCodes], false),
                    http_profile(vec![ReferenceForm::List, ReferenceForm::VqCodes], false),
                ],
                http_requirement(
                    vec![
                        ReferenceForm::Direct,
                        ReferenceForm::List,
                        ReferenceForm::VqCodes,
                    ],
                    false,
                ),
                http_profile(vec![ReferenceForm::None], false),
                http_requirement(vec![ReferenceForm::None], false),
                http_profile(
                    vec![
                        ReferenceForm::Direct,
                        ReferenceForm::List,
                        ReferenceForm::VqCodes,
                    ],
                    true,
                ),
                http_requirement(
                    vec![
                        ReferenceForm::Direct,
                        ReferenceForm::List,
                        ReferenceForm::VqCodes,
                    ],
                    true,
                ),
            ),
            (
                websocket_profile(
                    vec![
                        ReferenceForm::Direct,
                        ReferenceForm::List,
                        ReferenceForm::VqCodes,
                    ],
                    false,
                ),
                [
                    websocket_profile(vec![ReferenceForm::Direct, ReferenceForm::List], false),
                    websocket_profile(vec![ReferenceForm::Direct, ReferenceForm::VqCodes], false),
                    websocket_profile(vec![ReferenceForm::List, ReferenceForm::VqCodes], false),
                ],
                websocket_requirement(
                    vec![
                        ReferenceForm::Direct,
                        ReferenceForm::List,
                        ReferenceForm::VqCodes,
                    ],
                    false,
                ),
                websocket_profile(vec![ReferenceForm::None], false),
                websocket_requirement(vec![ReferenceForm::None], false),
                websocket_profile(
                    vec![
                        ReferenceForm::Direct,
                        ReferenceForm::List,
                        ReferenceForm::VqCodes,
                    ],
                    true,
                ),
                websocket_requirement(
                    vec![
                        ReferenceForm::Direct,
                        ReferenceForm::List,
                        ReferenceForm::VqCodes,
                    ],
                    true,
                ),
            ),
        ];

        for (
            complete_profile,
            incomplete_profiles,
            mixed_requirement,
            none_profile,
            none_requirement,
            managed_profile,
            managed_requirement,
        ) in cases
        {
            assert!(mixed_requirement.is_well_formed());
            assert!(complete_profile.matches(&mixed_requirement, None));
            assert!(
                incomplete_profiles
                    .iter()
                    .all(|profile| !profile.matches(&mixed_requirement, None))
            );

            assert!(none_requirement.is_well_formed());
            assert!(none_profile.matches(&none_requirement, None));

            assert!(managed_requirement.is_well_formed());
            assert!(managed_profile.matches(&managed_requirement, None));
            assert!(!complete_profile.matches(&managed_requirement, None));
            assert!(!managed_profile.matches(&mixed_requirement, None));

            for invalid_forms in [
                Vec::new(),
                vec![ReferenceForm::Direct, ReferenceForm::Direct],
                vec![ReferenceForm::None, ReferenceForm::Direct],
            ] {
                let mut invalid = mixed_requirement.clone();
                match &mut invalid {
                    ProfileRequirement::SpeechHttp {
                        reference_forms, ..
                    }
                    | ProfileRequirement::SpeechWebsocket {
                        reference_forms, ..
                    } => *reference_forms = invalid_forms,
                    ProfileRequirement::GenerationHttp { .. }
                    | ProfileRequirement::SpeechBatch { .. }
                    | ProfileRequirement::TranscriptionHttp { .. }
                    | ProfileRequirement::RealtimeWebsocket { .. }
                    | ProfileRequirement::VoiceControl => unreachable!(),
                }
                assert!(!invalid.is_well_formed());
            }
        }
    }

    #[test]
    fn malformed_paths_missing_capacity_and_absent_required_classes_fail_closed() {
        let mut malformed = worker(0);
        malformed.health_path = String::from("//authority");
        assert!(validate_workers(&[malformed], &[ServiceClass::GenerationHttp], None).is_err());

        let mut missing_capacity = worker(0);
        missing_capacity.capacity.generation_http = None;
        assert!(
            validate_workers(&[missing_capacity], &[ServiceClass::GenerationHttp], None,).is_err()
        );

        let mut dead_capacity = worker(0);
        dead_capacity.capacity.speech_http = Some(1);
        assert!(validate_workers(&[dead_capacity], &[ServiceClass::GenerationHttp], None).is_err());

        assert!(validate_workers(&[worker(0)], &[ServiceClass::SpeechHttp], None).is_err());
    }

    #[test]
    fn voice_owner_requires_exact_worker_and_control_profile() {
        let worker = worker(0);
        assert!(
            validate_workers(
                std::slice::from_ref(&worker),
                &[ServiceClass::GenerationHttp],
                Some("missing"),
            )
            .is_err()
        );
        assert!(
            validate_workers(&[worker], &[ServiceClass::GenerationHttp], Some("worker-0"),)
                .is_err()
        );
    }

    #[test]
    fn worker_default_is_required_for_model_services_and_forbidden_for_control_only() {
        let mut missing = worker(0);
        missing.default_model_id = None;
        assert!(validate_workers(&[missing], &[ServiceClass::GenerationHttp], None).is_err());

        let mut empty = worker(0);
        empty.default_model_id = Some(String::new());
        assert!(validate_workers(&[empty], &[ServiceClass::GenerationHttp], None).is_err());

        let mut oversized = worker(0);
        oversized.default_model_id = Some("x".repeat(257));
        assert!(validate_workers(&[oversized], &[ServiceClass::GenerationHttp], None).is_err());

        let mut control = worker(0);
        control.default_model_id = Some(String::from("forbidden"));
        control.capacity.generation_http = None;
        control.capacity.control = Some(1);
        control.service_profiles = vec![ServiceProfile::VoiceControl];
        assert!(
            validate_workers(&[control], &[ServiceClass::VoiceControl], Some("worker-0")).is_err()
        );
    }

    #[test]
    fn worker_default_membership_is_validated_independently_for_every_service() {
        let mut multi = worker(0);
        multi.capacity.speech_http = Some(1);
        multi.capacity.speech_batch = Some(1);
        multi.capacity.transcription_http = Some(1);
        multi.capacity.speech_websocket = Some(1);
        multi.service_profiles.extend([
            ServiceProfile::SpeechHttp {
                model_ids: vec![String::from("omni")],
                response_formats: vec![SpeechResponseFormat::Pcm],
                stream_modes: vec![StreamMode::NonStreaming],
                tasks: vec![SpeechTask::TextToSpeech],
                reference_forms: vec![ReferenceForm::None],
                managed_voice: false,
            },
            ServiceProfile::SpeechBatch {
                model_ids: vec![String::from("omni")],
                response_formats: vec![SpeechResponseFormat::Wav],
                tasks: vec![SpeechTask::TextToSpeech],
                reference_forms: vec![ReferenceForm::None],
                managed_voice: false,
                max_batch_size: 2,
                effective_features: vec![super::BatchFeature::Model],
            },
            ServiceProfile::TranscriptionHttp {
                model_ids: vec![String::from("omni")],
                response_formats: vec![super::TranscriptionResponseFormat::Json],
                media_profiles: vec![super::MediaProfile::Audio],
                stream_modes: vec![StreamMode::NonStreaming],
            },
            speech_websocket_profile(&[StreamMode::NonStreaming]),
        ]);
        assert!(validate_workers(&[multi.clone()], &[ServiceClass::GenerationHttp], None).is_err());
        if let ServiceProfile::SpeechWebsocket { model_ids, .. } = multi
            .service_profiles
            .last_mut()
            .expect("speech websocket row")
        {
            *model_ids = vec![String::from("omni")];
        }
        assert!(validate_workers(&[multi.clone()], &[ServiceClass::GenerationHttp], None).is_ok());
        for index in 1..multi.service_profiles.len() {
            let mut wrong = multi.clone();
            if let Some(model_ids) = wrong.service_profiles[index]
                .model_ids()
                .map(<[String]>::to_vec)
            {
                let profile = &mut wrong.service_profiles[index];
                match profile {
                    ServiceProfile::SpeechHttp { model_ids: ids, .. }
                    | ServiceProfile::SpeechBatch { model_ids: ids, .. }
                    | ServiceProfile::TranscriptionHttp { model_ids: ids, .. }
                    | ServiceProfile::SpeechWebsocket { model_ids: ids, .. } => {
                        *ids = model_ids
                            .into_iter()
                            .map(|_| String::from("other"))
                            .collect();
                    }
                    ServiceProfile::GenerationHttp { .. }
                    | ServiceProfile::RealtimeWebsocket { .. }
                    | ServiceProfile::VoiceControl => {}
                }
            }
            assert!(validate_workers(&[wrong], &[ServiceClass::GenerationHttp], None).is_err());
        }
    }

    #[test]
    fn generation_sets_and_cross_field_relationships_are_exact() {
        let capability = ServiceProfile::GenerationHttp {
            model_ids: vec![String::from("omni")],
            message_content_forms: vec![MessageContentForm::String, MessageContentForm::TypedParts],
            media_placements: vec![MediaPlacement::TopLevel, MediaPlacement::TypedParts],
            input_modalities: vec![InputModality::Text, InputModality::Image],
            output_modalities: vec![OutputModality::Text, OutputModality::Audio],
            chat_audio_formats: vec![
                ChatAudioFormat::Wav,
                ChatAudioFormat::Mp3,
                ChatAudioFormat::Flac,
                ChatAudioFormat::Pcm,
                ChatAudioFormat::Aac,
                ChatAudioFormat::Opus,
            ],
            stream_modes: vec![StreamMode::NonStreaming],
        };
        assert!(capability.validate().is_ok());
        for audio_format in [
            ChatAudioFormat::Wav,
            ChatAudioFormat::Mp3,
            ChatAudioFormat::Flac,
            ChatAudioFormat::Pcm,
            ChatAudioFormat::Aac,
            ChatAudioFormat::Opus,
        ] {
            let requirement = ProfileRequirement::GenerationHttp {
                model: ModelSelection::Explicit(String::from("omni")),
                message_content_forms: vec![
                    MessageContentForm::String,
                    MessageContentForm::TypedParts,
                ],
                media_placements: vec![MediaPlacement::TopLevel, MediaPlacement::TypedParts],
                input_modalities: vec![InputModality::Text, InputModality::Image],
                output_modalities: vec![OutputModality::Audio],
                audio_format: Some(audio_format),
                stream_mode: StreamMode::NonStreaming,
            };
            assert!(requirement.is_well_formed());
            assert!(capability.matches(&requirement, Some("wrong-default")));
        }

        let empty_known_input = ProfileRequirement::GenerationHttp {
            model: ModelSelection::Explicit(String::from("omni")),
            message_content_forms: vec![MessageContentForm::TypedParts],
            media_placements: Vec::new(),
            input_modalities: Vec::new(),
            output_modalities: vec![OutputModality::Text],
            audio_format: None,
            stream_mode: StreamMode::NonStreaming,
        };
        assert!(empty_known_input.is_well_formed());
        assert!(
            capability.matches(&empty_known_input, Some("wrong-default")),
            "an empty request constraint must match a nonempty worker modality profile"
        );

        let invalid = [
            ProfileRequirement::GenerationHttp {
                model: ModelSelection::Explicit(String::from("omni")),
                message_content_forms: vec![MessageContentForm::String],
                media_placements: vec![MediaPlacement::TypedParts],
                input_modalities: vec![InputModality::Image],
                output_modalities: vec![OutputModality::Text],
                audio_format: None,
                stream_mode: StreamMode::NonStreaming,
            },
            ProfileRequirement::GenerationHttp {
                model: ModelSelection::Explicit(String::from("omni")),
                message_content_forms: vec![MessageContentForm::String],
                media_placements: Vec::new(),
                input_modalities: vec![InputModality::Image],
                output_modalities: vec![OutputModality::Text],
                audio_format: None,
                stream_mode: StreamMode::NonStreaming,
            },
            ProfileRequirement::GenerationHttp {
                model: ModelSelection::Explicit(String::from("omni")),
                message_content_forms: vec![MessageContentForm::String],
                media_placements: Vec::new(),
                input_modalities: vec![InputModality::Text],
                output_modalities: vec![OutputModality::Audio],
                audio_format: None,
                stream_mode: StreamMode::NonStreaming,
            },
        ];
        assert!(
            invalid
                .iter()
                .all(|requirement| !requirement.is_well_formed())
        );

        let mut reordered = capability.clone();
        if let ServiceProfile::GenerationHttp {
            message_content_forms,
            media_placements,
            input_modalities,
            output_modalities,
            chat_audio_formats,
            ..
        } = &mut reordered
        {
            message_content_forms.reverse();
            media_placements.reverse();
            input_modalities.reverse();
            output_modalities.reverse();
            chat_audio_formats.reverse();
        }
        assert!(capability.semantically_eq(&reordered));
    }

    #[test]
    fn explicit_and_defaulted_model_provenance_is_preserved_in_batch_matching() {
        let profile = ServiceProfile::SpeechBatch {
            model_ids: vec![String::from("D")],
            response_formats: vec![SpeechResponseFormat::Wav],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::None],
            managed_voice: false,
            max_batch_size: 2,
            effective_features: vec![super::BatchFeature::Model],
        };
        let requirement = ProfileRequirement::SpeechBatch {
            models: vec![
                ModelSelection::Explicit(String::from("D")),
                ModelSelection::WorkerDefault {
                    expected_model_id: String::from("D"),
                },
            ],
            response_formats: vec![SpeechResponseFormat::Wav],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::None],
            managed_voice: false,
            batch_size: 2,
            effective_features: vec![super::BatchFeature::Model],
        };
        assert!(profile.matches(&requirement, Some("D")));
        assert!(!profile.matches(&requirement, Some("other")));
        let ProfileRequirement::SpeechBatch { models, .. } = requirement else {
            unreachable!()
        };
        assert_ne!(models[0], models[1]);
        assert_eq!(models[0].model_id(), models[1].model_id());
    }

    #[test]
    fn shared_model_selection_applies_to_every_client_selectable_surface() {
        let surfaces = [
            (
                ServiceProfile::SpeechHttp {
                    model_ids: vec![String::from("D")],
                    response_formats: vec![SpeechResponseFormat::Wav],
                    stream_modes: vec![StreamMode::NonStreaming],
                    tasks: vec![SpeechTask::TextToSpeech],
                    reference_forms: vec![ReferenceForm::None],
                    managed_voice: false,
                },
                ProfileRequirement::SpeechHttp {
                    model: ModelSelection::WorkerDefault {
                        expected_model_id: String::from("D"),
                    },
                    response_format: SpeechResponseFormat::Wav,
                    stream_mode: StreamMode::NonStreaming,
                    task: SpeechTask::TextToSpeech,
                    reference_forms: vec![ReferenceForm::None],
                    managed_voice: false,
                },
            ),
            (
                ServiceProfile::TranscriptionHttp {
                    model_ids: vec![String::from("D")],
                    response_formats: vec![super::TranscriptionResponseFormat::Json],
                    media_profiles: vec![super::MediaProfile::Audio],
                    stream_modes: vec![StreamMode::NonStreaming],
                },
                ProfileRequirement::TranscriptionHttp {
                    model: ModelSelection::WorkerDefault {
                        expected_model_id: String::from("D"),
                    },
                    response_format: super::TranscriptionResponseFormat::Json,
                    media_profile: super::MediaProfile::Audio,
                    stream_mode: StreamMode::NonStreaming,
                },
            ),
            (
                ServiceProfile::SpeechWebsocket {
                    model_ids: vec![String::from("D")],
                    input_profiles: vec![SpeechWebsocketInput::Text],
                    response_formats: vec![SpeechResponseFormat::Pcm],
                    stream_modes: vec![StreamMode::NonStreaming],
                    tasks: vec![SpeechTask::TextToSpeech],
                    reference_forms: vec![ReferenceForm::None],
                    managed_voice: false,
                },
                ProfileRequirement::SpeechWebsocket {
                    model: ModelSelection::WorkerDefault {
                        expected_model_id: String::from("D"),
                    },
                    input_profile: SpeechWebsocketInput::Text,
                    response_format: SpeechResponseFormat::Pcm,
                    stream_mode: StreamMode::NonStreaming,
                    task: SpeechTask::TextToSpeech,
                    reference_forms: vec![ReferenceForm::None],
                    managed_voice: false,
                },
            ),
        ];
        for (profile, defaulted) in surfaces {
            assert!(profile.matches(&defaulted, Some("D")));
            assert!(!profile.matches(&defaulted, Some("other")));
            let mut explicit = defaulted;
            match &mut explicit {
                ProfileRequirement::SpeechHttp { model, .. }
                | ProfileRequirement::TranscriptionHttp { model, .. }
                | ProfileRequirement::SpeechWebsocket { model, .. } => {
                    *model = ModelSelection::Explicit(String::from("D"));
                }
                ProfileRequirement::GenerationHttp { .. }
                | ProfileRequirement::SpeechBatch { .. }
                | ProfileRequirement::RealtimeWebsocket { .. }
                | ProfileRequirement::VoiceControl => unreachable!(),
            }
            assert!(profile.matches(&explicit, Some("other")));
        }
    }

    #[test]
    fn heterogeneous_defaults_are_valid_for_explicit_model_services() {
        let make_worker = |id: usize, model_id: &str| {
            let mut worker = worker(id);
            worker.default_model_id = Some(model_id.to_owned());
            worker.capacity.speech_batch = Some(2);
            worker.service_profiles = vec![
                profile(&[model_id], &[InputModality::Text]),
                ServiceProfile::SpeechBatch {
                    model_ids: vec![model_id.to_owned()],
                    response_formats: vec![SpeechResponseFormat::Wav],
                    tasks: vec![SpeechTask::TextToSpeech],
                    reference_forms: vec![ReferenceForm::None],
                    managed_voice: false,
                    max_batch_size: 2,
                    effective_features: vec![super::BatchFeature::Model],
                },
            ];
            worker
        };
        assert!(
            validate_workers(
                &[make_worker(0, "A"), make_worker(1, "B")],
                &[ServiceClass::GenerationHttp, ServiceClass::SpeechBatch],
                None,
            )
            .is_ok()
        );
    }
}
