use std::fmt;

use serde::de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};

use crate::error::HttpFault;
use crate::speech_facts::{
    SpeechFields, effective_reference_forms, exceeds_nesting_limit,
    managed_voice as classify_managed_voice, read_field as read_speech_field, reference_forms,
    response_format as classify_response_format, task as classify_task,
};
use crate::worker_pool::{
    BatchFeature, DefaultModelResolution, ModelSelection, ProfileRequirement, ReferenceForm,
    RouteRequirement, ServiceClass, SpeechResponseFormat, SpeechToTextTask, StreamMode,
    TranscriptionResponseFormat, TrustDomain, WorkerPool,
};

use super::headers::SuccessProfile;
use super::multipart;

pub(super) struct Classified {
    pub(super) requirement: RouteRequirement,
    pub(super) success: SuccessProfile,
    pub(super) credits: u32,
}

#[cfg(test)]
pub(super) fn speech(
    bytes: &[u8],
    pool: &WorkerPool,
    trust: &TrustDomain,
) -> Result<Classified, HttpFault> {
    speech_with_hints(bytes, None, None, pool, trust)
}

pub(super) fn speech_with_hints(
    bytes: &[u8],
    route_model: Option<&str>,
    route_stream: Option<bool>,
    pool: &WorkerPool,
    trust: &TrustDomain,
) -> Result<Classified, HttpFault> {
    let fields = parse_speech(bytes)?;
    let model = model_selection(
        fields.model.clone().flatten(),
        route_model,
        pool,
        trust,
        ServiceClass::SpeechHttp,
        None,
    )?;
    let format = classify_response_format(
        fields
            .response_format
            .as_ref()
            .and_then(Option::as_deref)
            .unwrap_or("wav"),
    )
    .ok_or(HttpFault::MalformedRequest)?;
    let body_stream = fields.stream.as_ref().and_then(|value| *value);
    let stream = merge_stream(body_stream, route_stream)?;
    if stream && format != SpeechResponseFormat::Pcm {
        return Err(HttpFault::MalformedRequest);
    }
    let stream_mode = if stream {
        StreamMode::Streaming
    } else {
        StreamMode::NonStreaming
    };
    let task = classify_task(
        fields
            .task
            .as_ref()
            .and_then(Option::as_deref)
            .unwrap_or("Base"),
    )
    .ok_or(HttpFault::MalformedRequest)?;
    let references = reference_forms(&fields);
    let managed_voice = classify_managed_voice(&fields, &references);
    Ok(Classified {
        requirement: RouteRequirement::new(
            ProfileRequirement::SpeechHttp {
                model,
                response_format: format,
                stream_mode,
                task,
                reference_forms: references,
                managed_voice,
            },
            trust.clone(),
        ),
        success: SuccessProfile::Speech(format, stream_mode),
        credits: 1,
    })
}

#[cfg(test)]
pub(super) fn batch(
    bytes: &[u8],
    pool: &WorkerPool,
    trust: &TrustDomain,
) -> Result<Classified, HttpFault> {
    batch_with_hints(bytes, None, None, pool, trust)
}

pub(super) fn batch_with_hints(
    bytes: &[u8],
    route_model: Option<&str>,
    route_stream: Option<bool>,
    pool: &WorkerPool,
    trust: &TrustDomain,
) -> Result<Classified, HttpFault> {
    let (defaults, items) = parse_batch(bytes)?;
    if merge_stream(
        defaults.stream.as_ref().and_then(|value| *value),
        route_stream,
    )? {
        return Err(HttpFault::MalformedRequest);
    }
    if items.is_empty() || items.len() > usize::from(u16::MAX) {
        return Err(HttpFault::MalformedRequest);
    }
    let mut models = Vec::with_capacity(items.len());
    let mut formats = Vec::new();
    let mut tasks = Vec::new();
    let mut references = Vec::new();
    let mut features = Vec::new();
    let default_references = reference_forms(&defaults);
    let mut managed_voice = classify_managed_voice(&defaults, &default_references);
    for item in items {
        if item.model.as_ref().is_some_and(Option::is_some) {
            insert_once(&mut features, BatchFeature::Model);
        }
        if item.response_format.as_ref().is_some_and(Option::is_some) {
            insert_once(&mut features, BatchFeature::Format);
        }
        if item.task.as_ref().is_some_and(Option::is_some) {
            insert_once(&mut features, BatchFeature::Task);
        }
        if item.ref_audio.as_ref().is_some_and(Option::is_some)
            || item.references.as_ref().is_some_and(Option::is_some)
        {
            insert_once(&mut features, BatchFeature::Reference);
        }
        if item.voice.as_ref().is_some_and(Option::is_some) {
            insert_once(&mut features, BatchFeature::Voice);
        }
        let effective_model = item
            .model
            .clone()
            .flatten()
            .or_else(|| defaults.model.clone().flatten());
        models.push(model_selection(
            effective_model,
            route_model,
            pool,
            trust,
            ServiceClass::SpeechBatch,
            None,
        )?);
        let format = classify_response_format(
            item.response_format
                .clone()
                .flatten()
                .or_else(|| defaults.response_format.clone().flatten())
                .as_deref()
                .unwrap_or("wav"),
        )
        .ok_or(HttpFault::MalformedRequest)?;
        insert_once(&mut formats, format);
        let task = classify_task(
            item.task
                .clone()
                .flatten()
                .or_else(|| defaults.task.clone().flatten())
                .as_deref()
                .unwrap_or("Base"),
        )
        .ok_or(HttpFault::MalformedRequest)?;
        insert_once(&mut tasks, task);
        let effective_references = effective_reference_forms(&defaults, &item);
        let explicit_reference = effective_references != [ReferenceForm::None];
        for form in effective_references {
            insert_once(&mut references, form);
        }
        let voice = item
            .voice
            .clone()
            .flatten()
            .or_else(|| defaults.voice.clone().flatten());
        managed_voice |= !explicit_reference
            && voice
                .is_some_and(|value| !value.is_empty() && !value.eq_ignore_ascii_case("default"));
    }
    let batch_size = u16::try_from(models.len()).map_err(|_| HttpFault::MalformedRequest)?;
    Ok(Classified {
        requirement: RouteRequirement::new(
            ProfileRequirement::SpeechBatch {
                models,
                response_formats: formats,
                tasks,
                reference_forms: references,
                managed_voice,
                batch_size,
                effective_features: features,
            },
            trust.clone(),
        ),
        success: SuccessProfile::Json,
        credits: u32::from(batch_size),
    })
}

pub(super) fn transcription_with_hints(
    bytes: &[u8],
    boundary: &[u8],
    route_model: Option<&str>,
    route_stream: Option<bool>,
    pool: &WorkerPool,
    trust: &TrustDomain,
) -> Result<Classified, HttpFault> {
    speech_to_text(
        bytes,
        boundary,
        route_model,
        route_stream,
        pool,
        trust,
        SpeechToTextTask::Transcribe,
    )
}

pub(super) fn translation_with_hints(
    bytes: &[u8],
    boundary: &[u8],
    route_model: Option<&str>,
    route_stream: Option<bool>,
    pool: &WorkerPool,
    trust: &TrustDomain,
) -> Result<Classified, HttpFault> {
    speech_to_text(
        bytes,
        boundary,
        route_model,
        route_stream,
        pool,
        trust,
        SpeechToTextTask::Translate,
    )
}

fn speech_to_text(
    bytes: &[u8],
    boundary: &[u8],
    route_model: Option<&str>,
    route_stream: Option<bool>,
    pool: &WorkerPool,
    trust: &TrustDomain,
    task: SpeechToTextTask,
) -> Result<Classified, HttpFault> {
    let facts = multipart::scan(bytes, boundary)?;
    let model = model_selection(
        facts.model,
        route_model,
        pool,
        trust,
        ServiceClass::TranscriptionHttp,
        Some(task),
    )?;
    let stream = merge_stream(facts.stream, route_stream)?;
    let (format, success) = if stream {
        if !matches!(
            facts
                .response_format
                .as_deref()
                .unwrap_or("json")
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "json" | "text"
        ) {
            return Err(HttpFault::MalformedRequest);
        }
        (TranscriptionResponseFormat::Sse, SuccessProfile::Sse)
    } else {
        match facts
            .response_format
            .as_deref()
            .unwrap_or("json")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "json" => (
                TranscriptionResponseFormat::Json,
                SuccessProfile::TranscriptionJson,
            ),
            "text" => (TranscriptionResponseFormat::Text, SuccessProfile::Text),
            "verbose_json" => (
                TranscriptionResponseFormat::VerboseJson,
                SuccessProfile::TranscriptionJson,
            ),
            "srt" => (TranscriptionResponseFormat::Srt, SuccessProfile::Text),
            "vtt" => (TranscriptionResponseFormat::Vtt, SuccessProfile::Text),
            _ => return Err(HttpFault::MalformedRequest),
        }
    };
    Ok(Classified {
        requirement: RouteRequirement::new(
            ProfileRequirement::TranscriptionHttp {
                model,
                task,
                response_format: format,
                media_profile: facts.media_profile,
                stream_mode: if stream {
                    StreamMode::Streaming
                } else {
                    StreamMode::NonStreaming
                },
            },
            trust.clone(),
        ),
        success,
        credits: 1,
    })
}

fn model_selection(
    model: Option<String>,
    route_assertion: Option<&str>,
    pool: &WorkerPool,
    trust: &TrustDomain,
    service: ServiceClass,
    task: Option<SpeechToTextTask>,
) -> Result<ModelSelection, HttpFault> {
    let model = model.filter(|value| !value.is_empty());
    match (model, route_assertion) {
        (Some(model), Some(asserted)) if model != asserted => Err(HttpFault::MalformedRequest),
        (Some(model), _) => Ok(ModelSelection::Explicit(model)),
        (None, Some(asserted)) => Ok(ModelSelection::WorkerDefault {
            expected_model_id: asserted.to_owned(),
        }),
        (None, None) => match pool.resolve_default_model_id(trust, service, task) {
            DefaultModelResolution::Unique(model) => Ok(ModelSelection::WorkerDefault {
                expected_model_id: model.to_owned(),
            }),
            DefaultModelResolution::Ambiguous => Err(HttpFault::AmbiguousModel),
            DefaultModelResolution::NoService => Err(HttpFault::RouterUnavailable),
        },
    }
}

fn merge_stream(body: Option<bool>, route: Option<bool>) -> Result<bool, HttpFault> {
    let effective = body.unwrap_or(false);
    if route.is_some_and(|asserted| asserted != effective) {
        Err(HttpFault::MalformedRequest)
    } else {
        Ok(effective)
    }
}

fn insert_once<T: Eq>(values: &mut Vec<T>, value: T) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn parse_speech(bytes: &[u8]) -> Result<SpeechFields, HttpFault> {
    parse(bytes, RootMode::Speech).map(|parsed| parsed.0)
}

fn parse_batch(bytes: &[u8]) -> Result<(SpeechFields, Vec<SpeechFields>), HttpFault> {
    parse(bytes, RootMode::Batch)
}

fn parse(bytes: &[u8], mode: RootMode) -> Result<(SpeechFields, Vec<SpeechFields>), HttpFault> {
    if exceeds_nesting_limit(bytes) {
        return Err(HttpFault::MalformedRequest);
    }
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let parsed = RootSeed(mode)
        .deserialize(&mut deserializer)
        .map_err(|_| HttpFault::MalformedRequest)?;
    deserializer
        .end()
        .map_err(|_| HttpFault::MalformedRequest)?;
    Ok(parsed)
}

#[derive(Clone, Copy)]
enum RootMode {
    Speech,
    Batch,
}

struct RootSeed(RootMode);

impl<'de> DeserializeSeed<'de> for RootSeed {
    type Value = (SpeechFields, Vec<SpeechFields>);

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(RootVisitor(self.0))
    }
}

struct RootVisitor(RootMode);

impl<'de> Visitor<'de> for RootVisitor {
    type Value = (SpeechFields, Vec<SpeechFields>);

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a speech request object")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut fields = SpeechFields::default();
        let mut items = None;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "stream" => set_once(&mut fields.stream, map.next_value()?, "stream")?,
                "items" if matches!(self.0, RootMode::Batch) => {
                    set_once(&mut items, map.next_value_seed(ItemsSeed)?, "items")?
                }
                _ => {
                    if !read_speech_field(&key, &mut map, &mut fields)? {
                        let _ignored = map.next_value::<IgnoredAny>()?;
                    }
                }
            }
        }
        match self.0 {
            RootMode::Speech => Ok((fields, Vec::new())),
            RootMode::Batch => Ok((
                fields,
                items.ok_or_else(|| de::Error::missing_field("items"))?,
            )),
        }
    }
}

struct ItemsSeed;

impl<'de> DeserializeSeed<'de> for ItemsSeed {
    type Value = Vec<SpeechFields>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ItemsVisitor;
        impl<'de> Visitor<'de> for ItemsVisitor {
            type Value = Vec<SpeechFields>;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a speech batch item array")
            }
            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut result = Vec::new();
                while let Some(item) = sequence.next_element_seed(ItemSeed)? {
                    result.push(item);
                    if result.len() > usize::from(u16::MAX) {
                        return Err(de::Error::custom("too many batch items"));
                    }
                }
                Ok(result)
            }
        }
        deserializer.deserialize_seq(ItemsVisitor)
    }
}

struct ItemSeed;

impl<'de> DeserializeSeed<'de> for ItemSeed {
    type Value = SpeechFields;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ItemVisitor;
        impl<'de> Visitor<'de> for ItemVisitor {
            type Value = SpeechFields;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a speech batch item object")
            }
            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut fields = SpeechFields::default();
                while let Some(key) = map.next_key::<String>()? {
                    if !read_speech_field(&key, &mut map, &mut fields)? {
                        let _ignored = map.next_value::<IgnoredAny>()?;
                    }
                }
                Ok(fields)
            }
        }
        deserializer.deserialize_map(ItemVisitor)
    }
}

fn set_once<E, T>(slot: &mut Option<T>, value: T, field: &'static str) -> Result<(), E>
where
    E: de::Error,
{
    if slot.is_some() {
        return Err(E::duplicate_field(field));
    }
    *slot = Some(value);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::config::Config;
    use crate::worker_pool::{
        BatchFeature, ModelSelection, ProfileRequirement, ReferenceForm, SpeechResponseFormat,
        SpeechTask, StreamMode, TrustDomain, WorkerPool,
    };

    use super::{HttpFault, batch, merge_stream, speech, speech_with_hints};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn pool() -> WorkerPool {
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "sgl-omni-media-classify-{}-{sequence}.toml",
            std::process::id()
        ));
        let config = r#"
schema_version = 1
[server]
listen = "127.0.0.1:30000"
[shutdown]
drain_timeout_ms = 1000
[logging]
format = "json"
filter = "error"
[router]
strategy = "round_robin"
[admission]
global = 8
generation_http = 1
speech_http = 4
transcription_http = 4
speech_batch = 16
[health]
interval_ms = 100
timeout_ms = 50
success_threshold = 1
failure_threshold = 1
max_concurrent_probes = 1
[http_generation]
trust_domain = "local"
buffered_request_max_bytes = 1024
buffered_request_total_bytes = 4096
streamed_request_max_bytes = 8192
connect_timeout_ms = 100
request_timeout_ms = 1000
pool_idle_timeout_ms = 1000
pool_max_idle_per_host = 1
[[workers]]
worker_id = "worker"
base_url = "http://127.0.0.1:9"
trust_domain = "local"
default_model_id = "tts"
[workers.capacity]
generation_http = 1
speech_http = 4
speech_batch = 16
transcription_http = 4
[[workers.service_profiles]]
service = "generation_http"
model_ids = ["tts"]
message_content_forms = ["string"]
media_placements = []
input_modalities = ["text"]
output_modalities = ["text"]
chat_audio_formats = []
stream_modes = ["non_streaming"]
[[workers.service_profiles]]
service = "speech_http"
model_ids = ["tts", "other"]
response_formats = ["mp3", "opus", "aac", "flac", "wav"]
stream_modes = ["non_streaming"]
tasks = ["text_to_speech", "voice_clone", "voice_design"]
reference_forms = ["none", "direct", "list", "vq_codes"]
managed_voice = false
[[workers.service_profiles]]
service = "speech_http"
model_ids = ["tts", "other"]
response_formats = ["pcm"]
stream_modes = ["non_streaming", "streaming"]
tasks = ["text_to_speech", "voice_clone", "voice_design"]
reference_forms = ["none", "direct", "list", "vq_codes"]
managed_voice = false
[[workers.service_profiles]]
service = "speech_batch"
model_ids = ["tts", "other"]
response_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
tasks = ["text_to_speech", "voice_clone", "voice_design"]
reference_forms = ["none", "direct", "list", "vq_codes"]
managed_voice = false
max_batch_size = 16
effective_features = ["model", "format", "task", "reference", "voice"]
[[workers.service_profiles]]
service = "transcription_http"
model_ids = ["tts", "other"]
task = "transcribe"
response_formats = ["json", "text", "verbose_json", "srt", "vtt", "sse"]
media_profiles = ["audio", "audio_video"]
stream_modes = ["non_streaming", "streaming"]
"#;
        fs::write(&path, config).expect("write classifier config");
        let parsed = Config::load(&path).expect("load classifier config");
        let _removed = fs::remove_file(path);
        WorkerPool::build(&parsed).expect("build classifier pool")
    }

    #[test]
    fn speech_classifies_every_format_mode_task_and_mixed_reference_set() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        for (name, expected) in [
            ("mp3", SpeechResponseFormat::Mp3),
            ("opus", SpeechResponseFormat::Opus),
            ("aac", SpeechResponseFormat::Aac),
            ("flac", SpeechResponseFormat::Flac),
            ("wav", SpeechResponseFormat::Wav),
            ("pcm", SpeechResponseFormat::Pcm),
        ] {
            let body =
                format!("{{\"model\":\"tts\",\"input\":\"x\",\"response_format\":\"{name}\"}}");
            let classified = speech(body.as_bytes(), &pool, &trust).expect("classify format");
            let ProfileRequirement::SpeechHttp {
                response_format, ..
            } = classified.requirement.profile()
            else {
                panic!("speech requirement")
            };
            assert_eq!(*response_format, expected);
        }
        let mixed = br#"{"model":"tts","input":"x","response_format":"pcm","stream":true,"task_type":"CustomVoice","ref_audio":"direct","references":[{"audio_path":"list"},{"vq_codes":[1]}]}"#;
        let classified = speech(mixed, &pool, &trust).expect("classify mixed speech");
        let ProfileRequirement::SpeechHttp {
            stream_mode,
            task,
            reference_forms,
            managed_voice,
            ..
        } = classified.requirement.profile()
        else {
            panic!("speech requirement")
        };
        assert_eq!(*stream_mode, StreamMode::Streaming);
        assert_eq!(*task, SpeechTask::VoiceClone);
        assert_eq!(
            reference_forms,
            &[
                ReferenceForm::Direct,
                ReferenceForm::List,
                ReferenceForm::VqCodes,
            ]
        );
        assert!(!managed_voice);
        assert_eq!(
            speech(br#"{"model":"a","model":"b","input":"x"}"#, &pool, &trust).err(),
            Some(HttpFault::MalformedRequest)
        );
    }

    #[test]
    fn batch_builds_whole_ordered_effective_unions_and_features() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        let body = br#"{
            "model":"tts","response_format":"wav","task_type":"Base","voice":"default",
            "items":[
                {"input":"first"},
                {"input":"second","model":"other","response_format":"mp3","task_type":"VoiceDesign","ref_audio":"direct","voice":"named"},
                {"input":"third","references":[{"audio":"list"},{"vq_codes":[1]}],"voice":"managed"}
            ]
        }"#;
        let classified = batch(body, &pool, &trust).expect("classify complete batch");
        let ProfileRequirement::SpeechBatch {
            models,
            response_formats,
            tasks,
            reference_forms,
            managed_voice,
            batch_size,
            effective_features,
        } = classified.requirement.profile()
        else {
            panic!("batch requirement")
        };
        assert_eq!(*batch_size, 3);
        assert_eq!(models[0].model_id(), "tts");
        assert!(matches!(models[0], ModelSelection::Explicit(_)));
        assert_eq!(models[1].model_id(), "other");
        assert_eq!(models[2].model_id(), "tts");
        assert_eq!(
            response_formats,
            &[SpeechResponseFormat::Wav, SpeechResponseFormat::Mp3]
        );
        assert_eq!(tasks, &[SpeechTask::TextToSpeech, SpeechTask::VoiceDesign]);
        assert_eq!(
            reference_forms,
            &[
                ReferenceForm::None,
                ReferenceForm::Direct,
                ReferenceForm::List,
                ReferenceForm::VqCodes,
            ]
        );
        assert!(
            !managed_voice,
            "explicit references avoid managed voice routing"
        );
        assert_eq!(
            effective_features,
            &[
                BatchFeature::Model,
                BatchFeature::Format,
                BatchFeature::Task,
                BatchFeature::Reference,
                BatchFeature::Voice,
            ]
        );
    }

    #[test]
    fn batch_default_named_voice_is_required_before_item_reference_overrides() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        let body = br#"{
            "model":"tts",
            "voice":"named-default",
            "items":[
                {"input":"first","ref_audio":"direct"},
                {"input":"second","references":[{"audio":"list"}]}
            ]
        }"#;
        let classified = batch(body, &pool, &trust).expect("classify default managed voice");
        let ProfileRequirement::SpeechBatch { managed_voice, .. } =
            classified.requirement.profile()
        else {
            panic!("batch requirement")
        };
        assert!(*managed_voice);
    }

    #[test]
    fn batch_reference_overrides_follow_exclude_none_semantics() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        let inherited = batch(
            br#"{
                "model":"tts",
                "ref_audio":"default-direct",
                "references":[{"audio":"default-list"}],
                "items":[{"input":"first","ref_audio":null,"references":null}]
            }"#,
            &pool,
            &trust,
        )
        .expect("null item references inherit batch defaults");
        let ProfileRequirement::SpeechBatch {
            reference_forms,
            effective_features,
            ..
        } = inherited.requirement.profile()
        else {
            panic!("batch requirement")
        };
        assert_eq!(
            reference_forms,
            &[ReferenceForm::Direct, ReferenceForm::List]
        );
        assert!(!effective_features.contains(&BatchFeature::Reference));

        let suppressed = batch(
            br#"{
                "model":"tts",
                "references":[{"audio":"default-list"}],
                "items":[{"input":"first","references":[]}]
            }"#,
            &pool,
            &trust,
        )
        .expect("empty item reference list overrides batch default");
        let ProfileRequirement::SpeechBatch {
            reference_forms,
            effective_features,
            ..
        } = suppressed.requirement.profile()
        else {
            panic!("batch requirement")
        };
        assert_eq!(reference_forms, &[ReferenceForm::None]);
        assert!(effective_features.contains(&BatchFeature::Reference));
    }

    #[test]
    fn preserves_empty_whitespace_model_and_voice_semantics() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        for (body, defaulted, expected_model, managed) in [
            (
                br#"{"model":"","input":"x","voice":""}"#.as_slice(),
                true,
                "tts",
                false,
            ),
            (
                br#"{"model":" ","input":"x","voice":" default "}"#.as_slice(),
                false,
                " ",
                true,
            ),
            (
                br#"{"input":"x","voice":"DeFaUlT"}"#.as_slice(),
                true,
                "tts",
                false,
            ),
        ] {
            let classified = speech(body, &pool, &trust).expect("classify model and voice facts");
            let ProfileRequirement::SpeechHttp {
                model,
                managed_voice,
                ..
            } = classified.requirement.profile()
            else {
                panic!("speech requirement")
            };
            assert_eq!(model.model_id(), expected_model);
            assert_eq!(
                matches!(model, ModelSelection::WorkerDefault { .. }),
                defaulted
            );
            assert_eq!(*managed_voice, managed);
        }

        for (voice, expected) in [("", false), ("DEFAULT", false), (" default ", true)] {
            let body = format!(r#"{{"voice":"{voice}","items":[{{"input":"x"}}]}}"#);
            let classified =
                batch(body.as_bytes(), &pool, &trust).expect("classify batch managed voice fact");
            let ProfileRequirement::SpeechBatch { managed_voice, .. } =
                classified.requirement.profile()
            else {
                panic!("batch requirement")
            };
            assert_eq!(*managed_voice, expected);
        }
    }

    #[test]
    fn route_header_assertions_cannot_override_worker_semantics() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        let asserted =
            speech_with_hints(br#"{"input":"x"}"#, Some("tts"), Some(false), &pool, &trust)
                .expect("matching default assertions");
        let ProfileRequirement::SpeechHttp {
            model, stream_mode, ..
        } = asserted.requirement.profile()
        else {
            panic!("speech requirement")
        };
        assert!(matches!(model, ModelSelection::WorkerDefault { .. }));
        assert_eq!(*stream_mode, StreamMode::NonStreaming);

        let empty_asserted = speech_with_hints(
            br#"{"model":"","input":"x"}"#,
            Some("tts"),
            None,
            &pool,
            &trust,
        )
        .expect("empty model retains default semantics");
        let ProfileRequirement::SpeechHttp { model, .. } = empty_asserted.requirement.profile()
        else {
            panic!("speech requirement")
        };
        assert!(matches!(model, ModelSelection::WorkerDefault { .. }));

        let explicit = speech_with_hints(
            br#"{"model":"tts","input":"x","stream":true,"response_format":"pcm"}"#,
            Some("tts"),
            Some(true),
            &pool,
            &trust,
        )
        .expect("matching explicit assertions");
        let ProfileRequirement::SpeechHttp {
            model, stream_mode, ..
        } = explicit.requirement.profile()
        else {
            panic!("speech requirement")
        };
        assert!(matches!(model, ModelSelection::Explicit(_)));
        assert_eq!(*stream_mode, StreamMode::Streaming);

        assert_eq!(
            speech_with_hints(
                br#"{"model":"tts","input":"x"}"#,
                Some("other"),
                None,
                &pool,
                &trust,
            )
            .err(),
            Some(HttpFault::MalformedRequest)
        );
        assert_eq!(
            merge_stream(None, Some(true)),
            Err(HttpFault::MalformedRequest)
        );
        assert_eq!(merge_stream(None, Some(false)), Ok(false));
        assert_eq!(
            merge_stream(Some(false), Some(true)),
            Err(HttpFault::MalformedRequest)
        );
        assert_eq!(
            speech_with_hints(
                br#"{"model":"tts","input":"x","stream":true,"response_format":"wav"}"#,
                None,
                None,
                &pool,
                &trust,
            )
            .err(),
            Some(HttpFault::MalformedRequest)
        );
    }
}
