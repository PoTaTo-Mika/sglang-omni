use std::fmt;

use serde::de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};

use crate::error::HttpFault;
use crate::worker_pool::profile::{InputModality, OutputModality, StreamMode};
use crate::worker_pool::{
    ChatAudioFormat, DefaultModelResolution, MediaPlacement, MessageContentForm, ModelSelection,
    ProfileRequirement, RouteRequirement, ServiceClass, TrustDomain, WorkerPool,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExpectedSuccess {
    Json,
    Sse,
}

pub(crate) struct ClassifiedRequest {
    pub(crate) requirement: RouteRequirement,
    pub(crate) expected_success: ExpectedSuccess,
}

#[derive(Default)]
struct Facts {
    model: Option<Option<String>>,
    messages: Option<MessageFacts>,
    top_audio: bool,
    top_image: bool,
    top_video: bool,
    modalities: Option<Vec<OutputModality>>,
    audio_format: Option<Option<ChatAudioFormat>>,
    stream: Option<bool>,
}

#[derive(Default)]
struct MessageFacts {
    count: usize,
    string: bool,
    typed: bool,
    text: bool,
    audio: bool,
    image: bool,
    video: bool,
    typed_media: bool,
}

pub(crate) fn classify(
    bytes: &[u8],
    pool: &WorkerPool,
    trust: &TrustDomain,
) -> Result<ClassifiedRequest, HttpFault> {
    if exceeds_nesting_limit(bytes) {
        return Err(HttpFault::MalformedRequest);
    }
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let facts = RootSeed
        .deserialize(&mut deserializer)
        .map_err(|_| HttpFault::MalformedRequest)?;
    deserializer
        .end()
        .map_err(|_| HttpFault::MalformedRequest)?;

    let messages = facts.messages.ok_or(HttpFault::MalformedRequest)?;
    if messages.count == 0 {
        return Err(HttpFault::MalformedRequest);
    }
    let model = match facts.model.flatten() {
        Some(model) if model.is_empty() => return Err(HttpFault::MalformedRequest),
        Some(model) => ModelSelection::Explicit(model),
        None => match pool.resolve_default_model_id(trust, ServiceClass::GenerationHttp) {
            DefaultModelResolution::Unique(model) => ModelSelection::WorkerDefault {
                expected_model_id: model.to_owned(),
            },
            DefaultModelResolution::Ambiguous => return Err(HttpFault::AmbiguousModel),
            DefaultModelResolution::NoService => return Err(HttpFault::RouterUnavailable),
        },
    };

    let mut message_content_forms = Vec::with_capacity(2);
    if messages.string {
        message_content_forms.push(MessageContentForm::String);
    }
    if messages.typed {
        message_content_forms.push(MessageContentForm::TypedParts);
    }
    let top_media = facts.top_audio || facts.top_image || facts.top_video;
    let mut media_placements = Vec::with_capacity(2);
    if top_media {
        media_placements.push(MediaPlacement::TopLevel);
    }
    if messages.typed_media {
        media_placements.push(MediaPlacement::TypedParts);
    }
    let mut input_modalities = Vec::with_capacity(4);
    if messages.text {
        input_modalities.push(InputModality::Text);
    }
    if facts.top_audio || messages.audio {
        input_modalities.push(InputModality::Audio);
    }
    if facts.top_image || messages.image {
        input_modalities.push(InputModality::Image);
    }
    if facts.top_video || messages.video {
        input_modalities.push(InputModality::Video);
    }
    let output_modalities = facts
        .modalities
        .filter(|modalities| !modalities.is_empty())
        .unwrap_or_else(|| vec![OutputModality::Text]);
    let audio_requested = output_modalities.contains(&OutputModality::Audio);
    let audio_format = if audio_requested {
        Some(facts.audio_format.flatten().unwrap_or(ChatAudioFormat::Wav))
    } else {
        None
    };
    let stream_mode = if facts.stream.unwrap_or(false) {
        StreamMode::Streaming
    } else {
        StreamMode::NonStreaming
    };
    let expected_success = if stream_mode == StreamMode::Streaming {
        ExpectedSuccess::Sse
    } else {
        ExpectedSuccess::Json
    };
    Ok(ClassifiedRequest {
        requirement: RouteRequirement::new(
            ProfileRequirement::GenerationHttp {
                model,
                message_content_forms,
                media_placements,
                input_modalities,
                output_modalities,
                audio_format,
                stream_mode,
            },
            trust.clone(),
        ),
        expected_success,
    })
}

fn exceeds_nesting_limit(bytes: &[u8]) -> bool {
    let mut depth = 0_u16;
    let mut in_string = false;
    let mut escaped = false;
    for byte in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else if *byte == b'\\' {
                escaped = true;
            } else if *byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match *byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth = depth.saturating_add(1);
                if depth > 128 {
                    return true;
                }
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    false
}

struct RootSeed;

impl<'de> DeserializeSeed<'de> for RootSeed {
    type Value = Facts;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(RootVisitor)
    }
}

struct RootVisitor;

impl<'de> Visitor<'de> for RootVisitor {
    type Value = Facts;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a chat request object")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut facts = Facts::default();
        let mut seen_audios = false;
        let mut seen_images = false;
        let mut seen_videos = false;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "model" => set_once(&mut facts.model, map.next_value()?, "model")?,
                "messages" => set_once(
                    &mut facts.messages,
                    map.next_value_seed(MessagesSeed)?,
                    "messages",
                )?,
                "audios" => {
                    reject_seen(&mut seen_audios, "audios")?;
                    facts.top_audio = map.next_value_seed(NonEmptyArraySeed)?;
                }
                "images" => {
                    reject_seen(&mut seen_images, "images")?;
                    facts.top_image = map.next_value_seed(NonEmptyArraySeed)?;
                }
                "videos" => {
                    reject_seen(&mut seen_videos, "videos")?;
                    facts.top_video = map.next_value_seed(NonEmptyArraySeed)?;
                }
                "modalities" => set_once(
                    &mut facts.modalities,
                    map.next_value_seed(NullableModalitiesSeed)?,
                    "modalities",
                )?,
                "audio" => set_once(
                    &mut facts.audio_format,
                    map.next_value_seed(NullableAudioSeed)?,
                    "audio",
                )?,
                "stream" => set_once(&mut facts.stream, map.next_value()?, "stream")?,
                _ => {
                    let _ignored = map.next_value::<IgnoredAny>()?;
                }
            }
        }
        Ok(facts)
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

fn reject_seen<E>(seen: &mut bool, field: &'static str) -> Result<(), E>
where
    E: de::Error,
{
    if *seen {
        return Err(E::duplicate_field(field));
    }
    *seen = true;
    Ok(())
}

struct NonEmptyArraySeed;

impl<'de> DeserializeSeed<'de> for NonEmptyArraySeed {
    type Value = bool;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ArrayVisitor;
        impl<'de> Visitor<'de> for ArrayVisitor {
            type Value = bool;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an array")
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<bool, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut nonempty = false;
                while seq.next_element::<IgnoredAny>()?.is_some() {
                    nonempty = true;
                }
                Ok(nonempty)
            }
        }
        deserializer.deserialize_seq(ArrayVisitor)
    }
}

struct MessagesSeed;

impl<'de> DeserializeSeed<'de> for MessagesSeed {
    type Value = MessageFacts;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct MessagesVisitor;
        impl<'de> Visitor<'de> for MessagesVisitor {
            type Value = MessageFacts;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a nonempty message array")
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<MessageFacts, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut all = MessageFacts::default();
                while let Some(message) = seq.next_element_seed(MessageSeed)? {
                    all.count += 1;
                    all.string |= message.string;
                    all.typed |= message.typed;
                    all.text |= message.text;
                    all.audio |= message.audio;
                    all.image |= message.image;
                    all.video |= message.video;
                    all.typed_media |= message.typed_media;
                }
                Ok(all)
            }
        }
        deserializer.deserialize_seq(MessagesVisitor)
    }
}

struct MessageSeed;

impl<'de> DeserializeSeed<'de> for MessageSeed {
    type Value = MessageFacts;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct MessageVisitor;
        impl<'de> Visitor<'de> for MessageVisitor {
            type Value = MessageFacts;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a message object")
            }
            fn visit_map<A>(self, mut map: A) -> Result<MessageFacts, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut content = None;
                while let Some(key) = map.next_key::<String>()? {
                    if key == "content" {
                        set_once(&mut content, map.next_value_seed(ContentSeed)?, "content")?;
                    } else {
                        let _ignored = map.next_value::<IgnoredAny>()?;
                    }
                }
                content.ok_or_else(|| de::Error::missing_field("content"))
            }
        }
        deserializer.deserialize_map(MessageVisitor)
    }
}

struct ContentSeed;

impl<'de> DeserializeSeed<'de> for ContentSeed {
    type Value = MessageFacts;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ContentVisitor;
        impl<'de> Visitor<'de> for ContentVisitor {
            type Value = MessageFacts;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a content string or typed-part array")
            }
            fn visit_str<E>(self, _value: &str) -> Result<MessageFacts, E> {
                Ok(MessageFacts {
                    string: true,
                    text: true,
                    ..MessageFacts::default()
                })
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<MessageFacts, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut facts = MessageFacts {
                    typed: true,
                    ..MessageFacts::default()
                };
                while let Some(part) = seq.next_element_seed(PartSeed)? {
                    facts.text |= part.text;
                    facts.audio |= part.audio;
                    facts.image |= part.image;
                    facts.video |= part.video;
                    facts.typed_media |= part.audio || part.image || part.video;
                }
                Ok(facts)
            }
        }
        deserializer.deserialize_any(ContentVisitor)
    }
}

struct PartSeed;

impl<'de> DeserializeSeed<'de> for PartSeed {
    type Value = MessageFacts;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct PartVisitor;
        impl<'de> Visitor<'de> for PartVisitor {
            type Value = MessageFacts;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a typed content-part object")
            }
            fn visit_map<A>(self, mut map: A) -> Result<MessageFacts, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut part_type: Option<String> = None;
                while let Some(key) = map.next_key::<String>()? {
                    if key == "type" {
                        set_once(&mut part_type, map.next_value()?, "type")?;
                    } else {
                        let _ignored = map.next_value::<IgnoredAny>()?;
                    }
                }
                let part_type = part_type.ok_or_else(|| de::Error::missing_field("type"))?;
                let mut facts = MessageFacts::default();
                match part_type.as_str() {
                    "text" => facts.text = true,
                    "audio_url" | "input_audio" => facts.audio = true,
                    "image_url" | "image" => facts.image = true,
                    "video_url" | "video" => facts.video = true,
                    _ => {}
                }
                Ok(facts)
            }
        }
        deserializer.deserialize_map(PartVisitor)
    }
}

struct NullableModalitiesSeed;

impl<'de> DeserializeSeed<'de> for NullableModalitiesSeed {
    type Value = Vec<OutputModality>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ModalitiesVisitor;
        impl<'de> Visitor<'de> for ModalitiesVisitor {
            type Value = Vec<OutputModality>;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("null or an output-modality array")
            }
            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(Vec::new())
            }
            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(Vec::new())
            }
            fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                deserializer.deserialize_seq(self)
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut result = Vec::with_capacity(2);
                while let Some(value) = seq.next_element::<String>()? {
                    let modality = match value.as_str() {
                        "text" => OutputModality::Text,
                        "audio" => OutputModality::Audio,
                        _ => return Err(de::Error::custom("unknown output modality")),
                    };
                    if result.contains(&modality) {
                        return Err(de::Error::custom("duplicate output modality"));
                    }
                    result.push(modality);
                }
                Ok(result)
            }
        }
        deserializer.deserialize_option(ModalitiesVisitor)
    }
}

struct NullableAudioSeed;

impl<'de> DeserializeSeed<'de> for NullableAudioSeed {
    type Value = Option<ChatAudioFormat>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct AudioVisitor;
        impl<'de> Visitor<'de> for AudioVisitor {
            type Value = Option<ChatAudioFormat>;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("null or an audio configuration object")
            }
            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(None)
            }
            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(None)
            }
            fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                deserializer.deserialize_map(self)
            }
            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut format: Option<Option<String>> = None;
                while let Some(key) = map.next_key::<String>()? {
                    if key == "format" {
                        set_once(&mut format, map.next_value()?, "format")?;
                    } else {
                        let _ignored = map.next_value::<IgnoredAny>()?;
                    }
                }
                let Some(Some(format)) = format else {
                    return Ok(None);
                };
                let value = match format.as_str() {
                    "wav" => ChatAudioFormat::Wav,
                    "mp3" => ChatAudioFormat::Mp3,
                    "flac" => ChatAudioFormat::Flac,
                    "pcm" => ChatAudioFormat::Pcm,
                    "aac" => ChatAudioFormat::Aac,
                    "opus" => ChatAudioFormat::Opus,
                    _ => return Err(de::Error::custom("unknown audio format")),
                };
                Ok(Some(value))
            }
        }
        deserializer.deserialize_option(AudioVisitor)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::fs;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicU64, Ordering};

    use http_body::Body as _;

    use crate::config::Config;
    use crate::worker_pool::WorkerPool;

    use super::{ClassifiedRequest, ExpectedSuccess, HttpFault, TrustDomain, classify};

    static NEXT_CONFIG: AtomicU64 = AtomicU64::new(0);

    fn pool(extra_worker: &str) -> WorkerPool {
        let sequence = NEXT_CONFIG.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "sgl-omni-chat-classify-{}-{sequence}.toml",
            std::process::id()
        ));
        let text = format!(
            r#"schema_version = 1
[server]
listen = "127.0.0.1:30000"
[shutdown]
drain_timeout_ms = 30000
[logging]
format = "json"
filter = "info"
[router]
[admission]
global = 8
generation_http = 4
[health]
interval_ms = 100
timeout_ms = 50
success_threshold = 1
failure_threshold = 1
max_concurrent_probes = 1
[http_generation]
trust_domain = "local"
[[workers]]
worker_id = "worker-a"
base_url = "http://127.0.0.1:9"
trust_domain = "local"
default_model_id = "omni"
[workers.capacity]
generation_http = 4
[[workers.service_profiles]]
service = "generation_http"
model_ids = ["omni", "other"]
message_content_forms = ["string", "typed_parts"]
media_placements = ["top_level", "typed_parts"]
input_modalities = ["text", "audio", "image", "video"]
output_modalities = ["text", "audio"]
chat_audio_formats = ["wav", "mp3", "flac", "pcm", "aac", "opus"]
stream_modes = ["non_streaming", "streaming"]
{extra_worker}
"#
        );
        fs::write(&path, text).expect("write classifier config");
        let config = Config::load(&path).expect("load classifier config");
        let _removed = fs::remove_file(path);
        WorkerPool::build(&config).expect("build classifier pool")
    }

    fn classify_with(body: &str, pool: &WorkerPool) -> Result<ExpectedSuccess, HttpFault> {
        classify_full(body.as_bytes(), pool).map(|classified| classified.expected_success)
    }

    fn classify_full(bytes: &[u8], pool: &WorkerPool) -> Result<ClassifiedRequest, HttpFault> {
        classify(bytes, pool, &TrustDomain::new(String::from("local")))
    }

    fn permutations(fields: &mut [&str], index: usize, output: &mut Vec<String>) {
        if index == fields.len() {
            output.push(format!("{{{}}}", fields.join(",")));
            return;
        }
        for swap in index..fields.len() {
            fields.swap(index, swap);
            permutations(fields, index + 1, output);
            fields.swap(index, swap);
        }
    }

    fn next_xorshift(state: &mut u64) -> u64 {
        let mut value = *state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        *state = value;
        value
    }

    #[test]
    fn model_and_output_defaults_match_direct_omni() {
        let pool = pool("");
        for model in ["", r#","model":null"#, r#","model":"omni""#] {
            let body = format!(r#"{{"messages":[{{"content":"hello"}}]{model}}}"#);
            assert_eq!(classify_with(&body, &pool), Ok(ExpectedSuccess::Json));
        }
        for modalities in ["", r#","modalities":null"#, r#","modalities":[]"#] {
            let body = format!(r#"{{"messages":[{{"content":"hello"}}]{modalities}}}"#);
            assert!(classify_with(&body, &pool).is_ok());
        }
        assert_eq!(
            classify_with(
                r#"{"messages":[{"content":"hello"}],"modalities":["audio"],"audio":null,"stream":true}"#,
                &pool,
            ),
            Ok(ExpectedSuccess::Sse)
        );
    }

    #[test]
    fn mixed_message_and_media_forms_are_classified_without_field_order_dependence() {
        let pool = pool("");
        let requests = [
            r#"{"videos":[1],"messages":[{"content":"text"},{"content":[{"image_url":{},"type":"image_url"},{"type":"audio_url"},{"type":"video"},{"type":"unknown"}]}],"images":[1],"audios":[1],"model":"omni"}"#,
            r#"{"model":"omni","audios":[1],"images":[1],"messages":[{"content":"text"},{"content":[{"type":"image"},{"type":"input_audio"},{"type":"video_url"}]}],"videos":[1]}"#,
        ];
        for request in requests {
            assert!(classify_with(request, &pool).is_ok());
        }
        for format in ["wav", "mp3", "flac", "pcm", "aac", "opus"] {
            let body = format!(
                r#"{{"messages":[{{"content":"x"}}],"modalities":["text","audio"],"audio":{{"format":"{format}"}}}}"#
            );
            assert!(classify_with(&body, &pool).is_ok());
        }
    }

    #[test]
    fn only_routing_duplicates_and_invalid_routing_values_are_rejected() {
        let pool = pool("");
        let rejected = [
            r#"{"model":"omni","model":"omni","messages":[{"content":"x"}]}"#,
            r#"{"messages":[{"content":"x","content":"y"}]}"#,
            r#"{"messages":[{"content":[{"type":"text","type":"text"}]}]}"#,
            r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{"format":"wav","format":"wav"}}"#,
            r#"{"messages":[]}"#,
            r#"{"model":"","messages":[{"content":"x"}]}"#,
            r#"{"messages":[{"content":null}]}"#,
            r#"{"messages":[{"content":3}]}"#,
            r#"{"messages":[{"content":{}}]}"#,
            r#"{"messages":[{"content":["raw"]}]}"#,
            r#"{"messages":[{"content":"x"}],"modalities":["text","text"]}"#,
            r#"{"messages":[{"content":"x"}],"modalities":["video"]}"#,
            r#"{"messages":[{"content":"x"}],"stream":"true"}"#,
            r#"{"messages":[{"content":"x"}],"audios":{}}"#,
            r#"{"messages":[{"content":"x"}]} trailing"#,
        ];
        for request in rejected {
            assert_eq!(
                classify_with(request, &pool),
                Err(HttpFault::MalformedRequest)
            );
        }
        assert!(
            classify_with(
                r#"{"messages":[{"content":"x"}],"vendor":{"temperature":1,"temperature":2}}"#,
                &pool,
            )
            .is_ok()
        );
        assert!(
            classify_with(
                r#"{"messages":[{"content":[{"type":"vendor_future_part"}]}]}"#,
                &pool,
            )
            .is_ok()
        );
        let long_model = format!(
            r#"{{"model":"{}","messages":[{{"content":"x"}}]}}"#,
            "m".repeat(257)
        );
        assert!(
            classify_with(&long_model, &pool).is_ok(),
            "nonempty model IDs remain dispatch-owned even when no row will match"
        );
    }

    #[test]
    fn nesting_limit_and_ambiguous_default_fail_closed() {
        let base_pool = pool("");
        let nested = format!(
            "{{\"messages\":[{{\"content\":\"x\"}}],\"ignored\":{}0{}}}",
            "[".repeat(129),
            "]".repeat(129)
        );
        assert_eq!(
            classify_with(&nested, &base_pool),
            Err(HttpFault::MalformedRequest)
        );
        let second = r#"
[[workers]]
worker_id = "worker-b"
base_url = "http://127.0.0.1:10"
trust_domain = "local"
default_model_id = "other"
[workers.capacity]
generation_http = 4
[[workers.service_profiles]]
service = "generation_http"
model_ids = ["omni", "other"]
message_content_forms = ["string", "typed_parts"]
media_placements = ["top_level", "typed_parts"]
input_modalities = ["text", "audio", "image", "video"]
output_modalities = ["text", "audio"]
chat_audio_formats = ["wav", "mp3", "flac", "pcm", "aac", "opus"]
stream_modes = ["non_streaming", "streaming"]
"#;
        let ambiguous = pool(second);
        assert_eq!(
            classify_with(r#"{"messages":[{"content":"x"}]}"#, &ambiguous),
            Err(HttpFault::AmbiguousModel)
        );
        assert!(
            classify_with(
                r#"{"model":"omni","messages":[{"content":"x"}]}"#,
                &ambiguous,
            )
            .is_ok()
        );
    }

    #[test]
    fn all_root_field_permutations_and_buffer_chunk_replays_are_identical() {
        let pool = pool("");
        let mut fields = [
            r#""model":"omni""#,
            r#""messages":[{"content":"text"},{"content":[{"type":"input_audio"},{"type":"image_url"},{"type":"video"}]}]"#,
            r#""audios":[1]"#,
            r#""modalities":["text","audio"]"#,
            r#""audio":{"format":"opus"}"#,
            r#""stream":true"#,
        ];
        let mut requests = Vec::new();
        permutations(&mut fields, 0, &mut requests);
        assert_eq!(requests.len(), 720);

        let baseline = classify_full(requests[0].as_bytes(), &pool)
            .expect("baseline permutation must classify");
        for request in &requests[1..] {
            let classified = classify_full(request.as_bytes(), &pool)
                .expect("every root permutation must classify");
            assert_eq!(classified.requirement, baseline.requirement);
            assert_eq!(classified.expected_success, baseline.expected_success);
        }

        for width in [1, 2, 3, 7, 31, requests[0].len()] {
            let replay: Vec<u8> = requests[0]
                .as_bytes()
                .chunks(width)
                .flat_map(|chunk| chunk.iter().copied())
                .collect();
            let classified = classify_full(&replay, &pool).expect("chunk replay must classify");
            assert_eq!(classified.requirement, baseline.requirement);
            assert_eq!(classified.expected_success, baseline.expected_success);
        }
    }

    #[test]
    fn routing_duplicate_corpus_and_nesting_boundaries_replay_deterministically() {
        let pool = pool("");
        let duplicates = [
            r#"{"model":"omni","model":"other","messages":[{"content":"x"}]}"#,
            r#"{"messages":[{"content":"x"}],"messages":[{"content":"y"}]}"#,
            r#"{"messages":[{"content":"x"}],"audios":[],"audios":[1]}"#,
            r#"{"messages":[{"content":"x"}],"images":[],"images":[1]}"#,
            r#"{"messages":[{"content":"x"}],"videos":[],"videos":[1]}"#,
            r#"{"messages":[{"content":"x"}],"modalities":[],"modalities":["text"]}"#,
            r#"{"messages":[{"content":"x"}],"audio":null,"audio":{"format":"wav"}}"#,
            r#"{"messages":[{"content":"x"}],"stream":false,"stream":true}"#,
            r#"{"messages":[{"content":"x","content":"y"}]}"#,
            r#"{"messages":[{"content":[{"type":"text","type":"image"}]}]}"#,
            r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{"format":"wav","format":"opus"}}"#,
        ];
        for request in duplicates {
            for width in [1, 2, 5, request.len()] {
                let replay: Vec<u8> = request
                    .as_bytes()
                    .chunks(width)
                    .flat_map(|chunk| chunk.iter().copied())
                    .collect();
                assert_eq!(
                    classify_full(&replay, &pool).err(),
                    Some(HttpFault::MalformedRequest),
                    "duplicate replay width {width} was accepted: {request}"
                );
            }
        }

        for (nested_arrays, accepted) in [(126, true), (127, true), (128, false), (129, false)] {
            let request = format!(
                "{{\"messages\":[{{\"content\":\"x\"}}],\"ignored\":{}0{}}}",
                "[".repeat(nested_arrays),
                "]".repeat(nested_arrays)
            );
            assert_eq!(
                classify_full(request.as_bytes(), &pool).is_ok(),
                accepted,
                "unexpected result at root plus {nested_arrays} nested arrays"
            );
        }
    }

    #[test]
    fn typed_modality_and_default_equivalence_corpus_is_exact() {
        let pool = pool("");
        let equivalent_groups: &[&[&str]] = &[
            &[
                r#"{"messages":[{"content":"x"}]}"#,
                r#"{"messages":[{"content":"x"}],"model":null}"#,
            ],
            &[
                r#"{"messages":[{"content":"x"}]}"#,
                r#"{"messages":[{"content":"x"}],"modalities":null}"#,
                r#"{"messages":[{"content":"x"}],"modalities":[]}"#,
                r#"{"messages":[{"content":"x"}],"modalities":["text"]}"#,
            ],
            &[
                r#"{"messages":[{"content":"x"}],"modalities":["audio"]}"#,
                r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":null}"#,
                r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{}}"#,
                r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{"format":null}}"#,
                r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{"format":"wav"}}"#,
            ],
            &[
                r#"{"messages":[{"content":[{"type":"audio_url"}]}]}"#,
                r#"{"messages":[{"content":[{"type":"input_audio"}]}]}"#,
            ],
            &[
                r#"{"messages":[{"content":[{"type":"image_url"}]}]}"#,
                r#"{"messages":[{"content":[{"type":"image"}]}]}"#,
            ],
            &[
                r#"{"messages":[{"content":[{"type":"video_url"}]}]}"#,
                r#"{"messages":[{"content":[{"type":"video"}]}]}"#,
            ],
            &[
                r#"{"messages":[{"content":[]}]}"#,
                r#"{"messages":[{"content":[{"type":"vendor_future_part"}]}]}"#,
                r#"{"messages":[{"content":[{"type":"unknown_a"},{"type":"unknown_b"}]}]}"#,
            ],
            &[
                r#"{"messages":[{"content":"x"}]}"#,
                r#"{"messages":[{"content":"x"}],"stream":false}"#,
            ],
        ];

        for group in equivalent_groups {
            let baseline = classify_full(group[0].as_bytes(), &pool)
                .expect("equivalence baseline must classify");
            for request in &group[1..] {
                let classified = classify_full(request.as_bytes(), &pool)
                    .expect("equivalent request must classify");
                assert_eq!(classified.requirement, baseline.requirement, "{request}");
                assert_eq!(
                    classified.expected_success, baseline.expected_success,
                    "{request}"
                );
            }
        }

        let defaulted = classify_full(br#"{"messages":[{"content":"x"}]}"#, &pool)
            .expect("defaulted model request");
        let explicit = classify_full(br#"{"model":"omni","messages":[{"content":"x"}]}"#, &pool)
            .expect("explicit model request");
        assert_ne!(
            explicit.requirement, defaulted.requirement,
            "equal effective IDs retain explicit/default routing provenance"
        );

        let streaming = classify_full(br#"{"messages":[{"content":"x"}],"stream":true}"#, &pool)
            .expect("streaming request");
        assert_eq!(streaming.expected_success, ExpectedSuccess::Sse);

        for request in [
            r#"{"model":1,"messages":[{"content":"x"}]}"#,
            r#"{"messages":"x"}"#,
            r#"{"messages":[{"content":"x"}],"modalities":"text"}"#,
            r#"{"messages":[{"content":"x"}],"audio":"wav"}"#,
            r#"{"messages":[{"content":"x"}],"stream":1}"#,
            r#"{"messages":[{"content":[{"type":null}]}]}"#,
        ] {
            assert_eq!(
                classify_with(request, &pool),
                Err(HttpFault::MalformedRequest),
                "wrong routing type was accepted: {request}"
            );
        }
    }

    #[test]
    fn fixed_seed_adversarial_mutation_replay_is_panic_free_and_deterministic() {
        const SEED: u64 = 0x4f4d_4e49_524f_5554;
        const MAX_MUTATED_BYTES: usize = 1024;
        const ADVERSARIAL_BYTES: &[u8] = &[
            b'{', b'}', b'[', b']', b'"', b'\\', b':', b',', 0, 0x1f, 0x7f, 0xff, b'n', b't', b'-',
        ];
        let corpus: &[&[u8]] = &[
            br#"{"messages":[{"content":"x"}]}"#,
            br#"{"model":"omni","model":"other","messages":[{"content":"x"}]}"#,
            br#"{"messages":[{"content":[{"type":"input_audio"},{"type":"image_url"},{"type":"video"}]}],"modalities":["text","audio"],"audio":{"format":"opus"},"stream":true}"#,
            br#"{"messages":[{"content":"x","content":"y"}],"ignored":{"vendor":[1,2,3]}}"#,
            br#"{"messages":[{"content":"x"}],"modalities":null,"audio":null,"stream":false}"#,
            br#"{"messages":[]}"#,
            br#"{"messages":[{"content":"x"}]} trailing"#,
            b"{\"messages\":[{\"content\":\"unterminated",
            b"\xff\x00[{\"messages\":true}]",
        ];
        let pool = pool("");
        let mut state = SEED;

        for case in 0..2048 {
            let corpus_index = next_xorshift(&mut state) as usize % corpus.len();
            let mut bytes = corpus[corpus_index].to_vec();
            let mutation_count = 1 + next_xorshift(&mut state) as usize % 12;
            for _ in 0..mutation_count {
                let operation = next_xorshift(&mut state) % 6;
                let random_byte =
                    ADVERSARIAL_BYTES[next_xorshift(&mut state) as usize % ADVERSARIAL_BYTES.len()];
                let index = if bytes.is_empty() {
                    0
                } else {
                    next_xorshift(&mut state) as usize % bytes.len()
                };
                match operation {
                    0 if !bytes.is_empty() => {
                        bytes[index] ^= 1 << (next_xorshift(&mut state) % 8);
                    }
                    1 if bytes.len() < MAX_MUTATED_BYTES => {
                        let insertion = if bytes.is_empty() {
                            0
                        } else {
                            next_xorshift(&mut state) as usize % (bytes.len() + 1)
                        };
                        bytes.insert(insertion, random_byte);
                    }
                    2 if !bytes.is_empty() => {
                        bytes.remove(index);
                    }
                    3 if !bytes.is_empty() => bytes[index] = random_byte,
                    4 if !bytes.is_empty() && bytes.len() < MAX_MUTATED_BYTES => {
                        let remaining = MAX_MUTATED_BYTES - bytes.len();
                        let width = (1 + next_xorshift(&mut state) as usize % 16)
                            .min(bytes.len() - index)
                            .min(remaining);
                        let duplicate = bytes[index..index + width].to_vec();
                        bytes.extend_from_slice(&duplicate);
                    }
                    _ if bytes.len() < MAX_MUTATED_BYTES => bytes.push(random_byte),
                    _ => {
                        bytes.truncate(index);
                    }
                }
            }

            let first = catch_unwind(AssertUnwindSafe(|| {
                classify_full(&bytes, &pool)
                    .map(|classified| (classified.requirement, classified.expected_success))
            }));
            assert!(
                first.is_ok(),
                "classifier panicked for seed=0x{SEED:016x} case={case} bytes={bytes:?}"
            );
            let first = first.expect("panic result checked above");
            let second = catch_unwind(AssertUnwindSafe(|| {
                classify_full(&bytes, &pool)
                    .map(|classified| (classified.requirement, classified.expected_success))
            }));
            assert!(
                second.is_ok(),
                "classifier replay panicked for seed=0x{SEED:016x} case={case} bytes={bytes:?}"
            );
            assert_eq!(
                first,
                second.expect("panic result checked above"),
                "classifier was nondeterministic for seed=0x{SEED:016x} case={case} bytes={bytes:?}"
            );
        }
    }

    #[test]
    fn fixed_error_envelopes_are_bounded_and_canonical() {
        let faults = [
            HttpFault::MalformedRequest,
            HttpFault::AmbiguousModel,
            HttpFault::RequestTimeout,
            HttpFault::RequestBodyTooLarge,
            HttpFault::UnsupportedMediaType,
            HttpFault::UnsupportedContentEncoding,
            HttpFault::ExpectationFailed,
            HttpFault::NoCompatibleWorker,
            HttpFault::RouterOverloaded,
            HttpFault::InternalError,
            HttpFault::UpstreamProtocolError,
            HttpFault::RouterUnavailable,
            HttpFault::UpstreamTimeout,
            HttpFault::HttpVersionNotSupported,
        ];
        for fault in faults {
            let response = fault.into_response();
            let exact = response
                .body()
                .size_hint()
                .exact()
                .expect("fixed router envelope has an exact body length");
            assert_eq!(
                response.headers().get("cache-control"),
                Some(&axum::http::HeaderValue::from_static("no-store"))
            );
            assert!(exact <= 256);
            assert_eq!(
                response
                    .headers()
                    .get("content-length")
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.parse::<u64>().ok()),
                Some(exact)
            );
        }
    }
}
