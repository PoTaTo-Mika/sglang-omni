use axum::http::header::{
    CACHE_CONTROL, CONTENT_DISPOSITION, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE, EXPECT,
    EXPIRES, HeaderName, HeaderValue, TRAILER, TRANSFER_ENCODING, VARY,
};
use axum::http::{HeaderMap, StatusCode};

use crate::error::HttpFault;
use crate::http_generation::headers::{
    connection_tokens, is_request_media_type, parse_content_length, valid_generic_content_type,
};
use crate::worker_pool::{SpeechResponseFormat, StreamMode};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RequestKind {
    Json,
    Multipart,
}

pub(super) struct RequestFraming {
    pub(super) content_length: Option<u64>,
    pub(super) transfer_framed: bool,
    pub(super) has_route_hint: bool,
    pub(super) has_content_encoding: bool,
    pub(super) content_type: HeaderValue,
    pub(super) boundary: Option<Vec<u8>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SuccessProfile {
    Speech(SpeechResponseFormat, StreamMode),
    Json,
    TranscriptionJson,
    Text,
    Sse,
    OpaqueSpeech,
    OpaqueTranscription,
}

pub(super) fn validate_request(
    headers: &HeaderMap,
    kind: RequestKind,
) -> Result<RequestFraming, HttpFault> {
    let content_types: Vec<_> = headers.get_all(CONTENT_TYPE).iter().collect();
    if content_types.len() != 1 {
        return Err(HttpFault::UnsupportedMediaType);
    }
    let content_type = content_types[0].clone();
    let text = content_type
        .to_str()
        .map_err(|_| HttpFault::UnsupportedMediaType)?;
    let boundary = match kind {
        RequestKind::Json => {
            if !is_request_media_type(text, "application/json") {
                return Err(HttpFault::UnsupportedMediaType);
            }
            None
        }
        RequestKind::Multipart => Some(
            multipart_boundary(text)
                .ok_or(HttpFault::UnsupportedMediaType)?
                .into_bytes(),
        ),
    };
    let encodings: Vec<_> = headers.get_all(CONTENT_ENCODING).iter().collect();
    if encodings.len() > 1
        || encodings.first().is_some_and(|value| {
            !value
                .to_str()
                .is_ok_and(|text| text.eq_ignore_ascii_case("identity"))
        })
    {
        return Err(HttpFault::UnsupportedContentEncoding);
    }
    if headers.contains_key(EXPECT) {
        return Err(HttpFault::ExpectationFailed);
    }
    if headers.contains_key(TRAILER) {
        return Err(HttpFault::MalformedRequest);
    }
    let transfer_framed = headers.contains_key(TRANSFER_ENCODING);
    let lengths: Vec<_> = headers.get_all(CONTENT_LENGTH).iter().collect();
    if lengths.len() > 1 || (transfer_framed && !lengths.is_empty()) {
        return Err(HttpFault::MalformedRequest);
    }
    let content_length = lengths
        .first()
        .map(|value| parse_content_length(value).ok_or(HttpFault::MalformedRequest))
        .transpose()?;
    let has_route_hint = headers.contains_key("x-smg-target-worker")
        || headers.contains_key("x-smg-routing-key")
        || headers.contains_key("x-sgl-decode-url");
    Ok(RequestFraming {
        content_length,
        transfer_framed,
        has_route_hint,
        has_content_encoding: !encodings.is_empty(),
        content_type,
        boundary,
    })
}

pub(super) fn validate_bodyless_request(headers: &HeaderMap) -> Result<(), HttpFault> {
    let encodings: Vec<_> = headers.get_all(CONTENT_ENCODING).iter().collect();
    if encodings.len() > 1
        || encodings.first().is_some_and(|value| {
            !value
                .to_str()
                .is_ok_and(|text| text.eq_ignore_ascii_case("identity"))
        })
    {
        return Err(HttpFault::UnsupportedContentEncoding);
    }
    if headers.contains_key(EXPECT) {
        return Err(HttpFault::ExpectationFailed);
    }
    if headers.contains_key(TRAILER) || headers.contains_key(TRANSFER_ENCODING) {
        return Err(HttpFault::MalformedRequest);
    }
    let lengths: Vec<_> = headers.get_all(CONTENT_LENGTH).iter().collect();
    if lengths.len() > 1
        || lengths
            .first()
            .is_some_and(|value| parse_content_length(value) != Some(0))
    {
        return Err(HttpFault::MalformedRequest);
    }
    Ok(())
}

pub(super) fn sanitize_response(
    status: StatusCode,
    source: &HeaderMap,
    profile: SuccessProfile,
) -> Result<HeaderMap, HttpFault> {
    let connection = connection_tokens(source)?;
    let mut preserve_pcm_metadata = false;
    if !(status.is_success() || status.is_client_error() || status.is_server_error()) {
        return Err(HttpFault::UpstreamProtocolError);
    }
    if source.contains_key(CONTENT_ENCODING) && connection.contains(CONTENT_ENCODING.as_str()) {
        return Err(HttpFault::UpstreamProtocolError);
    }
    let types: Vec<_> = source.get_all(CONTENT_TYPE).iter().collect();
    let lengths: Vec<_> = source.get_all(CONTENT_LENGTH).iter().collect();
    if types.len() > 1 || lengths.len() > 1 {
        return Err(HttpFault::UpstreamProtocolError);
    }
    if let Some(value) = types.first()
        && !value.to_str().is_ok_and(valid_generic_content_type)
    {
        return Err(HttpFault::UpstreamProtocolError);
    }
    if let Some(value) = lengths.first()
        && parse_content_length(value).is_none()
    {
        return Err(HttpFault::UpstreamProtocolError);
    }
    if status.is_success() {
        let content_type = types
            .first()
            .filter(|_| !connection.contains(CONTENT_TYPE.as_str()))
            .and_then(|value| value.to_str().ok())
            .ok_or(HttpFault::UpstreamProtocolError)?;
        if !valid_success_type(content_type, profile) {
            return Err(HttpFault::UpstreamProtocolError);
        }
        if let Some(required) = pcm_metadata_required(content_type, profile) {
            preserve_pcm_metadata = validate_pcm_metadata(source, &connection, required)?;
        }
    }

    let mut result = HeaderMap::new();
    copy_one(source, &connection, &mut result, CONTENT_TYPE);
    copy_one(source, &connection, &mut result, CONTENT_LENGTH);
    for name in [CONTENT_ENCODING, CACHE_CONTROL, EXPIRES, VARY] {
        copy_all(source, &connection, &mut result, name);
    }
    match profile {
        SuccessProfile::Speech(_, _) | SuccessProfile::OpaqueSpeech => {
            for name in [
                CONTENT_DISPOSITION,
                HeaderName::from_static("x-prompt-tokens"),
                HeaderName::from_static("x-completion-tokens"),
                HeaderName::from_static("x-engine-time"),
            ] {
                copy_one(source, &connection, &mut result, name);
            }
            if preserve_pcm_metadata {
                for name in [
                    HeaderName::from_static("x-sample-rate"),
                    HeaderName::from_static("x-channels"),
                    HeaderName::from_static("x-bit-depth"),
                ] {
                    copy_one(source, &connection, &mut result, name);
                }
            }
        }
        SuccessProfile::Json => {}
        SuccessProfile::TranscriptionJson
        | SuccessProfile::Text
        | SuccessProfile::Sse
        | SuccessProfile::OpaqueTranscription => {
            copy_one(
                source,
                &connection,
                &mut result,
                HeaderName::from_static("x-request-id"),
            );
        }
    }
    Ok(result)
}

fn valid_success_type(value: &str, profile: SuccessProfile) -> bool {
    let media = value.split(';').next().map(str::trim).unwrap_or_default();
    match profile {
        SuccessProfile::Speech(format, _) => media.eq_ignore_ascii_case(match format {
            SpeechResponseFormat::Mp3 => "audio/mpeg",
            SpeechResponseFormat::Opus => "audio/opus",
            SpeechResponseFormat::Aac => "audio/aac",
            SpeechResponseFormat::Flac => "audio/flac",
            SpeechResponseFormat::Wav => "audio/wav",
            SpeechResponseFormat::Pcm => "audio/pcm",
        }),
        SuccessProfile::Json | SuccessProfile::TranscriptionJson => {
            media.eq_ignore_ascii_case("application/json")
        }
        SuccessProfile::Text => media.eq_ignore_ascii_case("text/plain"),
        SuccessProfile::Sse => media.eq_ignore_ascii_case("text/event-stream"),
        SuccessProfile::OpaqueSpeech => [
            "audio/mpeg",
            "audio/opus",
            "audio/aac",
            "audio/flac",
            "audio/wav",
            "audio/pcm",
        ]
        .iter()
        .any(|expected| media.eq_ignore_ascii_case(expected)),
        SuccessProfile::OpaqueTranscription => {
            media.eq_ignore_ascii_case("application/json")
                || media.eq_ignore_ascii_case("text/plain")
                || media.eq_ignore_ascii_case("text/event-stream")
        }
    }
}

fn pcm_metadata_required(content_type: &str, profile: SuccessProfile) -> Option<bool> {
    match profile {
        SuccessProfile::Speech(SpeechResponseFormat::Pcm, StreamMode::Streaming) => Some(true),
        SuccessProfile::Speech(SpeechResponseFormat::Pcm, StreamMode::NonStreaming) => Some(false),
        SuccessProfile::OpaqueSpeech
            if content_type
                .split(';')
                .next()
                .is_some_and(|media| media.trim().eq_ignore_ascii_case("audio/pcm")) =>
        {
            Some(false)
        }
        _ => None,
    }
}

fn validate_pcm_metadata(
    source: &HeaderMap,
    connection: &std::collections::HashSet<String>,
    required: bool,
) -> Result<bool, HttpFault> {
    let names = [
        HeaderName::from_static("x-sample-rate"),
        HeaderName::from_static("x-channels"),
        HeaderName::from_static("x-bit-depth"),
    ];
    let any_present = names.iter().any(|name| source.contains_key(name));
    if !required && !any_present {
        return Ok(false);
    }
    for name in names {
        if connection.contains(name.as_str()) {
            return Err(HttpFault::UpstreamProtocolError);
        }
        let values = source.get_all(&name);
        let mut values = values.iter();
        let value = values.next().ok_or(HttpFault::UpstreamProtocolError)?;
        if values.next().is_some() {
            return Err(HttpFault::UpstreamProtocolError);
        }
        let value = value
            .to_str()
            .map_err(|_| HttpFault::UpstreamProtocolError)?;
        if value.is_empty()
            || !value.bytes().all(|byte| byte.is_ascii_digit())
            || !value.parse::<u64>().is_ok_and(|value| value > 0)
        {
            return Err(HttpFault::UpstreamProtocolError);
        }
    }
    Ok(true)
}

fn copy_one(
    source: &HeaderMap,
    connection: &std::collections::HashSet<String>,
    destination: &mut HeaderMap,
    name: HeaderName,
) {
    if !connection.contains(name.as_str())
        && let Some(value) = source.get(&name)
    {
        destination.insert(name, value.clone());
    }
}

fn copy_all(
    source: &HeaderMap,
    connection: &std::collections::HashSet<String>,
    destination: &mut HeaderMap,
    name: HeaderName,
) {
    if connection.contains(name.as_str()) {
        return;
    }
    for value in source.get_all(&name) {
        destination.append(name.clone(), value.clone());
    }
}

fn multipart_boundary(value: &str) -> Option<String> {
    let mut parts = value.split(';');
    if !parts
        .next()?
        .trim()
        .eq_ignore_ascii_case("multipart/form-data")
    {
        return None;
    }
    let mut boundary = None;
    for parameter in parts {
        let (name, raw) = parameter.trim().split_once('=')?;
        if !name.trim().eq_ignore_ascii_case("boundary") || boundary.is_some() {
            return None;
        }
        let raw = raw.trim();
        let quoted = raw.starts_with('"');
        let parsed = if quoted {
            raw.strip_prefix('"')?.strip_suffix('"')?
        } else {
            if !raw.bytes().all(is_boundary_byte) {
                return None;
            }
            raw
        };
        if parsed.is_empty()
            || parsed.len() > 70
            || !parsed.is_ascii()
            || parsed.as_bytes().last() == Some(&b' ')
            || !parsed
                .bytes()
                .all(|byte| is_boundary_byte(byte) || quoted && byte == b' ')
        {
            return None;
        }
        boundary = Some(parsed.to_owned());
    }
    boundary
}

fn is_boundary_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || b"'()+_,-./:=?".contains(&byte)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use axum::http::header::{CONNECTION, CONTENT_LENGTH, CONTENT_TYPE};
    use axum::http::{HeaderMap, HeaderValue, StatusCode};

    use super::{
        HttpFault, RequestKind, SpeechResponseFormat, StreamMode, SuccessProfile,
        sanitize_response, validate_request,
    };

    fn json_headers(content_type: &[u8]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_bytes(content_type).expect("representable content type fixture"),
        );
        headers.insert(CONTENT_LENGTH, HeaderValue::from_static("2"));
        headers
    }

    fn pcm_headers(include_metadata: bool) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("audio/pcm"));
        headers.insert(CONTENT_LENGTH, HeaderValue::from_static("4"));
        if include_metadata {
            headers.insert("x-sample-rate", HeaderValue::from_static("24000"));
            headers.insert("x-channels", HeaderValue::from_static("1"));
            headers.insert("x-bit-depth", HeaderValue::from_static("16"));
        }
        headers
    }

    fn assert_optional_pcm_metadata(profile: SuccessProfile) {
        let absent = sanitize_response(StatusCode::OK, &pcm_headers(false), profile)
            .expect("optional PCM metadata may be absent");
        assert!(!absent.contains_key("x-sample-rate"));
        assert!(!absent.contains_key("x-channels"));
        assert!(!absent.contains_key("x-bit-depth"));

        let complete_source = pcm_headers(true);
        let complete = sanitize_response(StatusCode::OK, &complete_source, profile)
            .expect("complete optional PCM metadata");
        assert_eq!(
            complete.get("x-sample-rate"),
            complete_source.get("x-sample-rate")
        );
        assert_eq!(
            complete.get("x-channels"),
            complete_source.get("x-channels")
        );
        assert_eq!(
            complete.get("x-bit-depth"),
            complete_source.get("x-bit-depth")
        );

        let mut partial = pcm_headers(false);
        partial.insert("x-sample-rate", HeaderValue::from_static("24000"));
        assert_eq!(
            sanitize_response(StatusCode::OK, &partial, profile),
            Err(HttpFault::UpstreamProtocolError)
        );
    }

    #[test]
    fn json_content_type_uses_the_strict_shared_request_parser() {
        for value in [
            b"application/json".as_slice(),
            b"application/json; charset=utf-8".as_slice(),
            b"application/json; charset=\"UTF-8\"".as_slice(),
        ] {
            validate_request(&json_headers(value), RequestKind::Json)
                .expect("valid JSON content type");
        }

        for value in [
            b"application/json; charset=\"utf-8".as_slice(),
            b"application/json;".as_slice(),
            b"application/json; charset=utf-8; charset=utf-8".as_slice(),
            b"application/json; version=1".as_slice(),
            b"application/json; charset=\"utf-8\"junk".as_slice(),
        ] {
            assert_eq!(
                validate_request(&json_headers(value), RequestKind::Json).err(),
                Some(HttpFault::UnsupportedMediaType),
                "malformed JSON content type {value:?}"
            );
        }
    }

    #[test]
    fn accepts_quoted_and_unquoted_multipart_boundaries() {
        for (value, expected) in [
            ("multipart/form-data; boundary=abc-123", "abc-123"),
            ("multipart/form-data; boundary=\"abc-123\"", "abc-123"),
            ("multipart/form-data; boundary=\"abc def\"", "abc def"),
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static(value));
            headers.insert(CONTENT_LENGTH, HeaderValue::from_static("10"));
            let framing = validate_request(&headers, RequestKind::Multipart)
                .expect("valid multipart request headers");
            assert_eq!(framing.boundary.as_deref(), Some(expected.as_bytes()));
        }

        for value in [
            "multipart/form-data; boundary=abc def",
            "multipart/form-data; boundary=\"abc def \"",
            "multipart/form-data; boundary=\"abc[def\"",
            "multipart/form-data; boundary=\"abc\\def\"",
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static(value));
            headers.insert(CONTENT_LENGTH, HeaderValue::from_static("10"));
            assert_eq!(
                validate_request(&headers, RequestKind::Multipart).err(),
                Some(HttpFault::UnsupportedMediaType)
            );
        }
        let mut control = HeaderMap::new();
        control.insert(
            CONTENT_TYPE,
            HeaderValue::from_bytes(b"multipart/form-data; boundary=\"abc\tdef\"")
                .expect("horizontal tab is representable in a field value"),
        );
        control.insert(CONTENT_LENGTH, HeaderValue::from_static("10"));
        assert_eq!(
            validate_request(&control, RequestKind::Multipart).err(),
            Some(HttpFault::UnsupportedMediaType)
        );
    }

    #[test]
    fn rejects_redirects_and_wrong_route_specific_success_types() {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        assert_eq!(
            sanitize_response(StatusCode::FOUND, &headers, SuccessProfile::Json),
            Err(HttpFault::UpstreamProtocolError)
        );
        assert_eq!(
            sanitize_response(StatusCode::OK, &headers, SuccessProfile::Text),
            Err(HttpFault::UpstreamProtocolError)
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("audio/x-private"));
        assert_eq!(
            sanitize_response(StatusCode::OK, &headers, SuccessProfile::OpaqueSpeech),
            Err(HttpFault::UpstreamProtocolError)
        );
    }

    #[test]
    fn opaque_speech_accepts_only_supported_media_types() {
        for content_type in [
            "audio/mpeg",
            "audio/opus",
            "audio/aac",
            "audio/flac",
            "audio/wav",
            "audio/pcm",
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(
                CONTENT_TYPE,
                HeaderValue::from_str(content_type).expect("static media type"),
            );
            sanitize_response(StatusCode::OK, &headers, SuccessProfile::OpaqueSpeech)
                .expect("supported opaque speech type");
        }
    }

    #[test]
    fn known_streaming_pcm_requires_complete_positive_unique_metadata() {
        let profile = SuccessProfile::Speech(SpeechResponseFormat::Pcm, StreamMode::Streaming);
        let complete_source = pcm_headers(true);
        let complete = sanitize_response(StatusCode::OK, &complete_source, profile)
            .expect("streaming PCM complete metadata");
        assert_eq!(
            complete.get("x-sample-rate"),
            complete_source.get("x-sample-rate")
        );

        assert_eq!(
            sanitize_response(StatusCode::OK, &pcm_headers(false), profile),
            Err(HttpFault::UpstreamProtocolError)
        );
        let mut partial = pcm_headers(true);
        partial.remove("x-channels");
        assert_eq!(
            sanitize_response(StatusCode::OK, &partial, profile),
            Err(HttpFault::UpstreamProtocolError)
        );
        for malformed in ["", "0", "-1", "+1", "1.0", " 1"] {
            let mut headers = pcm_headers(true);
            headers.insert(
                "x-bit-depth",
                HeaderValue::from_str(malformed).expect("valid header syntax"),
            );
            assert_eq!(
                sanitize_response(StatusCode::OK, &headers, profile),
                Err(HttpFault::UpstreamProtocolError),
                "malformed PCM value {malformed:?}"
            );
        }
        let mut duplicate = pcm_headers(true);
        duplicate.append("x-sample-rate", HeaderValue::from_static("48000"));
        assert_eq!(
            sanitize_response(StatusCode::OK, &duplicate, profile),
            Err(HttpFault::UpstreamProtocolError)
        );
        let mut nominated = pcm_headers(true);
        nominated.insert(CONNECTION, HeaderValue::from_static("x-sample-rate"));
        assert_eq!(
            sanitize_response(StatusCode::OK, &nominated, profile),
            Err(HttpFault::UpstreamProtocolError)
        );
    }

    #[test]
    fn known_non_streaming_pcm_metadata_is_optional_but_all_or_none() {
        assert_optional_pcm_metadata(SuccessProfile::Speech(
            SpeechResponseFormat::Pcm,
            StreamMode::NonStreaming,
        ));
    }

    #[test]
    fn opaque_pcm_metadata_is_optional_but_all_or_none() {
        assert_optional_pcm_metadata(SuccessProfile::OpaqueSpeech);
    }

    #[test]
    fn opus_mime_is_exact_and_non_pcm_drops_pcm_metadata() {
        for profile in [
            SuccessProfile::Speech(SpeechResponseFormat::Opus, StreamMode::NonStreaming),
            SuccessProfile::OpaqueSpeech,
        ] {
            let mut exact = pcm_headers(true);
            exact.insert(CONTENT_TYPE, HeaderValue::from_static("audio/opus"));
            let sanitized = sanitize_response(StatusCode::OK, &exact, profile)
                .expect("canonical Opus media type");
            assert!(!sanitized.contains_key("x-sample-rate"));
            assert!(!sanitized.contains_key("x-channels"));
            assert!(!sanitized.contains_key("x-bit-depth"));

            exact.insert(CONTENT_TYPE, HeaderValue::from_static("audio/ogg"));
            assert_eq!(
                sanitize_response(StatusCode::OK, &exact, profile),
                Err(HttpFault::UpstreamProtocolError)
            );
        }
    }
}
