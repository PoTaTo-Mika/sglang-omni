use std::collections::HashSet;

use axum::http::header::{
    CONNECTION, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE, EXPECT, TRAILER, TRANSFER_ENCODING,
};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};

use crate::error::HttpFault;

use super::classify::ExpectedSuccess;

pub(crate) struct RequestFraming {
    pub(crate) content_length: u64,
}

pub(crate) fn validate_request(headers: &HeaderMap) -> Result<RequestFraming, HttpFault> {
    let mut content_types = headers.get_all(CONTENT_TYPE).iter();
    let content_type = content_types.next();
    if content_types.next().is_some()
        || !content_type
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| is_request_media_type(value, "application/json"))
    {
        return Err(HttpFault::UnsupportedMediaType);
    }
    if headers.contains_key(CONTENT_ENCODING) {
        return Err(HttpFault::UnsupportedContentEncoding);
    }
    if headers.contains_key(EXPECT) {
        return Err(HttpFault::ExpectationFailed);
    }
    if headers.contains_key(TRAILER) {
        return Err(HttpFault::MalformedRequest);
    }
    if headers.contains_key(TRANSFER_ENCODING) {
        return Err(HttpFault::MalformedRequest);
    }
    let mut content_lengths = headers.get_all(CONTENT_LENGTH).iter();
    let content_length = content_lengths.next();
    if content_lengths.next().is_some() {
        return Err(HttpFault::MalformedRequest);
    }
    let content_length = content_length
        .and_then(parse_content_length)
        .ok_or(HttpFault::MalformedRequest)?;
    for forbidden in [
        "x-smg-target-worker",
        "x-smg-routing-key",
        "x-sgl-decode-url",
    ] {
        if headers.contains_key(forbidden) {
            return Err(HttpFault::MalformedRequest);
        }
    }
    Ok(RequestFraming { content_length })
}

pub(crate) fn sanitize_response(
    status: StatusCode,
    source: &HeaderMap,
    expected: Option<ExpectedSuccess>,
) -> Result<HeaderMap, HttpFault> {
    let connection_tokens = connection_tokens(source)?;
    if !(status.is_success() || status.is_client_error() || status.is_server_error()) {
        return Err(HttpFault::UpstreamProtocolError);
    }
    if source.contains_key(CONTENT_ENCODING)
        && connection_tokens.contains(CONTENT_ENCODING.as_str())
    {
        return Err(HttpFault::UpstreamProtocolError);
    }
    let mut content_types = source.get_all(CONTENT_TYPE).iter();
    let content_type = content_types.next();
    let mut content_lengths = source.get_all(CONTENT_LENGTH).iter();
    let content_length = content_lengths.next();
    if content_types.next().is_some() || content_lengths.next().is_some() {
        return Err(HttpFault::UpstreamProtocolError);
    }
    if let Some(value) = content_type
        && !value.to_str().is_ok_and(valid_generic_content_type)
    {
        return Err(HttpFault::UpstreamProtocolError);
    }
    if let Some(value) = content_length
        && parse_content_length(value).is_none()
    {
        return Err(HttpFault::UpstreamProtocolError);
    }
    if status.is_success() {
        let value = content_type
            .filter(|_| !connection_tokens.contains(CONTENT_TYPE.as_str()))
            .and_then(|value| value.to_str().ok())
            .ok_or(HttpFault::UpstreamProtocolError)?;
        let json = is_media_type(value, "application/json");
        let sse = is_media_type(value, "text/event-stream");
        if !json && !sse {
            return Err(HttpFault::UpstreamProtocolError);
        }
        if matches!(expected, Some(ExpectedSuccess::Json)) && !json
            || matches!(expected, Some(ExpectedSuccess::Sse)) && !sse
        {
            return Err(HttpFault::UpstreamProtocolError);
        }
    }

    let mut result = HeaderMap::new();
    for name in [CONTENT_TYPE, CONTENT_LENGTH] {
        if !connection_tokens.contains(name.as_str())
            && let Some(value) = source.get(&name)
        {
            result.insert(name, value.clone());
        }
    }
    for name in [
        CONTENT_ENCODING,
        axum::http::header::CACHE_CONTROL,
        axum::http::header::EXPIRES,
        axum::http::header::VARY,
    ] {
        if connection_tokens.contains(name.as_str()) {
            continue;
        }
        for value in source.get_all(&name) {
            result.append(name.clone(), value.clone());
        }
    }
    Ok(result)
}

pub(crate) fn connection_tokens(headers: &HeaderMap) -> Result<HashSet<String>, HttpFault> {
    let mut result = HashSet::new();
    for value in headers.get_all(CONNECTION) {
        let value = value
            .to_str()
            .map_err(|_| HttpFault::UpstreamProtocolError)?;
        for token in value.split(',') {
            let token = token.trim();
            let name = HeaderName::from_bytes(token.as_bytes())
                .map_err(|_| HttpFault::UpstreamProtocolError)?;
            result.insert(name.as_str().to_owned());
        }
    }
    Ok(result)
}

pub(crate) fn parse_content_length(value: &HeaderValue) -> Option<u64> {
    let text = value.to_str().ok()?;
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

fn is_media_type(value: &str, expected: &str) -> bool {
    parse_content_type(value, |_name, _parameter| {})
        .is_some_and(|media| media.eq_ignore_ascii_case(expected))
}

pub(crate) fn is_request_media_type(value: &str, expected: &str) -> bool {
    let mut charset_seen = false;
    let mut valid_parameters = true;
    let media = parse_content_type(value, |name, parameter| {
        if charset_seen || !name.eq_ignore_ascii_case("charset") {
            valid_parameters = false;
            return;
        }
        charset_seen = true;
        if !parameter.eq_ignore_ascii_case(b"utf-8") {
            valid_parameters = false;
        }
    });
    media.is_some_and(|media| media.eq_ignore_ascii_case(expected) && valid_parameters)
}

pub(crate) fn valid_generic_content_type(value: &str) -> bool {
    parse_content_type(value, |_name, _parameter| {}).is_some()
}

#[derive(Clone, Copy)]
enum ParameterValue<'a> {
    Token(&'a [u8]),
    Quoted(&'a [u8]),
}

impl ParameterValue<'_> {
    fn eq_ignore_ascii_case(self, expected: &[u8]) -> bool {
        let mut index = 0;
        let mut expected_index = 0;
        let bytes = match self {
            Self::Token(bytes) | Self::Quoted(bytes) => bytes,
        };
        while index < bytes.len() {
            if matches!(self, Self::Quoted(_)) && bytes[index] == b'\\' {
                index += 1;
            }
            if index >= bytes.len()
                || expected_index >= expected.len()
                || !bytes[index].eq_ignore_ascii_case(&expected[expected_index])
            {
                return false;
            }
            index += 1;
            expected_index += 1;
        }
        expected_index == expected.len()
    }
}

fn parse_content_type<'a>(
    value: &'a str,
    mut on_parameter: impl FnMut(&'a str, ParameterValue<'a>),
) -> Option<&'a str> {
    let bytes = value.as_bytes();
    let mut cursor = skip_ows(bytes, 0);
    let media_start = cursor;
    cursor = parse_token(bytes, cursor)?;
    if bytes.get(cursor) != Some(&b'/') {
        return None;
    }
    cursor = parse_token(bytes, cursor + 1)?;
    let media_end = cursor;

    loop {
        cursor = skip_ows(bytes, cursor);
        if cursor == bytes.len() {
            return Some(&value[media_start..media_end]);
        }
        if bytes.get(cursor) != Some(&b';') {
            return None;
        }
        cursor = skip_ows(bytes, cursor + 1);
        let name_start = cursor;
        cursor = parse_token(bytes, cursor)?;
        let name = &value[name_start..cursor];
        if bytes.get(cursor) != Some(&b'=') {
            return None;
        }
        cursor += 1;
        let parameter = if bytes.get(cursor) == Some(&b'"') {
            let quoted_start = cursor + 1;
            cursor = quoted_end(bytes, quoted_start)?;
            ParameterValue::Quoted(&bytes[quoted_start..cursor - 1])
        } else {
            let token_start = cursor;
            cursor = parse_token(bytes, cursor)?;
            ParameterValue::Token(&bytes[token_start..cursor])
        };
        on_parameter(name, parameter);
    }
}

fn parse_token(bytes: &[u8], start: usize) -> Option<usize> {
    let mut cursor = start;
    while bytes.get(cursor).is_some_and(|byte| is_tchar(*byte)) {
        cursor += 1;
    }
    (cursor > start).then_some(cursor)
}

const fn is_tchar(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

fn quoted_end(bytes: &[u8], mut cursor: usize) -> Option<usize> {
    while let Some(&byte) = bytes.get(cursor) {
        match byte {
            b'"' => return Some(cursor + 1),
            b'\\' => {
                cursor += 1;
                let escaped = *bytes.get(cursor)?;
                if !is_quoted_pair_byte(escaped) {
                    return None;
                }
            }
            byte if !is_qdtext(byte) => return None,
            _ => {}
        }
        cursor += 1;
    }
    None
}

const fn is_qdtext(byte: u8) -> bool {
    matches!(byte, b'\t' | b' ' | b'!' | b'#'..=b'[' | b']'..=b'~')
}

const fn is_quoted_pair_byte(byte: u8) -> bool {
    matches!(byte, b'\t' | b' '..=b'~')
}

fn skip_ows(bytes: &[u8], mut cursor: usize) -> usize {
    while bytes
        .get(cursor)
        .is_some_and(|byte| matches!(byte, b' ' | b'\t'))
    {
        cursor += 1;
    }
    cursor
}

pub(crate) fn canonical_content_type() -> HeaderValue {
    HeaderValue::from_static("application/json")
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use axum::http::header::{
        CACHE_CONTROL, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE, TRANSFER_ENCODING,
    };
    use axum::http::{HeaderMap, HeaderValue, StatusCode};

    use super::super::classify::ExpectedSuccess;
    use super::{HttpFault, sanitize_response, validate_request};

    fn valid_request_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/json; charset=UTF-8"),
        );
        headers.insert(CONTENT_LENGTH, HeaderValue::from_static("12"));
        headers
    }

    #[test]
    fn request_envelope_requires_fixed_unencoded_json() {
        let headers = valid_request_headers();
        assert_eq!(
            validate_request(&headers)
                .expect("valid fixed request")
                .content_length,
            12
        );

        for (name, value, fault) in [
            (CONTENT_TYPE, "text/plain", HttpFault::UnsupportedMediaType),
            (
                CONTENT_ENCODING,
                "identity",
                HttpFault::UnsupportedContentEncoding,
            ),
            (TRANSFER_ENCODING, "chunked", HttpFault::MalformedRequest),
        ] {
            let mut rejected = headers.clone();
            rejected.insert(name, HeaderValue::from_static(value));
            assert_eq!(validate_request(&rejected).err(), Some(fault));
        }

        let mut missing_length = headers;
        missing_length.remove(CONTENT_LENGTH);
        assert_eq!(
            validate_request(&missing_length).err(),
            Some(HttpFault::MalformedRequest)
        );

        let mut duplicate_type = valid_request_headers();
        duplicate_type.append(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        assert_eq!(
            validate_request(&duplicate_type).err(),
            Some(HttpFault::UnsupportedMediaType)
        );
        let mut duplicate_length = valid_request_headers();
        duplicate_length.append(CONTENT_LENGTH, HeaderValue::from_static("12"));
        assert_eq!(
            validate_request(&duplicate_length).err(),
            Some(HttpFault::MalformedRequest)
        );
    }

    #[test]
    fn response_allowlist_preserves_safe_duplicates_and_encoding() {
        let mut source = HeaderMap::new();
        source.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        source.append(CACHE_CONTROL, HeaderValue::from_static("private"));
        source.append(CACHE_CONTROL, HeaderValue::from_static("max-age=0"));
        source.insert(CONTENT_ENCODING, HeaderValue::from_static("gzip"));
        source.insert("set-cookie", HeaderValue::from_static("private=1"));
        source.insert("connection", HeaderValue::from_static("x-private"));
        source.insert("x-private", HeaderValue::from_static("secret"));

        let sanitized =
            sanitize_response(StatusCode::OK, &source, None).expect("valid worker response");
        assert_eq!(sanitized.get_all(CACHE_CONTROL).iter().count(), 2);
        assert_eq!(
            sanitized.get(CONTENT_ENCODING),
            source.get(CONTENT_ENCODING)
        );
        assert!(!sanitized.contains_key("set-cookie"));
        assert!(!sanitized.contains_key("x-private"));
    }

    #[test]
    fn response_rejects_redirects_and_invalid_success_media() {
        let mut plain = HeaderMap::new();
        plain.insert(CONTENT_TYPE, HeaderValue::from_static("text/plain"));
        assert_eq!(
            sanitize_response(StatusCode::OK, &plain, None).err(),
            Some(HttpFault::UpstreamProtocolError)
        );
        assert_eq!(
            sanitize_response(StatusCode::TEMPORARY_REDIRECT, &HeaderMap::new(), None).err(),
            Some(HttpFault::UpstreamProtocolError)
        );

        let mut duplicate_type = HeaderMap::new();
        duplicate_type.append(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        duplicate_type.append(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        assert_eq!(
            sanitize_response(StatusCode::OK, &duplicate_type, None).err(),
            Some(HttpFault::UpstreamProtocolError)
        );
        let mut duplicate_length = HeaderMap::new();
        duplicate_length.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        duplicate_length.append(CONTENT_LENGTH, HeaderValue::from_static("2"));
        duplicate_length.append(CONTENT_LENGTH, HeaderValue::from_static("2"));
        assert_eq!(
            sanitize_response(StatusCode::OK, &duplicate_length, None).err(),
            Some(HttpFault::UpstreamProtocolError)
        );
    }

    #[test]
    fn worker_errors_relay_without_a_success_content_type() {
        let headers = HeaderMap::new();
        let sanitized = sanitize_response(StatusCode::UNPROCESSABLE_ENTITY, &headers, None)
            .expect("worker error response is relayable");
        assert!(sanitized.is_empty());
    }

    #[test]
    fn classified_success_must_match_the_body_selected_stream_mode() {
        let mut json = HeaderMap::new();
        json.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let mut sse = HeaderMap::new();
        sse.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        assert!(sanitize_response(StatusCode::OK, &json, Some(ExpectedSuccess::Json)).is_ok());
        assert!(sanitize_response(StatusCode::OK, &sse, Some(ExpectedSuccess::Sse)).is_ok());
        assert_eq!(
            sanitize_response(StatusCode::OK, &json, Some(ExpectedSuccess::Sse)).err(),
            Some(HttpFault::UpstreamProtocolError)
        );
        assert_eq!(
            sanitize_response(StatusCode::OK, &sse, Some(ExpectedSuccess::Json)).err(),
            Some(HttpFault::UpstreamProtocolError)
        );
    }
}
