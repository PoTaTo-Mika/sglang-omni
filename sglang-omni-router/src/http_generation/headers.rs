use std::collections::HashSet;

use axum::http::header::{
    CONNECTION, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE, EXPECT, TRAILER, TRANSFER_ENCODING,
};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};

use crate::error::HttpFault;

use super::classify::ExpectedSuccess;

pub(crate) struct RequestFraming {
    pub(crate) content_length: Option<u64>,
    pub(crate) transfer_framed: bool,
    pub(crate) has_route_hint: bool,
    pub(crate) has_content_encoding: bool,
}

pub(crate) fn validate_request(headers: &HeaderMap) -> Result<RequestFraming, HttpFault> {
    let content_types: Vec<_> = headers.get_all(CONTENT_TYPE).iter().collect();
    if content_types.len() != 1
        || !content_types[0]
            .to_str()
            .is_ok_and(|value| is_request_media_type(value, "application/json"))
    {
        return Err(HttpFault::UnsupportedMediaType);
    }
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
    })
}

pub(crate) fn sanitize_response(
    status: StatusCode,
    source: &HeaderMap,
    expected: ExpectedSuccess,
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
    let content_types: Vec<_> = source.get_all(CONTENT_TYPE).iter().collect();
    let content_lengths: Vec<_> = source.get_all(CONTENT_LENGTH).iter().collect();
    if content_types.len() > 1 || content_lengths.len() > 1 {
        return Err(HttpFault::UpstreamProtocolError);
    }
    if let Some(value) = content_types.first()
        && !value.to_str().is_ok_and(valid_generic_content_type)
    {
        return Err(HttpFault::UpstreamProtocolError);
    }
    if let Some(value) = content_lengths.first()
        && parse_content_length(value).is_none()
    {
        return Err(HttpFault::UpstreamProtocolError);
    }
    if status.is_success() {
        let value = content_types
            .first()
            .filter(|_| !connection_tokens.contains(CONTENT_TYPE.as_str()))
            .and_then(|value| value.to_str().ok())
            .ok_or(HttpFault::UpstreamProtocolError)?;
        let json = is_media_type(value, "application/json");
        let sse = is_media_type(value, "text/event-stream");
        let valid = match expected {
            ExpectedSuccess::Json => json,
            ExpectedSuccess::Sse => sse,
            ExpectedSuccess::Either => json || sse,
        };
        if !valid {
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

fn connection_tokens(headers: &HeaderMap) -> Result<HashSet<String>, HttpFault> {
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

fn parse_content_length(value: &HeaderValue) -> Option<u64> {
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

fn is_request_media_type(value: &str, expected: &str) -> bool {
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

fn valid_generic_content_type(value: &str) -> bool {
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
        CACHE_CONTROL, CONNECTION, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE,
    };
    use axum::http::{HeaderMap, HeaderValue, StatusCode};

    use super::{
        ExpectedSuccess, HttpFault, sanitize_response, valid_generic_content_type, validate_request,
    };

    #[test]
    fn request_header_matrix_is_strict_at_the_hyper_normalized_boundary() {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/json; charset=UTF-8"),
        );
        headers.insert(CONTENT_LENGTH, HeaderValue::from_static("12"));
        let framing = validate_request(&headers).expect("valid canonical request");
        assert_eq!(framing.content_length, Some(12));
        assert!(!framing.transfer_framed);
        assert!(!framing.has_content_encoding);

        headers.insert(CONTENT_ENCODING, HeaderValue::from_static("identity"));
        let identity = validate_request(&headers).expect("identity encoding is buffered-valid");
        assert!(identity.has_content_encoding);
        headers.remove(CONTENT_ENCODING);

        for (name, value, expected) in [
            (
                "content-type",
                "text/plain",
                HttpFault::UnsupportedMediaType,
            ),
            (
                "content-encoding",
                "gzip",
                HttpFault::UnsupportedContentEncoding,
            ),
            ("expect", "100-continue", HttpFault::ExpectationFailed),
            ("trailer", "digest", HttpFault::MalformedRequest),
        ] {
            let mut rejected = headers.clone();
            rejected.insert(
                name,
                HeaderValue::from_str(value).expect("valid test header value"),
            );
            assert_eq!(validate_request(&rejected).err(), Some(expected));
        }
    }

    #[test]
    fn response_allowlist_preserves_ordered_safe_values_and_strips_tokens() {
        let mut source = HeaderMap::new();
        source.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        source.append(CACHE_CONTROL, HeaderValue::from_static("private"));
        source.append(CACHE_CONTROL, HeaderValue::from_static("max-age=0"));
        source.insert(CONTENT_ENCODING, HeaderValue::from_static("identity"));
        source.insert("set-cookie", HeaderValue::from_static("private=1"));
        source.insert("server", HeaderValue::from_static("worker-a"));
        source.insert("connection", HeaderValue::from_static("x-private"));
        source.insert("x-private", HeaderValue::from_static("secret"));
        let sanitized = sanitize_response(StatusCode::OK, &source, ExpectedSuccess::Json)
            .expect("valid worker response");
        assert_eq!(sanitized.get_all(CACHE_CONTROL).iter().count(), 2);
        assert_eq!(
            sanitized.get(CONTENT_ENCODING),
            source.get(CONTENT_ENCODING)
        );
        assert!(!sanitized.contains_key("set-cookie"));
        assert!(!sanitized.contains_key("server"));
        assert!(!sanitized.contains_key("x-private"));
    }

    #[test]
    fn success_mode_redirect_and_repeated_singletons_fail_before_commit() {
        for (content_type, expected) in [
            ("application/json", ExpectedSuccess::Sse),
            ("text/event-stream", ExpectedSuccess::Json),
            ("text/plain", ExpectedSuccess::Either),
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(
                CONTENT_TYPE,
                HeaderValue::from_str(content_type).expect("valid test content type"),
            );
            assert_eq!(
                sanitize_response(StatusCode::OK, &headers, expected).err(),
                Some(HttpFault::UpstreamProtocolError)
            );
        }
        assert_eq!(
            sanitize_response(
                StatusCode::TEMPORARY_REDIRECT,
                &HeaderMap::new(),
                ExpectedSuccess::Either
            )
            .err(),
            Some(HttpFault::UpstreamProtocolError)
        );
        let mut repeated = HeaderMap::new();
        repeated.append(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        repeated.append(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        assert_eq!(
            sanitize_response(StatusCode::OK, &repeated, ExpectedSuccess::Json).err(),
            Some(HttpFault::UpstreamProtocolError)
        );
    }

    #[test]
    fn only_2xx_4xx_and_5xx_statuses_are_accepted_and_nominated_encoding_fails() {
        let mut source = HeaderMap::new();
        source.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        for code in [200, 299, 400, 499, 500, 599] {
            let status = StatusCode::from_u16(code).expect("accepted fixture status");
            sanitize_response(status, &source, ExpectedSuccess::Json)
                .expect("accepted upstream status class");
        }
        for code in [100, 101, 199, 300, 399, 600, 999] {
            let status = StatusCode::from_u16(code).expect("representable rejected status");
            assert_eq!(
                sanitize_response(status, &source, ExpectedSuccess::Json).err(),
                Some(HttpFault::UpstreamProtocolError),
                "accepted forbidden upstream status {code}"
            );
        }

        source.insert(CONTENT_ENCODING, HeaderValue::from_static("gzip"));
        source.insert(
            CONNECTION,
            HeaderValue::from_static("content-encoding, x-worker-private"),
        );
        for status in [StatusCode::OK, StatusCode::UNPROCESSABLE_ENTITY] {
            assert_eq!(
                sanitize_response(status, &source, ExpectedSuccess::Json).err(),
                Some(HttpFault::UpstreamProtocolError),
                "Connection-nominated encoded bytes were accepted"
            );
        }
    }

    #[test]
    fn trusted_error_status_may_stream_without_success_content_type() {
        let mut source = HeaderMap::new();
        source.insert("www-authenticate", HeaderValue::from_static("private"));
        let sanitized = sanitize_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            &source,
            ExpectedSuccess::Sse,
        )
        .expect("trusted worker error is relayed");
        assert!(sanitized.is_empty());

        let mut invalid = HeaderMap::new();
        invalid.insert(
            CONTENT_TYPE,
            HeaderValue::from_bytes(b"application/json\xff").expect("opaque header bytes"),
        );
        assert_eq!(
            sanitize_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                &invalid,
                ExpectedSuccess::Json,
            )
            .err(),
            Some(HttpFault::UpstreamProtocolError)
        );
    }

    #[test]
    fn response_content_type_accepts_full_tchar_and_quoted_parameter_grammar() {
        let mut source = HeaderMap::new();
        source.insert(
            CONTENT_TYPE,
            HeaderValue::from_static(
                "application/json; p!#$%&'*+-.^_`|~=v!#$%&'*+-.^_`|~; quoted=\"semi;colon\\\\\\\"and\\\\\\\\slash\"",
            ),
        );
        let sanitized = sanitize_response(StatusCode::OK, &source, ExpectedSuccess::Json)
            .expect("complete RFC token and quoted-string parameter grammar is valid on success");
        assert_eq!(sanitized.get(CONTENT_TYPE), source.get(CONTENT_TYPE));

        source.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("a!#$%&'*+-.^_`|~/b!#$%&'*+-.^_`|~; quoted=\"safe;value\""),
        );
        sanitize_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            &source,
            ExpectedSuccess::Json,
        )
        .expect("complete RFC media-type tchar grammar is valid on trusted errors");

        source.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/json; charset=\"uTf\\-8\""),
        );
        sanitize_response(StatusCode::OK, &source, ExpectedSuccess::Json)
            .expect("escaped quoted UTF-8 charset is semantically valid");
    }

    #[test]
    fn response_content_type_rejects_adversarial_malformed_values() {
        for value in [
            "application/",
            "/json",
            "application /json",
            "application/json;",
            "application/json; charset",
            "application/json; charset =utf-8",
            "application/json; charset=",
            "application/json; charset=unquoted/slash",
            "application/json; charset=\"unterminated",
            "application/json; charset=\"dangling\\\"",
            "application/json; charset=\"valid\"junk",
        ] {
            let mut source = HeaderMap::new();
            source.insert(
                CONTENT_TYPE,
                HeaderValue::from_bytes(value.as_bytes()).expect("representable test header"),
            );
            assert_eq!(
                sanitize_response(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    &source,
                    ExpectedSuccess::Json,
                )
                .err(),
                Some(HttpFault::UpstreamProtocolError),
                "accepted malformed Content-Type {value:?}"
            );
        }
        assert!(!valid_generic_content_type(
            "application/json; charset=\"bad\""
        ));
    }

    #[test]
    fn connection_nomination_does_not_excuse_invalid_singletons() {
        for (name, value) in [(CONTENT_TYPE, "not-a-media-type"), (CONTENT_LENGTH, "+1")] {
            let mut source = HeaderMap::new();
            source.insert(
                name.clone(),
                HeaderValue::from_str(value).expect("test value"),
            );
            source.insert(
                CONNECTION,
                HeaderValue::from_str(name.as_str()).expect("connection token"),
            );
            assert_eq!(
                sanitize_response(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    &source,
                    ExpectedSuccess::Either,
                )
                .err(),
                Some(HttpFault::UpstreamProtocolError)
            );
        }

        let mut valid = HeaderMap::new();
        valid.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/problem+json; note=\"safe;quoted\""),
        );
        valid.insert(CONTENT_LENGTH, HeaderValue::from_static("7"));
        valid.insert(
            CONNECTION,
            HeaderValue::from_static("content-type, content-length"),
        );
        let sanitized = sanitize_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &valid,
            ExpectedSuccess::Json,
        )
        .expect("valid nominated singletons are parsed before being stripped");
        assert!(!sanitized.contains_key(CONTENT_TYPE));
        assert!(!sanitized.contains_key(CONTENT_LENGTH));
    }
}
