use crate::error::HttpFault;
use crate::worker_pool::MediaProfile;

const MAX_PARTS: usize = 64;
const MAX_HEADER_BYTES: usize = 65_536;
const MAX_ROUTING_VALUE_BYTES: usize = 8_192;

#[derive(Debug, Eq, PartialEq)]
pub(super) struct MultipartFacts {
    pub(super) model: Option<String>,
    pub(super) response_format: Option<String>,
    pub(super) stream: Option<bool>,
    pub(super) media_profile: MediaProfile,
}

#[derive(Default)]
struct Facts {
    model: Option<String>,
    response_format: Option<String>,
    stream: Option<bool>,
    file_profile: Option<MediaProfile>,
}

pub(super) fn scan(body: &[u8], boundary: &[u8]) -> Result<MultipartFacts, HttpFault> {
    if boundary.is_empty() || boundary.len() > 70 {
        return Err(HttpFault::MalformedRequest);
    }
    let mut delimiter = Vec::with_capacity(boundary.len() + 2);
    delimiter.extend_from_slice(b"--");
    delimiter.extend_from_slice(boundary);
    if !body.starts_with(&delimiter) {
        return Err(HttpFault::MalformedRequest);
    }
    let mut cursor = delimiter.len();
    let mut facts = Facts::default();
    let mut parts = 0_usize;
    let mut header_bytes = 0_usize;
    loop {
        if body.get(cursor..cursor + 2) == Some(b"--") {
            cursor += 2;
            if body
                .get(cursor..)
                .is_some_and(|tail| tail.is_empty() || tail == b"\r\n")
            {
                break;
            }
            return Err(HttpFault::MalformedRequest);
        }
        if body.get(cursor..cursor + 2) != Some(b"\r\n") {
            return Err(HttpFault::MalformedRequest);
        }
        cursor += 2;
        parts = parts.checked_add(1).ok_or(HttpFault::MalformedRequest)?;
        if parts > MAX_PARTS {
            return Err(HttpFault::MalformedRequest);
        }
        let header_end = find(body, cursor, b"\r\n\r\n").ok_or(HttpFault::MalformedRequest)?;
        let part_header_bytes = header_end
            .checked_sub(cursor)
            .ok_or(HttpFault::MalformedRequest)?;
        header_bytes = header_bytes
            .checked_add(part_header_bytes)
            .ok_or(HttpFault::MalformedRequest)?;
        if header_bytes > MAX_HEADER_BYTES {
            return Err(HttpFault::MalformedRequest);
        }
        let headers = parse_headers(&body[cursor..header_end])?;
        let data_start = header_end + 4;
        let mut marker = Vec::with_capacity(delimiter.len() + 2);
        marker.extend_from_slice(b"\r\n");
        marker.extend_from_slice(&delimiter);
        let mut search = data_start;
        let next = loop {
            let candidate = find(body, search, &marker).ok_or(HttpFault::MalformedRequest)?;
            let suffix = candidate + marker.len();
            if matches!(body.get(suffix..suffix + 2), Some(b"\r\n") | Some(b"--")) {
                break candidate;
            }
            search = candidate + 2;
        };
        classify_part(&headers, &body[data_start..next], &mut facts)?;
        cursor = next + 2 + delimiter.len();
    }
    if parts == 0 {
        return Err(HttpFault::MalformedRequest);
    }
    Ok(MultipartFacts {
        model: facts.model,
        response_format: facts.response_format,
        stream: facts.stream,
        media_profile: facts.file_profile.ok_or(HttpFault::MalformedRequest)?,
    })
}

#[derive(Default)]
struct PartHeaders {
    name: Option<String>,
    filename: Option<String>,
    content_type: Option<String>,
}

fn parse_headers(bytes: &[u8]) -> Result<PartHeaders, HttpFault> {
    let text = std::str::from_utf8(bytes).map_err(|_| HttpFault::MalformedRequest)?;
    let mut result = PartHeaders::default();
    let mut disposition_seen = false;
    let mut content_type_seen = false;
    for line in text.split("\r\n") {
        if line.is_empty() || line.starts_with([' ', '\t']) {
            return Err(HttpFault::MalformedRequest);
        }
        let (name, value) = line.split_once(':').ok_or(HttpFault::MalformedRequest)?;
        if name.eq_ignore_ascii_case("content-disposition") {
            if disposition_seen {
                return Err(HttpFault::MalformedRequest);
            }
            disposition_seen = true;
            parse_disposition(value.trim(), &mut result)?;
        } else if name.eq_ignore_ascii_case("content-type") {
            if content_type_seen {
                return Err(HttpFault::MalformedRequest);
            }
            content_type_seen = true;
            let value = value.trim();
            if value.is_empty() || !value.is_ascii() {
                return Err(HttpFault::MalformedRequest);
            }
            result.content_type = Some(value.to_ascii_lowercase());
        } else if !name.bytes().all(is_token_byte) {
            return Err(HttpFault::MalformedRequest);
        }
    }
    if !disposition_seen || result.name.is_none() {
        return Err(HttpFault::MalformedRequest);
    }
    Ok(result)
}

fn parse_disposition(value: &str, headers: &mut PartHeaders) -> Result<(), HttpFault> {
    let mut segments = disposition_segments(value)?.into_iter();
    if !segments
        .next()
        .is_some_and(|kind| kind.trim().eq_ignore_ascii_case("form-data"))
    {
        return Err(HttpFault::MalformedRequest);
    }
    for segment in segments {
        let (key, raw) = segment
            .trim()
            .split_once('=')
            .ok_or(HttpFault::MalformedRequest)?;
        let parsed = parse_parameter(raw.trim())?;
        if key.eq_ignore_ascii_case("name") {
            if headers.name.replace(parsed).is_some() {
                return Err(HttpFault::MalformedRequest);
            }
        } else if key.eq_ignore_ascii_case("filename") && headers.filename.replace(parsed).is_some()
        {
            return Err(HttpFault::MalformedRequest);
        }
    }
    Ok(())
}

fn disposition_segments(value: &str) -> Result<Vec<&str>, HttpFault> {
    let bytes = value.as_bytes();
    let mut result = Vec::new();
    let mut start = 0_usize;
    let mut quoted = false;
    let mut escaped = false;
    for (index, byte) in bytes.iter().copied().enumerate() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
        } else if byte == b'"' {
            quoted = true;
        } else if byte == b';' {
            result.push(&value[start..index]);
            start = index + 1;
        }
    }
    if quoted || escaped {
        return Err(HttpFault::MalformedRequest);
    }
    result.push(&value[start..]);
    Ok(result)
}

fn parse_parameter(raw: &str) -> Result<String, HttpFault> {
    if let Some(inner) = raw
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
    {
        let mut output = String::with_capacity(inner.len());
        let mut escaped = false;
        for character in inner.chars() {
            if escaped {
                output.push(character);
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' || character == '\r' || character == '\n' {
                return Err(HttpFault::MalformedRequest);
            } else {
                output.push(character);
            }
        }
        if escaped {
            return Err(HttpFault::MalformedRequest);
        }
        Ok(output)
    } else if !raw.is_empty() && raw.bytes().all(is_token_byte) {
        Ok(raw.to_owned())
    } else {
        Err(HttpFault::MalformedRequest)
    }
}

fn classify_part(headers: &PartHeaders, value: &[u8], facts: &mut Facts) -> Result<(), HttpFault> {
    let name = headers.name.as_deref().ok_or(HttpFault::MalformedRequest)?;
    match name {
        "model" => set_model(&mut facts.model, value)?,
        "response_format" => set_text(&mut facts.response_format, value)?,
        "stream" => {
            if facts.stream.is_some() || value.len() > MAX_ROUTING_VALUE_BYTES {
                return Err(HttpFault::MalformedRequest);
            }
            let text = std::str::from_utf8(value)
                .map_err(|_| HttpFault::MalformedRequest)?
                .trim();
            facts.stream = Some(
                if ["true", "1", "on", "yes", "y", "t"]
                    .iter()
                    .any(|value| text.eq_ignore_ascii_case(value))
                {
                    true
                } else if ["false", "0", "off", "no", "n", "f"]
                    .iter()
                    .any(|value| text.eq_ignore_ascii_case(value))
                {
                    false
                } else {
                    return Err(HttpFault::MalformedRequest);
                },
            );
        }
        "file" => {
            if facts.file_profile.is_some() {
                return Err(HttpFault::MalformedRequest);
            }
            facts.file_profile = Some(media_profile(headers)?);
        }
        _ => {}
    }
    Ok(())
}

fn set_model(slot: &mut Option<String>, value: &[u8]) -> Result<(), HttpFault> {
    if slot.is_some() || value.len() > MAX_ROUTING_VALUE_BYTES {
        return Err(HttpFault::MalformedRequest);
    }
    let text = std::str::from_utf8(value).map_err(|_| HttpFault::MalformedRequest)?;
    *slot = Some(text.to_owned());
    Ok(())
}

fn set_text(slot: &mut Option<String>, value: &[u8]) -> Result<(), HttpFault> {
    if slot.is_some() || value.len() > MAX_ROUTING_VALUE_BYTES {
        return Err(HttpFault::MalformedRequest);
    }
    let text = std::str::from_utf8(value)
        .map_err(|_| HttpFault::MalformedRequest)?
        .trim();
    if text.is_empty() {
        return Err(HttpFault::MalformedRequest);
    }
    *slot = Some(text.to_owned());
    Ok(())
}

fn media_profile(headers: &PartHeaders) -> Result<MediaProfile, HttpFault> {
    let content = headers
        .content_type
        .as_deref()
        .and_then(|value| value.split(';').next())
        .map(str::trim);
    let content_profile = match content {
        Some(value) if value.starts_with("audio/") => Some(MediaProfile::Audio),
        Some(value) if value.starts_with("video/") => Some(MediaProfile::AudioVideo),
        Some("application/octet-stream") | None => None,
        Some(_) => return Err(HttpFault::UnsupportedMediaType),
    };
    let extension_profile = headers.filename.as_deref().and_then(|filename| {
        let extension = filename.rsplit_once('.')?.1.to_ascii_lowercase();
        match extension.as_str() {
            "wav" | "mp3" | "flac" | "aac" | "m4a" | "ogg" | "opus" => Some(MediaProfile::Audio),
            "mp4" | "mov" | "mkv" | "webm" | "avi" => Some(MediaProfile::AudioVideo),
            _ => None,
        }
    });
    match (content_profile, extension_profile) {
        (Some(left), Some(right)) if left != right => Err(HttpFault::MalformedRequest),
        (Some(profile), _) | (None, Some(profile)) => Ok(profile),
        (None, None) => Err(HttpFault::UnsupportedMediaType),
    }
}

fn find(haystack: &[u8], start: usize, needle: &[u8]) -> Option<usize> {
    haystack
        .get(start..)?
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|position| start + position)
}

fn is_token_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::{HttpFault, MediaProfile, parse_headers, scan};

    fn body(parts: &[(&str, Option<&str>, &[u8])], close: bool) -> Vec<u8> {
        let mut output = Vec::new();
        for (name, content_type, value) in parts {
            output
                .extend_from_slice(b"--route-boundary\r\nContent-Disposition: form-data; name=\"");
            output.extend_from_slice(name.as_bytes());
            if *name == "file" {
                output.extend_from_slice(b"\"; filename=\"sample.wav");
            }
            output.extend_from_slice(b"\"\r\n");
            if let Some(content_type) = content_type {
                output.extend_from_slice(b"Content-Type: ");
                output.extend_from_slice(content_type.as_bytes());
                output.extend_from_slice(b"\r\n");
            }
            output.extend_from_slice(b"\r\n");
            output.extend_from_slice(value);
            output.extend_from_slice(b"\r\n");
        }
        output.extend_from_slice(b"--route-boundary");
        if close {
            output.extend_from_slice(b"--\r\n");
        }
        output
    }

    #[test]
    fn accepts_metadata_in_any_order_and_ignores_boundary_like_file_bytes() {
        let file = b"RIFF\r\n--route-boundary-not-a-delimiter\0audio";
        for parts in [
            vec![
                ("model", None, b"asr".as_slice()),
                ("file", Some("audio/wav"), file.as_slice()),
                ("response_format", None, b"text".as_slice()),
            ],
            vec![
                ("file", Some("audio/wav"), file.as_slice()),
                ("stream", None, b"true".as_slice()),
                ("model", None, b"asr".as_slice()),
            ],
        ] {
            let facts = scan(&body(&parts, true), b"route-boundary")
                .expect("valid complete multipart body");
            assert_eq!(facts.model.as_deref(), Some("asr"));
            assert_eq!(facts.media_profile, MediaProfile::Audio);
        }
    }

    #[test]
    fn rejects_duplicate_routing_fields_malformed_closure_and_identity_conflict() {
        let duplicate = body(
            &[
                ("model", None, b"a"),
                ("model", None, b"b"),
                ("file", Some("audio/wav"), b"RIFF"),
            ],
            true,
        );
        assert_eq!(
            scan(&duplicate, b"route-boundary"),
            Err(HttpFault::MalformedRequest)
        );
        let malformed = body(&[("file", Some("audio/wav"), b"RIFF")], false);
        assert_eq!(
            scan(&malformed, b"route-boundary"),
            Err(HttpFault::MalformedRequest)
        );
        let conflict = body(&[("file", Some("video/mp4"), b"data")], true);
        assert_eq!(
            scan(&conflict, b"route-boundary"),
            Err(HttpFault::MalformedRequest)
        );
    }

    #[test]
    fn enforces_part_and_routing_value_limits() {
        let many: Vec<_> = (0..65)
            .map(|_| ("ignored", None, b"x".as_slice()))
            .chain(std::iter::once((
                "file",
                Some("audio/wav"),
                b"x".as_slice(),
            )))
            .collect();
        assert_eq!(
            scan(&body(&many, true), b"route-boundary"),
            Err(HttpFault::MalformedRequest)
        );
        let large = vec![b'a'; 8_193];
        assert_eq!(
            scan(
                &body(
                    &[
                        ("model", None, large.as_slice()),
                        ("file", Some("audio/wav"), b"x"),
                    ],
                    true,
                ),
                b"route-boundary",
            ),
            Err(HttpFault::MalformedRequest)
        );
    }

    #[test]
    fn preserves_model_bytes_and_accepts_pydantic_boolean_spellings() {
        for (value, expected) in [
            ("true", true),
            ("1", true),
            ("ON", true),
            ("Yes", true),
            ("y", true),
            ("T", true),
            ("false", false),
            ("0", false),
            ("OFF", false),
            ("No", false),
            ("n", false),
            ("F", false),
        ] {
            let multipart = body(
                &[
                    ("model", None, b"  asr  "),
                    ("stream", None, value.as_bytes()),
                    ("file", Some("audio/wav"), b"RIFF"),
                ],
                true,
            );
            let facts = scan(&multipart, b"route-boundary").expect("accepted boolean spelling");
            assert_eq!(facts.model.as_deref(), Some("  asr  "));
            assert_eq!(facts.stream, Some(expected));
        }
        let empty_model = body(
            &[("model", None, b""), ("file", Some("audio/wav"), b"RIFF")],
            true,
        );
        assert_eq!(
            scan(&empty_model, b"route-boundary")
                .expect("empty model remains a worker-default marker")
                .model
                .as_deref(),
            Some("")
        );
        let invalid = body(
            &[
                ("stream", None, b"enabled"),
                ("file", Some("audio/wav"), b"RIFF"),
            ],
            true,
        );
        assert_eq!(
            scan(&invalid, b"route-boundary"),
            Err(HttpFault::MalformedRequest)
        );
    }

    #[test]
    fn disposition_keeps_quoted_semicolons_escapes_and_duplicate_checks() {
        let headers = parse_headers(
            br#"Content-Disposition: form-data; name="file"; filename="sample;one.wav""#,
        )
        .expect("quoted filename semicolon");
        assert_eq!(headers.filename.as_deref(), Some("sample;one.wav"));

        let escaped = parse_headers(
            br#"Content-Disposition: form-data; name="file"; filename="sample\";one.wav""#,
        )
        .expect("escaped quote in filename");
        assert_eq!(escaped.filename.as_deref(), Some("sample\";one.wav"));

        assert_eq!(
            parse_headers(
                br#"Content-Disposition: form-data; name="file"; name="again"; filename="sample.wav""#,
            )
            .err(),
            Some(HttpFault::MalformedRequest)
        );
        assert_eq!(
            parse_headers(
                br#"Content-Disposition: form-data; name="file"; filename="sample"one.wav""#,
            )
            .err(),
            Some(HttpFault::MalformedRequest)
        );
    }
}
