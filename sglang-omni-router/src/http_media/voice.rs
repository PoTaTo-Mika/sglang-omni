use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderValue, Method, Request, Response, Version};

use crate::config::VOICE_UPLOAD_MAX_BYTES;
use crate::error::HttpFault;
use crate::http_generation::headers::{
    REQUEST_ID_HEADER, canonical_request_id, validate_bodyless_request,
};
use crate::http_generation::request_body::BufferedBody;
use crate::http_generation::response_body::DirectResponseBody;
use crate::http_generation::{read_buffered, reserve_budget};
use crate::worker_pool::{AdmissionError, CapacityClass, DispatchError, RequestLease};

use super::HttpMedia;
use super::headers::{RequestKind, SuccessProfile, sanitize_response, validate_request};

struct UpstreamRequest<'a> {
    method: Method,
    query: Option<&'a str>,
    name: Option<&'a str>,
    request_id: HeaderValue,
}

pub(crate) async fn list(
    State(media): State<Arc<HttpMedia>>,
    request: Request<Body>,
) -> Response<Body> {
    outcome(handle_bodyless(media, request, Method::GET, None).await)
}

pub(crate) async fn upload(
    State(media): State<Arc<HttpMedia>>,
    request: Request<Body>,
) -> Response<Body> {
    outcome(handle_upload(media, request).await)
}

pub(crate) async fn delete(
    State(media): State<Arc<HttpMedia>>,
    Path(name): Path<String>,
    request: Request<Body>,
) -> Response<Body> {
    outcome(handle_bodyless(media, request, Method::DELETE, Some(name)).await)
}

fn outcome(result: Result<Response<Body>, HttpFault>) -> Response<Body> {
    result.unwrap_or_else(HttpFault::into_response)
}

async fn handle_bodyless(
    media: Arc<HttpMedia>,
    request: Request<Body>,
    method: Method,
    name: Option<String>,
) -> Result<Response<Body>, HttpFault> {
    validate_common(&media, &request)?;
    if name.as_ref().is_some_and(String::is_empty) {
        return Err(HttpFault::MalformedRequest);
    }
    validate_bodyless_request(request.headers())?;
    let request_id = canonical_request_id(request.headers())?;
    let deadline = tokio::time::Instant::now() + media.request_timeout;
    let lease = control_lease(&media)?;
    send_once(
        media,
        UpstreamRequest {
            method,
            query: request.uri().query(),
            name: name.as_deref(),
            request_id,
        },
        None,
        lease,
        deadline,
    )
    .await
}

async fn handle_upload(
    media: Arc<HttpMedia>,
    request: Request<Body>,
) -> Result<Response<Body>, HttpFault> {
    validate_common(&media, &request)?;
    let deadline = tokio::time::Instant::now() + media.request_timeout;
    let framing = validate_request(request.headers(), RequestKind::Multipart)?;
    let request_id = canonical_request_id(request.headers())?;
    if framing
        .content_length
        .is_some_and(|length| length > VOICE_UPLOAD_MAX_BYTES)
    {
        return Err(HttpFault::RequestBodyTooLarge);
    }
    let lease = control_lease(&media)?;
    let reserve = framing.content_length.unwrap_or(VOICE_UPLOAD_MAX_BYTES);
    let budget = reserve_budget(&media.buffered_budget, reserve)?;
    let query = request.uri().query().map(str::to_owned);
    let bytes = read_buffered(
        request.into_body(),
        framing.content_length,
        VOICE_UPLOAD_MAX_BYTES,
        deadline,
    )
    .await?;
    let length = u64::try_from(bytes.len()).map_err(|_| HttpFault::InternalError)?;
    let body = reqwest::Body::wrap(BufferedBody::new(bytes, budget));
    send_once(
        media,
        UpstreamRequest {
            method: Method::POST,
            query: query.as_deref(),
            name: None,
            request_id,
        },
        Some((body, length, framing.content_type)),
        lease,
        deadline,
    )
    .await
}

fn validate_common(media: &HttpMedia, request: &Request<Body>) -> Result<(), HttpFault> {
    if !media.voice_routes_enabled() {
        return Err(HttpFault::RouterUnavailable);
    }
    if request.version() != Version::HTTP_11 {
        return Err(HttpFault::HttpVersionNotSupported);
    }
    Ok(())
}

fn control_lease(media: &HttpMedia) -> Result<RequestLease, HttpFault> {
    let admission = media
        .pool
        .try_admit(CapacityClass::Control)
        .map_err(map_admission)?;
    media
        .pool
        .dispatch_voice_control(admission)
        .map_err(map_dispatch)
}

type Upload = (reqwest::Body, u64, axum::http::HeaderValue);

async fn send_once(
    media: Arc<HttpMedia>,
    upstream: UpstreamRequest<'_>,
    upload: Option<Upload>,
    lease: RequestLease,
    deadline: tokio::time::Instant,
) -> Result<Response<Body>, HttpFault> {
    let UpstreamRequest {
        method,
        query,
        name,
        request_id,
    } = upstream;
    if tokio::time::Instant::now() >= deadline {
        return Err(HttpFault::UpstreamTimeout);
    }
    let mut url = lease.target().base_url().clone();
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| HttpFault::InternalError)?;
        segments.clear().extend(["v1", "audio", "voices"]);
        if let Some(name) = name {
            segments.push(name);
        }
    }
    url.set_query(query);
    let mut request = media
        .client
        .request(method, url)
        .header(REQUEST_ID_HEADER, request_id);
    if let Some((body, length, content_type)) = upload {
        request = request
            .header(CONTENT_TYPE, content_type)
            .header(CONTENT_LENGTH, length)
            .body(body);
    }
    let response = tokio::select! {
        biased;
        result = request.send() => result,
        () = tokio::time::sleep_until(deadline) => return Err(HttpFault::UpstreamTimeout),
    };
    let response = match response {
        Ok(response) => response,
        Err(_source) => {
            lease.request_immediate_probe();
            return Err(HttpFault::UpstreamProtocolError);
        }
    };
    let response: axum::http::Response<reqwest::Body> = response.into();
    let (parts, body) = response.into_parts();
    let headers = match sanitize_response(parts.status, &parts.headers, SuccessProfile::Json) {
        Ok(headers) => headers,
        Err(fault) => {
            drop(body);
            lease.request_immediate_probe();
            return Err(fault);
        }
    };
    let mut downstream = Response::new(Body::new(DirectResponseBody::new(body, lease)));
    *downstream.status_mut() = parts.status;
    *downstream.headers_mut() = headers;
    Ok(downstream)
}

const fn map_admission(error: AdmissionError) -> HttpFault {
    match error {
        AdmissionError::Draining => HttpFault::RouterUnavailable,
        AdmissionError::Overloaded => HttpFault::RouterOverloaded,
        AdmissionError::Internal => HttpFault::InternalError,
    }
}

const fn map_dispatch(error: DispatchError) -> HttpFault {
    match error {
        DispatchError::Unavailable | DispatchError::Draining => HttpFault::RouterUnavailable,
        DispatchError::Overloaded => HttpFault::RouterOverloaded,
        DispatchError::NoEligibleProfile
        | DispatchError::AdmissionClassMismatch
        | DispatchError::Internal => HttpFault::InternalError,
    }
}
