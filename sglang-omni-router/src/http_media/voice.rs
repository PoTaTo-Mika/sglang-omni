use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Extension, Path, State};
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderValue, Method, Request, Response, Version};

use crate::config::VOICE_UPLOAD_BODY_MAX_BYTES;
use crate::error::HttpFault;
use crate::http_generation::{BufferedBody, DirectResponseBody, read_buffered, reserve_budget};
use crate::request_id::{CanonicalRequestId, REQUEST_ID_HEADER};
use crate::worker_pool::{AdmissionError, CapacityClass, DispatchError, RequestLease};

use super::HttpMedia;
use super::headers::{
    RequestKind, SuccessProfile, sanitize_response, validate_bodyless_request, validate_request,
};

pub(crate) async fn collection(
    State(media): State<Arc<HttpMedia>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    request: Request<Body>,
) -> Response<Body> {
    let request_id = request_id.into_header_value();
    let result = if request.method() == Method::GET {
        handle_bodyless(media, request, Method::GET, None, request_id).await
    } else if request.method() == Method::POST {
        handle_upload(media, request, request_id).await
    } else {
        return HttpFault::MethodNotAllowed
            .into_response_with_allow(HeaderValue::from_static("GET, POST"));
    };
    outcome(result)
}

pub(crate) async fn item(
    State(media): State<Arc<HttpMedia>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    Path(name): Path<String>,
    request: Request<Body>,
) -> Response<Body> {
    if request.method() != Method::DELETE {
        return HttpFault::MethodNotAllowed
            .into_response_with_allow(HeaderValue::from_static("DELETE"));
    }
    let result = handle_bodyless(
        media,
        request,
        Method::DELETE,
        Some(name),
        request_id.into_header_value(),
    )
    .await;
    outcome(result)
}

fn outcome(result: Result<Response<Body>, HttpFault>) -> Response<Body> {
    match result {
        Ok(response) => response,
        Err(fault) => fault.into_response(),
    }
}

async fn handle_bodyless(
    media: Arc<HttpMedia>,
    request: Request<Body>,
    method: Method,
    name: Option<String>,
    request_id: HeaderValue,
) -> Result<Response<Body>, HttpFault> {
    validate_common(&request)?;
    validate_bodyless_request(request.headers())?;
    let deadline = tokio::time::Instant::now() + media.request_timeout;
    let admission = media
        .pool
        .try_admit(CapacityClass::Control, 1)
        .map_err(map_admission)?;
    let lease = media
        .pool
        .dispatch_voice_control(admission)
        .map_err(map_dispatch)?;
    send_once(
        media,
        method,
        request.uri().query(),
        name.as_deref(),
        None,
        lease,
        request_id,
        deadline,
    )
    .await
}

async fn handle_upload(
    media: Arc<HttpMedia>,
    request: Request<Body>,
    request_id: HeaderValue,
) -> Result<Response<Body>, HttpFault> {
    validate_common(&request)?;
    let framing = validate_request(request.headers(), RequestKind::Multipart)?;
    if framing
        .content_length
        .is_some_and(|length| length > VOICE_UPLOAD_BODY_MAX_BYTES)
    {
        return Err(HttpFault::RequestBodyTooLarge);
    }
    let deadline = tokio::time::Instant::now() + media.request_timeout;
    let admission = media
        .pool
        .try_admit(CapacityClass::Control, 1)
        .map_err(map_admission)?;
    let reserved = framing
        .content_length
        .unwrap_or(VOICE_UPLOAD_BODY_MAX_BYTES);
    let budget = reserve_budget(&media.buffered_budget, reserved)?;
    let query = request.uri().query().map(str::to_owned);
    let bytes = read_buffered(
        request.into_body(),
        framing.content_length,
        VOICE_UPLOAD_BODY_MAX_BYTES,
        deadline,
    )
    .await?;
    let length = u64::try_from(bytes.len()).map_err(|_| HttpFault::InternalError)?;
    let upload = Upload {
        body: reqwest::Body::wrap(BufferedBody::new(bytes, budget)),
        length,
        content_type: framing.content_type,
    };
    let lease = media
        .pool
        .dispatch_voice_control(admission)
        .map_err(map_dispatch)?;
    send_once(
        media,
        Method::POST,
        query.as_deref(),
        None,
        Some(upload),
        lease,
        request_id,
        deadline,
    )
    .await
}

fn validate_common(request: &Request<Body>) -> Result<(), HttpFault> {
    if request.version() != Version::HTTP_11 {
        return Err(HttpFault::HttpVersionNotSupported);
    }
    Ok(())
}

struct Upload {
    body: reqwest::Body,
    length: u64,
    content_type: HeaderValue,
}

#[allow(clippy::too_many_arguments)]
async fn send_once(
    media: Arc<HttpMedia>,
    method: Method,
    query: Option<&str>,
    name: Option<&str>,
    upload: Option<Upload>,
    lease: RequestLease,
    request_id: HeaderValue,
    deadline: tokio::time::Instant,
) -> Result<Response<Body>, HttpFault> {
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
    let mut upstream = media
        .client
        .request(method, url)
        .header(REQUEST_ID_HEADER, request_id);
    if let Some(upload) = upload {
        upstream = upstream
            .header(CONTENT_TYPE, upload.content_type)
            .header(CONTENT_LENGTH, upload.length)
            .body(upload.body);
    }
    let response = tokio::select! {
        biased;
        result = upstream.send() => result,
        () = tokio::time::sleep_until(deadline) => {
            lease.request_immediate_probe();
            return Err(HttpFault::UpstreamTimeout);
        },
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
        DispatchError::NoEligibleProfile | DispatchError::Internal => HttpFault::InternalError,
    }
}
