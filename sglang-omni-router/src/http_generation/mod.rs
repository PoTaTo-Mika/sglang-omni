mod headers;
mod request_body;
mod response_body;

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::{Extension, State};
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderValue, Method, Request, Response, Version};

use crate::admission::Admission;
use crate::config::Config;
use crate::error::{HttpFault, RouterError};
use crate::request_id::{CanonicalRequestId, REQUEST_ID_HEADER};
use crate::upstream::Upstream;

use headers::{canonical_content_type, sanitize_response, validate_request};
use request_body::{DirectRequestBody, SharedUploadState, UploadState};
use response_body::DirectResponseBody;

pub(crate) const CHAT_PATH: &str = "/v1/chat/completions";

/// Complete one-worker generation relay state.
pub(crate) struct HttpGeneration {
    upstream: Upstream,
    admission: Arc<Admission>,
    maximum_request_bytes: u64,
    request_timeout: std::time::Duration,
}

impl HttpGeneration {
    pub(crate) fn build(
        config: &Config,
        admission: Arc<Admission>,
    ) -> Result<Arc<Self>, RouterError> {
        let worker = config
            .workers
            .first()
            .ok_or(RouterError::CoreHttpInvariant)?;
        Ok(Arc::new(Self {
            upstream: Upstream::build(worker, &config.http_generation)?,
            admission,
            maximum_request_bytes: config.http_generation.streamed_request_max_bytes,
            request_timeout: config.http_generation.request_timeout(),
        }))
    }
}

pub(crate) async fn chat(
    State(generation): State<Arc<HttpGeneration>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    request: Request<Body>,
) -> Response<Body> {
    match handle(generation, request, request_id.into_header_value()).await {
        Ok(response) => response,
        Err(fault) => fault.into_response(),
    }
}

async fn handle(
    generation: Arc<HttpGeneration>,
    request: Request<Body>,
    request_id: HeaderValue,
) -> Result<Response<Body>, HttpFault> {
    if request.method() != Method::POST {
        return Err(HttpFault::MethodNotAllowed);
    }
    if request.version() != Version::HTTP_11 {
        return Err(HttpFault::HttpVersionNotSupported);
    }
    if request.uri().path() != CHAT_PATH || request.uri().query().is_some() {
        return Err(HttpFault::MalformedRequest);
    }
    let framing = validate_request(request.headers())?;
    if framing.content_length > generation.maximum_request_bytes {
        return Err(HttpFault::RequestBodyTooLarge);
    }
    let lease = generation.admission.try_acquire()?;
    let deadline = tokio::time::Instant::now() + generation.request_timeout;
    let upload: SharedUploadState = Arc::new(Mutex::new(UploadState::Incomplete));
    let direct = DirectRequestBody::new(
        request.into_body(),
        framing.content_length,
        generation.maximum_request_bytes,
        Arc::clone(&upload),
        deadline,
    );
    let endpoint = generation.upstream.target.endpoint(CHAT_PATH);
    let outgoing = generation
        .upstream
        .client
        .post(endpoint)
        .header(CONTENT_TYPE, canonical_content_type())
        .header(CONTENT_LENGTH, framing.content_length)
        .header(REQUEST_ID_HEADER, request_id)
        .body(reqwest::Body::wrap(direct));

    let sent = tokio::select! {
        biased;
        result = outgoing.send() => result,
        () = tokio::time::sleep_until(deadline) => {
            return Err(deadline_fault(&upload));
        }
    };
    let response = match sent {
        Ok(response) => response,
        Err(source) => return Err(send_fault(&upload, &source)),
    };
    match upload_state(&upload)? {
        UploadState::Complete => {}
        UploadState::Failed(fault) => return Err(fault),
        UploadState::Incomplete => return Err(HttpFault::UpstreamProtocolError),
    }

    let response: axum::http::Response<reqwest::Body> = response.into();
    let (parts, body) = response.into_parts();
    let response_headers = sanitize_response(parts.status, &parts.headers)?;
    let relay = DirectResponseBody::new(body, lease);
    let mut downstream = Response::new(Body::new(relay));
    *downstream.status_mut() = parts.status;
    *downstream.headers_mut() = response_headers;
    Ok(downstream)
}

fn upload_state(upload: &SharedUploadState) -> Result<UploadState, HttpFault> {
    upload
        .lock()
        .map(|state| *state)
        .map_err(|_source| HttpFault::InternalError)
}

fn deadline_fault(upload: &SharedUploadState) -> HttpFault {
    match upload_state(upload) {
        Ok(UploadState::Incomplete | UploadState::Failed(HttpFault::RequestTimeout)) => {
            HttpFault::RequestTimeout
        }
        Ok(UploadState::Failed(fault)) => fault,
        Ok(UploadState::Complete) => HttpFault::UpstreamTimeout,
        Err(fault) => fault,
    }
}

fn send_fault(upload: &SharedUploadState, source: &reqwest::Error) -> HttpFault {
    match upload_state(upload) {
        Ok(UploadState::Failed(fault)) => fault,
        Err(fault) => fault,
        Ok(UploadState::Incomplete | UploadState::Complete) if source.is_timeout() => {
            HttpFault::UpstreamTimeout
        }
        Ok(UploadState::Incomplete) if source.is_body() => HttpFault::MalformedRequest,
        Ok(UploadState::Incomplete | UploadState::Complete) => HttpFault::UpstreamProtocolError,
    }
}
