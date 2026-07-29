mod classify;
mod headers;
mod multipart;
pub(crate) mod voice;

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::State;
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderValue, Request, Response, Version};
use tokio::sync::Semaphore;

use crate::config::{Config, HttpMediaRoute};
use crate::error::{HttpFault, RouterError};
use crate::http_generation::request_body::{
    BufferedBody, DirectRequestBody, SharedUploadState, UploadState,
};
use crate::http_generation::response_body::DirectResponseBody;
use crate::http_generation::{
    classify_with_cpu_slot, read_buffered, reserve_budget, snapshot_upload,
};
use crate::telemetry::Telemetry;
use crate::worker_pool::{
    AdmissionError, CapacityClass, DispatchError, RequestLease, ServiceClass, TrustDomain,
    WorkerPool,
};

use crate::http_generation::headers::{REQUEST_ID_HEADER, canonical_request_id};
use classify::Classified;
use headers::{RequestKind, SuccessProfile};

const SPEECH_PATH: &str = "/v1/audio/speech";
const BATCH_PATH: &str = "/v1/audio/speech/batch";
const TRANSCRIPTION_PATH: &str = "/v1/audio/transcriptions";

pub(crate) struct HttpMedia {
    pool: Arc<WorkerPool>,
    client: reqwest::Client,
    ordinary_trust: Option<TrustDomain>,
    enabled_services: Box<[ServiceClass]>,
    buffered_max: u64,
    streamed_max: u64,
    buffered_budget: Arc<Semaphore>,
    classification_slots: Arc<Semaphore>,
    telemetry: Arc<Telemetry>,
    request_timeout: std::time::Duration,
}

struct RelayMetadata {
    content_type: HeaderValue,
    route: HttpMediaRoute,
    success: SuccessProfile,
    request_id: HeaderValue,
}

impl HttpMediaRoute {
    const fn path(self) -> &'static str {
        match self {
            Self::Speech => SPEECH_PATH,
            Self::SpeechBatch => BATCH_PATH,
            Self::Transcription => TRANSCRIPTION_PATH,
        }
    }

    const fn capacity(self) -> CapacityClass {
        match self {
            Self::Speech => CapacityClass::SpeechHttp,
            Self::SpeechBatch => CapacityClass::SpeechBatch,
            Self::Transcription => CapacityClass::TranscriptionHttp,
        }
    }

    const fn request_kind(self) -> RequestKind {
        match self {
            Self::Speech | Self::SpeechBatch => RequestKind::Json,
            Self::Transcription => RequestKind::Multipart,
        }
    }

    const fn opaque_success(self) -> SuccessProfile {
        match self {
            Self::Speech => SuccessProfile::OpaqueSpeech,
            Self::SpeechBatch => SuccessProfile::Json,
            Self::Transcription => SuccessProfile::OpaqueTranscription,
        }
    }
}

impl HttpMedia {
    pub(crate) fn build(
        config: &Config,
        pool: Arc<WorkerPool>,
        classification_slots: Arc<Semaphore>,
    ) -> Result<Option<Arc<Self>>, RouterError> {
        if config.http_media.is_none() && !pool.voice_state_enabled() {
            return Ok(None);
        }
        let media = config.http_media.clone().unwrap_or_default();
        let buffered_total = media.buffered_total_usize().map_err(RouterError::Config)?;
        let client = pool
            .media_client()
            .ok_or(RouterError::WorkerPoolInvariant)?;
        Ok(Some(Arc::new(Self {
            telemetry: Arc::clone(pool.telemetry()),
            pool,
            client,
            ordinary_trust: config
                .http_media
                .as_ref()
                .map(|media| TrustDomain::new(media.trust_domain.clone())),
            enabled_services: media
                .routes
                .iter()
                .map(|route| route.service_class())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            buffered_max: media.buffered_request_max_bytes,
            streamed_max: media.streamed_request_max_bytes,
            buffered_budget: Arc::new(Semaphore::new(buffered_total)),
            classification_slots,
            request_timeout: media.request_timeout(),
        })))
    }

    pub(crate) fn enables(&self, route: HttpMediaRoute) -> bool {
        self.enabled_services.contains(&route.service_class())
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.ordinary_trust
            .as_ref()
            .is_none_or(|trust| self.pool.media_http_ready(trust, &self.enabled_services))
            && self.pool.voice_owner_ready()
    }

    pub(crate) fn voice_routes_enabled(&self) -> bool {
        self.pool.voice_state_enabled()
    }
}

pub(crate) async fn speech(
    State(media): State<Arc<HttpMedia>>,
    request: Request<Body>,
) -> Response<Body> {
    outcome(handle(media, request, HttpMediaRoute::Speech).await)
}

pub(crate) async fn batch(
    State(media): State<Arc<HttpMedia>>,
    request: Request<Body>,
) -> Response<Body> {
    outcome(handle(media, request, HttpMediaRoute::SpeechBatch).await)
}

pub(crate) async fn transcription(
    State(media): State<Arc<HttpMedia>>,
    request: Request<Body>,
) -> Response<Body> {
    outcome(handle(media, request, HttpMediaRoute::Transcription).await)
}

fn outcome(result: Result<Response<Body>, HttpFault>) -> Response<Body> {
    match result {
        Ok(response) => response,
        Err(fault) => fault.into_response(),
    }
}

async fn handle(
    media: Arc<HttpMedia>,
    request: Request<Body>,
    route: HttpMediaRoute,
) -> Result<Response<Body>, HttpFault> {
    if request.version() != Version::HTTP_11
        || request.uri().path() != route.path()
        || request.uri().query().is_some()
    {
        return Err(if request.version() != Version::HTTP_11 {
            HttpFault::HttpVersionNotSupported
        } else {
            HttpFault::MalformedRequest
        });
    }
    let deadline = tokio::time::Instant::now() + media.request_timeout;
    let framing = headers::validate_request(request.headers(), route.request_kind())?;
    let request_id = canonical_request_id(request.headers())?;
    let admission = media
        .pool
        .try_admit(route.capacity())
        .map_err(map_admission)?;
    let direct_proof = if route != HttpMediaRoute::SpeechBatch
        && framing.content_length.is_some_and(|length| {
            length > media.buffered_max
                && length <= media.streamed_max
                && !framing.transfer_framed
                && !framing.has_route_hint
                && !framing.has_content_encoding
        }) {
        media.pool.homogeneous_media_http(
            media
                .ordinary_trust
                .as_ref()
                .ok_or(HttpFault::InternalError)?,
            route.service_class(),
        )
    } else {
        None
    };
    if let Some(proof) = direct_proof {
        let length = framing.content_length.ok_or(HttpFault::InternalError)?;
        let lease = proof.dispatch(admission).map_err(map_dispatch)?;
        return relay_direct(
            media,
            request.into_body(),
            length,
            lease,
            RelayMetadata {
                content_type: framing.content_type,
                route,
                success: route.opaque_success(),
                request_id,
            },
            deadline,
        )
        .await;
    }
    if framing
        .content_length
        .is_some_and(|length| length > media.buffered_max)
    {
        return Err(HttpFault::RequestBodyTooLarge);
    }
    let reserve = framing.content_length.unwrap_or(media.buffered_max);
    let budget = reserve_budget(&media.buffered_budget, reserve)?;
    let bytes = read_buffered(
        request.into_body(),
        framing.content_length,
        media.buffered_max,
        deadline,
    )
    .await?;
    let content_type = framing.content_type;
    let boundary = framing.boundary;
    let classify_pool = Arc::clone(&media.pool);
    let classify_trust = media
        .ordinary_trust
        .clone()
        .ok_or(HttpFault::InternalError)?;
    let (bytes, budget, classified) = classify_with_cpu_slot(
        &media.classification_slots,
        &media.telemetry,
        route.capacity(),
        deadline,
        move || {
            let classified = classify(
                route,
                &bytes,
                boundary.as_deref(),
                &classify_pool,
                &classify_trust,
            )?;
            Ok((bytes, budget, classified))
        },
    )
    .await?;
    let lease = media
        .pool
        .dispatch(admission, &classified.requirement)
        .map_err(map_dispatch)?;
    let length = u64::try_from(bytes.len()).map_err(|_| HttpFault::InternalError)?;
    let body = reqwest::Body::wrap(BufferedBody::new(bytes, budget));
    send_once(
        media,
        body,
        length,
        lease,
        RelayMetadata {
            content_type,
            route,
            success: classified.success,
            request_id,
        },
        None,
        deadline,
    )
    .await
}

fn classify(
    route: HttpMediaRoute,
    bytes: &[u8],
    boundary: Option<&[u8]>,
    pool: &WorkerPool,
    trust: &TrustDomain,
) -> Result<Classified, HttpFault> {
    match route {
        HttpMediaRoute::Speech => classify::speech(bytes, pool, trust),
        HttpMediaRoute::SpeechBatch => classify::batch(bytes, pool, trust),
        HttpMediaRoute::Transcription => classify::transcription(
            bytes,
            boundary.ok_or(HttpFault::UnsupportedMediaType)?,
            pool,
            trust,
        ),
    }
}

async fn relay_direct(
    media: Arc<HttpMedia>,
    body: Body,
    length: u64,
    lease: RequestLease,
    metadata: RelayMetadata,
    deadline: tokio::time::Instant,
) -> Result<Response<Body>, HttpFault> {
    let state: SharedUploadState = Arc::new(Mutex::new(UploadState::Incomplete));
    let direct = DirectRequestBody::new(
        body,
        length,
        media.streamed_max,
        Arc::clone(&state),
        deadline,
    );
    send_once(
        media,
        reqwest::Body::wrap(direct),
        length,
        lease,
        metadata,
        Some(state),
        deadline,
    )
    .await
}

async fn send_once(
    media: Arc<HttpMedia>,
    body: reqwest::Body,
    length: u64,
    lease: RequestLease,
    metadata: RelayMetadata,
    upload: Option<SharedUploadState>,
    deadline: tokio::time::Instant,
) -> Result<Response<Body>, HttpFault> {
    let RelayMetadata {
        content_type,
        route,
        success,
        request_id,
    } = metadata;
    if tokio::time::Instant::now() >= deadline {
        return Err(deadline_fault(upload.as_ref()));
    }
    let mut url = lease.target().base_url().clone();
    url.set_path(route.path());
    url.set_query(None);
    let request = media
        .client
        .post(url)
        .header(CONTENT_TYPE, content_type)
        .header(CONTENT_LENGTH, length)
        .header(REQUEST_ID_HEADER, request_id)
        .body(body);
    let upstream_headers = lease.upstream_headers_timer();
    let sent = tokio::select! {
        biased;
        result = request.send() => result,
        () = tokio::time::sleep_until(deadline) => return Err(deadline_fault(upload.as_ref())),
    };
    drop(upstream_headers);
    let response = match sent {
        Ok(response) => response,
        Err(_source) => {
            let fault = upload_fault(upload.as_ref())?.unwrap_or(HttpFault::UpstreamProtocolError);
            if fault == HttpFault::UpstreamProtocolError {
                lease.request_immediate_probe();
            }
            return Err(fault);
        }
    };
    if let Some(state) = upload.as_ref() {
        match snapshot_upload(state)? {
            UploadState::Complete => {}
            UploadState::Failed(fault) => return Err(fault),
            UploadState::Incomplete => {
                lease.request_immediate_probe();
                return Err(HttpFault::UpstreamProtocolError);
            }
        }
    }
    let response: axum::http::Response<reqwest::Body> = response.into();
    let (parts, body) = response.into_parts();
    let sanitized = match headers::sanitize_response(parts.status, &parts.headers, success) {
        Ok(headers) => headers,
        Err(fault) => {
            drop(body);
            lease.request_immediate_probe();
            return Err(fault);
        }
    };
    let relay = DirectResponseBody::new(body, lease);
    let mut downstream = Response::new(Body::new(relay));
    *downstream.status_mut() = parts.status;
    *downstream.headers_mut() = sanitized;
    Ok(downstream)
}

fn upload_fault(upload: Option<&SharedUploadState>) -> Result<Option<HttpFault>, HttpFault> {
    upload
        .map(snapshot_upload)
        .transpose()
        .map(|state| match state {
            Some(UploadState::Failed(fault)) => Some(fault),
            Some(UploadState::Incomplete | UploadState::Complete) | None => None,
        })
}

fn deadline_fault(upload: Option<&SharedUploadState>) -> HttpFault {
    match upload.map(snapshot_upload).transpose() {
        Err(fault) => fault,
        Ok(Some(UploadState::Incomplete | UploadState::Failed(HttpFault::RequestTimeout))) => {
            HttpFault::RequestTimeout
        }
        Ok(Some(UploadState::Failed(fault))) => fault,
        Ok(Some(UploadState::Complete) | None) => HttpFault::UpstreamTimeout,
    }
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
        DispatchError::NoEligibleProfile => HttpFault::NoCompatibleWorker,
        DispatchError::Unavailable | DispatchError::Draining => HttpFault::RouterUnavailable,
        DispatchError::Overloaded => HttpFault::RouterOverloaded,
        DispatchError::AdmissionClassMismatch | DispatchError::Internal => HttpFault::InternalError,
    }
}

#[allow(dead_code)]
pub(crate) mod benchmark {
    use super::{classify, multipart};

    pub(crate) fn classify_speech(bytes: &[u8]) -> bool {
        classify::benchmark_speech(bytes)
    }

    pub(crate) fn scan_multipart(bytes: &[u8], boundary: &[u8]) -> bool {
        multipart::scan(bytes, boundary).is_ok()
    }
}
