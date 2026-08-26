mod classify;
mod headers;
mod multipart;
pub(crate) mod voice;

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::{Extension, State};
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderValue, Method, Request, Response, Version};
use tokio::sync::Semaphore;

use crate::config::{Config, HttpMediaRoute};
use crate::error::{HttpFault, RouterError};
use crate::http_generation::{
    BufferedBody, DirectRequestBody, DirectResponseBody, SharedUploadState, UploadState,
    classify_with_cpu_slot, read_buffered, reserve_budget, snapshot_upload,
};
use crate::request_id::{CanonicalRequestId, REQUEST_ID_HEADER};
use crate::worker_pool::{
    AdmissionError, CapacityClass, DispatchError, RequestLease, TrustDomain, WorkerPool,
};

use classify::Classified;
pub(crate) use headers::validate_bodyless_request;
use headers::{RequestKind, SuccessProfile};

const SPEECH_PATH: &str = "/v1/audio/speech";
const BATCH_PATH: &str = "/v1/audio/speech/batch";
const TRANSCRIPTION_PATH: &str = "/v1/audio/transcriptions";
const TRANSLATION_PATH: &str = "/v1/audio/translations";

pub(crate) struct HttpMedia {
    pool: Arc<WorkerPool>,
    client: reqwest::Client,
    ordinary_trust: Option<TrustDomain>,
    enabled_routes: Box<[HttpMediaRoute]>,
    buffered_max: u64,
    streamed_max: u64,
    buffered_budget: Arc<Semaphore>,
    classification_slots: Arc<Semaphore>,
    request_timeout: std::time::Duration,
}

impl HttpMediaRoute {
    pub(crate) const fn path(self) -> &'static str {
        match self {
            Self::Speech => SPEECH_PATH,
            Self::SpeechBatch => BATCH_PATH,
            Self::Transcription => TRANSCRIPTION_PATH,
            Self::Translation => TRANSLATION_PATH,
        }
    }

    const fn capacity(self) -> CapacityClass {
        match self {
            Self::Speech => CapacityClass::SpeechHttp,
            Self::SpeechBatch => CapacityClass::SpeechBatch,
            Self::Transcription | Self::Translation => CapacityClass::TranscriptionHttp,
        }
    }

    const fn request_kind(self) -> RequestKind {
        match self {
            Self::Speech | Self::SpeechBatch => RequestKind::Json,
            Self::Transcription | Self::Translation => RequestKind::Multipart,
        }
    }

    const fn opaque_success(self) -> SuccessProfile {
        match self {
            Self::Speech => SuccessProfile::OpaqueSpeech,
            Self::SpeechBatch => SuccessProfile::Json,
            Self::Transcription | Self::Translation => SuccessProfile::OpaqueTranscription,
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
            pool,
            client,
            ordinary_trust: config
                .http_media
                .as_ref()
                .map(|configured| TrustDomain::new(configured.trust_domain.clone())),
            enabled_routes: media.routes.clone().into_boxed_slice(),
            buffered_max: media.buffered_request_max_bytes,
            streamed_max: media.streamed_request_max_bytes,
            buffered_budget: Arc::new(Semaphore::new(buffered_total)),
            classification_slots,
            request_timeout: media.request_timeout(),
        })))
    }

    pub(crate) fn enables(&self, route: HttpMediaRoute) -> bool {
        self.enabled_routes.contains(&route)
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.ordinary_trust
            .as_ref()
            .is_none_or(|trust| self.pool.media_http_ready(trust, &self.enabled_routes))
            && self.pool.voice_owner_ready()
    }

    pub(crate) fn voice_routes_enabled(&self) -> bool {
        self.pool.voice_state_enabled()
    }
}

pub(crate) async fn speech(
    State(media): State<Arc<HttpMedia>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    request: Request<Body>,
) -> Response<Body> {
    outcome(media, request, HttpMediaRoute::Speech, request_id).await
}

pub(crate) async fn batch(
    State(media): State<Arc<HttpMedia>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    request: Request<Body>,
) -> Response<Body> {
    outcome(media, request, HttpMediaRoute::SpeechBatch, request_id).await
}

pub(crate) async fn transcription(
    State(media): State<Arc<HttpMedia>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    request: Request<Body>,
) -> Response<Body> {
    outcome(media, request, HttpMediaRoute::Transcription, request_id).await
}

pub(crate) async fn translation(
    State(media): State<Arc<HttpMedia>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    request: Request<Body>,
) -> Response<Body> {
    outcome(media, request, HttpMediaRoute::Translation, request_id).await
}

async fn outcome(
    media: Arc<HttpMedia>,
    request: Request<Body>,
    route: HttpMediaRoute,
    request_id: CanonicalRequestId,
) -> Response<Body> {
    match handle(media, request, route, request_id.into_header_value()).await {
        Ok(response) => response,
        Err(fault) => fault.into_response(),
    }
}

async fn handle(
    media: Arc<HttpMedia>,
    request: Request<Body>,
    route: HttpMediaRoute,
    request_id: HeaderValue,
) -> Result<Response<Body>, HttpFault> {
    if request.method() != Method::POST {
        return Err(HttpFault::MethodNotAllowed);
    }
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
    let trust = media
        .ordinary_trust
        .as_ref()
        .ok_or(HttpFault::InternalError)?;
    let framing = headers::validate_request(request.headers(), route.request_kind())?;
    let direct_proof = if route != HttpMediaRoute::SpeechBatch
        && framing.content_length.is_some_and(|length| {
            length <= media.streamed_max && !framing.transfer_framed && !framing.has_route_hint
        }) {
        media.pool.content_blind_media_http(trust, route)
    } else {
        None
    };
    let maximum = if direct_proof.is_some() {
        media.streamed_max
    } else {
        media.buffered_max
    };
    if framing
        .content_length
        .is_some_and(|length| length > maximum)
    {
        return Err(HttpFault::RequestBodyTooLarge);
    }
    let mut envelope = Some(media.pool.try_admit_envelope().map_err(map_admission)?);
    let mut admission = if route == HttpMediaRoute::SpeechBatch {
        None
    } else {
        Some(
            media
                .pool
                .try_admit_class(
                    envelope.take().ok_or(HttpFault::InternalError)?,
                    route.capacity(),
                    1,
                )
                .map_err(map_admission)?,
        )
    };
    if let Some(proof) = direct_proof {
        let length = framing.content_length.ok_or(HttpFault::InternalError)?;
        let lease = proof
            .dispatch(admission.take().ok_or(HttpFault::InternalError)?)
            .map_err(map_dispatch)?;
        let state: SharedUploadState = Arc::new(Mutex::new(UploadState::Incomplete));
        let direct = DirectRequestBody::new(
            request.into_body(),
            length,
            media.streamed_max,
            Arc::clone(&state),
            deadline,
        );
        return send_once(
            media,
            reqwest::Body::wrap(direct),
            length,
            framing.content_type,
            lease,
            route,
            route.opaque_success(),
            request_id,
            Some(state),
            deadline,
        )
        .await;
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
    let route_model = framing.route_model;
    let route_stream = framing.route_stream;
    let classify_pool = Arc::clone(&media.pool);
    let classify_trust = trust.clone();
    let (bytes, budget, classified) =
        classify_with_cpu_slot(&media.classification_slots, deadline, move || {
            let classified = classify(
                route,
                &bytes,
                boundary.as_deref(),
                route_model.as_deref(),
                route_stream,
                &classify_pool,
                &classify_trust,
            )?;
            Ok((bytes, budget, classified))
        })
        .await?;
    let admission = match admission {
        Some(admission) => admission,
        None => media
            .pool
            .try_admit_class(
                envelope.take().ok_or(HttpFault::InternalError)?,
                route.capacity(),
                classified.credits,
            )
            .map_err(map_admission)?,
    };
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
        content_type,
        lease,
        route,
        classified.success,
        request_id,
        None,
        deadline,
    )
    .await
}

fn classify(
    route: HttpMediaRoute,
    bytes: &[u8],
    boundary: Option<&[u8]>,
    route_model: Option<&str>,
    route_stream: Option<bool>,
    pool: &WorkerPool,
    trust: &TrustDomain,
) -> Result<Classified, HttpFault> {
    match route {
        HttpMediaRoute::Speech => {
            classify::speech_with_hints(bytes, route_model, route_stream, pool, trust)
        }
        HttpMediaRoute::SpeechBatch => {
            classify::batch_with_hints(bytes, route_model, route_stream, pool, trust)
        }
        HttpMediaRoute::Transcription => classify::transcription_with_hints(
            bytes,
            boundary.ok_or(HttpFault::UnsupportedMediaType)?,
            route_model,
            route_stream,
            pool,
            trust,
        ),
        HttpMediaRoute::Translation => classify::translation_with_hints(
            bytes,
            boundary.ok_or(HttpFault::UnsupportedMediaType)?,
            route_model,
            route_stream,
            pool,
            trust,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
async fn send_once(
    media: Arc<HttpMedia>,
    body: reqwest::Body,
    length: u64,
    content_type: axum::http::HeaderValue,
    lease: RequestLease,
    route: HttpMediaRoute,
    success: SuccessProfile,
    request_id: HeaderValue,
    upload: Option<SharedUploadState>,
    deadline: tokio::time::Instant,
) -> Result<Response<Body>, HttpFault> {
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
        .header(REQUEST_ID_HEADER, request_id);
    let request = request.body(body);
    let sent = tokio::select! {
        biased;
        result = request.send() => result,
        () = tokio::time::sleep_until(deadline) => {
            let fault = deadline_fault(upload.as_ref());
            if fault == HttpFault::UpstreamTimeout {
                lease.request_immediate_probe();
            }
            return Err(fault);
        },
    };
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
        DispatchError::Internal => HttpFault::InternalError,
    }
}
