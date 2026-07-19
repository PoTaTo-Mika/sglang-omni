use std::error::Error as _;
use std::fmt;
use std::future::Future;
use std::sync::Arc;

use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::header::{CONNECTION, UPGRADE};
use axum::http::{HeaderMap, HeaderName, StatusCode, Uri, Version};
use axum::response::{IntoResponse, Response};
use futures_util::{SinkExt, StreamExt};
use serde::Deserializer as _;
use serde::de::{self, DeserializeSeed, IgnoredAny, MapAccess, Visitor};
use tokio::sync::watch;
use tokio_tungstenite::tungstenite::Message as UpstreamMessage;
use tokio_tungstenite::tungstenite::error::CapacityError;
use tracing::error;

use crate::classification;
use crate::config::{Config, WebsocketConfig};
use crate::error::HttpFault;
use crate::speech_facts::{
    SpeechFields, exceeds_nesting_limit, managed_voice as classify_managed_voice,
    read_field as read_shared_speech_field, reference_forms,
    response_format as classify_response_format, task as classify_task,
};
use crate::worker_pool::{
    AdmissionError, CapacityClass, DefaultModelResolution, DispatchError, ModelSelection,
    ProfileRequirement, RealtimeProtocol, RouteRequirement, ServiceClass, SpeechResponseFormat,
    SpeechWebsocketInput, StreamMode, TrustDomain, WorkerPool,
};

mod session;
mod upstream;

pub(crate) use session::SessionTracker;
use session::{DrainState, PendingSession, SessionSupervisor, close_message};

const SPEECH_PATH: &str = "/v1/audio/speech/stream";
const REALTIME_PATH: &str = "/v1/realtime";

fn downstream_text_to_upstream(
    text: axum::extract::ws::Utf8Bytes,
) -> Result<tokio_tungstenite::tungstenite::Utf8Bytes, ()> {
    let bytes: bytes::Bytes = text.into();
    bytes.try_into().map_err(|_| ())
}

fn upstream_text_to_downstream(
    text: tokio_tungstenite::tungstenite::Utf8Bytes,
) -> Result<axum::extract::ws::Utf8Bytes, ()> {
    let bytes: bytes::Bytes = text.into();
    bytes.try_into().map_err(|_| ())
}

/// Immutable terminating WebSocket route state.
pub(crate) struct WebsocketGateway {
    pool: Arc<WorkerPool>,
    policy: WebsocketConfig,
    speech: Option<TrustDomain>,
    realtime: Option<TrustDomain>,
    tracker: SessionTracker,
    classification_slots: Arc<tokio::sync::Semaphore>,
}

impl WebsocketGateway {
    pub(crate) fn build(
        config: &Config,
        pool: Arc<WorkerPool>,
        tracker: SessionTracker,
        classification_slots: Arc<tokio::sync::Semaphore>,
    ) -> Option<Arc<Self>> {
        let policy = config.websocket.clone()?;
        Some(Arc::new(Self {
            pool,
            speech: policy
                .speech
                .as_ref()
                .map(|route| TrustDomain::new(route.trust_domain.clone())),
            realtime: policy
                .realtime
                .as_ref()
                .map(|route| TrustDomain::new(route.trust_domain.clone())),
            policy,
            tracker,
            classification_slots,
        }))
    }

    pub(crate) const fn speech_enabled(&self) -> bool {
        self.speech.is_some()
    }

    pub(crate) const fn realtime_enabled(&self) -> bool {
        self.realtime.is_some()
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.speech.as_ref().is_none_or(|trust| {
            self.pool
                .service_ready(trust, ServiceClass::SpeechWebsocket)
        }) && self.realtime.as_ref().is_none_or(|trust| {
            self.pool
                .service_ready(trust, ServiceClass::RealtimeWebsocket)
        })
    }
}

pub(crate) async fn speech(
    State(gateway): State<Arc<WebsocketGateway>>,
    version: Version,
    uri: Uri,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    let Some(trust) = gateway.speech.clone() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if validate_upgrade(version, &uri, &headers, SPEECH_PATH, &gateway.policy).is_err() {
        return HttpFault::MalformedRequest.into_response();
    }
    let admission = match gateway.pool.try_admit(CapacityClass::SpeechWebsocket) {
        Ok(admission) => admission,
        Err(error) => return admission_fault(error).into_response(),
    };
    let Some(registration) = gateway.tracker.register() else {
        return HttpFault::RouterUnavailable.into_response();
    };
    let pending = PendingSession::new(registration, admission);
    let path_and_query = uri
        .path_and_query()
        .map_or(SPEECH_PATH, axum::http::uri::PathAndQuery::as_str)
        .to_owned();
    let frame_limit = match usize::try_from(gateway.policy.frame_max_bytes) {
        Ok(limit) => limit,
        Err(_) => return HttpFault::InternalError.into_response(),
    };
    let message_limit = match usize::try_from(gateway.policy.speech_config_max_bytes) {
        Ok(limit) => limit,
        Err(_) => return HttpFault::InternalError.into_response(),
    };
    upgrade
        .max_frame_size(frame_limit)
        .max_message_size(message_limit)
        .on_upgrade(move |socket| async move {
            run_speech(socket, gateway, pending, trust, path_and_query, headers).await;
        })
        .into_response()
}

async fn run_speech(
    mut downstream: WebSocket,
    gateway: Arc<WebsocketGateway>,
    pending: PendingSession,
    trust: TrustDomain,
    path_and_query: String,
    headers: HeaderMap,
) {
    let mut drain = pending.drain_receiver();
    let config_text = match setup_or_drain(
        &mut drain,
        receive_speech_config(&mut downstream, &gateway.policy),
    )
    .await
    {
        Ok(Ok(text)) => text,
        Ok(Err(close)) => {
            send_setup_close(&mut downstream, &gateway.policy, &mut drain, close).await;
            return;
        }
        Err(state) => {
            close_setup_for_drain(&mut downstream, &gateway.policy, &mut drain, state).await;
            return;
        }
    };
    let classify_pool = Arc::clone(&gateway.pool);
    let classify_slots = Arc::clone(&gateway.classification_slots);
    let classify_trust = trust.clone();
    let classified = setup_or_drain(
        &mut drain,
        classification::run(&classify_slots, move || {
            let requirement =
                classify_speech(config_text.as_bytes(), &classify_pool, &classify_trust);
            (config_text, requirement)
        }),
    )
    .await;
    let (config_text, requirement) = match classified {
        Ok(Ok((text, Ok(requirement)))) => (text, requirement),
        Ok(Ok((_text, Err(())))) => {
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                close_message(1008, "invalid session.config"),
            )
            .await;
            return;
        }
        Ok(Err(source)) => {
            match source {
                classification::ClassificationError::Unavailable => {
                    error!("classification semaphore closed");
                }
                classification::ClassificationError::Join(source) => {
                    error!(error = %source, "classification task failed");
                }
            }
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                close_message(1011, "internal setup failure"),
            )
            .await;
            return;
        }
        Err(state) => {
            close_setup_for_drain(&mut downstream, &gateway.policy, &mut drain, state).await;
            return;
        }
    };
    let upstream_config = match downstream_text_to_upstream(config_text) {
        Ok(text) => text,
        Err(()) => {
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                close_message(1011, "internal setup failure"),
            )
            .await;
            return;
        }
    };
    let Some((registration, admission)) = pending.into_admitted() else {
        send_setup_close(
            &mut downstream,
            &gateway.policy,
            &mut drain,
            close_message(1011, "internal setup failure"),
        )
        .await;
        return;
    };
    let lease = match gateway.pool.dispatch(admission, &requirement) {
        Ok(lease) => lease,
        Err(error) => {
            let _registration = registration;
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                dispatch_close(error),
            )
            .await;
            return;
        }
    };
    let connected = setup_or_drain(
        &mut drain,
        upstream::connect(lease.target(), &path_and_query, &headers, &gateway.policy),
    )
    .await;
    let upstream = match connected {
        Ok(Ok(upstream)) => upstream,
        Ok(Err(_)) => {
            lease.request_immediate_probe();
            let _registration = registration;
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                close_message(1011, "upstream setup failure"),
            )
            .await;
            return;
        }
        Err(state) => {
            let _registration = registration;
            close_setup_for_drain(&mut downstream, &gateway.policy, &mut drain, state).await;
            return;
        }
    };
    let mut supervisor = SessionSupervisor::from_admitted(registration, lease, upstream);
    let sent = {
        let Some(upstream) = supervisor.upstream_mut() else {
            return;
        };
        setup_or_drain(
            &mut drain,
            upstream.send(UpstreamMessage::Text(upstream_config)),
        )
        .await
    };
    match sent {
        Ok(Ok(())) => {}
        Ok(Err(_)) => {
            supervisor.request_immediate_probe();
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                close_message(1011, "upstream setup failure"),
            )
            .await;
            return;
        }
        Err(state) => {
            close_setup_for_drain(&mut downstream, &gateway.policy, &mut drain, state).await;
            return;
        }
    }
    let first = {
        let Some(upstream) = supervisor.upstream_mut() else {
            return;
        };
        setup_or_drain(&mut drain, next_worker_application(upstream)).await
    };
    let text = match first {
        Ok(Some(Ok(UpstreamMessage::Text(text)))) if is_speech_setup_event(text.as_bytes()) => text,
        Ok(_) => {
            supervisor.request_immediate_probe();
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                close_message(1011, "upstream setup failure"),
            )
            .await;
            return;
        }
        Err(state) => {
            close_setup_for_drain(&mut downstream, &gateway.policy, &mut drain, state).await;
            return;
        }
    };
    let text = match upstream_text_to_downstream(text) {
        Ok(text) => text,
        Err(()) => {
            supervisor.request_immediate_probe();
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                close_message(1011, "upstream setup failure"),
            )
            .await;
            return;
        }
    };
    match setup_or_drain(&mut drain, downstream.send(Message::Text(text))).await {
        Ok(Ok(())) => {}
        Ok(Err(_)) => return,
        Err(state) => {
            close_setup_for_drain(&mut downstream, &gateway.policy, &mut drain, state).await;
            return;
        }
    }
    supervisor
        .relay(downstream, &gateway.policy, false, true)
        .await;
}

pub(crate) async fn realtime(
    State(gateway): State<Arc<WebsocketGateway>>,
    version: Version,
    uri: Uri,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    let Some(trust) = gateway.realtime.clone() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if validate_upgrade(version, &uri, &headers, REALTIME_PATH, &gateway.policy).is_err() {
        return HttpFault::MalformedRequest.into_response();
    }
    let model = match gateway
        .pool
        .resolve_default_model_id(&trust, ServiceClass::RealtimeWebsocket)
    {
        DefaultModelResolution::Unique(model) => model.to_owned(),
        DefaultModelResolution::Ambiguous => return HttpFault::AmbiguousModel.into_response(),
        DefaultModelResolution::NoService => return HttpFault::RouterUnavailable.into_response(),
    };
    let admission = match gateway.pool.try_admit(CapacityClass::RealtimeWebsocket) {
        Ok(admission) => admission,
        Err(error) => return admission_fault(error).into_response(),
    };
    let Some(registration) = gateway.tracker.register() else {
        return HttpFault::RouterUnavailable.into_response();
    };
    let mut drain = registration.drain_receiver();
    let requirement = RouteRequirement::new(
        ProfileRequirement::RealtimeWebsocket {
            protocol: RealtimeProtocol::OpenaiRealtimeV1,
            expected_default_model_id: model,
        },
        trust,
        false,
    );
    let lease = match gateway.pool.dispatch(admission, &requirement) {
        Ok(lease) => lease,
        Err(error) => {
            let _registration = registration;
            return dispatch_fault(error).into_response();
        }
    };
    let path_and_query = uri
        .path_and_query()
        .map_or(REALTIME_PATH, axum::http::uri::PathAndQuery::as_str);
    let connected = setup_or_drain(
        &mut drain,
        upstream::connect(lease.target(), path_and_query, &headers, &gateway.policy),
    )
    .await;
    let upstream = match connected {
        Ok(Ok(upstream)) => upstream,
        Ok(Err(_)) => {
            lease.request_immediate_probe();
            let _registration = registration;
            return HttpFault::RouterUnavailable.into_response();
        }
        Err(_) => {
            let _registration = registration;
            return HttpFault::RouterUnavailable.into_response();
        }
    };
    let supervisor = SessionSupervisor::from_admitted(registration, lease, upstream);
    let frame_limit = match usize::try_from(gateway.policy.frame_max_bytes) {
        Ok(limit) => limit,
        Err(_) => return HttpFault::InternalError.into_response(),
    };
    let message_limit = match usize::try_from(gateway.policy.realtime_message_max_bytes) {
        Ok(limit) => limit,
        Err(_) => return HttpFault::InternalError.into_response(),
    };
    upgrade
        .max_frame_size(frame_limit)
        .max_message_size(message_limit)
        .on_upgrade(move |socket| async move {
            run_realtime(socket, gateway, supervisor).await;
        })
        .into_response()
}

async fn run_realtime(
    mut downstream: WebSocket,
    gateway: Arc<WebsocketGateway>,
    mut supervisor: SessionSupervisor,
) {
    let mut drain = supervisor.drain_receiver();
    let first = {
        let Some(upstream) = supervisor.upstream_mut() else {
            return;
        };
        setup_or_drain(&mut drain, next_worker_application(upstream)).await
    };
    let text = match first {
        Ok(Some(Ok(UpstreamMessage::Text(text))))
            if text.len()
                <= usize::try_from(gateway.policy.worker_message_max_bytes).unwrap_or(0)
                && is_event_type(text.as_bytes(), "session.created") =>
        {
            text
        }
        Ok(_) => {
            supervisor.request_immediate_probe();
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                close_message(1011, "invalid session.created"),
            )
            .await;
            return;
        }
        Err(state) => {
            close_setup_for_drain(&mut downstream, &gateway.policy, &mut drain, state).await;
            return;
        }
    };
    let text = match upstream_text_to_downstream(text) {
        Ok(text) => text,
        Err(()) => {
            supervisor.request_immediate_probe();
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                close_message(1011, "invalid session.created"),
            )
            .await;
            return;
        }
    };
    match setup_or_drain(&mut drain, downstream.send(Message::Text(text))).await {
        Ok(Ok(())) => {}
        Ok(Err(_)) => return,
        Err(state) => {
            close_setup_for_drain(&mut downstream, &gateway.policy, &mut drain, state).await;
            return;
        }
    }
    supervisor
        .relay(downstream, &gateway.policy, true, false)
        .await;
}

async fn setup_or_drain<T>(
    drain: &mut watch::Receiver<DrainState>,
    operation: impl Future<Output = T>,
) -> Result<T, DrainState> {
    let initial = *drain.borrow();
    if initial != DrainState::Serving {
        return Err(initial);
    }
    tokio::pin!(operation);
    tokio::select! {
        biased;
        result = &mut operation => Ok(result),
        changed = drain.changed() => {
            if changed.is_err() {
                Err(DrainState::Forced)
            } else {
                Err(*drain.borrow())
            }
        }
    }
}

async fn close_setup_for_drain(
    downstream: &mut WebSocket,
    policy: &WebsocketConfig,
    drain: &mut watch::Receiver<DrainState>,
    state: DrainState,
) {
    if state == DrainState::Draining {
        send_setup_close(
            downstream,
            policy,
            drain,
            close_message(1012, "service restart"),
        )
        .await;
    }
}

async fn send_setup_close(
    downstream: &mut WebSocket,
    policy: &WebsocketConfig,
    drain: &mut watch::Receiver<DrainState>,
    message: Message,
) {
    tokio::select! {
        _ = tokio::time::timeout(policy.close_timeout(), downstream.send(message)) => {}
        _ = drain.wait_for(|state| *state == DrainState::Forced) => {}
    }
}

async fn receive_speech_config(
    downstream: &mut WebSocket,
    policy: &WebsocketConfig,
) -> Result<axum::extract::ws::Utf8Bytes, Message> {
    let received = tokio::time::timeout(policy.speech_config_timeout(), downstream.next())
        .await
        .map_err(|_| close_message(1008, "session.config timeout"))?;
    match received {
        Some(Ok(Message::Text(text)))
            if text.len() <= usize::try_from(policy.speech_config_max_bytes).unwrap_or(0) =>
        {
            Ok(text)
        }
        Some(Ok(Message::Text(_))) => Err(close_message(1009, "session.config too large")),
        Some(Ok(Message::Binary(bytes)))
            if bytes.len() > usize::try_from(policy.speech_config_max_bytes).unwrap_or(0) =>
        {
            Err(close_message(1009, "session.config too large"))
        }
        Some(Ok(Message::Binary(_))) => Err(close_message(1008, "invalid session.config")),
        Some(Err(error)) if websocket_message_too_large(&error) => {
            Err(close_message(1009, "session.config too large"))
        }
        _ => Err(close_message(1008, "invalid session.config")),
    }
}

fn websocket_message_too_large(error: &axum::Error) -> bool {
    error
        .source()
        .and_then(|source| source.downcast_ref::<tokio_tungstenite::tungstenite::Error>())
        .is_some_and(|error| {
            matches!(
                error,
                tokio_tungstenite::tungstenite::Error::Capacity(
                    CapacityError::MessageTooLong { .. }
                )
            )
        })
}

async fn next_worker_application(
    upstream: &mut upstream::UpstreamSocket,
) -> Option<Result<UpstreamMessage, tokio_tungstenite::tungstenite::Error>> {
    loop {
        match upstream.next().await {
            Some(Ok(
                UpstreamMessage::Ping(_) | UpstreamMessage::Pong(_) | UpstreamMessage::Frame(_),
            )) => {}
            other => return other,
        }
    }
}

fn validate_upgrade(
    version: Version,
    uri: &Uri,
    headers: &HeaderMap,
    path: &str,
    policy: &WebsocketConfig,
) -> Result<(), ()> {
    if version != Version::HTTP_11
        || uri.path() != path
        || uri.to_string().len() > usize::try_from(policy.uri_max_bytes).map_err(|_| ())?
        || headers.len() > usize::try_from(policy.header_max_fields).map_err(|_| ())?
        || headers.contains_key("sec-websocket-protocol")
        || headers.contains_key("sec-websocket-extensions")
    {
        return Err(());
    }
    if !connection_requests_upgrade(headers)? || !is_exact_websocket_upgrade(headers)? {
        return Err(());
    }
    let total = headers.iter().try_fold(0_usize, |total, (name, value)| {
        total
            .checked_add(name.as_str().len())?
            .checked_add(value.as_bytes().len())
    });
    if total.is_none_or(|total| total > usize::try_from(policy.header_max_bytes).unwrap_or(0)) {
        return Err(());
    }
    let origins: Vec<_> = headers.get_all("origin").iter().collect();
    if origins.len() > 1 {
        return Err(());
    }
    if let Some(origin) = origins.first() {
        let origin = origin.to_str().map_err(|_| ())?;
        if origin != "null" {
            let parsed: Uri = origin.parse().map_err(|_| ())?;
            if parsed.scheme().is_none() || parsed.authority().is_none() || parsed.path() != "/" {
                return Err(());
            }
        }
    }
    Ok(())
}

fn connection_requests_upgrade(headers: &HeaderMap) -> Result<bool, ()> {
    let mut found = false;
    let mut count = 0_usize;
    for value in headers.get_all(CONNECTION) {
        let value = value.to_str().map_err(|_| ())?;
        for token in value.split(',') {
            let token = token.trim();
            if token.is_empty() || HeaderName::from_bytes(token.as_bytes()).is_err() {
                return Err(());
            }
            count = count.checked_add(1).ok_or(())?;
            found |= token.eq_ignore_ascii_case("upgrade");
        }
    }
    Ok(count != 0 && found)
}

fn is_exact_websocket_upgrade(headers: &HeaderMap) -> Result<bool, ()> {
    let mut count = 0_usize;
    for value in headers.get_all(UPGRADE) {
        let value = value.to_str().map_err(|_| ())?;
        for token in value.split(',') {
            let token = token.trim();
            if token.is_empty()
                || HeaderName::from_bytes(token.as_bytes()).is_err()
                || !token.eq_ignore_ascii_case("websocket")
            {
                return Err(());
            }
            count = count.checked_add(1).ok_or(())?;
        }
    }
    Ok(count == 1)
}

fn admission_fault(error: AdmissionError) -> HttpFault {
    match error {
        AdmissionError::Overloaded => HttpFault::RouterOverloaded,
        AdmissionError::Draining => HttpFault::RouterUnavailable,
        AdmissionError::Internal => HttpFault::InternalError,
    }
}

fn dispatch_fault(error: DispatchError) -> HttpFault {
    match error {
        DispatchError::NoEligibleProfile => HttpFault::NoCompatibleWorker,
        DispatchError::Overloaded => HttpFault::RouterOverloaded,
        DispatchError::Unavailable | DispatchError::Draining => HttpFault::RouterUnavailable,
        DispatchError::AdmissionClassMismatch | DispatchError::Internal => HttpFault::InternalError,
    }
}

fn dispatch_close(error: DispatchError) -> Message {
    match error {
        DispatchError::Overloaded | DispatchError::Unavailable => {
            close_message(1013, "route unavailable")
        }
        DispatchError::Draining => close_message(1012, "service restart"),
        DispatchError::NoEligibleProfile => close_message(1008, "no compatible worker"),
        DispatchError::AdmissionClassMismatch | DispatchError::Internal => {
            close_message(1011, "internal setup failure")
        }
    }
}

fn classify_speech(
    bytes: &[u8],
    pool: &WorkerPool,
    trust: &TrustDomain,
) -> Result<RouteRequirement, ()> {
    let fields = parse_speech_config(bytes)?;
    let model = match fields.model.clone().flatten() {
        Some(model) if !model.is_empty() => ModelSelection::Explicit(model),
        Some(_) | None => match pool.resolve_default_model_id(trust, ServiceClass::SpeechWebsocket)
        {
            DefaultModelResolution::Unique(model) => ModelSelection::WorkerDefault {
                expected_model_id: model.to_owned(),
            },
            DefaultModelResolution::NoService | DefaultModelResolution::Ambiguous => return Err(()),
        },
    };
    speech_requirement(fields, model, trust)
}

fn speech_requirement(
    fields: SpeechFields,
    model: ModelSelection,
    trust: &TrustDomain,
) -> Result<RouteRequirement, ()> {
    let format = classify_response_format(
        fields
            .response_format
            .as_ref()
            .and_then(Option::as_deref)
            .unwrap_or("pcm"),
    )
    .ok_or(())?;
    let stream_mode = if fields
        .stream
        .as_ref()
        .and_then(|value| *value)
        .unwrap_or(false)
    {
        if format != SpeechResponseFormat::Pcm {
            return Err(());
        }
        StreamMode::Streaming
    } else {
        StreamMode::NonStreaming
    };
    let task = classify_task(
        fields
            .task
            .as_ref()
            .and_then(Option::as_deref)
            .unwrap_or("Base"),
    )
    .ok_or(())?;
    let references = reference_forms(&fields);
    let managed_voice = classify_managed_voice(&fields, &references);
    Ok(RouteRequirement::new(
        ProfileRequirement::SpeechWebsocket {
            model,
            input_profile: SpeechWebsocketInput::Text,
            response_format: format,
            stream_mode,
            task,
            reference_forms: references,
            managed_voice,
        },
        trust.clone(),
        managed_voice,
    ))
}

fn parse_speech_config(bytes: &[u8]) -> Result<SpeechFields, ()> {
    if exceeds_nesting_limit(bytes) {
        return Err(());
    }
    let source = select_speech_config_source(bytes)?;
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let parsed = ConfigEnvelopeSeed(source)
        .deserialize(&mut deserializer)
        .map_err(|_| ())?;
    deserializer.end().map_err(|_| ())?;
    parsed
}

#[derive(Clone, Copy)]
enum SpeechConfigSource {
    Flat,
    Nested,
}

fn select_speech_config_source(bytes: &[u8]) -> Result<SpeechConfigSource, ()> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let envelope = <ConfigSourceEnvelope as serde::Deserialize>::deserialize(&mut deserializer)
        .map_err(|_| ())?;
    deserializer.end().map_err(|_| ())?;
    if envelope.message_type.as_deref() != Some("session.config") {
        return Err(());
    }
    Ok(if envelope.session.is_some() {
        SpeechConfigSource::Nested
    } else {
        SpeechConfigSource::Flat
    })
}

#[derive(serde::Deserialize)]
struct ConfigSourceEnvelope {
    #[serde(rename = "type")]
    message_type: Option<String>,
    session: Option<IgnoredSessionObject>,
}

#[derive(serde::Deserialize)]
struct IgnoredSessionObject {}

struct ConfigEnvelopeSeed(SpeechConfigSource);

impl<'de> DeserializeSeed<'de> for ConfigEnvelopeSeed {
    type Value = Result<SpeechFields, ()>;
    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(ConfigEnvelopeVisitor(self.0))
    }
}

struct ConfigEnvelopeVisitor(SpeechConfigSource);

impl<'de> Visitor<'de> for ConfigEnvelopeVisitor {
    type Value = Result<SpeechFields, ()>;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a session.config object")
    }
    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut nested = None;
        let mut flat = SpeechFields::default();
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "session" => set_once(
                    &mut nested,
                    map.next_value_seed(NullableSpeechFieldsSeed)?,
                    "session",
                )?,
                _ if matches!(self.0, SpeechConfigSource::Flat) => {
                    read_speech_field(&key, &mut map, &mut flat)?;
                }
                _ => {
                    let _ignored = map.next_value::<IgnoredAny>()?;
                }
            }
        }
        match self.0 {
            SpeechConfigSource::Flat => Ok(Ok(flat)),
            SpeechConfigSource::Nested => Ok(nested.flatten().ok_or(())),
        }
    }
}

struct NullableSpeechFieldsSeed;

impl<'de> DeserializeSeed<'de> for NullableSpeechFieldsSeed {
    type Value = Option<SpeechFields>;
    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct OptionalVisitor;
        impl<'de> Visitor<'de> for OptionalVisitor {
            type Value = Option<SpeechFields>;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("null or a session object")
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
                deserializer.deserialize_map(SpeechFieldsVisitor).map(Some)
            }
        }
        deserializer.deserialize_option(OptionalVisitor)
    }
}

struct SpeechFieldsVisitor;

impl<'de> Visitor<'de> for SpeechFieldsVisitor {
    type Value = SpeechFields;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a speech session object")
    }
    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut fields = SpeechFields::default();
        while let Some(key) = map.next_key::<String>()? {
            read_speech_field(&key, &mut map, &mut fields)?;
        }
        Ok(fields)
    }
}

fn read_speech_field<'de, A>(
    key: &str,
    map: &mut A,
    fields: &mut SpeechFields,
) -> Result<(), A::Error>
where
    A: MapAccess<'de>,
{
    if key == "stream_audio" {
        return set_once(&mut fields.stream, map.next_value()?, "stream_audio");
    }
    if !read_shared_speech_field(key, map, fields)? {
        let _ignored = map.next_value::<IgnoredAny>()?;
    }
    Ok(())
}

fn is_event_type(bytes: &[u8], expected: &str) -> bool {
    struct EventVisitor<'a>(&'a str);
    impl<'de> Visitor<'de> for EventVisitor<'_> {
        type Value = bool;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a realtime event object")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut event_type = None;
            while let Some(key) = map.next_key::<String>()? {
                if key == "type" {
                    set_once(&mut event_type, map.next_value::<Option<String>>()?, "type")?;
                } else {
                    let _ignored = map.next_value::<IgnoredAny>()?;
                }
            }
            Ok(event_type.flatten().as_deref() == Some(self.0))
        }
    }
    if exceeds_nesting_limit(bytes) {
        return false;
    }
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    deserializer
        .deserialize_map(EventVisitor(expected))
        .is_ok_and(|valid| valid)
        && deserializer.end().is_ok()
}

fn is_speech_setup_event(bytes: &[u8]) -> bool {
    is_event_type(bytes, "session.configured") || is_event_type(bytes, "error")
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

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use axum::http::header::{CONNECTION, UPGRADE};
    use axum::http::{HeaderMap, HeaderValue, Uri, Version};

    use crate::config::WebsocketConfig;
    use crate::worker_pool::{
        ModelSelection, ProfileRequirement, ReferenceForm, SpeechResponseFormat, SpeechTask,
        StreamMode, TrustDomain,
    };

    use super::{
        is_event_type, is_speech_setup_event, parse_speech_config, speech_requirement,
        validate_upgrade, websocket_message_too_large,
    };

    #[test]
    fn speech_config_accepts_flat_and_nested_and_rejects_duplicates() {
        assert!(parse_speech_config(br#"{"type":"session.config","model":"tts"}"#).is_ok());
        assert!(
            parse_speech_config(br#"{"type":"session.config","session":{"model":"tts"}}"#).is_ok()
        );
        assert!(
            parse_speech_config(br#"{"type":"session.config","model":"a","model":"b"}"#).is_err()
        );
        assert!(parse_speech_config(br#"{"type":"input.text"}"#).is_err());
    }

    #[test]
    fn nested_speech_config_ignores_all_flat_fallback_fields() {
        for bytes in [
            br#"{"type":"session.config","model":7,"model":{"ignored":true},"session":{"model":"nested"}}"#
                .as_slice(),
            br#"{"type":"session.config","session":{"model":"nested"},"stream_audio":"invalid","stream_audio":false}"#
                .as_slice(),
        ] {
            let fields = parse_speech_config(bytes).expect("nested session is authoritative");
            assert_eq!(
                fields.model.as_ref().and_then(Option::as_deref),
                Some("nested")
            );
        }
    }

    #[test]
    fn null_session_keeps_flat_fallback_strict_and_duplicate_aware() {
        assert!(
            parse_speech_config(br#"{"type":"session.config","model":7,"session":null}"#).is_err()
        );
        assert!(
            parse_speech_config(
                br#"{"type":"session.config","session":null,"model":"a","model":"b"}"#
            )
            .is_err()
        );
    }

    #[test]
    fn nested_session_retains_envelope_and_active_field_validation() {
        assert!(
            parse_speech_config(
                br#"{"type":"session.config","session":{"model":"a","model":"b"}}"#
            )
            .is_err()
        );
        assert!(
            parse_speech_config(br#"{"type":"session.config","session":{},"session":{}}"#).is_err()
        );
        assert!(
            parse_speech_config(
                br#"{"type":"session.config","type":"session.config","session":{}}"#
            )
            .is_err()
        );
        assert!(
            parse_speech_config(br#"{"type":"session.config","session":"not-an-object"}"#).is_err()
        );
    }

    #[test]
    fn speech_requirement_preserves_every_selection_dimension() {
        let fields = parse_speech_config(
            br#"{"type":"session.config","model":"tts","response_format":"pcm","stream_audio":true,"task_type":"CustomVoice","voice":"named","ref_audio":"direct","references":[{"audio":"list"},{"vq_codes":[1]}],"split_granularity":"clause"}"#,
        )
        .expect("valid mixed speech configuration");
        let requirement = speech_requirement(
            fields,
            ModelSelection::Explicit(String::from("tts")),
            &TrustDomain::new(String::from("local")),
        )
        .expect("valid speech requirement");
        let ProfileRequirement::SpeechWebsocket {
            model,
            response_format,
            stream_mode,
            task,
            reference_forms,
            managed_voice,
            ..
        } = requirement.profile()
        else {
            panic!("speech websocket requirement")
        };
        assert_eq!(model.model_id(), "tts");
        assert_eq!(*response_format, SpeechResponseFormat::Pcm);
        assert_eq!(*stream_mode, StreamMode::Streaming);
        assert_eq!(*task, SpeechTask::VoiceClone);
        assert_eq!(
            reference_forms,
            &[
                ReferenceForm::Direct,
                ReferenceForm::List,
                ReferenceForm::VqCodes
            ]
        );
        assert!(
            !managed_voice,
            "explicit references avoid managed voice routing"
        );

        let encoded_stream = parse_speech_config(
            br#"{"type":"session.config","response_format":"mp3","stream_audio":true}"#,
        )
        .expect("routing fields parse before relationship validation");
        assert!(
            speech_requirement(
                encoded_stream,
                ModelSelection::Explicit(String::from("tts")),
                &TrustDomain::new(String::from("local")),
            )
            .is_err()
        );
    }

    #[test]
    fn speech_reference_aliases_follow_direct_omni_precedence_without_rejection() {
        let fields = parse_speech_config(
            br#"{"type":"session.config","references":[{"audio_path":null,"ref_audio":"first","audio":"second","data":"third"}]}"#,
        )
        .expect("valid direct reference aliases");
        let requirement = speech_requirement(
            fields,
            ModelSelection::Explicit(String::from("tts")),
            &TrustDomain::new(String::from("local")),
        )
        .expect("valid speech requirement");
        let ProfileRequirement::SpeechWebsocket {
            reference_forms, ..
        } = requirement.profile()
        else {
            panic!("speech websocket requirement")
        };
        assert_eq!(reference_forms, &[ReferenceForm::List]);
    }

    #[test]
    fn realtime_first_event_is_duplicate_aware() {
        assert!(is_event_type(
            br#"{"type":"session.created","session":{}}"#,
            "session.created"
        ));
        assert!(!is_event_type(
            br#"{"type":"session.created","type":"other"}"#,
            "session.created"
        ));
        assert!(!is_event_type(
            br#"{"type":"session.updated"}"#,
            "session.created"
        ));
        assert!(is_event_type(
            br#"{"type":"session.configured"}"#,
            "session.configured"
        ));
        assert!(is_speech_setup_event(
            br#"{"type":"error","message":"worker validation"}"#
        ));
        assert!(!is_speech_setup_event(br#"{"type":"audio.start"}"#));
    }

    #[test]
    fn axum_capacity_error_is_recognized_as_message_too_large() {
        let error = axum::Error::new(tokio_tungstenite::tungstenite::Error::Capacity(
            tokio_tungstenite::tungstenite::error::CapacityError::MessageTooLong {
                size: 11,
                max_size: 10,
            },
        ));
        assert!(websocket_message_too_large(&error));
    }

    #[test]
    fn handshake_tokens_are_exact_not_axum_substrings() {
        let uri: Uri = "/v1/realtime".parse().expect("valid route URI");
        let policy = WebsocketConfig::default();
        let mut headers = HeaderMap::new();
        headers.insert(CONNECTION, HeaderValue::from_static("keep-alive, Upgrade"));
        headers.insert(UPGRADE, HeaderValue::from_static("websocket"));
        assert!(
            validate_upgrade(Version::HTTP_11, &uri, &headers, "/v1/realtime", &policy).is_ok()
        );

        headers.insert(CONNECTION, HeaderValue::from_static("xupgrade"));
        assert!(
            validate_upgrade(Version::HTTP_11, &uri, &headers, "/v1/realtime", &policy).is_err()
        );

        headers.insert(CONNECTION, HeaderValue::from_static("upgrade"));
        headers.append(UPGRADE, HeaderValue::from_static("websocket"));
        assert!(
            validate_upgrade(Version::HTTP_11, &uri, &headers, "/v1/realtime", &policy).is_err()
        );
    }
}
