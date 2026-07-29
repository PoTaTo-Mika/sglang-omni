use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::header::{CACHE_CONTROL, CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderValue, Method, Response, StatusCode, Version};
use axum::middleware::Next;
use bytes::Bytes;
use serde::Serialize;

use crate::config::Config;
use crate::error::{HttpFault, RouterError};
use crate::http_generation::headers::{REQUEST_ID_HEADER, validate_bodyless_request};
use crate::lifecycle::State as LifecycleState;
use crate::telemetry::Telemetry;
use crate::worker_pool::{CapacityClass, OperationsSnapshot};

const JSON_CONTENT_TYPE: &str = "application/json";
const METRICS_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";
const LIFECYCLE_STATES: [&str; 5] = ["starting", "serving", "draining", "stopped", "failed"];
const HEALTH_STATES: [&str; 3] = ["unknown", "healthy", "unhealthy"];
const DISPOSITIONS: [&str; 2] = ["serving", "draining"];
static NEXT_PROCESS_INSTANCE: AtomicU64 = AtomicU64::new(0);

pub(crate) struct Operations {
    models: Bytes,
}

impl Operations {
    pub(crate) fn build(config: &Config) -> Result<Self, RouterError> {
        let profile_ids = config.workers.iter().flat_map(|worker| {
            worker
                .service_profiles
                .iter()
                .filter_map(|profile| profile.model_ids())
        });
        let defaults = config
            .workers
            .iter()
            .filter_map(|worker| worker.default_model_id.as_deref());
        Ok(Self {
            models: render_model_sources(profile_ids, defaults)?,
        })
    }

    pub(crate) fn models_response(&self) -> Response<Body> {
        response(StatusCode::OK, JSON_CONTENT_TYPE, self.models.clone())
    }
}

fn render_model_sources<'a>(
    profile_ids: impl Iterator<Item = &'a [String]>,
    defaults: impl Iterator<Item = &'a str>,
) -> Result<Bytes, RouterError> {
    render_models(profile_ids.flatten().map(String::as_str).chain(defaults))
}

pub(crate) fn validate_get(request: &Request, path: &str) -> Result<(), HttpFault> {
    if request.version() != Version::HTTP_11 {
        return Err(HttpFault::HttpVersionNotSupported);
    }
    if request.method() != Method::GET
        || request.uri().path() != path
        || request.uri().query().is_some()
    {
        return Err(HttpFault::MalformedRequest);
    }
    validate_bodyless_request(request.headers())
}

pub(crate) fn metrics_response(
    lifecycle: LifecycleState,
    ready: bool,
    snapshot: &OperationsSnapshot,
    telemetry: &Telemetry,
) -> Response<Body> {
    response(
        StatusCode::OK,
        METRICS_CONTENT_TYPE,
        Bytes::from(render_metrics(lifecycle, ready, snapshot, telemetry)),
    )
}

pub(crate) fn diagnostics_response(
    lifecycle: LifecycleState,
    ready: bool,
    snapshot: &OperationsSnapshot,
) -> Result<Response<Body>, HttpFault> {
    let diagnostics = Diagnostics::from_snapshot(lifecycle, ready, snapshot);
    let bytes = serde_json::to_vec(&diagnostics).map_err(|_| HttpFault::InternalError)?;
    Ok(response(
        StatusCode::OK,
        JSON_CONTENT_TYPE,
        Bytes::from(bytes),
    ))
}

fn response(status: StatusCode, content_type: &'static str, bytes: Bytes) -> Response<Body> {
    let mut response = Response::new(Body::from(bytes.clone()));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    response
        .headers_mut()
        .insert(CONTENT_LENGTH, HeaderValue::from(bytes.len()));
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

#[derive(Serialize)]
struct ModelList<'a> {
    object: &'static str,
    data: Vec<ModelCard<'a>>,
}

#[derive(Serialize)]
struct ModelCard<'a> {
    id: &'a str,
    object: &'static str,
    created: u8,
    owned_by: &'static str,
    permission: [ModelPermission; 1],
    root: &'a str,
}

#[derive(Clone, Copy, Serialize)]
struct ModelPermission {
    id: &'static str,
    object: &'static str,
    allow_create_engine: bool,
    allow_sampling: bool,
    allow_logprobs: bool,
}

fn render_models<'a>(ids: impl Iterator<Item = &'a str>) -> Result<Bytes, RouterError> {
    let ids: BTreeSet<_> = ids.collect();
    let permission = ModelPermission {
        id: "modelperm-default",
        object: "model_permission",
        allow_create_engine: false,
        allow_sampling: true,
        allow_logprobs: true,
    };
    let list = ModelList {
        object: "list",
        data: ids
            .into_iter()
            .map(|id| ModelCard {
                id,
                object: "model",
                created: 0,
                owned_by: "sglang-omni",
                permission: [permission],
                root: id,
            })
            .collect(),
    };
    serde_json::to_vec(&list)
        .map(Bytes::from)
        .map_err(|_| RouterError::WorkerPoolInvariant)
}

fn render_metrics(
    lifecycle: LifecycleState,
    ready: bool,
    snapshot: &OperationsSnapshot,
    telemetry: &Telemetry,
) -> String {
    let mut output = String::new();
    output.push_str("# HELP sglang_omni_router_lifecycle Router lifecycle state.\n");
    output.push_str("# TYPE sglang_omni_router_lifecycle gauge\n");
    for state in LIFECYCLE_STATES {
        let value = u8::from(state == lifecycle.label());
        let _ = writeln!(
            output,
            "sglang_omni_router_lifecycle{{state=\"{state}\"}} {value}"
        );
    }
    output.push_str("# HELP sglang_omni_router_ready Router readiness state.\n");
    output.push_str("# TYPE sglang_omni_router_ready gauge\n");
    let _ = writeln!(output, "sglang_omni_router_ready {}", u8::from(ready));

    output.push_str("# HELP sglang_omni_router_workers_by_health Workers by health state.\n");
    output.push_str("# TYPE sglang_omni_router_workers_by_health gauge\n");
    for health in HEALTH_STATES {
        let count = snapshot
            .workers
            .iter()
            .filter(|worker| worker.health.label() == health)
            .count();
        let _ = writeln!(
            output,
            "sglang_omni_router_workers_by_health{{health=\"{health}\"}} {count}"
        );
    }
    output.push_str(
        "# HELP sglang_omni_router_workers_by_disposition Workers by disposition state.\n",
    );
    output.push_str("# TYPE sglang_omni_router_workers_by_disposition gauge\n");
    for disposition in DISPOSITIONS {
        let count = snapshot
            .workers
            .iter()
            .filter(|worker| worker.disposition.label() == disposition)
            .count();
        let _ = writeln!(
            output,
            "sglang_omni_router_workers_by_disposition{{disposition=\"{disposition}\"}} {count}"
        );
    }

    render_admission_metrics(&mut output, snapshot);
    render_worker_capacity_metrics(&mut output, snapshot);
    telemetry.render(&mut output);
    output
}

fn render_admission_metrics(output: &mut String, snapshot: &OperationsSnapshot) {
    output.push_str("# HELP sglang_omni_router_admission_limit Configured admission limit.\n");
    output.push_str("# TYPE sglang_omni_router_admission_limit gauge\n");
    for entry in &snapshot.admission {
        let class = entry.class.label();
        let _ = writeln!(
            output,
            "sglang_omni_router_admission_limit{{class=\"{class}\"}} {}",
            entry.limit
        );
    }
    output.push_str(
        "# HELP sglang_omni_router_admission_in_flight Current admitted requests and sessions.\n",
    );
    output.push_str("# TYPE sglang_omni_router_admission_in_flight gauge\n");
    for entry in &snapshot.admission {
        let class = entry.class.label();
        let _ = writeln!(
            output,
            "sglang_omni_router_admission_in_flight{{class=\"{class}\"}} {}",
            entry.in_flight
        );
    }
}

fn render_worker_capacity_metrics(output: &mut String, snapshot: &OperationsSnapshot) {
    let mut limits = [0_usize; 7];
    let mut in_flight = [0_usize; 7];
    for worker in &snapshot.workers {
        for capacity in &worker.capacity {
            let index = capacity_index(capacity.class);
            limits[index] += capacity.limit;
            in_flight[index] += capacity.in_flight;
        }
    }
    output.push_str(
        "# HELP sglang_omni_router_worker_capacity_limit Aggregate configured worker capacity.\n",
    );
    output.push_str("# TYPE sglang_omni_router_worker_capacity_limit gauge\n");
    for (index, class) in CapacityClass::ALL.into_iter().enumerate() {
        let _ = writeln!(
            output,
            "sglang_omni_router_worker_capacity_limit{{class=\"{}\"}} {}",
            class.label(),
            limits[index]
        );
    }
    output.push_str(
        "# HELP sglang_omni_router_worker_capacity_in_flight Aggregate current worker capacity use.\n",
    );
    output.push_str("# TYPE sglang_omni_router_worker_capacity_in_flight gauge\n");
    for (index, class) in CapacityClass::ALL.into_iter().enumerate() {
        let _ = writeln!(
            output,
            "sglang_omni_router_worker_capacity_in_flight{{class=\"{}\"}} {}",
            class.label(),
            in_flight[index]
        );
    }
}

const fn capacity_index(class: CapacityClass) -> usize {
    match class {
        CapacityClass::GenerationHttp => 0,
        CapacityClass::SpeechHttp => 1,
        CapacityClass::TranscriptionHttp => 2,
        CapacityClass::SpeechBatch => 3,
        CapacityClass::SpeechWebsocket => 4,
        CapacityClass::RealtimeWebsocket => 5,
        CapacityClass::Control => 6,
    }
}

#[derive(Serialize)]
struct Diagnostics<'a> {
    lifecycle: &'static str,
    ready: bool,
    admission: Vec<DiagnosticCapacity>,
    workers: Vec<DiagnosticWorker<'a>>,
}

#[derive(Serialize)]
struct DiagnosticCapacity {
    class: &'static str,
    limit: usize,
    in_flight: usize,
}

#[derive(Serialize)]
struct DiagnosticWorker<'a> {
    worker_id: &'a str,
    registration_ordinal: usize,
    health: &'static str,
    disposition: &'static str,
    capacity: Vec<DiagnosticCapacity>,
}

impl<'a> Diagnostics<'a> {
    fn from_snapshot(
        lifecycle: LifecycleState,
        ready: bool,
        snapshot: &'a OperationsSnapshot,
    ) -> Self {
        Self {
            lifecycle: lifecycle.label(),
            ready,
            admission: snapshot
                .admission
                .iter()
                .map(|entry| DiagnosticCapacity {
                    class: entry.class.label(),
                    limit: entry.limit,
                    in_flight: entry.in_flight,
                })
                .collect(),
            workers: snapshot
                .workers
                .iter()
                .map(|worker| DiagnosticWorker {
                    worker_id: &worker.worker_id,
                    registration_ordinal: worker.registration_ordinal,
                    health: worker.health.label(),
                    disposition: worker.disposition.label(),
                    capacity: worker
                        .capacity
                        .iter()
                        .map(|entry| DiagnosticCapacity {
                            class: entry.class.label(),
                            limit: entry.limit,
                            in_flight: entry.in_flight,
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

pub(crate) struct RequestContext {
    prefix: String,
    sequence: AtomicU64,
}

impl RequestContext {
    pub(crate) fn new() -> Result<Arc<Self>, RouterError> {
        let instance = NEXT_PROCESS_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| RouterError::WorkerPoolInvariant)?;
        Ok(Arc::new(Self {
            prefix: format!("sglang-omni-{}-{instance}", std::process::id()),
            sequence: AtomicU64::new(0),
        }))
    }

    fn generate(&self) -> Option<HeaderValue> {
        let sequence = self
            .sequence
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .ok()?;
        HeaderValue::from_str(&format!("{}-{sequence}", self.prefix)).ok()
    }
}

pub(crate) async fn request_context(
    State(context): State<Arc<RequestContext>>,
    mut request: Request,
    next: Next,
) -> Response<Body> {
    let canonical = {
        let mut values = request.headers().get_all(REQUEST_ID_HEADER).iter();
        match (values.next(), values.next()) {
            (None, None) => context.generate(),
            (Some(value), None) if valid_request_id(value) => Some(value.clone()),
            _ => {
                let Some(generated) = context.generate() else {
                    return HttpFault::InternalError.into_response();
                };
                let mut response = HttpFault::MalformedRequest.into_response();
                response.headers_mut().insert(REQUEST_ID_HEADER, generated);
                return response;
            }
        }
    };
    let Some(canonical) = canonical else {
        return HttpFault::InternalError.into_response();
    };
    request.headers_mut().remove(REQUEST_ID_HEADER);
    request
        .headers_mut()
        .insert(REQUEST_ID_HEADER, canonical.clone());
    let mut response = next.run(request).await;
    response.headers_mut().insert(REQUEST_ID_HEADER, canonical);
    response
}

fn valid_request_id(value: &HeaderValue) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty() && bytes.len() <= 128 && bytes.iter().all(|byte| matches!(byte, 0x21..=0x7e))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use std::collections::{BTreeSet, HashSet};
    use std::sync::Arc;

    use axum::http::HeaderValue;

    use crate::lifecycle::State as LifecycleState;
    use crate::worker_pool::{
        AdmissionClass, AdmissionSnapshot, CapacityClass, CapacitySnapshot, Disposition,
        HealthState, OperationsSnapshot, WorkerSnapshot,
    };

    use super::{
        Diagnostics, RequestContext, render_metrics, render_model_sources, render_models,
        valid_request_id,
    };

    fn admission(class: AdmissionClass, limit: usize, in_flight: usize) -> AdmissionSnapshot {
        AdmissionSnapshot {
            class,
            limit,
            in_flight,
        }
    }

    fn capacity(class: CapacityClass, limit: usize, in_flight: usize) -> CapacitySnapshot {
        CapacitySnapshot {
            class,
            limit,
            in_flight,
        }
    }

    fn representative_snapshot() -> OperationsSnapshot {
        OperationsSnapshot {
            admission: [
                admission(AdmissionClass::Global, 100, 0),
                admission(
                    AdmissionClass::Capacity(CapacityClass::GenerationHttp),
                    101,
                    1,
                ),
                admission(AdmissionClass::Capacity(CapacityClass::SpeechHttp), 102, 2),
                admission(
                    AdmissionClass::Capacity(CapacityClass::TranscriptionHttp),
                    103,
                    3,
                ),
                admission(AdmissionClass::Capacity(CapacityClass::SpeechBatch), 104, 4),
                admission(
                    AdmissionClass::Capacity(CapacityClass::SpeechWebsocket),
                    105,
                    5,
                ),
                admission(
                    AdmissionClass::Capacity(CapacityClass::RealtimeWebsocket),
                    106,
                    6,
                ),
                admission(AdmissionClass::Capacity(CapacityClass::Control), 107, 7),
            ],
            workers: vec![
                WorkerSnapshot {
                    worker_id: String::from("worker-a"),
                    registration_ordinal: 0,
                    health: HealthState::Unknown,
                    disposition: Disposition::Serving,
                    capacity: vec![
                        capacity(CapacityClass::GenerationHttp, 10, 1),
                        capacity(CapacityClass::SpeechHttp, 20, 2),
                        capacity(CapacityClass::TranscriptionHttp, 30, 3),
                        capacity(CapacityClass::SpeechBatch, 40, 4),
                        capacity(CapacityClass::SpeechWebsocket, 50, 5),
                        capacity(CapacityClass::RealtimeWebsocket, 60, 6),
                        capacity(CapacityClass::Control, 70, 7),
                    ],
                },
                WorkerSnapshot {
                    worker_id: String::from("worker-b"),
                    registration_ordinal: 1,
                    health: HealthState::Healthy,
                    disposition: Disposition::Serving,
                    capacity: vec![capacity(CapacityClass::GenerationHttp, 2, 2)],
                },
                WorkerSnapshot {
                    worker_id: String::from("worker-c"),
                    registration_ordinal: 2,
                    health: HealthState::Unhealthy,
                    disposition: Disposition::Draining,
                    capacity: vec![capacity(CapacityClass::Control, 3, 1)],
                },
            ],
        }
    }

    #[test]
    fn model_bytes_are_exact_sorted_and_deduplicated() {
        let bytes = render_models(["zeta", "alpha", "zeta"].into_iter())
            .expect("serialize fixed model schema");
        assert_eq!(
            bytes.as_ref(),
            br#"{"object":"list","data":[{"id":"alpha","object":"model","created":0,"owned_by":"sglang-omni","permission":[{"id":"modelperm-default","object":"model_permission","allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true}],"root":"alpha"},{"id":"zeta","object":"model","created":0,"owned_by":"sglang-omni","permission":[{"id":"modelperm-default","object":"model_permission","allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true}],"root":"zeta"}]}"#
        );
        assert_eq!(
            render_models(std::iter::empty())
                .expect("serialize empty model list")
                .as_ref(),
            br#"{"object":"list","data":[]}"#
        );
    }

    #[test]
    fn profile_and_realtime_default_models_form_one_sorted_exact_union() {
        let first = vec![String::from("zeta"), String::from("shared")];
        let second = vec![String::from("alpha"), String::from("shared")];
        let bytes = render_model_sources(
            [first.as_slice(), second.as_slice()].into_iter(),
            ["realtime-only", "alpha"].into_iter(),
        )
        .expect("serialize model sources");
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).expect("parse canonical models JSON");
        let ids: Vec<_> = value["data"]
            .as_array()
            .expect("model data array")
            .iter()
            .map(|card| card["id"].as_str().expect("model id"))
            .collect();
        assert_eq!(ids, ["alpha", "realtime-only", "shared", "zeta"]);
    }

    #[tokio::test]
    async fn metrics_text_is_complete_exact_and_fixed_order() {
        let telemetry =
            crate::telemetry::Telemetry::new(["worker-a", "worker-b", "worker-c"].into_iter());
        let rendered = render_metrics(
            LifecycleState::Serving,
            true,
            &representative_snapshot(),
            &telemetry,
        );
        assert!(
            rendered.starts_with(concat!(
                "# HELP sglang_omni_router_lifecycle Router lifecycle state.\n",
                "# TYPE sglang_omni_router_lifecycle gauge\n",
                "sglang_omni_router_lifecycle{state=\"starting\"} 0\n",
                "sglang_omni_router_lifecycle{state=\"serving\"} 1\n",
                "sglang_omni_router_lifecycle{state=\"draining\"} 0\n",
                "sglang_omni_router_lifecycle{state=\"stopped\"} 0\n",
                "sglang_omni_router_lifecycle{state=\"failed\"} 0\n",
                "# HELP sglang_omni_router_ready Router readiness state.\n",
                "# TYPE sglang_omni_router_ready gauge\n",
                "sglang_omni_router_ready 1\n",
                "# HELP sglang_omni_router_workers_by_health Workers by health state.\n",
                "# TYPE sglang_omni_router_workers_by_health gauge\n",
                "sglang_omni_router_workers_by_health{health=\"unknown\"} 1\n",
                "sglang_omni_router_workers_by_health{health=\"healthy\"} 1\n",
                "sglang_omni_router_workers_by_health{health=\"unhealthy\"} 1\n",
                "# HELP sglang_omni_router_workers_by_disposition Workers by disposition state.\n",
                "# TYPE sglang_omni_router_workers_by_disposition gauge\n",
                "sglang_omni_router_workers_by_disposition{disposition=\"serving\"} 2\n",
                "sglang_omni_router_workers_by_disposition{disposition=\"draining\"} 1\n",
                "# HELP sglang_omni_router_admission_limit Configured admission limit.\n",
                "# TYPE sglang_omni_router_admission_limit gauge\n",
                "sglang_omni_router_admission_limit{class=\"global\"} 100\n",
                "sglang_omni_router_admission_limit{class=\"generation_http\"} 101\n",
                "sglang_omni_router_admission_limit{class=\"speech_http\"} 102\n",
                "sglang_omni_router_admission_limit{class=\"transcription_http\"} 103\n",
                "sglang_omni_router_admission_limit{class=\"speech_batch\"} 104\n",
                "sglang_omni_router_admission_limit{class=\"speech_websocket\"} 105\n",
                "sglang_omni_router_admission_limit{class=\"realtime_websocket\"} 106\n",
                "sglang_omni_router_admission_limit{class=\"control\"} 107\n",
                "# HELP sglang_omni_router_admission_in_flight Current admitted requests and sessions.\n",
                "# TYPE sglang_omni_router_admission_in_flight gauge\n",
                "sglang_omni_router_admission_in_flight{class=\"global\"} 0\n",
                "sglang_omni_router_admission_in_flight{class=\"generation_http\"} 1\n",
                "sglang_omni_router_admission_in_flight{class=\"speech_http\"} 2\n",
                "sglang_omni_router_admission_in_flight{class=\"transcription_http\"} 3\n",
                "sglang_omni_router_admission_in_flight{class=\"speech_batch\"} 4\n",
                "sglang_omni_router_admission_in_flight{class=\"speech_websocket\"} 5\n",
                "sglang_omni_router_admission_in_flight{class=\"realtime_websocket\"} 6\n",
                "sglang_omni_router_admission_in_flight{class=\"control\"} 7\n",
                "# HELP sglang_omni_router_worker_capacity_limit Aggregate configured worker capacity.\n",
                "# TYPE sglang_omni_router_worker_capacity_limit gauge\n",
                "sglang_omni_router_worker_capacity_limit{class=\"generation_http\"} 12\n",
                "sglang_omni_router_worker_capacity_limit{class=\"speech_http\"} 20\n",
                "sglang_omni_router_worker_capacity_limit{class=\"transcription_http\"} 30\n",
                "sglang_omni_router_worker_capacity_limit{class=\"speech_batch\"} 40\n",
                "sglang_omni_router_worker_capacity_limit{class=\"speech_websocket\"} 50\n",
                "sglang_omni_router_worker_capacity_limit{class=\"realtime_websocket\"} 60\n",
                "sglang_omni_router_worker_capacity_limit{class=\"control\"} 73\n",
                "# HELP sglang_omni_router_worker_capacity_in_flight Aggregate current worker capacity use.\n",
                "# TYPE sglang_omni_router_worker_capacity_in_flight gauge\n",
                "sglang_omni_router_worker_capacity_in_flight{class=\"generation_http\"} 3\n",
                "sglang_omni_router_worker_capacity_in_flight{class=\"speech_http\"} 2\n",
                "sglang_omni_router_worker_capacity_in_flight{class=\"transcription_http\"} 3\n",
                "sglang_omni_router_worker_capacity_in_flight{class=\"speech_batch\"} 4\n",
                "sglang_omni_router_worker_capacity_in_flight{class=\"speech_websocket\"} 5\n",
                "sglang_omni_router_worker_capacity_in_flight{class=\"realtime_websocket\"} 6\n",
                "sglang_omni_router_worker_capacity_in_flight{class=\"control\"} 8\n",
            )),
            "existing metric prefix or order changed"
        );
        assert!(rendered.contains("# TYPE sglang_omni_router_worker_dispatch_total counter\n"));
        assert!(rendered.contains("# TYPE sglang_omni_router_runtime_workers gauge\n"));
    }

    #[test]
    fn maximum_diagnostics_snapshot_is_bounded_ordered_and_exactly_redacted() {
        let admission = representative_snapshot().admission;
        let workers = (0..256)
            .map(|registration_ordinal| WorkerSnapshot {
                worker_id: format!("worker-{registration_ordinal:03}"),
                registration_ordinal,
                health: match registration_ordinal % 3 {
                    0 => HealthState::Unknown,
                    1 => HealthState::Healthy,
                    _ => HealthState::Unhealthy,
                },
                disposition: if registration_ordinal % 2 == 0 {
                    Disposition::Serving
                } else {
                    Disposition::Draining
                },
                capacity: CapacityClass::ALL
                    .into_iter()
                    .enumerate()
                    .map(|(index, class)| capacity(class, index + 1, index))
                    .collect(),
            })
            .collect();
        let snapshot = OperationsSnapshot { admission, workers };
        let bytes = serde_json::to_vec(&Diagnostics::from_snapshot(
            LifecycleState::Draining,
            false,
            &snapshot,
        ))
        .expect("serialize maximum diagnostics snapshot");
        assert!(
            bytes.len() < 256 * 1024,
            "diagnostics grew to {} bytes",
            bytes.len()
        );

        let value: serde_json::Value =
            serde_json::from_slice(&bytes).expect("parse diagnostics JSON");
        let top_keys: BTreeSet<_> = value
            .as_object()
            .expect("diagnostics object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            top_keys,
            BTreeSet::from(["admission", "lifecycle", "ready", "workers"])
        );
        assert_eq!(value["lifecycle"], "draining");
        assert_eq!(value["ready"], false);

        let admission = value["admission"].as_array().expect("admission array");
        assert_eq!(admission.len(), 8);
        let admission_classes: Vec<_> = admission
            .iter()
            .map(|entry| entry["class"].as_str().expect("admission class"))
            .collect();
        assert_eq!(
            admission_classes,
            [
                "global",
                "generation_http",
                "speech_http",
                "transcription_http",
                "speech_batch",
                "speech_websocket",
                "realtime_websocket",
                "control",
            ]
        );
        for entry in admission {
            let keys: BTreeSet<_> = entry
                .as_object()
                .expect("admission object")
                .keys()
                .map(String::as_str)
                .collect();
            assert_eq!(keys, BTreeSet::from(["class", "in_flight", "limit"]));
        }

        let workers = value["workers"].as_array().expect("workers array");
        assert_eq!(workers.len(), 256);
        let expected_capacity_classes = [
            "generation_http",
            "speech_http",
            "transcription_http",
            "speech_batch",
            "speech_websocket",
            "realtime_websocket",
            "control",
        ];
        for (ordinal, worker) in workers.iter().enumerate() {
            let keys: BTreeSet<_> = worker
                .as_object()
                .expect("worker object")
                .keys()
                .map(String::as_str)
                .collect();
            assert_eq!(
                keys,
                BTreeSet::from([
                    "capacity",
                    "disposition",
                    "health",
                    "registration_ordinal",
                    "worker_id",
                ])
            );
            assert_eq!(
                worker["registration_ordinal"].as_u64(),
                Some(u64::try_from(ordinal).expect("bounded ordinal"))
            );
            let capacity = worker["capacity"].as_array().expect("capacity array");
            assert_eq!(capacity.len(), 7);
            let classes: Vec<_> = capacity
                .iter()
                .map(|entry| entry["class"].as_str().expect("capacity class"))
                .collect();
            assert_eq!(classes, expected_capacity_classes);
            for entry in capacity {
                let keys: BTreeSet<_> = entry
                    .as_object()
                    .expect("capacity object")
                    .keys()
                    .map(String::as_str)
                    .collect();
                assert_eq!(keys, BTreeSet::from(["class", "in_flight", "limit"]));
            }
        }
    }

    #[test]
    fn request_id_validation_is_exact_visible_ascii() {
        assert!(valid_request_id(&HeaderValue::from_static("request-1")));
        assert!(!valid_request_id(&HeaderValue::from_static("")));
        assert!(!valid_request_id(&HeaderValue::from_static("has space")));
        let oversized = HeaderValue::from_str(&"x".repeat(129)).expect("valid header syntax");
        assert!(!valid_request_id(&oversized));
    }

    #[test]
    fn generated_ids_are_unique_under_concurrency_and_do_not_wrap() {
        let context = RequestContext::new().expect("construct request context");
        let handles: Vec<_> = (0..16)
            .map(|_| {
                let context = Arc::clone(&context);
                std::thread::spawn(move || {
                    (0..64)
                        .map(|_| {
                            context
                                .generate()
                                .expect("sequence remains available")
                                .to_str()
                                .expect("generated ID is visible ASCII")
                                .to_owned()
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let ids: HashSet<_> = handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("join generator thread"))
            .collect();
        assert_eq!(ids.len(), 1024);

        let exhausted = RequestContext {
            prefix: String::from("test"),
            sequence: std::sync::atomic::AtomicU64::new(u64::MAX),
        };
        assert!(exhausted.generate().is_none());
        assert!(exhausted.generate().is_none());
    }
}
