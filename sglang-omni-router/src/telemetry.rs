use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::worker_pool::CapacityClass;

const CLASS_COUNT: usize = 7;
const TICKS_PER_SECOND: u64 = 100_000_000;
const HISTOGRAM_BOUNDS_10NS: [u64; 30] = [
    10,
    25,
    50,
    100,
    250,
    500,
    1_000,
    2_500,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_500_000,
    5_000_000,
    10_000_000,
    25_000_000,
    50_000_000,
    100_000_000,
    250_000_000,
    500_000_000,
    1_000_000_000,
    3_000_000_000,
    6_000_000_000,
    30_000_000_000,
    60_000_000_000,
    180_000_000_000,
];
const HISTOGRAM_BOUND_LABELS: [&str; 30] = [
    "0.0000001",
    "0.00000025",
    "0.0000005",
    "0.000001",
    "0.0000025",
    "0.000005",
    "0.00001",
    "0.000025",
    "0.00005",
    "0.0001",
    "0.00025",
    "0.0005",
    "0.001",
    "0.0025",
    "0.005",
    "0.01",
    "0.025",
    "0.05",
    "0.1",
    "0.25",
    "0.5",
    "1",
    "2.5",
    "5",
    "10",
    "30",
    "60",
    "300",
    "600",
    "1800",
];
const RAW_BUCKET_COUNT: usize = HISTOGRAM_BOUNDS_10NS.len() + 1;

struct Histogram {
    raw_buckets: [AtomicU64; RAW_BUCKET_COUNT],
    sum_10ns: AtomicU64,
}

impl Histogram {
    fn new() -> Self {
        Self {
            raw_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            sum_10ns: AtomicU64::new(0),
        }
    }

    fn observe(&self, duration: Duration) {
        let (sum_ticks, bucket_ticks) = duration_ticks(duration);
        self.observe_ticks(sum_ticks, bucket_ticks);
    }

    fn observe_ticks(&self, sum_ticks: u64, bucket_ticks: u64) {
        let bucket = HISTOGRAM_BOUNDS_10NS
            .iter()
            .position(|bound| bucket_ticks <= *bound)
            .unwrap_or(HISTOGRAM_BOUNDS_10NS.len());
        self.raw_buckets[bucket].fetch_add(1, Ordering::Relaxed);
        saturating_add(&self.sum_10ns, sum_ticks);
    }

    fn snapshot(&self) -> HistogramSnapshot {
        HistogramSnapshot {
            raw_buckets: std::array::from_fn(|index| {
                self.raw_buckets[index].load(Ordering::Relaxed)
            }),
            sum_10ns: self.sum_10ns.load(Ordering::Relaxed),
        }
    }
}

struct DurationTotals {
    count: AtomicU64,
    sum_10ns: AtomicU64,
}

impl DurationTotals {
    fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            sum_10ns: AtomicU64::new(0),
        }
    }

    fn observe_ticks(&self, sum_ticks: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        saturating_add(&self.sum_10ns, sum_ticks);
    }

    fn snapshot(&self) -> DurationTotalsSnapshot {
        DurationTotalsSnapshot {
            count: self.count.load(Ordering::Relaxed),
            sum_10ns: self.sum_10ns.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct DurationTotalsSnapshot {
    count: u64,
    sum_10ns: u64,
}

#[derive(Debug, Eq, PartialEq)]
struct HistogramSnapshot {
    raw_buckets: [u64; RAW_BUCKET_COUNT],
    sum_10ns: u64,
}

impl HistogramSnapshot {
    fn count(&self) -> u64 {
        self.raw_buckets
            .iter()
            .copied()
            .fold(0_u64, u64::saturating_add)
    }
}

struct WorkerTelemetry {
    worker_id: Box<str>,
    dispatch: [AtomicU64; CLASS_COUNT],
    request_duration: [DurationTotals; CLASS_COUNT],
}

impl WorkerTelemetry {
    fn new(worker_id: &str) -> Self {
        Self {
            worker_id: worker_id.into(),
            dispatch: std::array::from_fn(|_| AtomicU64::new(0)),
            request_duration: std::array::from_fn(|_| DurationTotals::new()),
        }
    }
}

/// Fixed-cardinality router performance telemetry shared by immutable owners.
pub(crate) struct Telemetry {
    workers: Box<[WorkerTelemetry]>,
    request_lease_duration: [Histogram; CLASS_COUNT],
    dispatch_duration: [Histogram; CLASS_COUNT],
    upstream_headers: [Histogram; CLASS_COUNT],
    classification_wait: [Histogram; CLASS_COUNT],
    classification_run: [Histogram; CLASS_COUNT],
    classification_waiters: AtomicU64,
    classification_in_flight: AtomicU64,
    connections_accepted: AtomicU64,
    connections_in_use: AtomicU64,
    connection_capacity_waits: AtomicU64,
}

impl Telemetry {
    pub(crate) fn new<'a>(worker_ids: impl Iterator<Item = &'a str>) -> Self {
        Self {
            workers: worker_ids
                .map(WorkerTelemetry::new)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            request_lease_duration: std::array::from_fn(|_| Histogram::new()),
            dispatch_duration: std::array::from_fn(|_| Histogram::new()),
            upstream_headers: std::array::from_fn(|_| Histogram::new()),
            classification_wait: std::array::from_fn(|_| Histogram::new()),
            classification_run: std::array::from_fn(|_| Histogram::new()),
            classification_waiters: AtomicU64::new(0),
            classification_in_flight: AtomicU64::new(0),
            connections_accepted: AtomicU64::new(0),
            connections_in_use: AtomicU64::new(0),
            connection_capacity_waits: AtomicU64::new(0),
        }
    }

    pub(crate) fn record_dispatch(&self, worker_ordinal: usize, class: CapacityClass) {
        self.workers[worker_ordinal].dispatch[class_index(class)].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_worker_request(
        &self,
        worker_ordinal: usize,
        class: CapacityClass,
        duration: Duration,
    ) {
        let (sum_ticks, bucket_ticks) = duration_ticks(duration);
        self.workers[worker_ordinal].request_duration[class_index(class)].observe_ticks(sum_ticks);
        self.request_lease_duration[class_index(class)].observe_ticks(sum_ticks, bucket_ticks);
    }

    pub(crate) fn dispatch_timer(&self, class: CapacityClass) -> DurationGuard<'_> {
        DurationGuard::new(&self.dispatch_duration[class_index(class)])
    }

    pub(crate) fn upstream_headers_timer(&self, class: CapacityClass) -> DurationGuard<'_> {
        DurationGuard::new(&self.upstream_headers[class_index(class)])
    }

    pub(crate) fn classification_wait(&self, class: CapacityClass) -> WaitGuard<'_> {
        self.classification_waiters.fetch_add(1, Ordering::Relaxed);
        WaitGuard {
            telemetry: self,
            class,
            started: Instant::now(),
        }
    }

    pub(crate) fn classification_run(self: Arc<Self>, class: CapacityClass) -> RunGuard {
        self.classification_in_flight
            .fetch_add(1, Ordering::Relaxed);
        RunGuard {
            telemetry: self,
            class,
            started: Instant::now(),
        }
    }

    pub(crate) fn record_connection_capacity_wait(&self) {
        self.connection_capacity_waits
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn connection_accepted(self: &Arc<Self>) -> ConnectionGuard {
        self.connections_accepted.fetch_add(1, Ordering::Relaxed);
        self.connections_in_use.fetch_add(1, Ordering::Relaxed);
        ConnectionGuard {
            telemetry: Arc::clone(self),
        }
    }

    pub(crate) fn render(&self, output: &mut String) {
        render_worker_metrics(output, &self.workers);
        render_class_histogram(
            output,
            "sglang_omni_router_request_lease_duration_seconds",
            "Exact worker lease lifetime by request class.",
            &self.request_lease_duration,
        );
        render_class_histogram(
            output,
            "sglang_omni_router_dispatch_duration_seconds",
            "Router worker dispatch duration.",
            &self.dispatch_duration,
        );
        render_class_histogram(
            output,
            "sglang_omni_router_upstream_headers_seconds",
            "Time from upstream HTTP send to response headers.",
            &self.upstream_headers,
        );
        render_class_histogram(
            output,
            "sglang_omni_router_classification_wait_seconds",
            "Time waiting for a classification slot.",
            &self.classification_wait,
        );
        render_class_histogram(
            output,
            "sglang_omni_router_classification_run_seconds",
            "CPU classification closure duration.",
            &self.classification_run,
        );
        render_gauge(
            output,
            "sglang_omni_router_classification_waiters",
            "Classifications currently waiting for a slot.",
            self.classification_waiters.load(Ordering::Relaxed),
        );
        render_gauge(
            output,
            "sglang_omni_router_classification_in_flight",
            "Classification closures currently running.",
            self.classification_in_flight.load(Ordering::Relaxed),
        );
        render_counter(
            output,
            "sglang_omni_router_connections_accepted_total",
            "Accepted client connections.",
            self.connections_accepted.load(Ordering::Relaxed),
        );
        render_gauge(
            output,
            "sglang_omni_router_connections_in_use",
            "Accepted client connections whose socket owner is alive.",
            self.connections_in_use.load(Ordering::Relaxed),
        );
        render_counter(
            output,
            "sglang_omni_router_connection_capacity_waits_total",
            "Listener accepts that encountered exhausted connection capacity.",
            self.connection_capacity_waits.load(Ordering::Relaxed),
        );

        let runtime = tokio::runtime::Handle::current().metrics();
        render_gauge(
            output,
            "sglang_omni_router_runtime_workers",
            "Tokio runtime worker threads.",
            runtime.num_workers(),
        );
        render_gauge(
            output,
            "sglang_omni_router_runtime_alive_tasks",
            "Tokio runtime tasks currently alive.",
            runtime.num_alive_tasks(),
        );
        render_gauge(
            output,
            "sglang_omni_router_runtime_global_queue_depth",
            "Tokio runtime global injection queue depth.",
            runtime.global_queue_depth(),
        );
    }

    #[cfg(test)]
    pub(crate) fn worker_dispatch_count(&self, worker_ordinal: usize, class: CapacityClass) -> u64 {
        self.workers[worker_ordinal].dispatch[class_index(class)].load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn worker_request_count(&self, worker_ordinal: usize, class: CapacityClass) -> u64 {
        self.workers[worker_ordinal].request_duration[class_index(class)]
            .snapshot()
            .count
    }

    #[cfg(test)]
    pub(crate) fn dispatch_count(&self, class: CapacityClass) -> u64 {
        self.dispatch_duration[class_index(class)]
            .snapshot()
            .count()
    }

    #[cfg(test)]
    pub(crate) fn upstream_headers_count(&self, class: CapacityClass) -> u64 {
        self.upstream_headers[class_index(class)].snapshot().count()
    }

    #[cfg(test)]
    pub(crate) fn classification_counts(&self) -> (u64, u64) {
        (
            self.classification_waiters.load(Ordering::Relaxed),
            self.classification_in_flight.load(Ordering::Relaxed),
        )
    }

    #[cfg(test)]
    pub(crate) fn classification_observations(&self, class: CapacityClass) -> (u64, u64) {
        (
            self.classification_wait[class_index(class)]
                .snapshot()
                .count(),
            self.classification_run[class_index(class)]
                .snapshot()
                .count(),
        )
    }

    #[cfg(test)]
    pub(crate) fn connection_counts(&self) -> (u64, u64, u64) {
        (
            self.connections_accepted.load(Ordering::Relaxed),
            self.connections_in_use.load(Ordering::Relaxed),
            self.connection_capacity_waits.load(Ordering::Relaxed),
        )
    }
}

pub(crate) struct DurationGuard<'a> {
    histogram: &'a Histogram,
    started: Instant,
}

impl<'a> DurationGuard<'a> {
    fn new(histogram: &'a Histogram) -> Self {
        Self {
            histogram,
            started: Instant::now(),
        }
    }
}

impl Drop for DurationGuard<'_> {
    fn drop(&mut self) {
        self.histogram.observe(self.started.elapsed());
    }
}

pub(crate) struct WaitGuard<'a> {
    telemetry: &'a Telemetry,
    class: CapacityClass,
    started: Instant,
}

impl Drop for WaitGuard<'_> {
    fn drop(&mut self) {
        self.telemetry
            .classification_waiters
            .fetch_sub(1, Ordering::Relaxed);
        self.telemetry.classification_wait[class_index(self.class)].observe(self.started.elapsed());
    }
}

pub(crate) struct RunGuard {
    telemetry: Arc<Telemetry>,
    class: CapacityClass,
    started: Instant,
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        self.telemetry
            .classification_in_flight
            .fetch_sub(1, Ordering::Relaxed);
        self.telemetry.classification_run[class_index(self.class)].observe(self.started.elapsed());
    }
}

pub(crate) struct ConnectionGuard {
    telemetry: Arc<Telemetry>,
}

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.telemetry
            .connections_in_use
            .fetch_sub(1, Ordering::Relaxed);
    }
}

fn render_worker_metrics(output: &mut String, workers: &[WorkerTelemetry]) {
    output.push_str(
        "# HELP sglang_omni_router_worker_dispatch_total Successful worker dispatches.\n",
    );
    output.push_str("# TYPE sglang_omni_router_worker_dispatch_total counter\n");
    for worker in workers {
        let worker_id = escape_label(&worker.worker_id);
        for class in CapacityClass::ALL {
            let value = worker.dispatch[class_index(class)].load(Ordering::Relaxed);
            let _ = writeln!(
                output,
                "sglang_omni_router_worker_dispatch_total{{worker_id=\"{worker_id}\",class=\"{}\"}} {value}",
                class.label()
            );
        }
    }

    output.push_str(
        "# HELP sglang_omni_router_worker_request_duration_seconds Exact worker lease lifetime totals.\n",
    );
    output.push_str("# TYPE sglang_omni_router_worker_request_duration_seconds summary\n");
    for worker in workers {
        let worker_id = escape_label(&worker.worker_id);
        for class in CapacityClass::ALL {
            let snapshot = worker.request_duration[class_index(class)].snapshot();
            let labels = format!("worker_id=\"{worker_id}\",class=\"{}\"", class.label());
            let _ = writeln!(
                output,
                "sglang_omni_router_worker_request_duration_seconds_count{{{labels}}} {}",
                snapshot.count
            );
            let _ = writeln!(
                output,
                "sglang_omni_router_worker_request_duration_seconds_sum{{{labels}}} {}",
                ticks_as_seconds(snapshot.sum_10ns)
            );
        }
    }
}

fn render_class_histogram(
    output: &mut String,
    name: &str,
    help: &str,
    histograms: &[Histogram; CLASS_COUNT],
) {
    let _ = writeln!(output, "# HELP {name} {help}");
    let _ = writeln!(output, "# TYPE {name} histogram");
    for class in CapacityClass::ALL {
        render_histogram(
            output,
            name,
            &format!("class=\"{}\"", class.label()),
            &histograms[class_index(class)].snapshot(),
        );
    }
}

fn render_histogram(output: &mut String, name: &str, labels: &str, snapshot: &HistogramSnapshot) {
    let mut cumulative = 0_u64;
    for (index, label) in HISTOGRAM_BOUND_LABELS.iter().enumerate() {
        cumulative = cumulative.saturating_add(snapshot.raw_buckets[index]);
        let _ = writeln!(
            output,
            "{name}_bucket{{{labels},le=\"{label}\"}} {cumulative}"
        );
    }
    cumulative = cumulative.saturating_add(snapshot.raw_buckets[HISTOGRAM_BOUNDS_10NS.len()]);
    let _ = writeln!(output, "{name}_bucket{{{labels},le=\"+Inf\"}} {cumulative}");
    let _ = writeln!(output, "{name}_count{{{labels}}} {}", snapshot.count());
    let _ = writeln!(
        output,
        "{name}_sum{{{labels}}} {}",
        ticks_as_seconds(snapshot.sum_10ns)
    );
}

fn duration_ticks(duration: Duration) -> (u64, u64) {
    let nanoseconds = duration.as_nanos();
    // Sums use fixed-point floor quantization. Buckets use ceiling quantization
    // so any positive time above an exact boundary advances to the next bucket.
    let sum = nanoseconds / 10;
    let bucket = nanoseconds.saturating_add(9) / 10;
    (
        sum.min(u128::from(u64::MAX)) as u64,
        bucket.min(u128::from(u64::MAX)) as u64,
    )
}

fn saturating_add(atomic: &AtomicU64, value: u64) {
    let _previous = atomic.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(value))
    });
}

fn ticks_as_seconds(ticks: u64) -> f64 {
    ticks as f64 / TICKS_PER_SECOND as f64
}

fn render_gauge(output: &mut String, name: &str, help: &str, value: impl std::fmt::Display) {
    let _ = writeln!(output, "# HELP {name} {help}");
    let _ = writeln!(output, "# TYPE {name} gauge");
    let _ = writeln!(output, "{name} {value}");
}

fn render_counter(output: &mut String, name: &str, help: &str, value: u64) {
    let _ = writeln!(output, "# HELP {name} {help}");
    let _ = writeln!(output, "# TYPE {name} counter");
    let _ = writeln!(output, "{name} {value}");
}

fn escape_label(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            other => escaped.push(other),
        }
    }
    escaped
}

const fn class_index(class: CapacityClass) -> usize {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use super::{
        CLASS_COUNT, CapacityClass, HISTOGRAM_BOUND_LABELS, HISTOGRAM_BOUNDS_10NS, Histogram,
        Telemetry,
    };

    #[test]
    fn histogram_buckets_use_ceiling_ten_nanosecond_ticks() {
        let boundary_index = 1;
        assert_eq!(HISTOGRAM_BOUNDS_10NS[boundary_index], 25);
        for (nanoseconds, expected_index) in [
            (249_u64, boundary_index),
            (250, boundary_index),
            (251, boundary_index + 1),
            (259, boundary_index + 1),
        ] {
            let histogram = Histogram::new();
            histogram.observe(Duration::from_nanos(nanoseconds));
            let snapshot = histogram.snapshot();
            assert_eq!(
                snapshot.raw_buckets[expected_index], 1,
                "{nanoseconds} nanoseconds used the wrong raw bucket"
            );
            assert_eq!(snapshot.count(), 1);
        }
    }

    #[tokio::test]
    async fn worker_summary_and_aggregate_histogram_are_exact() {
        let telemetry = Arc::new(Telemetry::new(["worker\"\\\n"].into_iter()));
        telemetry.record_dispatch(0, CapacityClass::GenerationHttp);
        telemetry.record_worker_request(
            0,
            CapacityClass::GenerationHttp,
            Duration::from_nanos(300),
        );
        let mut rendered = String::new();
        telemetry.render(&mut rendered);

        assert!(rendered.contains("worker_id=\"worker\\\"\\\\\\n\""));
        assert!(
            rendered
                .contains("# TYPE sglang_omni_router_worker_request_duration_seconds summary\n")
        );
        assert!(rendered.contains(
            "sglang_omni_router_worker_request_duration_seconds_count{worker_id=\"worker\\\"\\\\\\n\",class=\"generation_http\"} 1"
        ));
        assert!(rendered.contains(
            "sglang_omni_router_worker_request_duration_seconds_sum{worker_id=\"worker\\\"\\\\\\n\",class=\"generation_http\"} 0.0000003"
        ));
        assert!(
            !rendered
                .contains("sglang_omni_router_worker_request_duration_seconds_bucket{worker_id=")
        );
        assert!(rendered.contains(
            "sglang_omni_router_request_lease_duration_seconds_bucket{class=\"generation_http\",le=\"0.00000025\"} 0"
        ));
        assert!(rendered.contains(
            "sglang_omni_router_request_lease_duration_seconds_bucket{class=\"generation_http\",le=\"0.0000005\"} 1"
        ));
        assert!(rendered.contains(
            "sglang_omni_router_request_lease_duration_seconds_count{class=\"generation_http\"} 1"
        ));
        assert!(rendered.contains(
            "sglang_omni_router_request_lease_duration_seconds_sum{class=\"generation_http\"} 0.0000003"
        ));
    }

    #[tokio::test]
    async fn help_and_type_lines_are_unique() {
        let telemetry = Telemetry::new(["one", "two"].into_iter());
        let mut rendered = String::new();
        telemetry.render(&mut rendered);
        for name in [
            "sglang_omni_router_worker_dispatch_total",
            "sglang_omni_router_worker_request_duration_seconds",
            "sglang_omni_router_request_lease_duration_seconds",
            "sglang_omni_router_dispatch_duration_seconds",
            "sglang_omni_router_upstream_headers_seconds",
            "sglang_omni_router_classification_wait_seconds",
            "sglang_omni_router_classification_run_seconds",
            "sglang_omni_router_classification_waiters",
            "sglang_omni_router_classification_in_flight",
            "sglang_omni_router_connections_accepted_total",
            "sglang_omni_router_connections_in_use",
            "sglang_omni_router_connection_capacity_waits_total",
            "sglang_omni_router_runtime_workers",
            "sglang_omni_router_runtime_alive_tasks",
            "sglang_omni_router_runtime_global_queue_depth",
        ] {
            assert_eq!(rendered.matches(&format!("# HELP {name} ")).count(), 1);
            assert_eq!(rendered.matches(&format!("# TYPE {name} ")).count(), 1);
        }
        let samples: Vec<_> = rendered
            .lines()
            .filter(|line| !line.starts_with('#'))
            .filter_map(|line| line.split_once(' ').map(|(series, _value)| series))
            .collect();
        let unique: std::collections::HashSet<_> = samples.iter().copied().collect();
        assert_eq!(samples.len(), unique.len(), "metric series are unique");
        assert!(rendered.contains("sglang_omni_router_runtime_workers "));
        assert!(rendered.contains("sglang_omni_router_runtime_alive_tasks "));
        assert!(rendered.contains("sglang_omni_router_runtime_global_queue_depth "));
    }

    #[tokio::test]
    async fn maximum_worker_render_is_bounded_and_structurally_fixed() {
        const WORKERS: usize = 256;
        const CLASS_HISTOGRAMS: usize = 5;
        const HISTOGRAM_SERIES: usize = HISTOGRAM_BOUND_LABELS.len() + 3;
        const EXPECTED_SERIES: usize = WORKERS * CLASS_COUNT
            + WORKERS * CLASS_COUNT * 2
            + CLASS_HISTOGRAMS * CLASS_COUNT * HISTOGRAM_SERIES
            + 8;
        let worker_ids: Vec<_> = (0..WORKERS)
            .map(|ordinal| format!("worker-{ordinal:03}"))
            .collect();
        let telemetry = Telemetry::new(worker_ids.iter().map(String::as_str));
        let started = Instant::now();
        let mut rendered = String::new();
        telemetry.render(&mut rendered);
        let elapsed = started.elapsed();

        let series = rendered
            .lines()
            .filter(|line| !line.starts_with('#'))
            .count();
        assert_eq!(series, EXPECTED_SERIES);
        assert!(
            rendered.len() < 900 * 1024,
            "maximum telemetry scrape rendered {} bytes",
            rendered.len()
        );
        assert!(
            elapsed < Duration::from_secs(1),
            "maximum telemetry scrape took {elapsed:?}"
        );
    }
}
