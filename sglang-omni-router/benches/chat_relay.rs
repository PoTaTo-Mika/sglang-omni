#![forbid(unsafe_code)]
#![allow(clippy::expect_used)] // Fixture construction is the benchmark assertion boundary.
#![allow(dead_code, unreachable_pub, unused_imports)] // Production modules are included directly.
#![allow(missing_docs)] // Criterion generates benchmark entry points.

//! Chat classification and relay mechanism microbenchmarks.

#[path = "../src/classification.rs"]
mod classification;
#[path = "../src/config.rs"]
mod config;
#[path = "../src/error.rs"]
mod error;
#[path = "../src/http_generation/mod.rs"]
mod http_generation;
#[path = "../src/telemetry.rs"]
mod telemetry;
#[path = "../src/worker_pool/mod.rs"]
mod worker_pool;

use bytes::Bytes;
use config::RoutingStrategy;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use http_generation::benchmark::{classify_request, consume_direct, consume_plain};
use worker_pool::benchmark::Fixture;

fn request_of_size(size: usize) -> Vec<u8> {
    let prefix = br#"{"model":"omni","messages":[{"content":"hello"}],"ignored":""#;
    let suffix = br#""}"#;
    let filler = size.saturating_sub(prefix.len() + suffix.len());
    let mut request = Vec::with_capacity(prefix.len() + filler + suffix.len());
    request.extend_from_slice(prefix);
    request.resize(request.len() + filler, b'x');
    request.extend_from_slice(suffix);
    request
}

fn chat_relay(criterion: &mut Criterion) {
    let fixture =
        Fixture::new(8, RoutingStrategy::RoundRobin).expect("chat benchmark fixture must be valid");
    for size in [1_024_usize, 65_536, 1_048_576, 8_388_608] {
        let request = request_of_size(size);
        criterion.bench_with_input(
            BenchmarkId::new("chat_classify_bytes", size),
            &request,
            |bencher, request| {
                bencher.iter(|| {
                    black_box(classify_request(
                        black_box(request),
                        fixture.pool(),
                        fixture.trust_domain(),
                    ))
                });
            },
        );
    }
    criterion.bench_function("chat_dispatch/normal", |bencher| {
        bencher.iter(|| black_box(fixture.normal_dispatch()));
    });
    criterion.bench_function("chat_dispatch/homogeneous", |bencher| {
        bencher.iter(|| black_box(fixture.homogeneous_dispatch()));
    });

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("build relay benchmark runtime");
    let relay_bytes = Bytes::from(vec![b'x'; 1_048_576]);
    criterion.bench_function("chat_relay/direct_body_1mib", |bencher| {
        bencher.iter(|| runtime.block_on(consume_direct(black_box(relay_bytes.clone()))));
    });
    criterion.bench_function("chat_relay/plain_body_1mib", |bencher| {
        bencher.iter(|| runtime.block_on(consume_plain(black_box(relay_bytes.clone()))));
    });
}

criterion_group!(benches, chat_relay);
criterion_main!(benches);
