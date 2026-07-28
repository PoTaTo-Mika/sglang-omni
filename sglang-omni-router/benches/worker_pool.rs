#![forbid(unsafe_code)]
#![allow(clippy::expect_used)] // Fixture construction is the benchmark assertion boundary.
#![allow(dead_code, unreachable_pub, unused_imports)] // Included production modules expose crate APIs.
#![allow(missing_docs)] // Criterion generates benchmark entry points.

//! Worker-pool candidate, policy, and exact-permit microbenchmarks.

#[path = "../src/config.rs"]
mod config;
#[path = "../src/error.rs"]
mod error;
#[path = "../src/worker_pool/mod.rs"]
mod worker_pool;

use config::RoutingStrategy;
use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use worker_pool::benchmark::Fixture;

fn worker_pool(c: &mut Criterion) {
    for size in [1_usize, 2, 8, 64, 256] {
        let rr = Fixture::new(size, RoutingStrategy::RoundRobin)
            .expect("round-robin benchmark fixture must be valid");
        c.bench_with_input(BenchmarkId::new("candidates", size), &size, |b, _| {
            b.iter(|| black_box(rr.candidate_construction()));
        });
        c.bench_with_input(BenchmarkId::new("round_robin", size), &size, |b, _| {
            b.iter_batched_ref(
                || rr.policy_input(),
                |input| black_box(rr.policy_order(input)),
                BatchSize::SmallInput,
            );
        });
        c.bench_with_input(
            BenchmarkId::new("exact_permit_scan", size),
            &size,
            |b, _| {
                b.iter_batched_ref(
                    || rr.exact_input(),
                    |input| black_box(rr.exact_permit_scan(input)),
                    BatchSize::SmallInput,
                );
            },
        );
        c.bench_with_input(
            BenchmarkId::new("round_robin_normal_dispatch", size),
            &size,
            |b, _| {
                b.iter(|| black_box(rr.normal_dispatch()));
            },
        );
        c.bench_with_input(
            BenchmarkId::new("round_robin_homogeneous_dispatch", size),
            &size,
            |b, _| {
                b.iter(|| black_box(rr.homogeneous_dispatch()));
            },
        );

        let least = Fixture::new(size, RoutingStrategy::LeastRequests)
            .expect("least-requests benchmark fixture must be valid");
        c.bench_with_input(BenchmarkId::new("least_requests", size), &size, |b, _| {
            b.iter_batched_ref(
                || least.policy_input(),
                |input| black_box(least.policy_order(input)),
                BatchSize::SmallInput,
            );
        });
        c.bench_with_input(
            BenchmarkId::new("least_requests_normal_dispatch", size),
            &size,
            |b, _| {
                b.iter(|| black_box(least.normal_dispatch()));
            },
        );
        c.bench_with_input(
            BenchmarkId::new("least_requests_homogeneous_dispatch", size),
            &size,
            |b, _| {
                b.iter(|| black_box(least.homogeneous_dispatch()));
            },
        );
    }
}

criterion_group!(benches, worker_pool);
criterion_main!(benches);
