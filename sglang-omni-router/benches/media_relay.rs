#![forbid(unsafe_code)]
#![allow(clippy::expect_used)]
#![allow(dead_code, unreachable_pub, unused_imports)]
#![allow(missing_docs)]

//! Media classifier and direct-body mechanism smoke benchmarks.

#[path = "../src/config.rs"]
mod config;
#[path = "../src/error.rs"]
mod error;
#[path = "../src/http_generation/mod.rs"]
mod http_generation;
#[path = "../src/http_media/mod.rs"]
mod http_media;
#[path = "../src/worker_pool/mod.rs"]
mod worker_pool;

use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use http_generation::benchmark::{consume_direct, consume_plain};
use http_media::benchmark::{classify_speech, scan_multipart};

const PAYLOAD_SIZES: [usize; 4] = [1_024, 65_536, 1_048_576, 8_388_608];

fn speech_request(size: usize) -> Vec<u8> {
    let prefix = br#"{"model":"tts","input":""#;
    let suffix = br#"","response_format":"wav"}"#;
    let filler = size.saturating_sub(prefix.len() + suffix.len());
    let mut request = Vec::with_capacity(prefix.len() + filler + suffix.len());
    request.extend_from_slice(prefix);
    request.resize(request.len() + filler, b'x');
    request.extend_from_slice(suffix);
    request
}

fn multipart(file_size: usize) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(b"--bench\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\nContent-Type: audio/wav\r\n\r\n");
    body.extend(std::iter::repeat_n(b'a', file_size));
    body.extend_from_slice(b"\r\n--bench\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nasr\r\n--bench--\r\n");
    body
}

fn media_relay(criterion: &mut Criterion) {
    let mut speech_group = criterion.benchmark_group("media_classify/speech_json_bytes");
    for size in PAYLOAD_SIZES {
        let request = speech_request(size);
        assert_eq!(request.len(), size);
        assert!(classify_speech(&request));
        speech_group.throughput(Throughput::Bytes(request.len() as u64));
        speech_group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &request,
            |bencher, request| {
                bencher.iter(|| black_box(classify_speech(black_box(request))));
            },
        );
    }
    speech_group.finish();

    let mut multipart_group = criterion.benchmark_group("media_classify/multipart_file_bytes");
    for size in PAYLOAD_SIZES {
        let body = multipart(size);
        assert!(scan_multipart(&body, b"bench"));
        multipart_group.throughput(Throughput::Bytes(body.len() as u64));
        multipart_group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &body,
            |bencher, body| {
                bencher.iter(|| black_box(scan_multipart(black_box(body), b"bench")));
            },
        );
    }
    multipart_group.finish();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("build media relay benchmark runtime");
    let mut relay_group = criterion.benchmark_group("media_relay/bytes");
    for size in PAYLOAD_SIZES {
        let bytes = Bytes::from(vec![b'x'; size]);
        assert_eq!(runtime.block_on(consume_direct(Bytes::clone(&bytes))), size);
        assert_eq!(runtime.block_on(consume_plain(Bytes::clone(&bytes))), size);
        relay_group.throughput(Throughput::Bytes(size as u64));
        relay_group.bench_with_input(
            BenchmarkId::new("direct_upload", size),
            &bytes,
            |bencher, bytes| {
                bencher.iter(|| runtime.block_on(consume_direct(black_box(Bytes::clone(bytes)))));
            },
        );
        relay_group.bench_with_input(
            BenchmarkId::new("plain_body_baseline", size),
            &bytes,
            |bencher, bytes| {
                bencher.iter(|| runtime.block_on(consume_plain(black_box(Bytes::clone(bytes)))));
            },
        );
    }
    relay_group.finish();
}

criterion_group!(benches, media_relay);
criterion_main!(benches);
