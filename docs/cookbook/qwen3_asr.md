# Qwen3-ASR

[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) is a multilingual
audio transcription model served through the OpenAI-compatible transcription
API.

## Overview

| Item | Value |
|---|---|
| Task | ASR |
| Checkpoint(s) | `Qwen/Qwen3-ASR-1.7B` |
| Endpoint(s) | `/v1/audio/transcriptions` |
| Pipeline | audio preprocessing → ASR engine → response formatting |
| Input / output | One uploaded audio file → text, JSON, or verbose JSON transcript |
| Streaming | SSE transcript output; complete uploaded-file input, up to 1,200 seconds |
| Validated hardware | H100; RTX 4090 24 GB |

Qwen3-ASR does not support `/v1/audio/translations`; that route returns HTTP
400. See the [audio translation matrix](../basic_usage/audio_translations.md)
for models that support it.

## Prerequisites

Follow [Installation](../get_started/installation.md). No additional
model-specific package is required.

## Deploy

Qwen3-ASR runs one ASR stage on one GPU:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --port 8000
```

## Send a request

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=json
```

See the [Transcription API](../user_guide/serving/transcription_api.md) for
shared request fields, response formats, usage, and errors.

## Capabilities

### Language hints

When `language` is omitted, Qwen3-ASR detects the spoken language. You can pass
a case-insensitive code or canonical name for these 30 languages:

| Codes | Canonical names |
|---|---|
| `ar`, `yue`, `zh`, `cs`, `da`, `nl`, `en`, `fil`, `fi`, `fr` | Arabic, Cantonese, Chinese, Czech, Danish, Dutch, English, Filipino, Finnish, French |
| `de`, `el`, `hi`, `hu`, `id`, `it`, `ja`, `ko`, `mk`, `ms` | German, Greek, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Macedonian, Malay |
| `fa`, `pl`, `pt`, `ro`, `ru`, `es`, `sv`, `th`, `tr`, `vi` | Persian, Polish, Portuguese, Romanian, Russian, Spanish, Swedish, Thai, Turkish, Vietnamese |

The legacy `cn` and regional `zh-*` spellings map to Chinese. Unsupported hints
return HTTP 400. The model recognizes additional Chinese dialects, but they are
not separate forced hints; use `Chinese` or `zh`.

### Long audio

Non-streaming uploads are split into engine requests and reassembled in order.
These model-owned defaults are declared by `Qwen3ASRPipelineConfig`:

| Setting | Value | Behavior |
|---|---:|---|
| `max_audio_clip_s` | 60 | Engine chunk length |
| `max_native_clip_s` | 1,200 | Native and streaming request limit |
| `max_total_audio_s` | 3,600 | Whole non-streaming upload limit |
| `max_concurrent_chunks` | 8 | Per-upload engine concurrency |
| `min_tail_s` | 0.5 | Minimum final chunk length |

`verbose_json` returns one segment per chunk with chunk-level start and end
times, not word timestamps. Formats without a readable duration fall back to
the non-chunked path.

### Streaming

Streaming does not currently use long-audio chunking, so uploads above 1,200
seconds return HTTP 400. Use non-streaming mode for longer files. See
[Streaming](../user_guide/advanced_features/streaming.md) for the shared SSE
event contract.

## Configuration

The checked-in `examples/configs/qwen3_asr_rtx4090.yaml` profile keeps BF16,
limits the stage to 16 running requests, and sets `mem_fraction_static` to
`0.65`; it was validated on one 24 GB RTX 4090. This is not a minimum-memory
claim.

The default `auto` dtype follows the BF16 checkpoint configuration. Pass
`--asr.factory.dtype float16` only when you intentionally need FP16. Per-stage
config files and dotted CLI overrides follow the shared
[configuration contract](../developer_reference/config.md); command-line
overrides take precedence over the checked-in profile.

`prompt` is accepted for OpenAI compatibility but Qwen3-ASR ignores it. Audio
is resampled to 16 kHz before transcription.

## Limitations

- The endpoint accepts one uploaded file per request.
- `/v1/audio/translations` is unsupported.
- Streaming is limited to 1,200 seconds and does not use long-audio chunking.
- Timestamps are chunk-level; the model does not emit word timestamps.
- `prompt` does not affect transcription.

## Benchmark

Run the Seed-TTS ASR benchmark against the deployed server:

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --concurrencies 1,2,4,8,16,32,64 \
  --repeats 3 \
  --warmup
```

See the
[Qwen3-ASR concurrency profile](../developer_reference/qwen3_asr_concurrency_profile.md)
for the measured tuning study and bottleneck decomposition, and follow the
[benchmark methodology](../benchmarks/methodology.md) when publishing results.

## Related documentation

- [Transcription API](../user_guide/serving/transcription_api.md)
- [Streaming](../user_guide/advanced_features/streaming.md)
- [Admission control](../user_guide/advanced_features/admission_control.md)
- [Benchmark methodology](../benchmarks/methodology.md)
- [Audio translation support](../basic_usage/audio_translations.md)
- [MPS/DP deployment](../basic_usage/mps_dp.md)
- [Supported models](../supported_models.md)
- [Qwen3-ASR concurrency profile](../developer_reference/qwen3_asr_concurrency_profile.md)
