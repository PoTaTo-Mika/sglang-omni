# Whisper ASR

Whisper ASR checkpoints can be started through the OpenAI-compatible `/v1/audio/transcriptions` endpoint, but this path is experimental in the current SGLang-Omni tree. Prefer [Qwen3-ASR](qwen3_asr.md) for validated ASR serving.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download a Whisper checkpoint:

```bash
hf download openai/whisper-large-v3
```

## Server Configuration

Whisper ASR runs a single ASR stage on one GPU.
Async decode is enabled by default for all decode batch sizes, allowing the
shared one-step-lookahead path to overlap host-side result processing with the
next GPU decode forward even for a single request. Use `--decode-mode sync` to
disable it, or tune the crossover with `--async-lookahead-min-batch-size`.

```bash
sgl-omni serve \
  --model-path openai/whisper-large-v3 \
  --port 8000
```

For example, force synchronous decode when comparing modes:

```bash
sgl-omni serve \
  --model-path openai/whisper-large-v3 \
  --decode-mode sync \
  --port 8000
```

## Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=openai/whisper-large-v3 \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "openai/whisper-large-v3",
            "response_format": "json",
        },
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
print(resp.json()["text"])
```

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | unset | Optional language hint |
| `response_format` | string | `json` | Use `json` for the current Whisper path |
| `temperature` | float | `0.0` | Sampling temperature; defaults to greedy decoding |

The request builder also supports `task` (`transcribe` by default) and
`max_new_tokens`, but the public transcription endpoint currently exposes only
the fields above. The route uses the ASR stage default unless the pipeline is
configured another way. For smoke tests, keep the request minimal and use
`response_format=json`.

## Benchmarking

Use `benchmarks/eval/benchmark_asr_seedtts.py` to sweep ASR concurrency on
SeedTTS reference audio through `/v1/audio/transcriptions`. Compare sync and
async decode with `--decode-mode` while keeping the model, sample subset,
concurrency list, and repeat count identical across arms.

Higher concurrency currently needs atomic encoder-prefix admission
(`chunked_prefill_size=0`; see [#1412](https://github.com/sgl-project/sglang-omni/pull/1412));
otherwise concurrent prefills can crash the Whisper stage. The profile below
sets that override via `runtime_overrides`.

```bash
MODEL_PATH=$(hf download openai/whisper-base)
cat > /tmp/whisper_async_ab.yaml <<EOF
config_cls: WhisperASRPipelineConfig
name: whisper-async-ab
model_path: ${MODEL_PATH}
runtime_overrides:
  asr:
    server_args_overrides:
      chunked_prefill_size: 0
EOF

PORT_SYNC=8101
PORT_ASYNC=8102

# Arm A: sync decode (baseline)
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --config /tmp/whisper_async_ab.yaml \
  --model-name openai/whisper-base \
  --decode-mode sync \
  --port "${PORT_SYNC}"

# Arm B: async decode with min batch size 1 (default)
CUDA_VISIBLE_DEVICES=1 sgl-omni serve \
  --config /tmp/whisper_async_ab.yaml \
  --model-name openai/whisper-base \
  --decode-mode async \
  --async-lookahead-min-batch-size 1 \
  --port "${PORT_ASYNC}"

for PORT in "${PORT_SYNC}" "${PORT_ASYNC}"; do
  python -m benchmarks.eval.benchmark_asr_seedtts \
    --port "${PORT}" \
    --model-path openai/whisper-base \
    --max-samples 20 \
    --concurrencies 1,2,4,8 \
    --repeats 3 --warmup \
    --output "whisper_async_ab_port${PORT}.json"
done
```

On one H200 with `openai/whisper-base` and 20 SeedTTS EN clips, both arms
evaluated 60/60 requests at every concurrency with identical corpus WER
`0.0415`. Async lookahead (`min_batch_size=1`) improved throughput at every
measured level:

| Concurrency | Sync req/s | Async req/s | Throughput gain | Sync mean latency (s) | Async mean latency (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 20.440 | 21.592 | +5.6% | 0.049 | 0.046 |
| 2 | 28.827 | 29.314 | +1.7% | 0.069 | 0.068 |
| 4 | 36.177 | 38.154 | +5.5% | 0.109 | 0.104 |
| 8 | 42.686 | 44.550 | +4.4% | 0.182 | 0.174 |

## Known Limitations

- This path is experimental and not yet correctness-validated. Prefer Qwen3-ASR
  for validated ASR serving.
- Keep Whisper ASR at encoder batch size 1.
- Use `response_format=json`; other response formats are not validated for this
  experimental path.
- First startup can take several minutes.
- The endpoint accepts one uploaded file per request.
- Audio is resampled to 16 kHz before transcription.
- `prompt` is accepted by the HTTP endpoint for OpenAI compatibility, but
  Whisper ASR currently does not pass it into decoding.
