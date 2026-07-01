# Qwen3-ASR

[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) is an audio transcription model served through the OpenAI-compatible `/v1/audio/transcriptions` endpoint. It accepts one uploaded audio file per request and returns text.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

```bash
hf download Qwen/Qwen3-ASR-1.7B
```

## Server Configuration

Qwen3-ASR runs a single ASR stage on one GPU.

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --port 8000
```

## Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "Qwen/Qwen3-ASR-1.7B",
            "language": "en",
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
| `language` | string | `en` | Language hint; `zh`/`cn` select Chinese, other values use English prompting |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `temperature` | float | `0.01` effective | Sampling temperature; `0` is converted to near-greedy `0.01` |

`verbose_json` is accepted, but currently returns the same minimal JSON shape as `json`:
`{"text": "..."}`.

`max_new_tokens` is supported inside the model request builder, but the public transcription endpoint does not currently expose it as a form field. The route uses the ASR stage default unless the pipeline is configured another way.

## Benchmarking

SeedTTS EN concurrency/WER benchmarking for Qwen3-ASR lives in
`benchmarks/eval/benchmark_asr_concurrency.py` (default
`--model-path Qwen/Qwen3-ASR-1.7B`). The per-model HTTP knobs are resolved by
`benchmarks.tasks.tts.make_asr_send_fn`.

```bash
sgl-omni serve --model-path Qwen/Qwen3-ASR-1.7B --port 8000

# Sweep the full SeedTTS EN set (1088 clips) at 1..64 concurrency, 3 repeats:
python -m benchmarks.eval.benchmark_asr_concurrency \
  --port 8000 --concurrencies 1,2,4,8,16,32,64 --repeats 3
```

Qwen3-ASR correctness and concurrency coverage is in
`tests/test_model/test_qwen3_asr_ci.py`, which transcribes SeedTTS samples
through `/v1/audio/transcriptions` and checks WER, throughput, latency, and RTF.

```bash
QWEN3_ASR_CI_CONCURRENCY=32 pytest tests/test_model/test_qwen3_asr_ci.py -s
```

## Known Limitations

- The endpoint accepts one uploaded file per request.
- `prompt` is accepted by the HTTP endpoint for OpenAI compatibility, but Qwen3-ASR currently ignores it.
- Audio is resampled to 16 kHz before transcription.
