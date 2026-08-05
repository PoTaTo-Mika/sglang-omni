# dots.tts

[dots.tts-mf](https://huggingface.co/dots-studio/dots.tts-mf) is a 2B continuous
AR TTS model (MeanFlow distillation of dots.tts-soar). The current SGLang-Omni
path supports **zero-shot continuation cloning** through `/v1/audio/speech` and
produces **48 kHz** audio.

```text
preprocessing -> reference_encode -> tts_engine -> audio_decode
```

`reference_encode` loads and caches the reference waveform, then builds the
generation schedule. `tts_engine` runs the in-tree MeanFlow AR loop and emits
continuous latents. `audio_decode` converts those latents with the AudioVAE.

## Prerequisites

```bash
hf download dots-studio/dots.tts-mf --local-dir /path/to/dots.tts-mf
```

## Server

```bash
sgl-omni serve \
  --model-path /path/to/dots.tts-mf \
  --config examples/configs/dots_tts.yaml \
  --allowed-local-media-path /path/to/references \
  --port 8000
```

## Voice cloning

Provide one local reference clip and its transcript:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dots-tts-mf",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "file:///path/to/references/prompt.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "response_format": "wav"
  }' \
  --output cloned.wav
```

`ref_audio` / `ref_text` are accepted as shorthand for the single reference.

## Generation parameters

| Parameter | Default | Notes |
|---|---|---|
| `input` | required | Non-empty text to synthesize |
| `references` | required | Exactly one local clip with non-empty `text` |
| `num_steps` | `4` | MeanFlow NFE; pass via `extra_body` / generation params |
| `guidance_scale` | `1.2` | Ignored on mf (CFG fused into the student) |
| `speaker_scale` | `1.5` | X-vector scale |
| `seed` | `null` | Optional deterministic noise seed |
| `stream` | `false` | Streaming is not supported yet |

## Correctness check

Official mf Seed-TTS-Eval **test-en WER is 1.29%** (NFE=4). Bring the server
up first (single GPU is enough), then generate against it and score with ASR:

```bash
# Terminal A — TTS (1 GPU)
sgl-omni serve \
  --model-path /path/to/dots.tts-mf \
  --config examples/configs/dots_tts.yaml \
  --allowed-local-media-path / \
  --port 8000

# Terminal B — generate EN 1088
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model dots-tts-mf \
  --base-url http://127.0.0.1:8000 \
  --use-existing-server \
  --generate-only \
  --lang en \
  --concurrency 1 \
  --output-dir results/dots_tts_seedtts_en

# Terminal C — ASR WER (separate GPU)
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model dots-tts-mf \
  --transcribe-only \
  --lang en \
  --port 8001 \
  --output-dir results/dots_tts_seedtts_en
```

Serving corpus WER should stay close to the official number. Values that jump
above a few percent usually mean a broken feedback / latent / VAE path.

## Known limitations

- Non-streaming only (`BatchVocoderBase`).
- No OmniScheduler / SGLang KV path yet — the AR loop uses the in-tree
  StaticCache implementation (Audar-style stage executor).
- soar checkpoint and x-vector-only / text-only modes are follow-ups.
- `torch.compile` / DiT CUDA Graph are not enabled.
