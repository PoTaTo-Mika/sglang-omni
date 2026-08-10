# dots.tts Low-Fruit Optimization Research Log

Last updated: 2026-08-10.

This log records bounded, math-preserving serving optimizations found after the
larger dots.tts CUDA Graph, batching, and KV-cache work had landed. The local
microbenchmarks isolate the changed operator path; they are not end-to-end RTF
or throughput claims.

## Source of Truth

- Repository: [sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni)
- Audit base: `main` at `1bb0f15b`
- Performance roadmap: [#1367](https://github.com/sgl-project/sglang-omni/issues/1367)
- Model operators: [`dots.tts==0.2.1`](https://pypi.org/project/dots.tts/0.2.1/)
- Checkpoints: [`dots.tts-mf`](https://huggingface.co/dots-studio/dots.tts-mf)
  and [`dots.tts-soar`](https://huggingface.co/dots-studio/dots.tts-soar)

| Optimization | Pull request | Head commit | Status |
| --- | --- | --- | --- |
| Reuse one batched latent denormalization | [#1438](https://github.com/sgl-project/sglang-omni/pull/1438) | `17e9ffe4` | Draft |
| Stack feedback directly into the CUDA Graph buffer | [#1439](https://github.com/sgl-project/sglang-omni/pull/1439) | `c230a111` | Draft |
| Bypass redundant vocoder staging | [#1440](https://github.com/sgl-project/sglang-omni/pull/1440) | `a544e4bf` | Draft |
| Reuse the CFG null-projection bias | [#1441](https://github.com/sgl-project/sglang-omni/pull/1441) | `2b1ab3da` | Draft |

No dataset, checkpoint, adapter, or model artifact was produced. The PR
branches and this log are the complete reusable state; raw benchmark logs are
transient validation output rather than source-of-truth artifacts.

## Experiment Contract

- Keep RNG, tensor shapes, request ordering, solver parameters, and model
  outputs unchanged.
- Prefer removing redundant allocations, copies, or mathematically constant
  work over introducing a new kernel or execution mode.
- Validate the owning unit-test file and run an RTX A6000 CUDA microbenchmark
  with PyTorch `2.11.0+cu130`, SGLang `0.5.16`, and `dots.tts==0.2.1`.
- Report isolated path latency only; require a full serving benchmark before
  claiming an end-to-end latency or throughput change.

## Results and Decisions

| Hypothesis | Change | Isolated result | Correctness evidence | Decision |
| --- | --- | --- | --- | --- |
| Batched MeanFlow denormalizes the same latent repeatedly | Reuse one batch result for semantic-encoder input and per-request views | At batch 16, 17 calls became 1; 558.794 to 77.494 us/step (-86.13%) | `test_flow_head.py`: 9 passed; batch-call regression | Keep |
| Graph feedback staging allocates and copies an intermediate tensor | Use `torch.stack(..., out=persistent_buffer)` | Batch 1-16: about 49-58% lower staging latency | `test_model_runner.py`: 9 passed; output-storage regression | Keep |
| Vocoder always pads singleton and equal-length buckets | Direct singleton input, one `cat` for equal lengths, retain padding for mixed lengths | Batch 1/4/8/16 assembly: 39.951/85.686/187.155/357.004 to 1.082/14.653/17.626/18.554 us | Vocoder tests: 14 passed; singleton storage regressions | Keep |
| SOAR/base computes `Linear(zeros)` every token | Expand the linear bias for the CFG null branch | Actual 1536-to-1024 projection: 118.875 to 74.577 us/append (-37.26%) | Bitwise CUDA parity; `test_flow_head.py`: 10 passed | Keep |
| Binary EOS softmax could become a logit-difference comparison | No code change | Would change rounding near a user-visible threshold | Static review | Drop |
| Tail index and mask tensors could be cached more aggressively | No code change | Frequent batch 8/16 shapes are already CUDA Graph captured; expected win is small outside fallback shapes | Static review | Drop |
| Incremental acoustic-tail KV allocation could reduce memory | No code change | Material design and lifecycle work, outside the low-fruit scope | Roadmap review | Defer |

## Failure Notes

- A local macOS test environment had mismatched Torch/Torchaudio packages and
  could not collect model-level tests. Validation moved to an isolated venv in
  the existing Aries canonical container; the shared container environment was
  not modified.
- One foreground SSH test lost its connection. The retry ran detached inside
  the same container and persisted its exit code; all nine model-runner tests
  passed.

## Reproduction Checks

Run the owning tests for each PR:

```bash
python -m pytest -q tests/unit_test/dots_tts/test_flow_head.py
python -m pytest -q tests/unit_test/dots_tts/test_model_runner.py
python -m pytest -q \
  tests/unit_test/dots_tts/test_vocoder.py \
  tests/unit_test/dots_tts/test_vocoder_streaming.py
```

Before quoting serving-level gains, run the canonical Seed-TTS benchmark from
the [dots.tts cookbook](../../cookbook/dots_tts.md#performance) on the same GPU
for both revisions.
