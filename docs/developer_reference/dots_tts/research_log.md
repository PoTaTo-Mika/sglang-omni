# dots.tts Low-Fruit Optimization Research Log

Last updated: 2026-08-11.

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
| Reuse one batched latent denormalization | [#1438](https://github.com/sgl-project/sglang-omni/pull/1438) | `76d0816d` | Merged |
| Stack feedback directly into the CUDA Graph buffer | [#1439](https://github.com/sgl-project/sglang-omni/pull/1439) | `eede6786` | Merged |
| Bypass redundant vocoder staging | [#1440](https://github.com/sgl-project/sglang-omni/pull/1440) | `25ae3d95` | Merged |
| Reuse the CFG null-projection bias | [#1441](https://github.com/sgl-project/sglang-omni/pull/1441) | `5f530dc5` | Merged |
| Batch compatible streaming AudioVAE steps | [#1444](https://github.com/sgl-project/sglang-omni/pull/1444) | `7f5529ad` | Merged |

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

## 2026-08-11 Streaming SeedTTS Concurrency Sweep

- Hypothesis: the merged cross-request AudioVAE batching in #1444 should retain
  its c=8 streaming gain on the full SeedTTS EN set and reveal the saturation
  point across c=1,2,4,8,16,32.
- Contract: [contract.json](runs/2026-08-11-streaming-seedtts-sweep/contract.json)
- Allocation: [rollout-plan.json](runs/2026-08-11-streaming-seedtts-sweep/rollout-plan.json)
- Execution: hyper00 and hyper01, exclusive H200 access; c=1 uses the first 50
  samples, all other concurrency levels use all 1,088 English samples.
- Failure: the first detached coordinators exited before model setup because
  the bind-mounted checkout tripped Git's `safe.directory` ownership check.
  The launcher now scopes `safe.directory` to its three read-only Git commands;
  no benchmark request ran in the failed attempt.
- Failure: the reused hyper01 image lacked the declared `jiwer` dependency, so
  its first workers exited during module import. The launch script now installs
  `jiwer` before starting workers; no request ran in that attempt.
- Failure: the previous reusable containers used an older runtime that lacked
  `dots_tts` and `msgpack`. Both canonical containers were recreated with the
  current H200 profile image, preserving their persistent `/data` mount. The
  launcher now pins `dots.tts`, `jiwer`, and `openai-whisper`; no request ran in
  the incompatible-runtime attempts.
- Allocation correction: use only high-numbered GPUs: hyper00 GPUs 7,6,5,4 for
  c=1,2,4,8 and hyper01 GPUs 7,6 for c=16,32.
- Pause: the first valid sweep was stopped on request after c=1 had completed
  both synthesis and ASR. The retained c=1 artifacts contain 50/50 successful
  requests and 50/50 evaluated WER samples; the other concurrency runs were
  incomplete and are not used as results.
- Resume allocation:
  [resume-rollout-plan.json](runs/2026-08-11-streaming-seedtts-sweep/resume-rollout-plan.json).
  It uses only GPUs reported free immediately before launch, ordered from high
  to low: hyper00 7,6,5 for c=2,4,8 and hyper01 7,6 for c=16,32. Each host uses
  a new task-scoped timestamp container exposing only its selected GPUs.
- Status: `RUNNING`.
