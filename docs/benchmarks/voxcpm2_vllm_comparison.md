# VoxCPM2 Serving Performance Comparison

This document defines a reproducible experiment for comparing
SGLang-Omni and vLLM-Omni serving `openbmb/VoxCPM2`. It is intended to answer
three separate questions:

1. Which framework gives better user-visible latency and streaming behavior?
2. Which framework delivers more useful audio per GPU while meeting an SLO?
3. Which SGLang-Omni optimizations account for any difference?

The experiment compares the common VoxCPM2 checkpoint only. Do not compare
SGLang-Omni VoxCPM 1.0 or 1.5 results with vLLM-Omni VoxCPM2 results.

## Rules for a Valid Comparison

Use the same:

- physical GPU, power limit, clocks, CUDA driver, and idle-host policy
- `openbmb/VoxCPM2` checkpoint revision and local weight files
- BF16 AR model and FP32-compatible output validation
- client implementation, request order, dataset revision, and network path
- reference audio bytes, reference transcript, and target text
- request-generation parameters and maximum generation budget
- 48 kHz, mono, PCM16 interpretation for VoxCPM2 output
- warmup policy, measured sample count, timeout, and retry policy
- server admission limit and KV-cache capacity for matched-configuration runs

Run one server at a time. Running the two servers concurrently on the same GPU
does not produce a valid comparison.

Record the full Git state, including uncommitted changes. A commit SHA alone is
not sufficient when benchmarking a dirty worktree.

## Workloads

Use `zhaochenyang20/seed-tts-eval-arrow` as the primary corpus. Download one
specific dataset snapshot before the experiment and use that same read-only
local snapshot for every run; the current benchmark CLI does not forward a
Hugging Face dataset revision. Do the same for the model checkpoint. Record
snapshot revisions and file hashes in the manifest. Report English and Chinese
separately.

The primary workload is continuation voice cloning:

```text
reference audio + reference transcript + target text
```

This is the shape used by the Seed-TTS benchmark. Report the following
secondary workloads separately:

- isolated cloning: reference audio without reference transcript
- text-only synthesis: no reference audio or transcript

Do not aggregate these modes. Their reference encoding and prompt construction
costs differ.

Use these sample budgets:

| Purpose | Minimum workload |
|---|---:|
| Smoke test | 20 requests |
| Serial latency | At least 1000 requests per language when reporting p99 |
| Closed-loop throughput | 500 requests or 5 minutes per point, whichever is longer |
| Open-loop capacity | 10 minutes per offered-load point |
| Quality | Full language split; use at least 200 samples for development |
| Soak | 30 minutes at 80% of the measured sustainable load |

Preserve a fixed sample order for paired comparisons. Also report results by
target-text length:

- short: fewer than 50 characters
- medium: 50-200 characters
- long: more than 200 characters

Add a separate synthetic long-input set at 500, 1000, and 2000 characters.
Do not mix synthetic long-input results into the Seed-TTS aggregate.

## Fixed Generation Semantics

The matched comparison uses:

| Parameter | Value |
|---|---:|
| CFM inference timesteps | 10 |
| CFG value | 2.0 |
| CFM temperature/noise scale | 1.0 |
| Speech speed | 1.0 |
| Maximum new tokens/decode steps | 2000 |
| Output sample rate | 48000 Hz |
| Output channels | 1 |
| Output sample width | 16 bit |

Do not use `top_p`, `top_k`, or `repetition_penalty` as comparison dimensions
unless both implementations are first verified to consume them in the VoxCPM2
generation path.

The implementations do not have identical random-noise and stop semantics.
Performance runs should not pass a request seed. Quality runs must use three
independent generations and report their distribution. Always report output
audio duration so that an implementation is not rewarded for stopping early.

The current neutral benchmark CLI forwards `max_new_tokens` but does not expose
all VoxCPM2 CFM parameters. vLLM-Omni also does not reliably apply the request
`max_new_tokens` field in its VoxCPM2 adapter. Before Phase 1, save each
server's resolved runtime configuration and assert the values in the table
above. Set generation caps in the server configuration as well as the request.
If a parameter cannot be set or verified on both sides, label it as an
implementation difference rather than silently assuming equivalent defaults.

## KV-Cache Capacity and High-Concurrency Admission

### KV bytes per token

VoxCPM2 has a 28-layer base LM and an 8-layer residual LM. With two KV heads,
head dimension 128, and BF16 cache elements:

```text
KV bytes/token
  = 2 (K and V)
    * 36 cached layers
    * 2 KV heads
    * 128 head dimension
    * 2 bytes
  = 36,864 bytes
  = 36 KiB/token
```

Do not omit the residual LM. Counting only 28 layers underestimates KV memory.

### Request token estimate

VoxCPM2 emits one AR patch for approximately 160 ms of audio:

```text
generated patch tokens ~= ceil(output audio seconds / 0.16)
                       ~= ceil(6.25 * output audio seconds)
```

Reference audio has approximately the same patch rate. Estimate a request as:

```text
T_request =
    text tokens
  + special tokens
  + reference patch tokens
  + generated patch tokens
```

Measure this value from successful requests when token headers are available.
Otherwise calculate it from the tokenizer and actual reference/output audio
duration. Store p50, p95, p99, and maximum request lengths.

### Capacity-derived admission

Let:

- `T_KV` be the actual KV token capacity reported after server startup
- `T_req_p95` be the measured p95 total tokens per request
- `alpha` be 0.80 for block rounding, fragmentation, and runtime variance

The typical-workload admission estimate is:

```text
C_typical = floor(alpha * T_KV / T_req_p95)
```

The full-context safety bound for the matched 4096-token context is:

```text
C_worst = floor(alpha * T_KV / 4096)
```

`C_typical` estimates useful benchmark concurrency. `C_worst` is the admission
limit that remains KV-safe if every request reaches 4096 tokens. Report both;
do not present the typical estimate as a worst-case guarantee.

For a matched-capacity run, choose a common token capacity first:

```text
T_common = min(T_KV_sglang, T_KV_vllm)
vLLM KV bytes = T_common * 36,864
SGLang max total tokens = T_common
```

Round the common capacity down to both frameworks' cache-block granularity.
The actual capacities in startup logs, not nominal GiB labels, must match.

KV capacity is not the only limit. LocDiT, CFM, AudioVAE, activations, request
state pools, and CUDA Graph allocations are outside or partly outside KV-cache
accounting. Every admission increase must pass an actual startup and load test.

### H100 80 GB capacity tiers

The following tiers use a 4096-token worst case and include approximately 20%
headroom. Rounded budgets are useful for capacity planning but are not exact
matched-capacity settings:

| Maximum concurrency | Raw worst-case KV | Rounded allocation budget |
|---:|---:|---:|
| 32 | 4.5 GiB | 6 GiB |
| 64 | 9 GiB | 12 GiB |
| 96 | 13.5 GiB | 18 GiB |
| 128 | 18 GiB | 24 GiB |

Use the following closed-loop sweep:

```text
1, 2, 4, 8, 16, 32, 48, 64
```

If concurrency 64 is stable and `C_typical >= 96`, extend to:

```text
96, 128
```

Do not capture graphs for 128 immediately. Calibrate in ascending order and
stop when any of the following occurs:

- startup or CUDA Graph capture OOM
- request failure rate exceeds 0.1%
- p95 RTF reaches or exceeds 1.0
- p95 first-audio latency grows by more than 2x from the preceding point
- throughput improves by less than 5% across two successive points
- queue depth grows for the duration of a fixed-load run

Report three limits separately:

- maximum stable concurrency: no OOM, crash, timeout burst, or growing queue
- throughput saturation point: the first point after which two increases add
  less than 5% throughput
- maximum SLO-valid concurrency: the highest point meeting all declared SLOs

Do not collapse these into one "failed concurrency."

## Server Profiles

Run two profiles. Do not combine them into one ranking.

### Production-default profile

Each framework uses its recommended optimizations. This answers which framework
a user would deploy by default.

Keep model semantics, dataset, generation limit, and output format fixed, but
allow framework-specific graph, fusion, and batching implementations.

Use the checked-in server configuration without capacity overrides:

```bash
export MODEL_PATH=/snapshots/openbmb-VoxCPM2/REVISION

# SGLang-Omni production default
sgl-omni serve \
  --model-path "$MODEL_PATH" \
  --config examples/configs/voxcpm2.yaml \
  --port 8000

# vLLM-Omni production default
vllm serve "$MODEL_PATH" \
  --omni \
  --deploy-config vllm_omni/deploy/voxcpm2.yaml \
  --port 8000
```

The checked-in defaults currently admit eight requests. Production-default
results therefore describe those defaults; high-concurrency results belong to
the capacity profiles below.

### Matched-capacity profile

Use identical:

- maximum admitted requests
- effective KV token capacity
- 4096-token comparison context
- maximum 2000 generated tokens
- graph coverage through the tested admission point, or eager mode on both

This profile helps separate scheduler/engine behavior from different memory
budgets.

## Starting SGLang-Omni

For a 64-request matched tier, use a common capacity of 327680 tokens. This is
the minimum theoretical capacity for 64 requests of 4096 tokens at `alpha=0.8`:

```bash
cd /path/to/sglang-omni

sgl-omni serve \
  --model-path "$MODEL_PATH" \
  --config examples/configs/voxcpm2.yaml \
  --max-running-requests 64 \
  --cuda-graph-max-bs 64 \
  --max-total-tokens 327680 \
  --port 8000
```

`327680` is a matched token cap, not a promise that the runtime can allocate
that many tokens. The server cannot raise the pool above its profiled capacity.
Record the startup log's actual `max_total_num_tokens`.

The VoxCPM preprocessing executor currently has a hard-coded concurrency of
eight. Keep it unchanged for the production-default profile. Before claiming
engine scaling above eight in the matched-capacity profile, make this limit
configurable and set it to at least the tested admission limit; otherwise
report it as an explicit front-end bottleneck. Record preprocessing queue time
separately from generation-engine queue time.

For 96 and 128, increase `max-running-requests`,
`cuda-graph-max-bs`, and the token cap only after the preceding tier is stable.
Use 491520 tokens for the 96 tier and 655360 for the 128 tier, subject to
profiled capacity.

SGLang-Omni VoxCPM diffusion graphs retain batches up to 8, multiples of 16,
and the maximum requested batch. Prefer 16/32/48/64/96/128 as high-concurrency
graph points.

### SGLang optimization environment

Record these variables explicitly:

```bash
export SGLANG_VOXCPM_ENABLE_VAE_COMPILE=0
export SGLANG_VOXCPM_ENABLE_ASYNC_DECODE=0
unset SGLANG_VOXCPM_DISABLE_DIFFUSION_GRAPH
```

Use the repository defaults for the production-default profile. Change one
variable at a time for ablation. Async decode should be evaluated only at
concurrency 16 or greater.

## Starting vLLM-Omni

Copy `vllm_omni/deploy/voxcpm2.yaml` to an experiment-specific file. For the
64-request matched tier set:

```yaml
stages:
  - stage_id: 0
    max_num_seqs: 64
    # 327680 tokens * 36864 bytes/token = 12079595520 bytes.
    kv_cache_memory_bytes: 12079595520
    max_num_batched_tokens: 4096
    max_model_len: 4096
    engine_extras:
      hf_overrides:
        voxcpm2_runtime_config:
          enable_unified_decode_graph: true
          unified_decode_graph_max_batch_size: 64
    default_sampling_params:
      max_tokens: 2000
```

Preserve the other VoxCPM2 defaults from the source YAML. Start with:

```bash
cd /path/to/vllm-omni

vllm serve "$MODEL_PATH" \
  --omni \
  --deploy-config /path/to/voxcpm2-c64.yaml \
  --port 8000
```

Use exactly 18119393280 bytes for 491520 tokens at concurrency 96 and
24159191040 bytes for 655360 tokens at concurrency 128, then round both
frameworks down if cache-block granularity requires it. Record the startup
log's available KV memory/token capacity and graph-capture peak. Do not infer
capacity from `gpu_memory_utilization` when `kv_cache_memory_bytes` is set.

## Neutral Client

Use one checkout and one version of
`benchmarks.eval.benchmark_tts_seedtts` to drive both servers. Only the target
server changes.

Non-streaming comparison:

```bash
cd /path/to/sglang-omni

export MODEL_PATH=/snapshots/openbmb-VoxCPM2/REVISION
export DATASET_PATH=/snapshots/seed-tts-eval-arrow/REVISION
export FRAMEWORK=sglang  # Set to vllm when targeting vLLM-Omni.

python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only \
  --use-existing-server \
  --base-url http://127.0.0.1:8000 \
  --model "$MODEL_PATH" \
  --meta "$DATASET_PATH" \
  --lang en \
  --max-concurrency 64 \
  --request-rate inf \
  --warmup 10 \
  --max-new-tokens 2000 \
  --output-dir "results/voxcpm2/${FRAMEWORK}/en/nonstream/c64/run-1"
```

Set `FRAMEWORK` to `sglang` or `vllm`. Repeat with `--lang zh`.

### Streaming compatibility requirement

SGLang-Omni returns raw PCM for `stream=true` and `response_format=pcm`.
vLLM-Omni requires `stream_format=audio` to select raw PCM instead of SSE.
The current neutral client does not send `stream_format`, defaults missing PCM
metadata to 24 kHz, and therefore is not yet valid for this cross-framework
streaming comparison. Before publishing streaming results:

1. add a target-independent `stream_format` client option
2. make vLLM-Omni return `X-Sample-Rate`, `X-Channels`, and `X-Bit-Depth`
3. change the neutral client to reject missing PCM metadata instead of falling
   back to 24 kHz
4. add per-stream playback-buffer/underrun calculation

Send:

```json
{
  "stream": true,
  "stream_format": "audio",
  "response_format": "pcm"
}
```

Both targets must return `audio/pcm` and metadata corresponding to 48 kHz,
mono, PCM16. The client must reject absent or inconsistent format metadata.

Do not compare SGLang raw PCM TTFA against vLLM SSE TTFA. Do not use a 24 kHz
fallback for VoxCPM2. If the vLLM benchmark client is used for an independent
cross-check, set:

```bash
export VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE=48000
```

## Experiment Matrix

### Phase 0: correctness and instrumentation

Run 20 requests per language and framework.

Verify:

- all responses decode to non-empty 48 kHz mono PCM/WAV
- the same dataset rows and reference bytes were sent
- generation parameters were accepted and applied
- actual prompt/completion lengths are recorded
- output duration differs by no more than an explainable model variance
- streaming chunk boundaries are preserved by the client
- GPU telemetry and server logs cover the measured interval

No performance claim is valid until this phase passes.

### Phase 1: serial latency

Use concurrency 1, at least 1000 requests per language when reporting p99, and
six fresh server starts.
Report:

- end-to-end latency p50/p95/p99
- streaming TTFA p50/p95/p99
- RTF p50/p95/p99
- output duration distribution
- prompt and completion token distributions
- reference preprocessing time if exposed

### Phase 2: closed-loop concurrency

For every framework and language, run:

```text
1, 2, 4, 8, 16, 32, 48, 64
```

Extend to 96 and 128 using the capacity criteria above. Use at least 500
requests or five minutes at each point. The request source should always have
enough work to keep the concurrency window full.

The current Seed-TTS runner consumes its sample list once. For a duration-based
run, materialize a deterministic repetition of the pinned sample-ID sequence
before dispatch so the measured interval reaches five minutes. Do not restart
the client repeatedly and then pool only the successful invocations.

Primary outputs:

- generated audio seconds per wall-clock second
- successful requests per second
- TTFA p95/p99
- RTF p95/p99
- end-to-end latency p95/p99
- failure and timeout rate
- average output audio duration
- peak resident GPU memory

Plot throughput and tail latency against concurrency on the same x-axis.

### Phase 3: open-loop capacity

Closed-loop testing measures maximum work conservation but hides queueing.
Use Poisson arrivals to find sustainable offered load.

The current Seed-TTS `BenchmarkRunner` is not a valid open-loop load generator:
requests blocked by its client semaphore wait outside the measured latency, and
it has no fixed-duration arrival loop. Do not use its `--request-rate` result
for a sustainable-capacity claim.

Before this phase, provide a neutral fixed-duration generator that:

- schedules arrivals independently of request completion
- records planned arrival, task creation, socket-send, first audio, and finish
- includes client-side queue wait in end-to-end latency
- reports load-generator lag and saturation
- runs for a fixed duration and stops scheduling at the deadline
- records server queue depth and running requests
- uses the same Seed-TTS sample cycle for both frameworks

First run a geometric sweep:

```text
0.5, 1, 2, 4, 8, 16 requests/s
```

After locating the knee, test intermediate points around it. Keep each point
for ten minutes. Use an admission cap at or above the stable closed-loop
concurrency.

A point is sustainable only if:

- success rate is at least 99.9%
- queue depth does not grow throughout the run
- p95 RTF is below 0.8
- p95 TTFA is at most 500 ms and p99 TTFA is at most 1 s
- at least 99% of streams meet the continuity threshold

Report the maximum offered RPS satisfying all criteria, not merely the highest
RPS that returns some successful responses.

### Phase 4: streaming continuity

For each successful stream collect:

- first-audio latency
- first audio payload bytes and playable duration
- chunk count
- inter-chunk p50/p95/p99
- maximum playback-buffer underrun
- continuity success using a 100 ms underrun threshold

Calculate inter-chunk and underrun statistics per stream first, then aggregate
the per-stream values. Pooling every chunk interval gives longer streams
disproportionate weight.

TTFA is not meaningful without first-payload size. A framework can report an
artificially low TTFA by flushing a tiny, non-useful first chunk.

### Phase 5: quality

Generate the complete EN and ZH splits at concurrency 1. Generate three
independent runs into their final quality directories:

```bash
for run in 1 2 3; do
  python -m benchmarks.eval.benchmark_tts_seedtts \
    --generate-only \
    --use-existing-server \
    --base-url http://127.0.0.1:8000 \
    --model "$MODEL_PATH" \
    --meta "$DATASET_PATH" \
    --lang en \
    --max-concurrency 1 \
    --request-rate inf \
    --warmup 10 \
    --max-new-tokens 2000 \
    --output-dir "results/voxcpm2/${FRAMEWORK}/en/quality/run-${run}"
done
```

Repeat for Chinese. Run quality evaluation after stopping the TTS server so
that ASR and similarity models do not contaminate serving measurements. Run
each evaluator against each generated directory:

```bash
for run in 1 2 3; do
  output_dir="results/voxcpm2/${FRAMEWORK}/en/quality/run-${run}"

  python -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only \
    --model "$MODEL_PATH" \
    --meta "$DATASET_PATH" \
    --lang en \
    --output-dir "$output_dir"

  python -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only \
    --model "$MODEL_PATH" \
    --meta "$DATASET_PATH" \
    --lang en \
    --output-dir "$output_dir"

  python -m benchmarks.eval.benchmark_tts_seedtts \
    --utmos-only \
    --model "$MODEL_PATH" \
    --meta "$DATASET_PATH" \
    --lang en \
    --output-dir "$output_dir"
done
```

Report WER/CER, speaker similarity, UTMOS, output duration, empty/truncated
audio rate, and repeated-audio rate. Pre-register quality non-inferiority
bounds before viewing performance results. Suggested development bounds are:

- WER/CER: no more than 1 absolute percentage point worse
- speaker similarity: no more than 0.01 lower
- UTMOS: no more than 0.10 lower
- empty/truncated audio: no increase above 0.1%

For a publication or release gate, replace these suggested bounds with values
validated by listening tests and the target product.

### Phase 6: SGLang-Omni ablation

Run ablations at concurrency 1, 32, 64, and the highest stable point:

| Dimension | Values |
|---|---|
| Diffusion CUDA Graph | on, off |
| VAE Snake compile | on, off |
| Async decode | on, off at concurrency >=16 |
| Streaming prefix length | 1, 2, 3, 4 |

Change one dimension at a time. Then test the best combined configuration.

Evaluate the quality/performance Pareto frontier separately:

```text
inference_timesteps = 4, 6, 8, 10, 12
```

Changing `inference_timesteps` can switch SGLang-Omni from a captured graph to
the eager CFM path. Label the execution path in every result.

## Resource Telemetry

Collect at one-second intervals:

- GPU utilization
- memory used
- SM and memory-clock frequency
- power draw and enforced power limit
- temperature and throttling reason
- host CPU utilization and RSS
- server queue/running request count, when available

Example GPU capture:

```bash
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active \
  --format=csv \
  --loop=1 \
  > results/gpu-telemetry.csv
```

Start telemetry before server startup and stop it after the measured run.
Mark warmup and measurement timestamps in the manifest.

Capture host and process telemetry separately, for example with
`pidstat -durh 1 -p <server-pid>`. Export queue/running-request metrics from
the server when available. If a driver does not expose
`clocks_event_reasons.active`, record the supported `nvidia-smi --query-gpu`
fields and use `nvidia-smi dmon` for throttling diagnostics.

## Repetition and Run Order

Use six fresh server starts per point. Randomize a balanced order within each
pair to reduce thermal, cache, and host-state bias:

```text
pair 1: SGLang -> vLLM
pair 2: vLLM -> SGLang
pair 3: randomized, then run the reverse order
```

Warm up with at least ten requests and do not include graph capture, compile,
or first model execution in measured results. Cold-start and first-request
latency are separate experiments.

For aggregate reporting:

- preserve every per-request record
- calculate percentiles independently for each repeat
- report the median of the six repeat-level values
- include the range and a bootstrap 95% confidence interval
- use paired sample IDs when comparing quality and output duration

Do not pool all repeats before computing p99; pooling hides run-to-run
instability.

Use a hierarchical bootstrap: resample server starts as clusters, then resample
paired request IDs within each selected start. A six-start interval is still
descriptive rather than definitive; preserve the repeat-level range and do not
claim tiny differences as significant.

## Required Artifacts

Every run directory must contain:

```text
manifest.json
speed_results.json
generated_audio_metadata.json
per-request records
server.log
client.log
gpu-telemetry.csv
environment.txt
git-state/
```

The manifest must record:

- framework and repository URL
- commit SHA, `git status`, and patch hash
- model ID, revision, and weight-file hashes
- dataset ID, revision, split, sample IDs, and ordering seed
- GPU name/UUID, driver, CUDA, PyTorch, and SGLang/vLLM versions
- server command and effective resolved configuration
- actual KV token capacity and KV byte budget
- graph buckets/capture status
- warmup and measured time range
- all request-generation parameters
- client commit and command

The current benchmark command does not create this complete artifact set.
Use an orchestration script to start telemetry, capture server/client logs,
write the manifest and Git state, invoke the benchmark, and terminate all
processes. Until such a script exists, treat this section as a mandatory manual
checklist and reject incomplete runs.

## Reporting

The primary comparison table should contain:

| Framework | Profile | Concurrency | Audio s/s | Req/s | TTFA p95 | TTFA p99 | RTF p95 | E2E p95 | Success |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

Add:

1. throughput and TTFA versus concurrency
2. throughput and RTF versus concurrency
3. sustainable RPS under the declared SLO
4. GPU memory and power versus concurrency
5. WER/CER, similarity, and UTMOS quality table
6. SGLang-Omni optimization ablation table

The headline system-efficiency metric is generated audio seconds per GPU
second while satisfying the latency, continuity, success-rate, and quality
guardrails. Requests per second is secondary because output duration varies.

Before starting the experiment, save the chosen SLO and quality bounds in the
manifest. The defaults in this document are p95 TTFA <=500 ms, p99 TTFA <=1 s,
p95 RTF <0.8, continuity success >=99%, and request success >=99.9%.

## Invalid Comparisons

Do not publish a comparison when any of the following is true:

- different checkpoint or dataset revisions
- different request clients or request order
- one side uses 24 kHz to interpret VoxCPM2 PCM
- one side receives SSE while the other receives raw PCM
- different maximum generation budgets or admission limits in the matched run
- quality or output duration is omitted from a throughput claim
- one result includes warmup/graph capture and the other does not
- the load generator saturates before the server
- p95/p99 is calculated from a 20-request smoke test
- the SGLang-Omni result omits uncommitted optimization changes

Treat existing repository baselines as framework-internal regression references,
not as cross-framework comparison results.
