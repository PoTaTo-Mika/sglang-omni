#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Recheck PR #1071's same-GPU DP3 x MPS upper-bound result with real
# /v1/audio/speech Higgs voice-clone requests.
set -Eeuo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
CLIENT="$HERE/load_client.py"

: "${PR_COMMIT:=6a5ac7a543294d747005a343f6d1336abdcb4bf0}"
: "${MODEL:=/user/sunao/models/higgs-tts-3-4b}"
: "${SEEDTTS_META:=/user/sunao/datasets/seed-tts-eval-50-arrow}"
: "${PYTHON:=$REPO/.venv/bin/python}"
: "${GPU_ID:=0}"
: "${MAX_RUNNING_REQUESTS:=160}"
: "${CONC_PER_REPLICA:=$MAX_RUNNING_REQUESTS}"
: "${MAX_TOTAL_TOKENS:=54000}"
: "${SECS:=110}"
: "${WARMUP_SECS:=20}"
: "${REPS:=2}"
: "${BASE_PORT:=8801}"
: "${DP1_SERVER_CORES:=0-29}"
: "${DP3_SERVER_CORES:=0-9 10-19 20-29}"
: "${DP1_CLIENT_CORES:=30}"
: "${DP3_CLIENT_CORES:=30 31 32}"
: "${OUT:=/user/sunao/pr1071_mps_speech_$(date +%Y%m%d_%H%M%S)}"
: "${TMP_ROOT:=/tmp/pr1071-mps-speech-$UID-$$}"

WT="$TMP_ROOT/pr1071"
STATE_BASE="/tmp/pr1071-mps-state-$UID-$$"
CONFIG_PATH="$OUT/higgs_pr1071_mps.yaml"
RESULTS_JSONL="$OUT/results.jsonl"
CURRENT_STATE_ROOT=""
CURRENT_RUN_ID=""
CLIENT_PIDS=()

die() {
  echo "error: $*" >&2
  exit 1
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${CLIENT_PIDS[@]:-}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  if [ -n "$CURRENT_STATE_ROOT" ] && [ -n "$CURRENT_RUN_ID" ] && [ -d "$WT" ]; then
    STATE_ROOT="$CURRENT_STATE_ROOT" \
      bash "$WT/examples/mps_dp/launch.sh" down "$CURRENT_RUN_ID" \
      >/dev/null 2>&1 || true
  fi
  git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true
  rm -rf "$TMP_ROOT"
  exit "$status"
}
trap cleanup EXIT INT TERM

[[ "$MAX_RUNNING_REQUESTS" =~ ^[1-9][0-9]*$ ]] \
  || die "MAX_RUNNING_REQUESTS must be positive"
[[ "$CONC_PER_REPLICA" =~ ^[1-9][0-9]*$ ]] \
  || die "CONC_PER_REPLICA must be positive"
[[ "$MAX_TOTAL_TOKENS" =~ ^[1-9][0-9]*$ ]] \
  || die "MAX_TOTAL_TOKENS must be positive"
[[ "$REPS" =~ ^[1-9][0-9]*$ ]] || die "REPS must be positive"
[ -x "$PYTHON" ] || die "Python is not executable: $PYTHON"
[ -f "$MODEL/config.json" ] || die "Higgs model not found: $MODEL"
[ -e "$SEEDTTS_META" ] || die "SeedTTS dataset not found: $SEEDTTS_META"

# Trust only this synced repository for this process. The remote checkout is
# commonly owned by the image's build UID rather than the devspace root user.
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$REPO"
git -C "$REPO" cat-file -e "$PR_COMMIT^{commit}" \
  || die "missing PR commit $PR_COMMIT"
mkdir -p "$TMP_ROOT" "$OUT"
OUT=$(cd -- "$OUT" && pwd)
CONFIG_PATH="$OUT/higgs_pr1071_mps.yaml"
RESULTS_JSONL="$OUT/results.jsonl"
: > "$RESULTS_JSONL"

git -C "$REPO" worktree add --detach "$WT" "$PR_COMMIT" >/dev/null

"$PYTHON" - "$CONFIG_PATH" "$MODEL" "$MAX_TOTAL_TOKENS" \
  "$MAX_RUNNING_REQUESTS" <<'PY'
import sys
from pathlib import Path

import yaml

path, model, tokens, requests = sys.argv[1:]
config = {
    "config_cls": "HiggsTtsPipelineConfig",
    "name": "higgs",
    "model_path": model,
    "runtime_overrides": {
        "tts_engine": {
            "server_args_overrides": {
                "max_total_tokens": int(tokens),
                "max_running_requests": int(requests),
                "cuda_graph_max_bs": int(requests),
            }
        }
    },
}
Path(path).write_text(yaml.safe_dump(config, sort_keys=False))
PY

cat > "$OUT/environment.txt" <<EOF
date=$(date --iso-8601=seconds)
commit=$PR_COMMIT
model=$MODEL
dataset=$SEEDTTS_META
gpu_id=$GPU_ID
max_running_requests=$MAX_RUNNING_REQUESTS
concurrency_per_replica=$CONC_PER_REPLICA
max_total_tokens_per_replica=$MAX_TOTAL_TOKENS
secs=$SECS
warmup_secs=$WARMUP_SECS
reps=$REPS
dp1_server_cores=$DP1_SERVER_CORES
dp3_server_cores=$DP3_SERVER_CORES
EOF
nvidia-smi -i "$GPU_ID" -q > "$OUT/nvidia-smi-q.txt"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY=localhost,127.0.0.1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
export REPO SEEDTTS_META

aggregate() {
  local condition_dir=$1 dp=$2 rep=$3 idle_mib=$4 load_mib=$5
  "$PYTHON" - "$condition_dir" "$dp" "$rep" "$idle_mib" "$load_mib" \
    "$RESULTS_JSONL" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
dp, rep, idle_mib, load_mib = map(int, sys.argv[2:6])
jsonl = Path(sys.argv[6])
clients = [json.loads(p.read_text()) for p in sorted(root.glob("client_*.json"))]
if len(clients) != dp:
    raise SystemExit(f"expected {dp} client results, found {len(clients)}")
for client in clients:
    check = client.get("payload_check") or {}
    if (
        not client.get("api_url", "").endswith("/v1/audio/speech")
        or check.get("input_chars", 0) <= 0
        or check.get("references") != 1
        or client["errors"]
    ):
        raise SystemExit("invalid endpoint, payload, or request result")
latencies = sorted(
    value for client in clients for value in client.get("latency_window_s", [])
)
if not latencies:
    raise SystemExit("no steady-window latency samples")


def percentile(q):
    pos = (len(latencies) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return latencies[lo]
    return latencies[lo] * (hi - pos) + latencies[hi] * (pos - lo)


qps = sum(client["qps_window"] for client in clients)
latency = statistics.fmean(latencies)
result = {
    "dp": dp,
    "rep": rep,
    "aggregate_concurrency": sum(client["conc"] for client in clients),
    "qps": qps,
    "latency_mean_s": latency,
    "latency_p50_s": percentile(0.50),
    "latency_p95_s": percentile(0.95),
    "qps_x_latency": qps * latency,
    "requests": len(latencies),
    "errors": sum(client["errors"] for client in clients),
    "prompt_tokens_mean": statistics.fmean(
        client["prompt_tokens_mean"] for client in clients
    ),
    "completion_tokens_mean": statistics.fmean(
        client["completion_tokens_mean"] for client in clients
    ),
    "audio_duration_mean_s": statistics.fmean(
        client["audio_duration_mean_s"] for client in clients
    ),
    "vram_idle_mib": idle_mib,
    "vram_load_mib": load_mib,
}
(root / "aggregate.json").write_text(json.dumps(result, indent=2) + "\n")
with jsonl.open("a") as handle:
    handle.write(json.dumps(result) + "\n")
print(
    f"DP{dp} rep{rep}: qps={qps:.2f} mean={latency:.3f}s "
    f"p95={result['latency_p95_s']:.3f}s qps*lat={result['qps_x_latency']:.1f}"
)
PY
}

run_condition() {
  local dp=$1 rep=$2 server_cores=$3 client_cores_text=$4
  local label="dp${dp}_r${rep}"
  local condition_dir="$OUT/$label"
  local state_root="$STATE_BASE/$label"
  local launcher="$WT/examples/mps_dp/launch.sh"
  local state run_id idle_mib load_mib status=0
  local -a client_cores=() pids=()
  read -r -a client_cores <<< "$client_cores_text"
  [ "${#client_cores[@]}" -eq "$dp" ] \
    || die "$label needs $dp client core entries"

  mkdir -p "$condition_dir"
  CURRENT_STATE_ROOT="$state_root"
  CURRENT_RUN_ID=""
  echo "=== $label: full PR #1071, corrected speech workload ==="
  STATE_ROOT="$state_root" CONFIG="$CONFIG_PATH" MODEL_NAME=higgs \
    PYTHON_BIN="$PYTHON" GPU_ID="$GPU_ID" N="$dp" BASE_PORT="$BASE_PORT" \
    CORE_BLOCKS="$server_cores" MAX_TOTAL_TOKENS="$MAX_TOTAL_TOKENS" \
    HEALTH_TRIES=100 bash "$launcher" up 2>&1 | tee "$condition_dir/up.log"

  state=$(STATE_ROOT="$state_root" bash "$launcher" list)
  [ -n "$state" ] || die "$label launch returned no state"
  run_id=$(basename "$state")
  CURRENT_RUN_ID="$run_id"
  cp "$state/manifest" "$condition_dir/manifest.txt"
  cp "$state/mps_attach.txt" "$condition_dir/mps_attach.txt"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    -i "$GPU_ID" > "$condition_dir/vram_idle_mib.txt"
  read -r idle_mib < "$condition_dir/vram_idle_mib.txt"

  CLIENT_PIDS=()
  for ((i=0; i<dp; i++)); do
    setsid numactl -C "${client_cores[$i]}" \
      "$PYTHON" "$CLIENT" --port "$((BASE_PORT + i))" --model higgs \
        --conc "$CONC_PER_REPLICA" --secs "$SECS" \
        --warmup-secs "$WARMUP_SECS" --voice-clone \
        --ref-format references --meta "$SEEDTTS_META" --save-latencies \
        --out "$condition_dir/client_$i.json" \
        > "$condition_dir/client_$i.log" 2>&1 &
    pids+=("$!")
    CLIENT_PIDS+=("$!")
  done

  sleep "$((WARMUP_SECS + 15))"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    -i "$GPU_ID" > "$condition_dir/vram_load_mib.txt"
  read -r load_mib < "$condition_dir/vram_load_mib.txt"
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  CLIENT_PIDS=()
  [ "$status" -eq 0 ] || die "$label has a failed client"

  cp -R "$state/logs" "$condition_dir/server_logs"
  aggregate "$condition_dir" "$dp" "$rep" "$idle_mib" "$load_mib"
  STATE_ROOT="$state_root" bash "$launcher" down "$run_id" \
    | tee "$condition_dir/down.log"
  CURRENT_STATE_ROOT=""
  CURRENT_RUN_ID=""
  sleep 5
}

for ((rep=1; rep<=REPS; rep++)); do
  if ((rep % 2 == 1)); then
    run_condition 1 "$rep" "$DP1_SERVER_CORES" "$DP1_CLIENT_CORES"
    run_condition 3 "$rep" "$DP3_SERVER_CORES" "$DP3_CLIENT_CORES"
  else
    run_condition 3 "$rep" "$DP3_SERVER_CORES" "$DP3_CLIENT_CORES"
    run_condition 1 "$rep" "$DP1_SERVER_CORES" "$DP1_CLIENT_CORES"
  fi
done

"$PYTHON" - "$RESULTS_JSONL" "$OUT" <<'PY'
import json
import statistics
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line]
out = Path(sys.argv[2])
by_dp = {dp: [row for row in rows if row["dp"] == dp] for dp in (1, 3)}
if any(len(values) == 0 for values in by_dp.values()):
    raise SystemExit("missing DP1 or DP3 results")
means = {
    dp: {
        key: statistics.fmean(row[key] for row in values)
        for key in ("qps", "latency_mean_s", "latency_p95_s", "vram_load_mib")
    }
    for dp, values in by_dp.items()
}
scaling = means[3]["qps"] / means[1]["qps"]
summary = {"means": means, "dp3_over_dp1_qps": scaling, "rows": rows}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
text = (
    "# Corrected PR #1071 MPS upper-bound result\n\n"
    "| condition | aggregate QPS | mean latency | p95 | load VRAM |\n"
    "|---|---:|---:|---:|---:|\n"
    f"| DP1 x MPS | {means[1]['qps']:.2f} | {means[1]['latency_mean_s']:.3f}s | "
    f"{means[1]['latency_p95_s']:.3f}s | {means[1]['vram_load_mib']:.0f} MiB |\n"
    f"| DP3 x MPS | {means[3]['qps']:.2f} | {means[3]['latency_mean_s']:.3f}s | "
    f"{means[3]['latency_p95_s']:.3f}s | {means[3]['vram_load_mib']:.0f} MiB |\n\n"
    f"DP3/DP1 aggregate QPS scaling: **{scaling:.2f}x**\n"
)
(out / "summary.md").write_text(text)
print(text)
print(f"Artifacts: {out}")
PY

trap - EXIT INT TERM
git -C "$REPO" worktree remove --force "$WT"
rm -rf "$TMP_ROOT"
