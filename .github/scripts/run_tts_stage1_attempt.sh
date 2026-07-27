#!/usr/bin/env bash

set -euo pipefail

: "${TTS_STAGE1_AUDIT_ROOT:?TTS_STAGE1_AUDIT_ROOT is required}"
: "${TTS_STAGE1_TOPOLOGY:?TTS_STAGE1_TOPOLOGY is required}"
: "${TTS_CI_MODEL:?TTS_CI_MODEL is required}"

if [[ "${TTS_STAGE1_TOPOLOGY}" == "mps_shared" ]]; then
  : "${TTS_STAGE1_MPS_STATE_ROOT:?TTS_STAGE1_MPS_STATE_ROOT is required}"
  gpu_id="${TTS_STAGE1_MPS_GPU_ID:-0}"
  shopt -s nullglob
  for state in "${TTS_STAGE1_MPS_STATE_ROOT}/gpu-${gpu_id}"/run-*; do
    STATE_ROOT="${TTS_STAGE1_MPS_STATE_ROOT}" \
      bash examples/mps_dp/launch.sh down "$(basename "${state}")"
  done
  shopt -u nullglob
fi

python .github/scripts/tts_stage1_runtime.py initialize \
  --output-dir "${TTS_STAGE1_AUDIT_ROOT}" \
  --topology "${TTS_STAGE1_TOPOLOGY}" \
  --model "${TTS_CI_MODEL}" \
  --configured-timeout-minutes "${TTS_STAGE1_CONFIGURED_TIMEOUT_MINUTES:-25}"

exec "$@"
