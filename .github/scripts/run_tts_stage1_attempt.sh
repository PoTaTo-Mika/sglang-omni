#!/usr/bin/env bash

set -euo pipefail

: "${TTS_STAGE1_AUDIT_ROOT:?TTS_STAGE1_AUDIT_ROOT is required}"
: "${TTS_STAGE1_TOPOLOGY:?TTS_STAGE1_TOPOLOGY is required}"
: "${TTS_CI_MODEL:?TTS_CI_MODEL is required}"

python .github/scripts/tts_stage1_runtime.py initialize \
  --output-dir "${TTS_STAGE1_AUDIT_ROOT}" \
  --topology "${TTS_STAGE1_TOPOLOGY}" \
  --model "${TTS_CI_MODEL}" \
  --configured-timeout-minutes "${TTS_STAGE1_CONFIGURED_TIMEOUT_MINUTES:-25}"

exec "$@"
