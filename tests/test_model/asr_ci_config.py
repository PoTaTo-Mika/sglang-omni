# SPDX-License-Identifier: Apache-2.0
"""Model presets and thresholds for the SeedTTS ASR CI (ASR CI stage 2).

Mirrors tests/test_model/tts_ci_config.py: each preset bundles the model
path and calibrated gate thresholds for one ASR model, and CI selects one
preset per run through the ASR_CI_MODEL environment variable (or the
--asr-ci-model pytest option). tests/test_model/test_asr_ci_seedtts.py is
model-agnostic and reads everything model-specific from the selected preset.

Threshold constants live here (not in the test file) so that the
tune-ci-thresholds skill can read and rewrite them; the skill locates this
file through the `threshold_file` key in
.claude/skills/tune-ci-thresholds/models/asr/config.yaml and claims each
preset's constants by name prefix (e.g. ^FUN_ASR_).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from benchmarks.tasks.asr import FUN_ASR_MODEL_PATH, QWEN3_ASR_MODEL_PATH
from tests.utils import apply_wer_slack


@dataclass(frozen=True)
class AsrCiThresholdPreset:
    """CI gate values with slack already applied."""

    en_corpus_wer_max: float
    en_sample_wer_max: float
    zh_corpus_wer_max: float
    zh_sample_wer_max: float
    throughput_min: float
    latency_mean_max_s: float
    latency_p95_max_s: float
    rtf_mean_max: float
    rtf_p95_max: float


@dataclass(frozen=True)
class AsrCiPreset:
    model_path: str
    display_name: str
    thresholds: AsrCiThresholdPreset
    # note (Jeffro): False means threshold assertions are skipped while Qwen3-ASR awaits threshold calibration
    gate_thresholds: bool = True


# Slack factors applied to worst-of-N reference values to derive CI gates.
# Higher-is-better metrics: threshold = reference * THRESHOLD_SLACK_HIGHER.
# Lower-is-better metrics: threshold = reference * THRESHOLD_SLACK_LOWER.
THRESHOLD_SLACK_HIGHER = 0.9
THRESHOLD_SLACK_LOWER = 1.1


# note (db-ol): references are the strict worst-of-5 from the official
# tune-ci-thresholds calibration run 20260719T2302Z_asr_fun_g25 on idle
# GPUs of the CI host (2x H100 80GB, DP=2 managed router, concurrency 32).
# EN WER was identical across all five repeats and speed CV stayed under
# 7 percent with no outliers. Two caveats the numbers alone do not show:
# heavily contended CI runs have measured 16 to 20 percent below the idle
# speed references, beyond the 10 percent slack, so a loaded runner leans
# on the flaky-pytest retry wrapper. The ZH per-sample bound keeps the more
# conservative worst of the two strict calibrations because that single
# utterance metric drifted between runs (0.75 in this run, 0.8333 in the
# previous strict worst-of-5 on the same topology).
FUN_ASR_EN_CORPUS_WER_MAX = 0.0172
FUN_ASR_EN_SAMPLE_WER_MAX = 0.2858
FUN_ASR_ZH_CORPUS_WER_MAX = 0.0136
FUN_ASR_ZH_SAMPLE_WER_MAX = 0.8334
FUN_ASR_THROUGHPUT_MIN = 64.27181519213258
FUN_ASR_LATENCY_MEAN_MAX_S = 0.49501860166257006
FUN_ASR_LATENCY_P95_MAX_S = 0.715
FUN_ASR_RTF_MEAN_MAX = 0.1076
FUN_ASR_RTF_P95_MAX = 0.1597

FUN_ASR_EN_CORPUS_WER_THRESHOLD = apply_wer_slack(
    FUN_ASR_EN_CORPUS_WER_MAX, THRESHOLD_SLACK_LOWER
)
FUN_ASR_EN_SAMPLE_WER_THRESHOLD = apply_wer_slack(
    FUN_ASR_EN_SAMPLE_WER_MAX, THRESHOLD_SLACK_LOWER
)
FUN_ASR_ZH_CORPUS_WER_THRESHOLD = apply_wer_slack(
    FUN_ASR_ZH_CORPUS_WER_MAX, THRESHOLD_SLACK_LOWER
)
FUN_ASR_ZH_SAMPLE_WER_THRESHOLD = apply_wer_slack(
    FUN_ASR_ZH_SAMPLE_WER_MAX, THRESHOLD_SLACK_LOWER
)
FUN_ASR_THROUGHPUT_THRESHOLD = round(FUN_ASR_THROUGHPUT_MIN * THRESHOLD_SLACK_HIGHER, 3)
FUN_ASR_LATENCY_MEAN_THRESHOLD_S = round(
    FUN_ASR_LATENCY_MEAN_MAX_S * THRESHOLD_SLACK_LOWER, 3
)
FUN_ASR_LATENCY_P95_THRESHOLD_S = round(
    FUN_ASR_LATENCY_P95_MAX_S * THRESHOLD_SLACK_LOWER, 3
)
FUN_ASR_RTF_MEAN_THRESHOLD = round(FUN_ASR_RTF_MEAN_MAX * THRESHOLD_SLACK_LOWER, 4)
FUN_ASR_RTF_P95_THRESHOLD = round(FUN_ASR_RTF_P95_MAX * THRESHOLD_SLACK_LOWER, 4)


# note (Jeffro): Qwen3-ASR runs report-only (gate_thresholds=False) until a
# fresh tune-ci-thresholds worst-of-5 calibration on the current 2x H100
# DP=2 topology lands (#1214). The EN references below are the last
# calibrated values from the pre-#1093 stage-2 gate and are kept for log
# context only; ZH has never been calibrated, so its bounds are inert
# placeholders. None of these values gate CI while the flag is False.
# ref: https://github.com/sgl-project/sglang-omni/pull/1093/changes#diff-806bffc1dc635f8f13348c3cc6ce6f1d5626e3cf32484339ebcc3058da71d189
QWEN3_ASR_EN_CORPUS_WER_MAX = 0.0122
QWEN3_ASR_EN_SAMPLE_WER_MAX = 0.1819
QWEN3_ASR_ZH_CORPUS_WER_MAX = 1.0  # placeholder
QWEN3_ASR_ZH_SAMPLE_WER_MAX = 1.0  # placeholder
QWEN3_ASR_THROUGHPUT_MIN = 89.15367011738545
QWEN3_ASR_LATENCY_MEAN_MAX_S = 0.357
QWEN3_ASR_LATENCY_P95_MAX_S = 0.493352619552752
QWEN3_ASR_RTF_MEAN_MAX = 0.07720349691994066
QWEN3_ASR_RTF_P95_MAX = 0.1084

QWEN3_ASR_EN_CORPUS_WER_THRESHOLD = apply_wer_slack(
    QWEN3_ASR_EN_CORPUS_WER_MAX, THRESHOLD_SLACK_LOWER
)
QWEN3_ASR_EN_SAMPLE_WER_THRESHOLD = apply_wer_slack(
    QWEN3_ASR_EN_SAMPLE_WER_MAX, THRESHOLD_SLACK_LOWER
)
QWEN3_ASR_ZH_CORPUS_WER_THRESHOLD = apply_wer_slack(
    QWEN3_ASR_ZH_CORPUS_WER_MAX, THRESHOLD_SLACK_LOWER
)
QWEN3_ASR_ZH_SAMPLE_WER_THRESHOLD = apply_wer_slack(
    QWEN3_ASR_ZH_SAMPLE_WER_MAX, THRESHOLD_SLACK_LOWER
)
QWEN3_ASR_THROUGHPUT_THRESHOLD = round(
    QWEN3_ASR_THROUGHPUT_MIN * THRESHOLD_SLACK_HIGHER, 3
)
QWEN3_ASR_LATENCY_MEAN_THRESHOLD_S = round(
    QWEN3_ASR_LATENCY_MEAN_MAX_S * THRESHOLD_SLACK_LOWER, 3
)
QWEN3_ASR_LATENCY_P95_THRESHOLD_S = round(
    QWEN3_ASR_LATENCY_P95_MAX_S * THRESHOLD_SLACK_LOWER, 3
)
QWEN3_ASR_RTF_MEAN_THRESHOLD = round(QWEN3_ASR_RTF_MEAN_MAX * THRESHOLD_SLACK_LOWER, 4)
QWEN3_ASR_RTF_P95_THRESHOLD = round(QWEN3_ASR_RTF_P95_MAX * THRESHOLD_SLACK_LOWER, 4)


ASR_CI_PRESETS: dict[str, AsrCiPreset] = {
    "fun": AsrCiPreset(
        model_path=FUN_ASR_MODEL_PATH,
        display_name="Fun-ASR",
        thresholds=AsrCiThresholdPreset(
            en_corpus_wer_max=FUN_ASR_EN_CORPUS_WER_THRESHOLD,
            en_sample_wer_max=FUN_ASR_EN_SAMPLE_WER_THRESHOLD,
            zh_corpus_wer_max=FUN_ASR_ZH_CORPUS_WER_THRESHOLD,
            zh_sample_wer_max=FUN_ASR_ZH_SAMPLE_WER_THRESHOLD,
            throughput_min=FUN_ASR_THROUGHPUT_THRESHOLD,
            latency_mean_max_s=FUN_ASR_LATENCY_MEAN_THRESHOLD_S,
            latency_p95_max_s=FUN_ASR_LATENCY_P95_THRESHOLD_S,
            rtf_mean_max=FUN_ASR_RTF_MEAN_THRESHOLD,
            rtf_p95_max=FUN_ASR_RTF_P95_THRESHOLD,
        ),
    ),
    "qwen3": AsrCiPreset(
        model_path=QWEN3_ASR_MODEL_PATH,
        display_name="Qwen3-ASR",
        thresholds=AsrCiThresholdPreset(
            en_corpus_wer_max=QWEN3_ASR_EN_CORPUS_WER_THRESHOLD,
            en_sample_wer_max=QWEN3_ASR_EN_SAMPLE_WER_THRESHOLD,
            zh_corpus_wer_max=QWEN3_ASR_ZH_CORPUS_WER_THRESHOLD,
            zh_sample_wer_max=QWEN3_ASR_ZH_SAMPLE_WER_THRESHOLD,
            throughput_min=QWEN3_ASR_THROUGHPUT_THRESHOLD,
            latency_mean_max_s=QWEN3_ASR_LATENCY_MEAN_THRESHOLD_S,
            latency_p95_max_s=QWEN3_ASR_LATENCY_P95_THRESHOLD_S,
            rtf_mean_max=QWEN3_ASR_RTF_MEAN_THRESHOLD,
            rtf_p95_max=QWEN3_ASR_RTF_P95_THRESHOLD,
        ),
        gate_thresholds=False,
    ),
}


def select_asr_ci_preset(model_name: str | None = None) -> tuple[str, AsrCiPreset]:
    selected = model_name or os.environ.get("ASR_CI_MODEL", "fun")
    preset = ASR_CI_PRESETS.get(selected)
    if preset is None:
        allowed = ", ".join(sorted(ASR_CI_PRESETS))
        raise ValueError(
            f"Unsupported ASR_CI_MODEL={selected!r}; expected one of: {allowed}"
        )
    return selected, preset
