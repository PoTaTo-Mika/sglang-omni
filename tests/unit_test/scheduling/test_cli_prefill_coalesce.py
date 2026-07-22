# SPDX-License-Identifier: Apache-2.0
"""CLI override tests for prefill admission coalescing.

``--prefill-coalesce-requests/--prefill-coalesce-wait-ms`` apply to the
async-decode factory set (a subset of the OmniScheduler AR stages); omitting
both is a no-op and unsupported pipelines are rejected.
"""

from __future__ import annotations

import pytest
import typer

from sglang_omni.cli.serve import apply_prefill_coalesce_cli_overrides
from sglang_omni.config import PipelineConfig, resolve_stage_factory_args
from sglang_omni.models.fun_asr.config import FunASRPipelineConfig
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.moss_transcribe_diarize.config import (
    MossTranscribeDiarizePipelineConfig,
)
from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig
from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig


def _ar_stage_args(config: PipelineConfig, stage_name: str) -> dict[str, object]:
    stage = next(s for s in config.stages if s.name == stage_name)
    return resolve_stage_factory_args(stage, config)


@pytest.mark.parametrize(
    ("config_cls", "stage_name"),
    [
        (HiggsTtsPipelineConfig, "tts_engine"),
        (MossTTSLocalPipelineConfig, "tts_engine"),
        (MossTranscribeDiarizePipelineConfig, "asr"),
        (FunASRPipelineConfig, "asr"),
        (Qwen3OmniPipelineConfig, "thinker"),
    ],
)
def test_cli_sets_coalesce_args(config_cls, stage_name):
    config = config_cls(model_path="dummy")
    apply_prefill_coalesce_cli_overrides(
        config, prefill_coalesce_requests=32, prefill_coalesce_wait_ms=300.0
    )
    args = _ar_stage_args(config, stage_name)
    assert args["prefill_coalesce_requests"] == 32
    assert args["prefill_coalesce_wait_ms"] == 300.0


def test_omitting_both_flags_is_a_noop():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    before = _ar_stage_args(config, "tts_engine")
    apply_prefill_coalesce_cli_overrides(
        config, prefill_coalesce_requests=None, prefill_coalesce_wait_ms=None
    )
    assert _ar_stage_args(config, "tts_engine") == before


def test_rejects_unsupported_pipeline():
    config = Qwen3TTSPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter):
        apply_prefill_coalesce_cli_overrides(
            config, prefill_coalesce_requests=32, prefill_coalesce_wait_ms=None
        )


def test_rejects_invalid_values():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter):
        apply_prefill_coalesce_cli_overrides(
            config, prefill_coalesce_requests=-1, prefill_coalesce_wait_ms=None
        )
    for bad_wait in (0.0, float("nan"), float("inf")):
        with pytest.raises(typer.BadParameter):
            apply_prefill_coalesce_cli_overrides(
                config,
                prefill_coalesce_requests=None,
                prefill_coalesce_wait_ms=bad_wait,
            )
