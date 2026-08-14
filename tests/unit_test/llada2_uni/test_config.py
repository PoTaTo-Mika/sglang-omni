# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import pytest

from sglang_omni.config import resolve_stage_factory_args
from sglang_omni.models.llada2_uni.bootstrap import create_dllm_thinker_scheduler
from sglang_omni.models.llada2_uni.config import (
    THINKER_STAGE,
    LLaDA2UniOmniPipelineConfig,
    LLaDA2UniPipelineConfig,
)
from sglang_omni.models.llada2_uni.stages import (
    create_sglang_dllm_thinker_executor_from_config,
)


def _stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def test_omni_thinker_injects_configured_total_gpu_memory_fraction() -> None:
    config = LLaDA2UniOmniPipelineConfig(model_path="unused-model")
    thinker = _stage(config, THINKER_STAGE)
    thinker.runtime.resources.total_gpu_memory_fraction = 0.55

    args = resolve_stage_factory_args(thinker, config)

    assert args["total_gpu_memory_fraction"] == pytest.approx(0.55)


def test_text_thinker_keeps_total_gpu_memory_fraction_unset() -> None:
    config = LLaDA2UniPipelineConfig(model_path="unused-model")
    thinker = _stage(config, THINKER_STAGE)

    args = resolve_stage_factory_args(thinker, config)
    factory_parameter = inspect.signature(
        create_sglang_dllm_thinker_executor_from_config
    ).parameters["total_gpu_memory_fraction"]
    bootstrap_parameter = inspect.signature(create_dllm_thinker_scheduler).parameters[
        "total_gpu_memory_fraction"
    ]

    assert "total_gpu_memory_fraction" not in args
    assert factory_parameter.default is None
    assert bootstrap_parameter.default is None
