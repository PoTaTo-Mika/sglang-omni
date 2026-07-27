# SPDX-License-Identifier: Apache-2.0
"""Stage factories shared by VoxCPM1.5 and VoxCPM2."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.voxcpm_common.request_builders import build_voxcpm_state
from sglang_omni.scheduling.pipeline_state import store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def create_voxcpm_preprocessing_executor(
    model_path: str, *, variant: str
) -> SimpleScheduler:
    del model_path
    if variant not in {"voxcpm", "voxcpm2"}:
        raise ValueError(f"unsupported VoxCPM variant: {variant}")
    return SimpleScheduler(
        lambda payload: store_state(
            payload, build_voxcpm_state(payload, variant=variant)
        ),
        max_concurrency=8,
    )


def create_voxcpm_tts_engine_executor(
    model_path: str,
    *,
    variant: str,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
    total_gpu_memory_fraction: float | None = None,
) -> Any:
    from sglang_omni.models.voxcpm_common.engine_builder import VoxCPMEngineBuilder

    return VoxCPMEngineBuilder(
        variant=variant,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


__all__ = [
    "create_voxcpm_preprocessing_executor",
    "create_voxcpm_tts_engine_executor",
]
