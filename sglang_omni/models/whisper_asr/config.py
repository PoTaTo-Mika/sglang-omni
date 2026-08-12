# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Whisper ASR."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.whisper_asr"


class WhisperASRPipelineConfig(PipelineConfig):
    """Single-stage batched ASR pipeline for Whisper checkpoints."""

    architecture: ClassVar[str] = "WhisperForConditionalGeneration"

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"asr": "asr"}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "asr"}

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        StageConfig(
            name="asr",
            process="asr",
            factory=f"{_PKG}.stages.create_sglang_whisper_asr_executor",
            factory_args={
                "device": "cuda:0",
                "enable_encoder_cuda_graph": True,
                "enable_pre_lm_encoder": True,
                "pre_lm_cache_max_entries": 4096,
                "pre_lm_cache_size_bytes": 2 * 1024**3,
                "pre_lm_max_batch_size": 8,
                "pre_lm_max_batch_wait_ms": 0,
            },
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = WhisperASRPipelineConfig
