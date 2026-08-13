# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for ARK-ASR-3B."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.arkasr"


class ArkasrPipelineConfig(PipelineConfig):
    """Single-stage batched ASR pipeline for ARK-ASR-3B checkpoints."""

    architecture: ClassVar[str] = "ArkasrForConditionalGeneration"

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        StageConfig(
            name="asr",
            process="asr",
            factory=f"{_PKG}.stages.create_sglang_arkasr_executor",
            factory_args={
                "device": "cuda:0",
                "max_running_requests": 32,
                "encoder_max_batch_size": 8,
                "max_new_tokens": 256,
                # encode_item blocks its request-build worker until the
                # embedding is attached, so the encoder only ever sees as many
                # concurrent items as there are build workers; two workers
                # would cap pre-LM encode groups at two (Qwen3-ASR #1341).
                "request_build_max_workers": 8,
                "request_build_max_pending": 32,
                "enable_pre_lm_encoder": True,
                "pre_lm_cache_max_entries": 4096,
                "pre_lm_cache_size_bytes": 2 * 1024**3,
                # one drained group maps to exactly one encoder microbatch at
                # the matching encoder_max_batch_size default.
                "pre_lm_max_batch_size": 8,
                "pre_lm_max_batch_wait_ms": 0,
            },
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = ArkasrPipelineConfig
