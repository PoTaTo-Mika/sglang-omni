# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Fun-ASR (Fun-ASR-Nano-2512).

Fun-ASR-Nano is an audio-LLM: a frozen SenseVoice encoder + low-frame-rate
downsampling adaptor projects audio embeddings into a Qwen3-0.6B decoder that
autoregressively emits the transcript (see Fun-ASR/model.py). Like Qwen3-ASR it
is a single AR stage with no thinker/talker/codec split, so it maps onto one
terminal SGLang stage. The executor factory is implemented in ``stages.py``.
"""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.fun_asr"


class FunASRPipelineConfig(PipelineConfig):
    """Single-stage batched ASR pipeline for Fun-ASR-Nano checkpoints."""

    # Matches ``architectures[0]`` in the HF-adapted config.json shipped with
    # Fun-ASR-Nano-2512 (checkpoints/Fun-ASR-Nano-2512/config.json), which is
    # what sglang_omni.utils.hf.architecture_from_hf_config resolves to. The
    # upstream funasr registry name is "FunASRNano" (Fun-ASR/model.py
    # @tables.register) — kept as an alias for anyone resolving via that path.
    architecture: ClassVar[str] = "FunAsrNanoForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "FunASRNano",
        "FunASRForConditionalGeneration",
    )

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        StageConfig(
            name="asr",
            process="asr",
            factory=f"{_PKG}.stages.create_sglang_fun_asr_executor",
            factory_args={"device": "cuda:0", "max_running_requests": 32},
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = FunASRPipelineConfig
