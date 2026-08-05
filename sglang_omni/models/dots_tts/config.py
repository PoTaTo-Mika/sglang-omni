# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for dots.tts."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.dots_tts"

PREPROCESSING_STAGE = "preprocessing"
REFERENCE_ENCODE_STAGE = "reference_encode"
TTS_ENGINE_STAGE = "tts_engine"
AUDIO_DECODE_STAGE = "audio_decode"


def _stages() -> list[StageConfig]:
    return [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next=REFERENCE_ENCODE_STAGE,
        ),
        StageConfig(
            name=REFERENCE_ENCODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_reference_encode_executor",
            gpu=0,
            next=TTS_ENGINE_STAGE,
        ),
        StageConfig(
            name=TTS_ENGINE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_tts_engine_executor",
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            next=AUDIO_DECODE_STAGE,
        ),
        StageConfig(
            name=AUDIO_DECODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_audio_decode_executor",
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            terminal=True,
        ),
    ]


class DotsTtsPipelineConfig(PipelineConfig):
    """dots.tts pipeline.

    preprocessing -> reference_encode -> tts_engine -> audio_decode.
    Continuation cloning makes reference audio plus its transcript mandatory,
    so the reference stage is always on the critical path.
    """

    architecture: ClassVar[str] = "DotsTTSForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True
    required_speech_reference_count: ClassVar[int | None] = 1
    speech_reference_text_required: ClassVar[bool] = True

    @classmethod
    def process_safe_edges(cls) -> frozenset[tuple[str, str]]:
        # note (chenyang): every stage rebuilds its inputs from the payload
        # state, and audio_decode owns its own vocoder handle, so all three
        # handoffs stay correct once they cross a process boundary. No memory
        # fractions are recommended yet; an operator that splits must declare
        # them explicitly.
        return frozenset(
            {
                (PREPROCESSING_STAGE, REFERENCE_ENCODE_STAGE),
                (REFERENCE_ENCODE_STAGE, TTS_ENGINE_STAGE),
                (TTS_ENGINE_STAGE, AUDIO_DECODE_STAGE),
            }
        )

    model_path: str
    entry_stage: str = PREPROCESSING_STAGE
    stages: list[StageConfig] = Field(default_factory=_stages)

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = DotsTtsPipelineConfig
