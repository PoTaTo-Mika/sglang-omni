# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for OpenBMB VoxCPM2."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.voxcpm_common"


def _stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_voxcpm_preprocessing_executor",
            factory_args={"variant": "voxcpm2"},
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_voxcpm_tts_engine_executor",
            factory_args={"variant": "voxcpm2"},
            gpu=0,
            terminal=True,
        ),
    ]


class VoxCPM2PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "voxcpm2"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "VoxCPM2",
        "VoxCPM2ForConditionalGeneration",
        "VoxCPM2TalkerForConditionalGeneration",
    )
    requires_model_capabilities: ClassVar[bool] = True

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "tts_engine"}

    model_path: str
    stages: list[StageConfig] = Field(default_factory=_stages)

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = VoxCPM2PipelineConfig
