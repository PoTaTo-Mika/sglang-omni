# SPDX-License-Identifier: Apache-2.0
"""Wire state shared by VoxCPM generations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class VoxCPMState(DeclarativeStateBase):
    variant: str = wire("voxcpm", codec="str_or")
    target_text: str = wire("", emit="truthy", codec="str")
    reference_text: str | None = None
    reference_audio: Any | None = None
    reference_mode: str = wire("none", codec="str_or")
    generation_kwargs: dict[str, Any] = wire(default_factory=dict, codec="dict")
    audio_waveform: Any | None = wire(None, codec="typed_tensor")
    # Filled from the loaded AudioVAE config by the terminal adapter.
    sample_rate: int = wire(0, codec="int_or")


__all__ = ["VoxCPMState"]
