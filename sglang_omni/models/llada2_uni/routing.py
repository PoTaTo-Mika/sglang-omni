# SPDX-License-Identifier: Apache-2.0
"""Dynamic routing functions for LLaDA2-Uni pipeline stages."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.llada2_uni.config import (
    DECODE_STAGE,
    IMAGE_DECODE_STAGE,
    THINKER_STAGE,
)
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState


def thinker_next(request_id: str, output: Any) -> str | list[str]:
    """Route thinker output: re-enter for thinking Phase 2, or fan-out to terminals."""
    if hasattr(output, "data") and isinstance(output.data, dict):
        state = LLaDA2UniPipelineState.from_dict(output.data)
        ss = state.stream_state
        if ss.get("thinking_needs_reentry"):
            return THINKER_STAGE
    return [DECODE_STAGE, IMAGE_DECODE_STAGE]
