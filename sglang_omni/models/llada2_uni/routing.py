# SPDX-License-Identifier: Apache-2.0
"""Dynamic routing functions for LLaDA2-Uni pipeline stages."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.models.llada2_uni.config import (
    DECODE_STAGE,
    IMAGE_DECODE_STAGE,
    INTERLEAVED_COLLECT_STAGE,
    THINKER_STAGE,
)
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState

logger = logging.getLogger(__name__)


def thinker_next(request_id: str, output: Any) -> str | list[str]:
    """Route thinker output: re-enter for thinking Phase 2, or fan-out to terminals."""
    if hasattr(output, "data") and isinstance(output.data, dict):
        state = LLaDA2UniPipelineState.from_dict(output.data)
        ss = state.stream_state
        if state.task_kind == "interleaved":
            emit_frame = bool(ss.get("interleaved_emit_frame"))
            done = bool(ss.get("interleaved_done"))
            frame_index = int(ss.get("interleaved_frame_index", 0))
            if emit_frame and done:
                targets = [IMAGE_DECODE_STAGE, INTERLEAVED_COLLECT_STAGE]
                logger.debug(
                    "Interleaved thinker route request=%s frame=%d targets=%s",
                    request_id,
                    frame_index,
                    targets,
                )
                return targets
            if emit_frame:
                targets = [IMAGE_DECODE_STAGE, THINKER_STAGE]
                logger.debug(
                    "Interleaved thinker route request=%s frame=%d targets=%s",
                    request_id,
                    frame_index,
                    targets,
                )
                return targets
            if ss.get("interleaved_needs_reentry"):
                return THINKER_STAGE
            if done:
                logger.debug(
                    "Interleaved thinker route request=%s final_text frame=%d",
                    request_id,
                    frame_index,
                )
                return INTERLEAVED_COLLECT_STAGE
            raise ValueError("interleaved thinker result has no valid route")
        if ss.get("thinking_needs_reentry"):
            return THINKER_STAGE
    return [DECODE_STAGE, IMAGE_DECODE_STAGE]
