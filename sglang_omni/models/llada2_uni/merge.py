# SPDX-License-Identifier: Apache-2.0
"""Merge helpers for LLaDA2-Uni pipelines."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.models.llada2_uni.components.preprocessor import IMAGE_TOKEN_OFFSET
from sglang_omni.models.llada2_uni.payload_types import (
    LLaDA2UniEvent,
    LLaDA2UniPipelineState,
)

logger = logging.getLogger(__name__)


def extract_image_vq_tokens(
    state: LLaDA2UniPipelineState,
) -> tuple[list[int], int, int, dict[str, Any]] | None:
    """Extract VQ token IDs, grid dimensions, and per-request params from state.

    Returns ``(vq_token_ids, h, w, params)`` where:
        - ``vq_token_ids`` are stripped of ``image_token_offset``
        - ``h, w`` are in 16-px semantic-patch units. Sources (in priority
          order, falling through if ``h * w != len(vq_tokens)``):
              1. preprocessor's ``image_info`` (set by the t2i/edit path)
              2. square ``sqrt(N)`` fallback, truncated to fit
        - ``params`` is the request's ``image_generation`` dict (empty if absent)

    Returns ``None`` if this request does not contain a decodable VQ frame.
    """
    # Skip cheaply when not a t2i / edit request — chat / mmu pipelines also
    # fan-out to image_decode but never produce VQ tokens.
    if state.task_kind not in ("t2i", "edit", "interleaved"):
        return None

    thinker_out = state.thinker_out or state.engine_outputs.get("thinker")
    if not isinstance(thinker_out, dict):
        return None

    output_ids = thinker_out.get("output_ids", [])
    if not output_ids:
        return None

    vq_tokens = [
        int(tid) - IMAGE_TOKEN_OFFSET
        for tid in output_ids
        if isinstance(tid, (int, float)) and int(tid) >= IMAGE_TOKEN_OFFSET
    ]
    if not vq_tokens:
        return None

    n = len(vq_tokens)

    params: dict[str, Any] = {}
    metadata = state.request_metadata or {}
    if isinstance(metadata, dict):
        metadata_key = (
            "interleaved_generation"
            if state.task_kind == "interleaved"
            else "image_generation"
        )
        ig = metadata.get(metadata_key)
        if isinstance(ig, dict):
            params = ig

    def _viable(h_: int | None, w_: int | None) -> bool:
        return (
            isinstance(h_, int)
            and isinstance(w_, int)
            and h_ > 0
            and w_ > 0
            and h_ * w_ == n
        )

    h: int | None = None
    w: int | None = None

    # Priority 1: preprocessor-provided image_info
    image_info = (
        state.stream_state.get("image_info", [])
        if isinstance(state.stream_state, dict)
        else []
    )
    if image_info:
        first_info = image_info[0]
        cand_h = first_info.get("grid_h")
        cand_w = first_info.get("grid_w")
        if _viable(cand_h, cand_w):
            h, w = int(cand_h), int(cand_w)

    # Priority 2: square sqrt(N) fallback
    if h is None:
        side = int(n**0.5)
        if side * side != n:
            logger.warning(
                "thinker produced %d VQ tokens (not a perfect square); "
                "truncating to %d (%dx%d) for decoder",
                n,
                side * side,
                side,
                side,
            )
            vq_tokens = vq_tokens[: side * side]
        h, w = side, side

    return vq_tokens, h, w, params


def decode_events(
    *,
    thinker_out: dict[str, Any],
    tokenizer: Any,
) -> list[LLaDA2UniEvent]:
    """Convert thinker output tokens to a text_final event."""
    # TODO: add streaming support
    output_ids = thinker_out.get("output_ids", [])
    if not output_ids:
        return []

    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return [
        LLaDA2UniEvent(
            type="text_final",
            modality="text",
            payload={"text": text},
            is_final=True,
        )
    ]
