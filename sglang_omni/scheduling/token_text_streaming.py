# SPDX-License-Identifier: Apache-2.0
"""Shared token→text streaming for ASR OmniScheduler builders."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Callable

from sglang_omni.scheduling.messages import OutgoingMessage

DecodeFn = Callable[[list[int]], str]
BuildMessageDataFn = Callable[[str], Any]
BuildMessageMetadataFn = Callable[[int | None], dict[str, Any] | None]


def _stream_state(req: Any) -> SimpleNamespace:
    state = getattr(req, "_asr_token_text_stream", None)
    if state is None:
        state = SimpleNamespace(pending_ids=[], last_emit_t=0.0)
        req._asr_token_text_stream = state
    return state


def make_token_text_stream_output_builder(
    *,
    decode_fn: DecodeFn,
    build_message_data: BuildMessageDataFn,
    build_message_metadata: BuildMessageMetadataFn,
    eos_token_id: int | None,
    min_emit_interval_s: float = 0.0,
    allow_terminal_flush: bool = False,
) -> Callable[[str, Any, Any], list[OutgoingMessage]]:
    def _build_stream_output(
        request_id: str, req_data: Any, req_output: Any
    ) -> list[OutgoingMessage]:
        req = req_data.req
        if req is None:
            return []

        # note (guozhihao): suppress while chunked prefill is still on prompt tokens.
        if req.inflight_middle_chunks > 0:
            return []

        stage_payload = req_data.stage_payload
        if stage_payload is None:
            return []
        if not (stage_payload.request.params or {}).get("stream", False):
            return []

        # note (guozhihao): Fun sets allow_terminal_flush so empty data can flush
        # when finished(); MOSS leaves it False and empty data stays silent.
        token_data = req_output.data
        is_terminal = bool(req.finished()) if allow_terminal_flush else False
        token_id: int | None = None
        if token_data is not None:
            try:
                token_id = int(token_data)
            except (TypeError, ValueError):
                return []
        elif not is_terminal:
            return []

        state = _stream_state(req)
        pending = state.pending_ids

        is_eos = (
            token_id is not None
            and eos_token_id is not None
            and token_id == eos_token_id
        )
        if token_id is not None and not is_eos:
            pending.append(token_id)
        if not pending:
            return []

        now = time.perf_counter()
        last_emit = state.last_emit_t

        # note (guozhihao): last_emit==0 → first delta immediate; EOS/terminal always flush.
        if (
            not is_eos
            and not is_terminal
            and min_emit_interval_s > 0.0
            and last_emit > 0.0
            and (now - last_emit) < min_emit_interval_s
        ):
            return []

        delta = decode_fn(pending)
        if delta.endswith("\ufffd") and not is_terminal:
            return []
        pending.clear()
        if not delta:
            return []

        state.last_emit_t = now

        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=None,
                data=build_message_data(delta),
                metadata=build_message_metadata(token_id),
            )
        ]

    if allow_terminal_flush:

        def _flush_stream_output(
            request_id: str, req_data: Any
        ) -> list[OutgoingMessage]:
            return _build_stream_output(
                request_id,
                req_data,
                SimpleNamespace(data=None),
            )

        _build_stream_output.flush = _flush_stream_output  # type: ignore[attr-defined]

    return _build_stream_output


__all__ = ["make_token_text_stream_output_builder"]
