# SPDX-License-Identifier: Apache-2.0
"""Interleaved text/image generation helpers for LLaDA2-Uni."""

from __future__ import annotations

import logging
import queue
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_INTERLEAVED = "You are a interleaved generation assistant."
UNCONDITION_TOKEN = "<uncondition>"


@dataclass(frozen=True)
class InterleavedGenerationConfig:
    """Validated per-request controls for the interleaved state machine."""

    max_frames: int = 10
    text_max_new_tokens: int = 8192
    dllm_steps: int = 32
    cfg_scale: float = 0.0
    cfg_text_scale: float = 7.5
    cfg_image_scale: float = 1.5
    cfg_rescale: float = 0.7
    decode_mode: str = "decoder-turbo"
    decoder_steps: int = 8
    seed: int | None = None
    max_image_tokens: int = 4096

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "InterleavedGenerationConfig":
        raw = metadata.get("interleaved_generation", {})
        if not isinstance(raw, dict):
            raise TypeError("interleaved_generation metadata must be a mapping")
        config = cls(
            max_frames=int(raw.get("max_frames", cls.max_frames)),
            text_max_new_tokens=int(
                raw.get("text_max_new_tokens", cls.text_max_new_tokens)
            ),
            dllm_steps=int(raw.get("dllm_steps", cls.dllm_steps)),
            cfg_scale=float(raw.get("cfg_scale", cls.cfg_scale)),
            cfg_text_scale=float(raw.get("cfg_text_scale", cls.cfg_text_scale)),
            cfg_image_scale=float(raw.get("cfg_image_scale", cls.cfg_image_scale)),
            cfg_rescale=float(raw.get("cfg_rescale", cls.cfg_rescale)),
            decode_mode=str(raw.get("decode_mode", cls.decode_mode)),
            decoder_steps=int(raw.get("decoder_steps", cls.decoder_steps)),
            seed=(int(raw["seed"]) if raw.get("seed") is not None else None),
            max_image_tokens=int(raw.get("max_image_tokens", cls.max_image_tokens)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.max_frames <= 0:
            raise ValueError("interleaved max_frames must be positive")
        if self.text_max_new_tokens <= 0:
            raise ValueError("interleaved text_max_new_tokens must be positive")
        if self.dllm_steps <= 0:
            raise ValueError("interleaved dllm_steps must be positive")
        if not 0.0 <= self.cfg_rescale <= 1.0:
            raise ValueError("interleaved cfg_rescale must be in [0, 1]")
        if self.decoder_steps <= 0:
            raise ValueError("interleaved decoder_steps must be positive")
        if self.max_image_tokens <= 0:
            raise ValueError("interleaved max_image_tokens must be positive")
        if self.decode_mode not in {"normal", "decoder-turbo"}:
            raise ValueError(
                "interleaved decode_mode must be 'normal' or 'decoder-turbo'"
            )

    def to_stream_state(
        self, *, prompt_length: int, max_seq_len: int
    ) -> dict[str, Any]:
        return {
            "interleaved_phase": "text",
            "interleaved_frame_index": 0,
            "interleaved_max_frames": self.max_frames,
            "interleaved_text_max_new_tokens": self.text_max_new_tokens,
            "interleaved_segment_start": prompt_length,
            "interleaved_max_seq_len": max_seq_len,
            "interleaved_prompt_length": prompt_length,
            "interleaved_image_dllm_steps": self.dllm_steps,
            "interleaved_cfg_scale": self.cfg_scale,
            "interleaved_cfg_text_scale": self.cfg_text_scale,
            "interleaved_cfg_image_scale": self.cfg_image_scale,
            "interleaved_cfg_rescale": self.cfg_rescale,
            "interleaved_decode_mode": self.decode_mode,
            "interleaved_decoder_steps": self.decoder_steps,
            "interleaved_seed": self.seed,
            "interleaved_max_image_tokens": self.max_image_tokens,
            "interleaved_segments": [],
        }


@dataclass(frozen=True)
class ImageHeader:
    soi_position: int
    boi_position: int
    grid_h: int
    grid_w: int
    token_ids: list[int]

    @property
    def image_token_count(self) -> int:
        return self.grid_h * self.grid_w


@dataclass(frozen=True)
class CFGBranchPlan:
    """Condition branches consumed together by one dLLM request."""

    mode: Literal["none", "simple", "editing"]
    branches: dict[str, list[int]]
    cfg_scale: float
    cfg_text_scale: float
    cfg_image_scale: float
    cfg_rescale: float


@dataclass(frozen=True)
class _CollectedFrame:
    frame: dict[str, Any]
    image_decode_interval_ns: tuple[int, int] | None


def _token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or int(token_id) < 0:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Expected one token for {token!r}, got {ids}")
        token_id = ids[0]
    return int(token_id)


def parse_reserved_dimension(tokenizer: Any, token_id: int) -> int:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    match = re.fullmatch(r"<\|reserved_token_(\d+)\|>", token or "")
    if match is None:
        raise ValueError(f"Expected a reserved dimension token, got {token!r}")
    return int(match.group(1))


def parse_image_header(generated_ids: list[int], tokenizer: Any) -> ImageHeader:
    """Parse the final ``SOI/H/W/BOI`` generated by the text phase."""

    soi_id = _token_id(tokenizer, "<|image|>")
    boi_id = _token_id(tokenizer, "<boi>")
    if not generated_ids or generated_ids[-1] != boi_id:
        raise ValueError("interleaved text phase did not end with <boi>")
    boi_position = len(generated_ids) - 1
    soi_positions = [
        index
        for index, token_id in enumerate(generated_ids[:boi_position])
        if token_id == soi_id
    ]
    if not soi_positions:
        raise ValueError("interleaved image header has no <|image|> token")
    soi_position = soi_positions[-1]
    token_ids = generated_ids[soi_position : boi_position + 1]
    if len(token_ids) != 4:
        raise ValueError(
            "interleaved image header must be exactly SOI/H/W/BOI, "
            f"got {len(token_ids)} tokens"
        )
    return ImageHeader(
        soi_position=soi_position,
        boi_position=boi_position,
        grid_h=parse_reserved_dimension(tokenizer, token_ids[1]),
        grid_w=parse_reserved_dimension(tokenizer, token_ids[2]),
        token_ids=token_ids,
    )


def build_cfg_plan(
    *,
    full_ids: list[int],
    header: ImageHeader,
    frame_index: int,
    tokenizer: Any,
    config: InterleavedGenerationConfig,
) -> CFGBranchPlan:
    """Build disabled, simple, or editing CFG branches for an image frame."""

    if (
        config.cfg_scale <= 0.0
        and config.cfg_text_scale <= 0.0
        and config.cfg_image_scale <= 0.0
    ):
        return CFGBranchPlan(
            mode="none",
            branches={},
            cfg_scale=0.0,
            cfg_text_scale=0.0,
            cfg_image_scale=0.0,
            cfg_rescale=config.cfg_rescale,
        )

    uncond_base = tokenizer.encode(
        f"<role>SYSTEM</role> {SYSTEM_PROMPT_INTERLEAVED} "
        f"<role>HUMAN</role>{UNCONDITION_TOKEN}<role>ASSISTANT</role>",
        add_special_tokens=False,
    )
    image_suffix = full_ids[len(full_ids) - len(header.token_ids) :]
    eoi_id = _token_id(tokenizer, "<|/image|>")
    eoi_positions = [
        index
        for index, token_id in enumerate(full_ids[: -len(image_suffix)])
        if token_id == eoi_id
    ]

    if frame_index > 0 and eoi_positions:
        last_eoi = eoi_positions[-1]
        history_context = full_ids[: last_eoi + 1]
        current_text = full_ids[last_eoi + 1 : len(full_ids) - len(image_suffix)]
        uncondition_ids = tokenizer.encode(UNCONDITION_TOKEN, add_special_tokens=False)
        no_text_ids = history_context + uncondition_ids + image_suffix
        if config.cfg_image_scale > 0.0:
            return CFGBranchPlan(
                mode="editing",
                branches={
                    "no_text": no_text_ids,
                    "no_image": uncond_base + current_text + image_suffix,
                },
                cfg_scale=1.0,
                cfg_text_scale=config.cfg_text_scale,
                cfg_image_scale=config.cfg_image_scale,
                cfg_rescale=config.cfg_rescale,
            )
        if config.cfg_text_scale > 0.0:
            return CFGBranchPlan(
                mode="simple",
                branches={"uncond": no_text_ids},
                cfg_scale=config.cfg_text_scale,
                cfg_text_scale=0.0,
                cfg_image_scale=0.0,
                cfg_rescale=config.cfg_rescale,
            )

    effective_scale = (
        config.cfg_scale if config.cfg_scale > 0 else config.cfg_text_scale
    )
    return CFGBranchPlan(
        mode="simple",
        branches={"uncond": uncond_base + image_suffix},
        cfg_scale=effective_scale,
        cfg_text_scale=0.0,
        cfg_image_scale=0.0,
        cfg_rescale=config.cfg_rescale,
    )


class InterleavedCollectorScheduler:
    """Collect decoded frames and emit one final response after thinker EOS."""

    _ABORT_TOMBSTONE_LIMIT = 4096

    def __init__(self) -> None:
        self.inbox: queue.Queue[IncomingMessage] = queue.Queue()
        self.outbox: queue.Queue[OutgoingMessage] = queue.Queue()
        self.requires_tp_work_fanout = False
        self._running = False
        self._lock = threading.Lock()
        self._aborted: OrderedDict[str, None] = OrderedDict()
        self._frames: dict[str, dict[int, _CollectedFrame]] = {}
        self._final_payloads: dict[str, StagePayload] = {}

    def start(self) -> None:
        self._running = True
        while self._running:
            try:
                message = self.inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            if message.type != "new_request":
                continue
            try:
                with self._lock:
                    if message.request_id in self._aborted:
                        continue
                    self._receive(message.request_id, message.data)
            except Exception as exc:
                self.outbox.put(
                    OutgoingMessage(
                        request_id=message.request_id,
                        type="error",
                        data=exc,
                    )
                )

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        with self._lock:
            self._aborted[request_id] = None
            self._aborted.move_to_end(request_id)
            while len(self._aborted) > self._ABORT_TOMBSTONE_LIMIT:
                self._aborted.popitem(last=False)
            self._cleanup(request_id)

    def _cleanup(self, request_id: str) -> None:
        self._frames.pop(request_id, None)
        self._final_payloads.pop(request_id, None)

    def _receive(self, request_id: str, payload: StagePayload) -> None:
        data = payload.data if isinstance(payload.data, dict) else {}
        if data.get("kind") == "interleaved_frame":
            frame = data.get("frame")
            if not isinstance(frame, dict):
                raise TypeError("decoded interleaved frame must be a mapping")
            frame_index = int(frame.get("index", 0))
            if frame_index <= 0:
                raise ValueError("decoded interleaved frame index must be positive")
            frames = self._frames.setdefault(request_id, {})
            if frame_index in frames:
                raise ValueError(f"duplicate decoded frame {frame_index}")
            frames[frame_index] = _CollectedFrame(
                frame=frame,
                image_decode_interval_ns=self._image_decode_interval_ns(payload.timing),
            )
            logger.debug(
                "Interleaved collector received frame request=%s frame=%d frames=%s",
                request_id,
                frame_index,
                sorted(frames),
            )
        else:
            state = LLaDA2UniPipelineState.from_dict(data)
            if state.task_kind != "interleaved" or not state.stream_state.get(
                "interleaved_done"
            ):
                raise ValueError("collector received a non-final thinker payload")
            self._final_payloads[request_id] = payload
            logger.debug(
                "Interleaved collector received final state request=%s expected=%d",
                request_id,
                int(state.stream_state.get("interleaved_frame_index", 0)),
            )
        self._maybe_finish(request_id)

    def _maybe_finish(self, request_id: str) -> None:
        final_payload = self._final_payloads.get(request_id)
        if final_payload is None:
            return
        state = LLaDA2UniPipelineState.from_dict(final_payload.data)
        stream_state = state.stream_state
        expected = int(stream_state.get("interleaved_frame_index", 0))
        frames = self._frames.get(request_id, {})
        expected_indexes = set(range(1, expected + 1))
        if set(frames) != expected_indexes:
            logger.debug(
                "Interleaved collector waiting request=%s expected=%s received=%s",
                request_id,
                sorted(expected_indexes),
                sorted(frames),
            )
            return

        ordered_frames = [frames[index] for index in range(1, expected + 1)]
        segments = stream_state.get("interleaved_segments", [])
        trailing_text = str(stream_state.get("interleaved_trailing_text", ""))
        content: list[dict[str, Any]] = []
        for index, collected_frame in enumerate(ordered_frames, start=1):
            frame = collected_frame.frame
            segment = segments[index - 1] if index - 1 < len(segments) else {}
            text = str(segment.get("text", frame.get("text", "")))
            if text:
                content.append({"type": "text", "text": text})
            image = {
                "data": frame["image"],
                "format": frame.get("format", "png"),
                "frame_index": index,
                "grid_h": frame.get("grid_h"),
                "grid_w": frame.get("grid_w"),
            }
            content.append({"type": "image", "image": image})
        if trailing_text:
            content.append({"type": "text", "text": trailing_text})

        for collected_frame in ordered_frames:
            interval_ns = collected_frame.image_decode_interval_ns
            if interval_ns is not None:
                self._append_image_decode_timing(final_payload.timing, interval_ns)

        final_payload.data = {
            "modality": "interleaved",
            "content": content,
            "finish_reason": stream_state.get("interleaved_finish_reason", "stop"),
            "usage": stream_state.get("interleaved_usage", {}),
        }
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=final_payload,
            )
        )
        self._cleanup(request_id)

    @staticmethod
    def _image_decode_interval_ns(
        timing: dict[str, int],
    ) -> tuple[int, int] | None:
        enter_ns = timing.get("image_decode.enter")
        done_ns = timing.get("image_decode.done")
        if enter_ns is None or done_ns is None:
            return None
        return enter_ns, done_ns

    @staticmethod
    def _append_image_decode_timing(
        timing: dict[str, int], interval_ns: tuple[int, int]
    ) -> None:
        pass_index = 1
        while True:
            suffix = "" if pass_index == 1 else f".{pass_index}"
            enter_key = f"image_decode.enter{suffix}"
            done_key = f"image_decode.done{suffix}"
            if enter_key not in timing and done_key not in timing:
                timing[enter_key], timing[done_key] = interval_ns
                return
            pass_index += 1


def create_interleaved_collector_executor(model_path: str):
    del model_path
    return InterleavedCollectorScheduler()
