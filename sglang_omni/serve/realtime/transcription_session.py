from __future__ import annotations

import asyncio
import base64
import binascii
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketState

from sglang_omni.client import Client
from sglang_omni.config import AudioChunkingConfig, RealtimeTranscriptionConfig
from sglang_omni.serve.realtime.audio_buffer import BufferOverflow, RealtimeAudioBuffer
from sglang_omni.serve.realtime.events import (
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    SessionUpdate,
    TranscriptionDone,
    TurnDetection,
    TurnDetectionType,
    make_event,
    parse_client_event,
)
from sglang_omni.serve.realtime.vad import (
    VAD_SAMPLE_RATE,
    StreamingVAD,
    VADConfig,
    VADEvent,
    offsets_to_ms,
)
from sglang_omni.serve.transcription_chunking import join_transcript_parts

_SILENT_PCM16_PEAK = 33


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(slots=True)
class TranscriptionSessionSettings:
    language: str | None = None
    decode_interval_ms: int = 2000
    turn_detection: TurnDetection | None = field(
        default_factory=lambda: TurnDetection(
            type=TurnDetectionType.SERVER_VAD,
            threshold=0.5,
            prefix_padding_ms=300,
            silence_duration_ms=500,
        )
    )


@dataclass(slots=True)
class ActiveTranscriptionSegment:
    segment_id: int
    start_sample: int
    strategy_state: object
    next_refresh_sample: int
    decode_attempt: int = 0
    last_text: str = ""
    dirty: bool = False
    finalizing: bool = False
    request_id: str | None = None
    decode_task: asyncio.Task[None] | None = None


@dataclass(frozen=True, slots=True)
class CommittedTranscriptionSegment:
    segment_id: int
    text: str


class RealtimeTranscriptionSession:
    handlers = {
        SessionUpdate: "handle_session_update",
        InputAudioBufferAppend: "handle_audio_append",
        InputAudioBufferCommit: "handle_audio_commit",
        TranscriptionDone: "handle_transcription_done",
    }

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        model_name: str,
        capability: RealtimeTranscriptionConfig,
        audio_chunking: AudioChunkingConfig,
        strategy: Any,
        session_id: str | None = None,
    ) -> None:
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
        self.session_id = session_id or _new_id("sess")
        self.closed = False
        self.event_index = 0
        self._send_lock = asyncio.Lock()
        self.capability = capability
        self.audio_chunking = audio_chunking
        self.strategy = strategy
        self.settings = TranscriptionSessionSettings(
            decode_interval_ms=capability.decode_interval_ms
        )
        max_buffer_seconds = max(
            audio_chunking.max_audio_clip_s * 4,
            audio_chunking.max_audio_clip_s + 4,
        )
        max_buffer_bytes = int(max_buffer_seconds * VAD_SAMPLE_RATE * 2)
        self.audio_buffer = RealtimeAudioBuffer(
            source_sr=VAD_SAMPLE_RATE,
            target_sr=VAD_SAMPLE_RATE,
            max_bytes=max_buffer_bytes,
        )
        self.vad: StreamingVAD | None = self._new_vad(self.settings.turn_detection)
        self.vad_origin_samples = 0
        self.buffer_origin_samples = 0
        self.active_segment: ActiveTranscriptionSegment | None = None
        self.committed_segments: list[CommittedTranscriptionSegment] = []
        self._next_segment_id = 0
        self._decode_lock = asyncio.Lock()
        self._final_tasks: set[asyncio.Task[None]] = set()
        self._inflight_request_ids: set[str] = set()
        self._input_done = False

    async def run(self) -> None:
        await self.send(self.initial_event())
        while not self.closed:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message["type"] != "websocket.receive":
                continue
            try:
                payload = json.loads(message["text"])
            except (KeyError, TypeError, json.JSONDecodeError):
                await self.send_error(
                    "invalid_request_error", "invalid_json", "Invalid JSON event."
                )
                continue
            if not isinstance(payload, dict):
                await self.send_error(
                    "invalid_request_error",
                    "invalid_event",
                    "Top-level payload must be a JSON object.",
                )
                continue
            await self.dispatch(payload)

    async def dispatch(self, payload: dict[str, Any]) -> None:
        try:
            event = parse_client_event(payload)
        except ValueError as exc:
            await self.send_error("invalid_request_error", "invalid_event", str(exc))
            return
        if event is None or type(event) not in self.handlers:
            await self.send_error(
                "invalid_request_error",
                "unsupported_event",
                f"Unsupported event type: {payload.get('type')!r}",
            )
            return
        await getattr(self, self.handlers[type(event)])(event)

    async def send(self, event: dict[str, Any]) -> None:
        if self.closed:
            return
        if self.websocket.application_state != WebSocketState.CONNECTED:
            return
        async with self._send_lock:
            self.event_index += 1
            event.setdefault("event_id", _new_id("evt"))
            event.setdefault("event_index", self.event_index)
            await self.websocket.send_text(json.dumps(event))

    async def send_error(self, type_: str, code: str, message: str) -> None:
        await self.send(
            make_event(
                "error",
                error={"type": type_, "code": code, "message": message},
            )
        )

    async def cancel_and_abort(
        self, task: asyncio.Task[Any] | None, request_id: str | None
    ) -> None:
        if task is None or task.done():
            return
        task.cancel()
        try:
            if request_id is not None:
                await self.client.abort(request_id)
        finally:
            await asyncio.gather(task, return_exceptions=True)

    def initial_event(self) -> dict[str, Any]:
        return make_event("session.created", session=self._session_payload())

    def _session_payload(self) -> dict[str, Any]:
        turn_detection = self.settings.turn_detection
        return {
            "id": self.session_id,
            "object": "realtime.session",
            "model": self.model_name,
            "intent": "transcription",
            "modalities": ["text"],
            "input_audio_format": "pcm16",
            "language": self.settings.language,
            "decode_interval_ms": self.settings.decode_interval_ms,
            "turn_detection": (
                turn_detection.model_dump(exclude_none=True)
                if turn_detection is not None
                else None
            ),
        }

    @staticmethod
    def _new_vad(turn_detection: TurnDetection | None) -> StreamingVAD | None:
        if turn_detection is None:
            return None
        return StreamingVAD(
            VADConfig(
                threshold=(
                    turn_detection.threshold
                    if turn_detection.threshold is not None
                    else 0.5
                ),
                prefix_padding_ms=(
                    turn_detection.prefix_padding_ms
                    if turn_detection.prefix_padding_ms is not None
                    else 300
                ),
                silence_duration_ms=(
                    turn_detection.silence_duration_ms
                    if turn_detection.silence_duration_ms is not None
                    else 500
                ),
            )
        )

    async def handle_session_update(self, event: SessionUpdate) -> None:
        update = event.session.model_dump(exclude_unset=True)
        if update.get("modalities") not in (None, ["text"]):
            await self.send_error(
                "invalid_request_error",
                "unsupported_modality",
                "Transcription sessions support only the text modality.",
            )
            return
        if update.get("input_audio_format") not in (None, "pcm16"):
            await self.send_error(
                "invalid_request_error",
                "unsupported_audio_format",
                "Realtime transcription supports only PCM16 input audio.",
            )
            return
        if not self.audio_buffer.is_empty() and any(
            key in update for key in ("language", "turn_detection")
        ):
            await self.send_error(
                "invalid_request_error",
                "session_active",
                "Language and VAD settings cannot change "
                "while uncommitted audio is buffered.",
            )
            return

        if "language" in update:
            language = update["language"]
            self.settings.language = language.strip() if language else None
        if "turn_detection" in update:
            turn_detection = event.session.turn_detection
            if (
                turn_detection is not None
                and turn_detection.type != TurnDetectionType.SERVER_VAD
            ):
                await self.send_error(
                    "invalid_request_error",
                    "unsupported_turn_detection",
                    "Realtime transcription supports only server_vad or null.",
                )
                return
            if self.vad is not None:
                self.vad.reset()
            self.settings.turn_detection = turn_detection
            self.vad = self._new_vad(turn_detection)
            self.vad_origin_samples = self.buffer_origin_samples
        await self.send(make_event("session.updated", session=self._session_payload()))

    async def handle_audio_append(self, event: InputAudioBufferAppend) -> None:
        if self._input_done:
            await self.send_error(
                "invalid_request_error",
                "input_already_done",
                "Audio cannot be appended after transcription.done.",
            )
            return
        try:
            pcm = base64.b64decode(event.audio, validate=False)
        except (ValueError, binascii.Error):
            await self.send_error(
                "invalid_request_error", "invalid_audio", "Audio must be base64 PCM16."
            )
            return
        if len(pcm) % 2:
            await self.send_error(
                "invalid_request_error",
                "invalid_audio",
                "PCM16 audio must contain complete 16-bit samples.",
            )
            return
        total_limit = self.audio_chunking.max_total_audio_s
        absolute_end = self.buffer_origin_samples + self.audio_buffer.num_samples
        if total_limit is not None and absolute_end + len(pcm) // 2 > int(
            total_limit * VAD_SAMPLE_RATE
        ):
            await self.send_error(
                "invalid_request_error",
                "audio_too_long",
                f"Realtime transcription accepts up to {total_limit:g} seconds.",
            )
            return

        append_start_sample = absolute_end
        try:
            self.audio_buffer.append_bytes(pcm)
        except BufferOverflow as exc:
            await self.send_error(
                "invalid_request_error", "audio_buffer_overflow", str(exc)
            )
            return
        if self.vad is None:
            if self.active_segment is None and pcm:
                self._start_segment(append_start_sample)
        else:
            emits = await asyncio.to_thread(self.vad.process, pcm)
            for emit in emits:
                await self._handle_vad_emit(emit)

        await self._enforce_hard_limit()
        self._maybe_schedule_partial()

    def _absolute_vad_sample(self, sample_offset: int) -> int:
        return self.vad_origin_samples + sample_offset

    def _absolute_buffer_end(self) -> int:
        return self.buffer_origin_samples + self.audio_buffer.num_samples

    async def _handle_vad_emit(self, emit: Any) -> None:
        absolute_sample = self._absolute_vad_sample(emit.sample_offset)
        if emit.event_type == VADEvent.SPEECH_STARTED:
            if self.active_segment is None:
                self._start_segment(absolute_sample)
            await self.send(
                make_event(
                    "input_audio_buffer.speech_started",
                    audio_start_ms=offsets_to_ms(absolute_sample),
                    segment_id=self.active_segment.segment_id,
                )
            )
            return
        if emit.event_type == VADEvent.SPEECH_STOPPED:
            segment_id = (
                self.active_segment.segment_id
                if self.active_segment is not None
                else None
            )
            await self.send(
                make_event(
                    "input_audio_buffer.speech_stopped",
                    audio_end_ms=offsets_to_ms(absolute_sample),
                    segment_id=segment_id,
                )
            )
            await self._finalize_through(absolute_sample)

    def _start_segment(self, start_sample: int) -> ActiveTranscriptionSegment:
        start_sample = min(
            max(start_sample, self.buffer_origin_samples), self._absolute_buffer_end()
        )
        interval_samples = self.settings.decode_interval_ms * VAD_SAMPLE_RATE // 1000
        segment = ActiveTranscriptionSegment(
            segment_id=self._next_segment_id,
            start_sample=start_sample,
            strategy_state=self.strategy.create_state(
                model_name=self.model_name,
                language=self.settings.language,
            ),
            next_refresh_sample=start_sample + interval_samples,
        )
        self._next_segment_id += 1
        self.active_segment = segment
        return segment

    async def _enforce_hard_limit(self) -> None:
        max_samples = int(self.audio_chunking.max_audio_clip_s * VAD_SAMPLE_RATE)
        end_sample = self._absolute_buffer_end()
        while (
            self.active_segment is not None
            and end_sample - self.active_segment.start_sample >= max_samples
        ):
            cut = self.active_segment.start_sample + max_samples
            await self._queue_final(cut)
            self._start_segment(cut)

    async def _finalize_through(self, end_sample: int) -> None:
        if self.active_segment is None:
            return
        end_sample = min(end_sample, self._absolute_buffer_end())
        max_samples = int(self.audio_chunking.max_audio_clip_s * VAD_SAMPLE_RATE)
        while end_sample - self.active_segment.start_sample > max_samples:
            cut = self.active_segment.start_sample + max_samples
            await self._queue_final(cut)
            self._start_segment(cut)
        if (
            self.active_segment is not None
            and end_sample > self.active_segment.start_sample
        ):
            await self._queue_final(end_sample)

    async def _queue_final(self, end_sample: int) -> None:
        segment = self.active_segment
        if segment is None or segment.finalizing or end_sample <= segment.start_sample:
            return
        start_byte = max(0, (segment.start_sample - self.buffer_origin_samples) * 2)
        end_byte = min(
            self.audio_buffer.num_bytes,
            max(start_byte, (end_sample - self.buffer_origin_samples) * 2),
        )
        pcm = bytes(self.audio_buffer.buf[start_byte:end_byte])
        wav_bytes = self.audio_buffer.to_sliced_wav_bytes(
            start_byte=start_byte, end_byte=end_byte
        )
        segment.finalizing = True
        task = asyncio.create_task(self._run_final(segment, wav_bytes, pcm))
        self._final_tasks.add(task)
        task.add_done_callback(self._handle_final_task_done)

        self.audio_buffer.drop_prefix(end_byte)
        self.buffer_origin_samples += end_byte // 2
        self.active_segment = None
        await self.send(
            make_event(
                "input_audio_buffer.committed",
                segment_id=segment.segment_id,
            )
        )

    def _handle_final_task_done(self, task: asyncio.Task[None]) -> None:
        self._final_tasks.discard(task)
        if not task.cancelled():
            task.exception()
        if self.active_segment is not None and self.active_segment.dirty:
            self._schedule_dirty_partial(self.active_segment)

    def _maybe_schedule_partial(self) -> None:
        segment = self.active_segment
        if segment is None or segment.finalizing:
            return
        end_sample = self._absolute_buffer_end()
        if end_sample < segment.next_refresh_sample:
            return
        interval_samples = self.settings.decode_interval_ms * VAD_SAMPLE_RATE // 1000
        while segment.next_refresh_sample <= end_sample:
            segment.next_refresh_sample += interval_samples
        if segment.decode_task is not None and not segment.decode_task.done():
            self._mark_dirty(segment)
            return
        if any(not task.done() for task in self._final_tasks):
            self._mark_dirty(segment)
            return
        segment.decode_task = asyncio.create_task(self._run_partial(segment))

    def _mark_dirty(self, segment: ActiveTranscriptionSegment) -> None:
        segment.dirty = True

    def _schedule_dirty_partial(self, segment: ActiveTranscriptionSegment) -> None:
        if segment.finalizing or self.active_segment is not segment:
            return
        if any(not task.done() for task in self._final_tasks):
            self._mark_dirty(segment)
            return
        if segment.decode_task is None or segment.decode_task.done():
            segment.dirty = False
            segment.decode_task = asyncio.create_task(self._run_partial(segment))

    async def _run_partial(self, segment: ActiveTranscriptionSegment) -> None:
        try:
            async with self._decode_lock:
                if segment.finalizing or self.active_segment is not segment:
                    return
                start_byte = max(
                    0, (segment.start_sample - self.buffer_origin_samples) * 2
                )
                end_byte = self.audio_buffer.num_bytes
                audio = self.audio_buffer.to_sliced_wav_bytes(
                    start_byte=start_byte, end_byte=end_byte
                )
                await self._decode_and_emit(segment, audio, is_final=False)
        finally:
            segment.decode_task = None
            if (
                segment.dirty
                and not segment.finalizing
                and self.active_segment is segment
            ):
                self._schedule_dirty_partial(segment)

    async def _run_final(
        self,
        segment: ActiveTranscriptionSegment,
        audio: bytes,
        pcm: bytes,
    ) -> None:
        await self.cancel_and_abort(segment.decode_task, segment.request_id)
        async with self._decode_lock:
            if self._is_silent(pcm):
                await self._emit_hypothesis(segment, "", is_final=True)
            else:
                await self._decode_and_emit(segment, audio, is_final=True)

    @staticmethod
    def _is_silent(pcm: bytes) -> bool:
        if not pcm:
            return True
        samples = np.frombuffer(pcm, dtype="<i2")
        return bool(
            samples.size == 0
            or np.max(np.abs(samples.astype(np.int32))) < _SILENT_PCM16_PEAK
        )

    async def _decode_and_emit(
        self,
        segment: ActiveTranscriptionSegment,
        audio: bytes,
        *,
        is_final: bool,
    ) -> None:
        segment.decode_attempt += 1
        request_id = f"{self.session_id}:{segment.segment_id}:{segment.decode_attempt}"
        segment.request_id = request_id
        request = self.strategy.build_decode_request(
            audio=audio,
            state=segment.strategy_state,
            is_final=is_final,
            request_id=request_id,
        )
        self._inflight_request_ids.add(request_id)
        try:
            result = await self.client.completion(request, request_id=request_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.send_error("server_error", "transcription_failed", str(exc))
            return
        finally:
            self._inflight_request_ids.discard(request_id)
            if segment.request_id == request_id:
                segment.request_id = None
        if not is_final and (segment.finalizing or self.active_segment is not segment):
            return
        text = self.strategy.update_hypothesis(
            generated_text=result.text,
            state=segment.strategy_state,
        )
        if not is_final and text == segment.last_text:
            return
        await self._emit_hypothesis(segment, text, is_final=is_final)

    async def _emit_hypothesis(
        self,
        segment: ActiveTranscriptionSegment,
        text: str,
        *,
        is_final: bool,
    ) -> None:
        segment.last_text = text
        await self.send(
            make_event(
                "transcription.segment",
                segment_id=segment.segment_id,
                text=text,
                is_final=is_final,
            )
        )
        if is_final:
            self.committed_segments.append(
                CommittedTranscriptionSegment(
                    segment_id=segment.segment_id,
                    text=text,
                )
            )

    async def handle_audio_commit(self, event: InputAudioBufferCommit) -> None:
        del event
        await self._commit_buffer("client_commit")

    async def _commit_buffer(self, reason: str) -> None:
        end_sample = self._absolute_buffer_end()
        if self.active_segment is None and not self.audio_buffer.is_empty():
            if reason == "session_end" and self.vad is not None:
                self.buffer_origin_samples = end_sample
                self.audio_buffer.clear()
                self.vad.reset()
                self.vad_origin_samples = self.buffer_origin_samples
                return
            self._start_segment(self.buffer_origin_samples)
        await self._finalize_through(end_sample)
        if self.vad is not None:
            self.vad.reset()
            self.vad_origin_samples = self.buffer_origin_samples

    async def handle_transcription_done(self, event: TranscriptionDone) -> None:
        del event
        if self._input_done:
            await self.send_error(
                "invalid_request_error",
                "input_already_done",
                "transcription.done was already received.",
            )
            return
        self._input_done = True
        await self._commit_buffer("session_end")
        while self._final_tasks:
            final_tasks = list(self._final_tasks)
            await asyncio.gather(*final_tasks, return_exceptions=True)
            self._final_tasks.difference_update(final_tasks)
        ordered = sorted(self.committed_segments, key=lambda item: item.segment_id)
        await self.send(
            make_event(
                "transcription.completed",
                text=join_transcript_parts(item.text for item in ordered),
            )
        )

    async def teardown(self) -> None:
        self.closed = True
        segment = self.active_segment
        if segment is not None:
            segment.finalizing = True
            self.active_segment = None
            await self.cancel_and_abort(segment.decode_task, segment.request_id)
        final_tasks = list(self._final_tasks)
        request_ids = list(self._inflight_request_ids)
        for task in final_tasks:
            task.cancel()
        if request_ids:
            await asyncio.gather(
                *(self.client.abort(request_id) for request_id in request_ids),
                return_exceptions=True,
            )
        for task in final_tasks:
            await asyncio.gather(task, return_exceptions=True)
        if self.websocket.client_state == WebSocketState.CONNECTED:
            await self.websocket.close()


__all__ = ["RealtimeTranscriptionSession"]
