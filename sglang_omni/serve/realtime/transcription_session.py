from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import WebSocket

from sglang_omni.client import Client
from sglang_omni.config import RealtimeTranscriptionConfig
from sglang_omni.serve.realtime.audio_buffer import BufferOverflow, RealtimeAudioBuffer
from sglang_omni.serve.realtime.base_session import BaseRealtimeSession
from sglang_omni.serve.realtime.events import (
    InputAudioBufferAppend,
    InputAudioBufferClear,
    InputAudioBufferCommit,
    SessionUpdate,
    TranscriptionDone,
    TurnDetection,
    TurnDetectionType,
    make_event,
)
from sglang_omni.serve.realtime.strategy import (
    StreamingASRConfig,
    StreamingASRStrategy,
    StreamingHypothesis,
)
from sglang_omni.serve.realtime.vad import (
    VAD_SAMPLE_RATE,
    StreamingVAD,
    VADConfig,
    VADEvent,
    offsets_to_ms,
)
from sglang_omni.serve.transcription_chunking import join_transcript_parts

_SEQUENCE_GAP_TOLERANCE = 32
_SILENT_PCM16_PEAK = 33


@dataclass(slots=True)
class TranscriptionSessionSettings:
    response_format: str = "json"
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
    revision: int = 0
    decode_attempt: int = 0
    last_text: str = ""
    dirty: bool = False
    finalizing: bool = False
    request_id: str | None = None
    decode_task: asyncio.Task[None] | None = None


@dataclass(frozen=True, slots=True)
class CommittedTranscriptionSegment:
    segment_id: int
    start_sample: int
    end_sample: int
    text: str
    language: str | None
    usage: dict[str, Any]
    decode_time_s: float
    boundary_reason: str


class RealtimeTranscriptionSession(BaseRealtimeSession):
    handlers = {
        SessionUpdate: "handle_session_update",
        InputAudioBufferAppend: "handle_audio_append",
        InputAudioBufferCommit: "handle_audio_commit",
        InputAudioBufferClear: "handle_audio_clear",
        TranscriptionDone: "handle_transcription_done",
    }

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        model_name: str,
        capability: RealtimeTranscriptionConfig,
        strategy: StreamingASRStrategy,
        session_id: str | None = None,
    ) -> None:
        super().__init__(
            websocket,
            client=client,
            model_name=model_name,
            session_id=session_id,
        )
        if capability.sample_rate != VAD_SAMPLE_RATE:
            raise ValueError(
                "Realtime transcription currently requires 16 kHz PCM16 audio."
            )
        self.capability = capability
        self.strategy = strategy
        self.settings = TranscriptionSessionSettings(
            decode_interval_ms=capability.decode_interval_ms
        )
        max_buffer_seconds = max(
            capability.max_audio_clip_s * 4,
            capability.max_audio_clip_s + 4,
        )
        max_buffer_bytes = int(max_buffer_seconds * capability.sample_rate * 2)
        self.audio_buffer = RealtimeAudioBuffer(
            source_sr=capability.sample_rate,
            target_sr=capability.sample_rate,
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
        self._sequence_hashes: dict[int, bytes] = {}
        self._highest_sequence: int | None = None
        self._committed_sequence_watermark: int | None = None
        self._decode_request_count = 0
        self._coalesced_refresh_count = 0
        self._cancelled_refresh_count = 0
        self._input_done = False

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
            "response_format": self.settings.response_format,
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
            key in update
            for key in ("language", "decode_interval_ms", "turn_detection")
        ):
            await self.send_error(
                "invalid_request_error",
                "session_active",
                "Language, decode interval, and VAD settings cannot change "
                "while uncommitted audio is buffered.",
            )
            return

        response_format = update.get("response_format")
        if response_format is not None:
            self.settings.response_format = response_format
        if "language" in update:
            language = update["language"]
            self.settings.language = language.strip() if language else None
        decode_interval_ms = update.get("decode_interval_ms")
        if decode_interval_ms is not None:
            if not (
                self.capability.min_decode_interval_ms
                <= decode_interval_ms
                <= self.capability.max_decode_interval_ms
            ):
                await self.send_error(
                    "invalid_request_error",
                    "invalid_decode_interval",
                    "decode_interval_ms must be between "
                    f"{self.capability.min_decode_interval_ms} and "
                    f"{self.capability.max_decode_interval_ms}.",
                )
                return
            self.settings.decode_interval_ms = decode_interval_ms
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
        if await self._handle_sequence(event.sequence, pcm):
            return

        total_limit = self.capability.max_total_audio_s
        absolute_end = self.buffer_origin_samples + self.audio_buffer.num_samples
        if total_limit is not None and absolute_end + len(pcm) // 2 > int(
            total_limit * self.capability.sample_rate
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
        self._record_sequence(event.sequence, pcm)

        if self.vad is None:
            if self.active_segment is None and pcm:
                self._start_segment(append_start_sample)
        else:
            emits = await asyncio.to_thread(self.vad.process, pcm)
            for emit in emits:
                await self._handle_vad_emit(emit)

        await self._enforce_hard_limit()
        self._maybe_schedule_partial()
        if event.sequence is not None:
            await self.send(
                make_event(
                    "input_audio_buffer.appended",
                    sequence=event.sequence,
                    duplicate=False,
                )
            )

    async def _handle_sequence(self, sequence: int | None, pcm: bytes) -> bool:
        if sequence is None:
            return False
        digest = hashlib.sha256(pcm).digest()
        previous = self._sequence_hashes.get(sequence)
        if previous is not None:
            if previous != digest:
                await self.send_error(
                    "invalid_request_error",
                    "idempotency_conflict",
                    f"Audio sequence {sequence} was reused with different data.",
                )
            else:
                await self.send(
                    make_event(
                        "input_audio_buffer.appended",
                        sequence=sequence,
                        duplicate=True,
                    )
                )
            return True
        watermark = self._committed_sequence_watermark
        if watermark is not None and sequence <= watermark:
            await self.send(
                make_event(
                    "input_audio_buffer.appended",
                    sequence=sequence,
                    duplicate=True,
                )
            )
            return True
        highest = self._highest_sequence
        if highest is not None and sequence <= highest:
            await self.send_error(
                "invalid_request_error",
                "out_of_order_sequence",
                f"Audio sequence {sequence} arrived out of order.",
            )
            return True
        if highest is not None and sequence - highest > _SEQUENCE_GAP_TOLERANCE:
            await self.send_error(
                "invalid_request_error",
                "sequence_gap",
                f"Audio sequence gap exceeds {_SEQUENCE_GAP_TOLERANCE}.",
            )
            return True
        return False

    def _record_sequence(self, sequence: int | None, pcm: bytes) -> None:
        if sequence is None:
            return
        self._sequence_hashes[sequence] = hashlib.sha256(pcm).digest()
        self._highest_sequence = sequence

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
            await self._finalize_through(absolute_sample, "vad")

    def _start_segment(self, start_sample: int) -> ActiveTranscriptionSegment:
        start_sample = min(
            max(start_sample, self.buffer_origin_samples), self._absolute_buffer_end()
        )
        config = StreamingASRConfig(
            model_name=self.model_name,
            sample_rate=self.capability.sample_rate,
            language=self.settings.language,
            rollback_tokens=self.capability.rollback_tokens,
            unfixed_chunk_num=self.capability.unfixed_chunk_num,
        )
        interval_samples = (
            self.settings.decode_interval_ms * self.capability.sample_rate // 1000
        )
        segment = ActiveTranscriptionSegment(
            segment_id=self._next_segment_id,
            start_sample=start_sample,
            strategy_state=self.strategy.create_state(config),
            next_refresh_sample=start_sample + interval_samples,
        )
        self._next_segment_id += 1
        self.active_segment = segment
        return segment

    async def _enforce_hard_limit(self) -> None:
        max_samples = int(
            self.capability.max_audio_clip_s * self.capability.sample_rate
        )
        end_sample = self._absolute_buffer_end()
        while (
            self.active_segment is not None
            and end_sample - self.active_segment.start_sample >= max_samples
        ):
            cut = self.active_segment.start_sample + max_samples
            await self._queue_final(cut, "hard_limit")
            self._start_segment(cut)

    async def _finalize_through(self, end_sample: int, reason: str) -> None:
        if self.active_segment is None:
            return
        end_sample = min(end_sample, self._absolute_buffer_end())
        max_samples = int(
            self.capability.max_audio_clip_s * self.capability.sample_rate
        )
        while end_sample - self.active_segment.start_sample > max_samples:
            cut = self.active_segment.start_sample + max_samples
            await self._queue_final(cut, "hard_limit")
            self._start_segment(cut)
        if (
            self.active_segment is not None
            and end_sample > self.active_segment.start_sample
        ):
            await self._queue_final(end_sample, reason)

    async def _queue_final(self, end_sample: int, reason: str) -> None:
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
        task = asyncio.create_task(
            self._run_final(segment, wav_bytes, pcm, end_sample, reason)
        )
        self._final_tasks.add(task)
        task.add_done_callback(self._handle_final_task_done)

        self.audio_buffer.drop_prefix(end_byte)
        self.buffer_origin_samples += end_byte // 2
        self.active_segment = None
        if self._highest_sequence is not None:
            self._committed_sequence_watermark = self._highest_sequence
            self._sequence_hashes.clear()
        await self.send(
            make_event(
                "input_audio_buffer.committed",
                segment_id=segment.segment_id,
                boundary_reason=reason,
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
        interval_samples = (
            self.settings.decode_interval_ms * self.capability.sample_rate // 1000
        )
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
        if not segment.dirty:
            self._coalesced_refresh_count += 1
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
                end_sample = self._absolute_buffer_end()
                audio = self.audio_buffer.to_sliced_wav_bytes(
                    start_byte=start_byte, end_byte=end_byte
                )
                await self._decode_and_emit(
                    segment, audio, end_sample=end_sample, is_final=False
                )
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
        end_sample: int,
        reason: str,
    ) -> None:
        if segment.decode_task is not None and not segment.decode_task.done():
            self._cancelled_refresh_count += 1
        await self.cancel_and_abort(segment.decode_task, segment.request_id)
        try:
            async with self._decode_lock:
                if self._is_silent(pcm):
                    hypothesis = StreamingHypothesis(
                        text="", stable_text="", language=self.settings.language
                    )
                    await self._emit_hypothesis(
                        segment,
                        hypothesis,
                        end_sample=end_sample,
                        is_final=True,
                        usage={},
                        decode_time_s=0.0,
                        boundary_reason=reason,
                    )
                else:
                    await self._decode_and_emit(
                        segment,
                        audio,
                        end_sample=end_sample,
                        is_final=True,
                        boundary_reason=reason,
                    )
        finally:
            self.strategy.reset_segment(segment.strategy_state)

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
        end_sample: int,
        is_final: bool,
        boundary_reason: str | None = None,
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
        started = time.perf_counter()
        self._decode_request_count += 1
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
        decode_time_s = time.perf_counter() - started
        hypothesis = self.strategy.update_hypothesis(
            generated_text=result.text,
            metadata=result.metadata,
            state=segment.strategy_state,
            is_final=is_final,
        )
        if not is_final and hypothesis.text == segment.last_text:
            return
        usage = result.usage.to_dict() if result.usage is not None else {}
        await self._emit_hypothesis(
            segment,
            hypothesis,
            end_sample=end_sample,
            is_final=is_final,
            usage=usage,
            decode_time_s=decode_time_s,
            boundary_reason=boundary_reason,
        )

    async def _emit_hypothesis(
        self,
        segment: ActiveTranscriptionSegment,
        hypothesis: StreamingHypothesis,
        *,
        end_sample: int,
        is_final: bool,
        usage: dict[str, Any],
        decode_time_s: float,
        boundary_reason: str | None,
    ) -> None:
        segment.revision += 1
        segment.last_text = hypothesis.text
        fields: dict[str, Any] = {
            "session_id": self.session_id,
            "segment_id": segment.segment_id,
            "revision": segment.revision,
            "text": hypothesis.text,
            "stable_text": hypothesis.stable_text,
            "is_final": is_final,
        }
        if self.settings.response_format == "verbose_json":
            fields.update(
                {
                    "language": hypothesis.language,
                    "audio_start_ms": offsets_to_ms(segment.start_sample),
                    "audio_end_ms": offsets_to_ms(end_sample),
                    "duration_ms": offsets_to_ms(end_sample - segment.start_sample),
                    "usage": usage,
                    "decode_time_s": decode_time_s,
                    "timestamp_source": "vad",
                    "boundary_reason": boundary_reason,
                }
            )
        await self.send(make_event("transcription.segment", **fields))
        if is_final:
            self.committed_segments.append(
                CommittedTranscriptionSegment(
                    segment_id=segment.segment_id,
                    start_sample=segment.start_sample,
                    end_sample=end_sample,
                    text=hypothesis.text,
                    language=hypothesis.language,
                    usage=usage,
                    decode_time_s=decode_time_s,
                    boundary_reason=boundary_reason or "unknown",
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
        await self._finalize_through(end_sample, reason)
        if self.vad is not None:
            self.vad.reset()
            self.vad_origin_samples = self.buffer_origin_samples

    async def handle_audio_clear(self, event: InputAudioBufferClear) -> None:
        del event
        segment = self.active_segment
        if segment is not None:
            segment.finalizing = True
            await self.cancel_and_abort(segment.decode_task, segment.request_id)
            self.strategy.reset_segment(segment.strategy_state)
        discarded = self.audio_buffer.num_samples
        self.buffer_origin_samples += discarded
        self.audio_buffer.clear()
        self.active_segment = None
        if self.vad is not None:
            self.vad.reset()
            self.vad_origin_samples = self.buffer_origin_samples
        await self.send(make_event("input_audio_buffer.cleared"))

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
        segments = [
            {
                "id": item.segment_id,
                "start": item.start_sample / self.capability.sample_rate,
                "end": item.end_sample / self.capability.sample_rate,
                "text": item.text,
            }
            for item in ordered
        ]
        fields: dict[str, Any] = {
            "session_id": self.session_id,
            "text": join_transcript_parts(item.text for item in ordered),
            "segments": segments,
        }
        if self.settings.response_format == "verbose_json":
            fields["usage"] = [item.usage for item in ordered]
            fields["timestamp_source"] = "vad"
            fields["statistics"] = {
                "decode_requests": self._decode_request_count,
                "coalesced_refreshes": self._coalesced_refresh_count,
                "cancelled_refreshes": self._cancelled_refresh_count,
            }
        await self.send(make_event("transcription.completed", **fields))

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
        await super().teardown()


__all__ = ["RealtimeTranscriptionSession"]
