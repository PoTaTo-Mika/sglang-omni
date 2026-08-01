# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for Qwen3-TTS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import (
    INITIAL_CODEC_CHUNK_FRAMES_PARAM,
    StreamingVocoderBase,
    resolve_initial_codec_chunk_frames,
)
from sglang_omni.utils.audio_payload import audio_waveform_payload

DEFAULT_QWEN3_TTS_STREAM_STRIDE = 16
DEFAULT_QWEN3_TTS_STREAM_FOLLOWUP_STRIDE = 64
DEFAULT_QWEN3_TTS_INITIAL_CHUNK_FRAMES = 1
DEFAULT_QWEN3_TTS_LEFT_CONTEXT_FRAMES = 25
_QWEN3_TTS_CODEBOOK_SIZE = 2048


@dataclass
class _Qwen3TTSStreamState:
    code_chunks: list[torch.Tensor] = field(default_factory=list)
    total_frames: int = 0
    ref_frames: int = 0
    emitted_generated_frames: int = 0
    next_decode_generated_frames: int = 0
    decoded_chunks: int = 0
    num_quantizers: int | None = None
    pending_ref_frames: int = 0
    initial_chunk_frames: int = DEFAULT_QWEN3_TTS_INITIAL_CHUNK_FRAMES


class Qwen3TTSStreamingVocoderScheduler(
    StreamingVocoderBase[_Qwen3TTSStreamState, None]
):
    """Decode Qwen3-TTS codec frames on a priority CUDA stream."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        device: str,
        stream_stride: int = DEFAULT_QWEN3_TTS_STREAM_STRIDE,
        stream_followup_stride: int = DEFAULT_QWEN3_TTS_STREAM_FOLLOWUP_STRIDE,
        initial_chunk_frames: int = DEFAULT_QWEN3_TTS_INITIAL_CHUNK_FRAMES,
        stream_left_context_frames: int = DEFAULT_QWEN3_TTS_LEFT_CONTEXT_FRAMES,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
    ) -> None:
        if stream_stride <= 0 or stream_followup_stride <= 0:
            raise ValueError("stream strides must be > 0")
        if initial_chunk_frames < 0:
            raise ValueError("initial_chunk_frames must be >= 0")
        if stream_left_context_frames < 0:
            raise ValueError("stream_left_context_frames must be >= 0")

        self._tokenizer = tokenizer
        self._device = torch.device(device)
        self._decoder = tokenizer.model.decoder
        self._samples_per_frame = int(self._decoder.total_upsample)
        self._stream_stride = int(stream_stride)
        self._stream_followup_stride = int(stream_followup_stride)
        self._default_initial_chunk_frames = int(initial_chunk_frames)
        self._stream_left_context_frames = int(stream_left_context_frames)
        self._decode_stream = (
            torch.cuda.Stream(device=self._device, priority=-1)
            if self._device.type == "cuda"
            else None
        )
        sample_rate = int(tokenizer.get_output_sample_rate())

        super().__init__(
            self._vocode_payload,
            batch_compute_fn=self._vocode_payloads,
            sample_rate=sample_rate,
            stream_source_hint="Qwen3-TTS",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def create_stream_state(self, request_id: str) -> _Qwen3TTSStreamState:
        del request_id
        return _Qwen3TTSStreamState(
            initial_chunk_frames=self._default_initial_chunk_frames
        )

    def latch_stream_contract(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        if origin == "payload":
            params = source.request.params
            if isinstance(params, Mapping):
                state.initial_chunk_frames = resolve_initial_codec_chunk_frames(
                    params,
                    steady_chunk_frames=self._stream_stride,
                    default_frames=self._default_initial_chunk_frames,
                )
            return

        metadata: Mapping[str, Any] = source
        if "num_quantizers" not in metadata and state.num_quantizers is None:
            raise RuntimeError(
                f"Qwen3-TTS stream chunk for {request_id!r} is missing num_quantizers"
            )
        if "num_quantizers" in metadata:
            num_quantizers = int(metadata["num_quantizers"])
            if num_quantizers <= 0:
                raise ValueError("Qwen3-TTS num_quantizers must be > 0")
            if (
                state.num_quantizers is not None
                and state.num_quantizers != num_quantizers
            ):
                raise ValueError(
                    f"Qwen3-TTS num_quantizers changed for {request_id!r}: "
                    f"{state.num_quantizers} -> {num_quantizers}"
                )
            state.num_quantizers = num_quantizers
        if "ref_code_len" in metadata:
            ref_frames = int(metadata["ref_code_len"])
            if ref_frames < 0:
                raise ValueError("Qwen3-TTS ref_code_len must be >= 0")
            if state.total_frames or state.ref_frames:
                raise ValueError(
                    f"Qwen3-TTS reference codes arrived after stream start for "
                    f"{request_id!r}"
                )
            state.pending_ref_frames = ref_frames
        if INITIAL_CODEC_CHUNK_FRAMES_PARAM in metadata:
            state.initial_chunk_frames = resolve_initial_codec_chunk_frames(
                metadata,
                steady_chunk_frames=self._stream_stride,
                default_frames=self._default_initial_chunk_frames,
            )

    def validate_chunk(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        chunk = codes.detach().to(device="cpu", dtype=torch.long)
        if chunk.ndim == 1:
            chunk = chunk.unsqueeze(0)
        elif chunk.ndim != 2:
            raise ValueError(
                f"Qwen3-TTS stream chunk must be [Q] or [T, Q], "
                f"got {tuple(chunk.shape)}"
            )
        if chunk.shape[0] == 0:
            raise ValueError("Qwen3-TTS stream chunk must not be empty")
        if state.num_quantizers is None:
            raise RuntimeError(
                f"Qwen3-TTS stream contract for {request_id!r} is missing "
                "num_quantizers"
            )
        if int(chunk.shape[1]) != state.num_quantizers:
            raise ValueError(
                f"Qwen3-TTS stream chunk has {int(chunk.shape[1])} quantizers, "
                f"expected {state.num_quantizers}"
            )
        if bool((chunk < 0).any()) or bool((chunk >= _QWEN3_TTS_CODEBOOK_SIZE).any()):
            raise ValueError(
                f"Qwen3-TTS stream chunk for {request_id!r} contains codec ids "
                f"outside [0, {_QWEN3_TTS_CODEBOOK_SIZE})"
            )
        return chunk

    def ingest(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        codes: torch.Tensor,
    ) -> None:
        del request_id
        if state.pending_ref_frames:
            if state.pending_ref_frames >= int(codes.shape[0]):
                raise ValueError(
                    "Qwen3-TTS first stream chunk must include at least one "
                    "generated codec frame after the reference"
                )
            state.ref_frames = state.pending_ref_frames
            state.pending_ref_frames = 0
        state.code_chunks.append(codes)
        state.total_frames += int(codes.shape[0])

    def should_decode(self, state: _Qwen3TTSStreamState, *, is_final: bool) -> bool:
        if is_final:
            return True
        generated_frames = state.total_frames - state.ref_frames
        next_frames = self._next_decode_threshold(state)
        return generated_frames >= next_frames

    def _next_decode_threshold(self, state: _Qwen3TTSStreamState) -> int:
        if state.next_decode_generated_frames:
            return state.next_decode_generated_frames
        return state.initial_chunk_frames or self._stream_stride

    def decode_delta(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        del request_id
        generated_frames = state.total_frames - state.ref_frames
        if generated_frames <= state.emitted_generated_frames:
            return None

        next_frames = self._next_decode_threshold(state)
        if not is_final and generated_frames < next_frames:
            state.next_decode_generated_frames = next_frames
            return None

        absolute_emitted = state.ref_frames + state.emitted_generated_frames
        window_start = max(0, absolute_emitted - self._stream_left_context_frames)
        codes = torch.cat(state.code_chunks, dim=0)
        codes_window = codes[window_start : state.total_frames]
        decoder_input = codes_window.transpose(0, 1).unsqueeze(0)
        with torch.inference_mode():
            if self._decode_stream is None:
                waveform = self._decoder.chunked_decode(decoder_input)
            else:
                with torch.cuda.stream(self._decode_stream):
                    waveform = self._decoder.chunked_decode(
                        decoder_input.to(self._device)
                    )
                torch.cuda.current_stream(self._device).wait_stream(self._decode_stream)
        if waveform.ndim == 3:
            waveform = waveform[0, 0]
        elif waveform.ndim == 2:
            waveform = waveform[0]
        else:
            raise ValueError(
                "Qwen3-TTS decoder returned unexpected waveform shape "
                f"{tuple(waveform.shape)}"
            )

        trim_frames = absolute_emitted - window_start
        trim_samples = min(
            trim_frames * self._samples_per_frame,
            int(waveform.shape[-1]),
        )
        new_frames = generated_frames - state.emitted_generated_frames
        emit_samples = new_frames * self._samples_per_frame
        delta = waveform[trim_samples : trim_samples + emit_samples]
        if delta.numel() == 0:
            return None

        state.emitted_generated_frames = generated_frames
        state.decoded_chunks += 1
        state.next_decode_generated_frames = (
            generated_frames + self._stream_followup_stride
        )
        return delta.detach().to(torch.float32).contiguous()

    def _decode_and_emit(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
    ) -> list[OutgoingMessage]:
        if not self.should_decode(state, is_final=False):
            return []
        waveform = self.decode_delta(request_id, state, is_final=False)
        if waveform is None:
            return []

        self._mark_stream_emitted(request_id)
        split_samples = state.initial_chunk_frames * self._samples_per_frame
        if (
            state.decoded_chunks == 1
            and split_samples > 0
            and split_samples < int(waveform.shape[-1])
        ):
            return [
                self._stream_chunk_message(request_id, waveform[:split_samples]),
                self._stream_chunk_message(request_id, waveform[split_samples:]),
            ]
        return [self._stream_chunk_message(request_id, waveform)]

    def fallback_full_decode(
        self,
        request_id: str,
        payload: StagePayload,
        state: _Qwen3TTSStreamState,
    ) -> torch.Tensor | None:
        del request_id, state
        return self._decode_state_audio(Qwen3TTSState.from_dict(payload.data))

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: _Qwen3TTSStreamState,
    ) -> dict[str, Any]:
        del request_id, state
        final_state = Qwen3TTSState.from_dict(payload.data)
        data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        usage = build_usage(final_state)
        if usage is not None:
            data["usage"] = usage
        return data

    async def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        return (await self._vocode_payloads([payload]))[0]

    async def _vocode_payloads(
        self, payloads: list[StagePayload]
    ) -> list[StagePayload]:
        states = [Qwen3TTSState.from_dict(payload.data) for payload in payloads]
        codes = []
        for state in states:
            if state.audio_codes is None:
                raise RuntimeError(
                    "Qwen3-TTS vocoder requires audio_codes from tts_engine"
                )
            codes.append(torch.as_tensor(state.audio_codes, dtype=torch.long))

        wavs, sample_rate = self._tokenizer.decode(
            [{"audio_codes": item} for item in codes]
        )
        if len(wavs) != len(payloads):
            raise RuntimeError(
                f"Qwen3-TTS speech tokenizer returned {len(wavs)} audios for "
                f"{len(payloads)} requests"
            )
        return [
            self._store_vocoder_result(payload, state, wav, sample_rate)
            for payload, state, wav in zip(payloads, states, wavs)
        ]

    def _store_vocoder_result(
        self,
        payload: StagePayload,
        state: Qwen3TTSState,
        waveform: Any,
        sample_rate: int,
    ) -> StagePayload:
        if waveform is None:
            raise RuntimeError("Qwen3-TTS speech tokenizer did not return audio")
        if state.ref_code_len:
            total_frames = len(state.audio_codes)
            cut = int(state.ref_code_len / max(total_frames, 1) * waveform.shape[0])
            waveform = waveform[cut:]

        data = audio_waveform_payload(
            waveform,
            sample_rate=int(sample_rate),
            modality="audio",
            source_hint="Qwen3-TTS",
        )
        usage = build_usage(state)
        if usage is not None:
            data["usage"] = usage
        payload.data = data
        return payload

    def _decode_state_audio(self, state: Qwen3TTSState) -> torch.Tensor | None:
        if state.audio_codes is None:
            return None
        codes = torch.as_tensor(state.audio_codes, dtype=torch.long)
        wavs, _ = self._tokenizer.decode([{"audio_codes": codes}])
        if not wavs:
            return None
        waveform = torch.as_tensor(wavs[0], dtype=torch.float32)
        if state.ref_code_len:
            cut = state.ref_code_len * self._samples_per_frame
            waveform = waveform[min(cut, int(waveform.shape[0])) :]
        return waveform.contiguous()


__all__ = ["Qwen3TTSStreamingVocoderScheduler"]
