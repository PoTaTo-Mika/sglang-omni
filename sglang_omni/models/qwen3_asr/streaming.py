from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.client import GenerateRequest
from sglang_omni.serve.speech_to_text import build_speech_to_text_generate_request

_ASR_TEXT = "<asr_text>"
_ROLLBACK_TOKENS = 5
_UNFIXED_CHUNK_NUM = 2


@dataclass(slots=True)
class Qwen3ASRStreamingState:
    model_name: str
    language: str | None = None
    chunk_id: int = 0
    transcript: str = ""


class Qwen3ASRStreamingStrategy:
    def create_state(
        self, *, model_name: str, language: str | None
    ) -> Qwen3ASRStreamingState:
        return Qwen3ASRStreamingState(model_name=model_name, language=language)

    def build_decode_request(
        self,
        *,
        audio: bytes,
        state: Qwen3ASRStreamingState,
        is_final: bool,
        request_id: str,
    ) -> GenerateRequest:
        del is_final, request_id
        use_prefix = state.chunk_id >= _UNFIXED_CHUNK_NUM and bool(state.transcript)
        request = build_speech_to_text_generate_request(
            audio_bytes=audio,
            filename="realtime-segment.wav",
            content_type="audio/wav",
            model=state.model_name,
            language=state.language,
            prompt=None,
            temperature=0.0,
            stream=False,
        )
        request.extra_params.update(
            {
                "_asr_streaming": True,
                "_asr_streaming_prefix_text": (
                    state.transcript if use_prefix else None
                ),
                "_asr_streaming_rollback_tokens": (
                    _ROLLBACK_TOKENS if use_prefix else 0
                ),
            }
        )
        return request

    def update_hypothesis(
        self,
        *,
        generated_text: str,
        state: Qwen3ASRStreamingState,
    ) -> str:
        prefix, marker, transcript = generated_text.partition(_ASR_TEXT)
        if marker:
            label, separator, value = prefix.partition(" ")
            if separator and label.casefold() == "language" and value.strip():
                state.language = value.strip()
            visible_text = transcript
        else:
            visible_text = generated_text
        state.transcript = visible_text
        state.chunk_id += 1
        return visible_text


__all__ = ["Qwen3ASRStreamingState", "Qwen3ASRStreamingStrategy"]
