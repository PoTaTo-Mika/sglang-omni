from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from sglang_omni.client import GenerateRequest
from sglang_omni.serve.realtime.strategy import StreamingASRConfig, StreamingHypothesis
from sglang_omni.serve.speech_to_text import build_speech_to_text_generate_request


@dataclass(slots=True)
class Qwen3ASRStreamingState:
    config: StreamingASRConfig
    chunk_id: int = 0
    raw_decoded: str = ""
    language: str | None = None


class Qwen3ASRStreamingStrategy:
    def create_state(self, config: StreamingASRConfig) -> object:
        return Qwen3ASRStreamingState(config=config, language=config.language)

    @staticmethod
    def _state(state: object) -> Qwen3ASRStreamingState:
        if not isinstance(state, Qwen3ASRStreamingState):
            raise TypeError("Qwen3-ASR received incompatible streaming state")
        return state

    def build_decode_request(
        self,
        *,
        audio: bytes,
        state: object,
        is_final: bool,
        request_id: str,
    ) -> GenerateRequest:
        del is_final, request_id
        qwen_state = self._state(state)
        use_prefix = (
            qwen_state.chunk_id >= qwen_state.config.unfixed_chunk_num
            and bool(qwen_state.raw_decoded)
        )
        request = build_speech_to_text_generate_request(
            audio_bytes=audio,
            filename="realtime-segment.wav",
            content_type="audio/wav",
            model=qwen_state.config.model_name,
            language=qwen_state.language,
            prompt=None,
            temperature=0.0,
            stream=False,
        )
        request.extra_params.update(
            {
                "_asr_streaming": True,
                "_asr_streaming_prefix_text": (
                    qwen_state.raw_decoded if use_prefix else None
                ),
                "_asr_streaming_rollback_tokens": (
                    qwen_state.config.rollback_tokens if use_prefix else 0
                ),
            }
        )
        return request

    def update_hypothesis(
        self,
        *,
        generated_text: str,
        metadata: Mapping[str, Any],
        state: object,
        is_final: bool,
    ) -> StreamingHypothesis:
        qwen_state = self._state(state)
        language = metadata.get("language")
        if isinstance(language, str) and language:
            qwen_state.language = language
        stable_text = metadata.get("streaming_prefix_text")
        if not isinstance(stable_text, str):
            stable_text = ""
        qwen_state.raw_decoded = generated_text
        qwen_state.chunk_id += 1
        return StreamingHypothesis(
            text=generated_text,
            stable_text=generated_text if is_final else stable_text,
            language=qwen_state.language,
        )

    def reset_segment(self, state: object) -> None:
        qwen_state = self._state(state)
        qwen_state.chunk_id = 0
        qwen_state.raw_decoded = ""
        qwen_state.language = qwen_state.config.language


__all__ = ["Qwen3ASRStreamingState", "Qwen3ASRStreamingStrategy"]
