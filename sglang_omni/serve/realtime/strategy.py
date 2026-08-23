from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from sglang_omni.client import GenerateRequest


@dataclass(frozen=True, slots=True)
class StreamingASRConfig:
    model_name: str
    sample_rate: int
    language: str | None
    rollback_tokens: int
    unfixed_chunk_num: int


@dataclass(frozen=True, slots=True)
class StreamingHypothesis:
    text: str
    stable_text: str
    language: str | None


class StreamingASRStrategy(Protocol):
    def create_state(self, config: StreamingASRConfig) -> object: ...

    def build_decode_request(
        self,
        *,
        audio: bytes,
        state: object,
        is_final: bool,
        request_id: str,
    ) -> GenerateRequest: ...

    def update_hypothesis(
        self,
        *,
        generated_text: str,
        metadata: Mapping[str, Any],
        state: object,
        is_final: bool,
    ) -> StreamingHypothesis: ...

    def reset_segment(self, state: object) -> None: ...
