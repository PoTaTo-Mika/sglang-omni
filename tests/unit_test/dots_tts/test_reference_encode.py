# SPDX-License-Identifier: Apache-2.0
"""Reference-cache and schedule contracts for dots.tts."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import pytest
import torch

from sglang_omni.models.dots_tts.payload_types import DotsTtsState
from sglang_omni.models.dots_tts.prompt_builder import (
    normalize_prompt_text,
    prompt_audio_patch_count,
)
from sglang_omni.models.dots_tts.reference_encode import (
    DotsTtsReferenceAudioHook,
    DotsTtsReferenceEncoder,
)
from sglang_omni.proto import OmniRequest, StagePayload

SAMPLE_RATE = 48000
SAMPLES_PER_PATCH = 4 * 1920


class _FakeTokenizer:
    """Minimal stand-in: one id per character, fixed special-token ids."""

    _SPECIAL_TOKEN_IDS = {
        "<|audio_gen_start|>": 90001,
        "<|audio_gen_span|>": 90002,
        "<|audio_gen_end|>": 90003,
        "<|text_cond_end|>": 90004,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._SPECIAL_TOKEN_IDS.get(token, -1)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(chr(token_id) for token_id in token_ids if token_id < 90000)


def _write_wav(path: Path, samples: bytes) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples)
    return path


def _make_payload(state: DotsTtsState) -> StagePayload:
    return StagePayload(
        request_id="req",
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )


def _make_encoder(tmp_path: Path) -> DotsTtsReferenceEncoder:
    return DotsTtsReferenceEncoder(
        tokenizer=_FakeTokenizer(),
        sample_rate=SAMPLE_RATE,
        samples_per_patch=SAMPLES_PER_PATCH,
        model_identity=str(tmp_path),
    )


def test_cache_key_differs_for_same_size_different_content(tmp_path: Path) -> None:
    hook = DotsTtsReferenceAudioHook(
        model_identity="dots.tts-mf", sample_rate=SAMPLE_RATE
    )
    first = _write_wav(tmp_path / "a.wav", b"\x01\x00" * 24000)
    second = _write_wav(tmp_path / "b.wav", b"\x02\x00" * 24000)

    assert first.stat().st_size == second.stat().st_size
    assert hook.input_key(str(first)) != hook.input_key(str(second))
    assert hook.cache_key(str(first)) != hook.cache_key(str(second))


def test_cache_key_is_none_for_missing_reference(tmp_path: Path) -> None:
    hook = DotsTtsReferenceAudioHook(
        model_identity="dots.tts-mf", sample_rate=SAMPLE_RATE
    )

    assert hook.input_key(str(tmp_path / "missing.wav")) is None
    assert hook.cache_key(str(tmp_path / "missing.wav")) is None


def test_encode_payload_attaches_prompt_audio_and_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _write_wav(tmp_path / "ref.wav", b"\x01\x00" * 24000)
    prompt_audio = torch.zeros((1, SAMPLES_PER_PATCH * 3), dtype=torch.float32)
    monkeypatch.setattr(
        "sglang_omni.models.dots_tts.reference_encode.load_reference_waveform",
        lambda path, *, target_sample_rate: prompt_audio,
    )
    payload = _make_payload(
        DotsTtsState(
            text="target",
            prompt_text="reference",
            ref_audio_path=str(reference),
            max_audio_patches=16,
        )
    )

    stored = _make_encoder(tmp_path).encode_payload(payload)
    state = DotsTtsState.from_dict(stored.data)

    assert state.prompt_audio.shape == (1, SAMPLES_PER_PATCH * 3)
    assert state.generation_schedule_ids is not None
    assert state.generation_schedule_ids.count(90002) == 16
    assert state.prompt_tokens == len(state.generation_schedule_ids)


def test_encode_payload_rejects_prompt_longer_than_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _write_wav(tmp_path / "long.wav", b"\x01\x00" * 24000)
    monkeypatch.setattr(
        "sglang_omni.models.dots_tts.reference_encode.load_reference_waveform",
        lambda path, *, target_sample_rate: torch.zeros(
            (1, SAMPLES_PER_PATCH * 4), dtype=torch.float32
        ),
    )
    payload = _make_payload(
        DotsTtsState(
            text="target",
            prompt_text="reference",
            ref_audio_path=str(reference),
            max_audio_patches=4,
        )
    )

    with pytest.raises(ValueError, match="prompt audio patch count"):
        _make_encoder(tmp_path).encode_payload(payload)


def test_encode_payload_requires_reference_audio(tmp_path: Path) -> None:
    payload = _make_payload(DotsTtsState(text="target", prompt_text="reference"))

    with pytest.raises(ValueError, match="requires reference audio"):
        _make_encoder(tmp_path).encode_payload(payload)


def test_prompt_text_gets_the_training_continuation_boundary() -> None:
    assert normalize_prompt_text("  reference  ") == "reference\n"
    assert normalize_prompt_text("   ") == ""
    assert normalize_prompt_text(None) == ""


def test_prompt_audio_patch_count_rounds_up() -> None:
    assert prompt_audio_patch_count(prompt_sample_count=1, samples_per_patch=100) == 1
    assert prompt_audio_patch_count(prompt_sample_count=100, samples_per_patch=100) == 1
    assert prompt_audio_patch_count(prompt_sample_count=101, samples_per_patch=100) == 2
