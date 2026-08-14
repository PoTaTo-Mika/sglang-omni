# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import wave

import pytest
from fastapi import HTTPException

from sglang_omni.serve import speech_to_text
from sglang_omni.serve.transcriptions import (
    build_transcription_generate_request as legacy_transcription_request_builder,
)


def _build_request(*, task: str = "transcribe"):
    return speech_to_text.build_speech_to_text_generate_request(
        audio_bytes=b"RIFF",
        filename="sample.wav",
        content_type="audio/wav",
        model="openai/whisper-large-v3",
        language="en",
        prompt=None,
        temperature=None,
        task=task,
    )


def _wav_bytes(*, duration_s: float = 0.25, sample_rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\0\0" * int(duration_s * sample_rate))
    return buffer.getvalue()


def test_transcription_builder_import_keeps_shared_callable() -> None:
    """Keep the pre-extraction import stable while stacked consumers migrate."""
    assert (
        legacy_transcription_request_builder
        is speech_to_text.build_speech_to_text_generate_request
    )


def test_build_request_defaults_to_transcribe_task() -> None:
    assert _build_request().extra_params["task"] == "transcribe"


def test_build_request_accepts_sibling_endpoint_task() -> None:
    assert _build_request(task="translate").extra_params["task"] == "translate"


def test_response_format_validation_preserves_endpoint_error_contract() -> None:
    with pytest.raises(HTTPException) as exc_info:
        speech_to_text.validate_speech_to_text_response_format(
            " SRT ",
            stream=False,
            endpoint_path="/v1/audio/transcriptions",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == (
        "Unsupported response_format for /v1/audio/transcriptions: ' SRT '"
    )


def test_assemble_uses_the_caller_probed_duration(monkeypatch) -> None:
    # A handler that already probed the upload passes the duration in;
    # assembling the response must not probe the same bytes again.
    monkeypatch.setattr(
        speech_to_text,
        "probe_audio_duration",
        lambda audio_bytes: pytest.fail("re-probed the upload"),
    )

    response = speech_to_text.assemble_speech_to_text_response(
        text="hello world",
        response_format="json",
        endpoint_path="/v1/audio/transcriptions",
        task="transcribe",
        language="en",
        audio_bytes=b"not-a-real-audio-file",
        architectures=None,
        duration_s=2.4,
    )

    assert json.loads(response.body)["usage"] == {"seconds": 3, "type": "duration"}


def test_probe_audio_duration_uses_soundfile_fast_path_for_wav(monkeypatch) -> None:
    import av

    monkeypatch.setattr(
        av,
        "open",
        lambda *args, **kwargs: pytest.fail("WAV duration probe fell back to PyAV"),
    )

    assert speech_to_text.probe_audio_duration(_wav_bytes()) == pytest.approx(0.25)


def test_probe_audio_duration_uses_pyav_for_non_wav(monkeypatch) -> None:
    import av
    import soundfile as sf

    class FakeContainer:
        duration = 2_500_000
        streams = ()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(
        sf,
        "info",
        lambda *args, **kwargs: pytest.fail(
            "non-WAV duration probe used the soundfile fast path"
        ),
    )
    monkeypatch.setattr(av, "open", lambda *args, **kwargs: FakeContainer())

    assert speech_to_text.probe_audio_duration(b"not-a-wav") == pytest.approx(2.5)


def test_probe_audio_duration_falls_back_when_wav_fast_path_fails(
    monkeypatch,
) -> None:
    import av
    import soundfile as sf

    class FakeContainer:
        duration = 1_500_000
        streams = ()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fail_soundfile_probe(*args, **kwargs):
        raise RuntimeError("bad WAV")

    monkeypatch.setattr(sf, "info", fail_soundfile_probe)
    monkeypatch.setattr(av, "open", lambda *args, **kwargs: FakeContainer())

    assert speech_to_text.probe_audio_duration(_wav_bytes()) == pytest.approx(1.5)


def test_verbose_response_uses_requested_task() -> None:
    response = speech_to_text.assemble_speech_to_text_response(
        text="hello world",
        response_format="verbose_json",
        endpoint_path="/v1/audio/transcriptions",
        task="translate",
        language="en",
        audio_bytes=b"not-a-real-audio-file",
        architectures=None,
    )

    assert json.loads(response.body)["task"] == "translate"
