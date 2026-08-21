# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Whisper segment-timestamp adapter."""

from __future__ import annotations

import pytest

from sglang_omni.serve.transcription_adapters.whisper_asr import WhisperASRAdapter


def test_parse_whisper_timestamp_segments() -> None:
    adapter = WhisperASRAdapter()

    response = adapter.build_timestamped_response(
        text="<|0.00|> Hello there.<|1.20|><|1.20|> Goodbye.<|2.40|>",
        language="en",
        audio_duration_s=2.4,
    )

    assert [
        (segment.id, segment.start, segment.end, segment.text)
        for segment in response.segments
    ] == [
        (0, 0.0, 1.2, "Hello there."),
        (1, 1.2, 2.4, "Goodbye."),
    ]
    assert response.text == "Hello there. Goodbye."


def test_postprocess_strips_whisper_timestamp_markers() -> None:
    adapter = WhisperASRAdapter()

    assert adapter.postprocess_text("<|0.00|> hello<|1.20|>") == "hello"


def test_verbose_json_falls_back_for_markerless_text() -> None:
    adapter = WhisperASRAdapter()

    response = adapter.build_verbose_response(
        text="plain transcript", language="en", audio_duration_s=3.5
    )

    assert [
        (segment.id, segment.start, segment.end, segment.text)
        for segment in response.segments
    ] == [(0, 0.0, 3.5, "plain transcript")]


def test_subtitle_response_rejects_markerless_text() -> None:
    adapter = WhisperASRAdapter()

    with pytest.raises(ValueError, match="model did not produce segment timestamps"):
        adapter.build_timestamped_response(
            text="plain transcript", language="en", audio_duration_s=3.5
        )


@pytest.mark.parametrize(
    "text",
    [
        "<|0.00|>hello<|1.20|> trailing unpaired text",
        "hello <|1.20|>",
        "<|0.00|> hello",
        "<|2.00|>foo<|1.00|>",
        "<|0.00|>hello<|1.20|><|1.00|>backwards<|2.00|>",
        "<|0.00|>hello<|1.20|><|1.20|>",
    ],
)
def test_subtitle_response_rejects_incomplete_timestamp_coverage(text: str) -> None:
    adapter = WhisperASRAdapter()

    with pytest.raises(ValueError, match="model did not produce segment timestamps"):
        adapter.build_timestamped_response(
            text=text, language="en", audio_duration_s=3.5
        )
