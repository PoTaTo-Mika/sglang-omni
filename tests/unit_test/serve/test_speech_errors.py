# SPDX-License-Identifier: Apache-2.0
"""Speech API error mapping helpers."""

from __future__ import annotations

from sglang_omni.serve.speech_errors import speech_generation_error


def test_speech_generation_error_maps_queue_full_to_503() -> None:
    err = speech_generation_error(RuntimeError("The request queue is full."))
    assert err.status_code == 503
    assert "queue is full" in err.message


def test_speech_generation_error_keeps_other_failures_as_500() -> None:
    err = speech_generation_error(RuntimeError("cuda out of memory"))
    assert err.status_code == 500
    assert "cuda out of memory" in err.message
