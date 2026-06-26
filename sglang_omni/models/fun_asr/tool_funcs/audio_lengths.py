# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

# Fun-ASR-Nano-2512 frontend constants (checkpoints/config.yaml, WavFrontend).
_LFR_M = 7  # stack 7 mel frames -> 560-dim (input_size); does not affect length
_LFR_N = 6  # LFR stride; output length = ceil(T / lfr_n)
# Number of stride-2 reductions in the low-frame-rate adaptor (model.py:404-407).
_LOW_FRAME_RATE_STAGES = 3


def fun_asr_lfr_length(mel_frames: int) -> int:
    """Length after the LFR frontend: ``ceil(mel_frames / lfr_n)``.

    ``lfr_m`` only widens the feature (7 * 80 = 560) and does not change the
    frame count, so the output length depends solely on the stride ``lfr_n``.
    """
    return (mel_frames + _LFR_N - 1) // _LFR_N


def fun_asr_low_frame_rate_length(lfr_frames: int) -> int:
    """Adaptor output length from LFR frames: three ``ceil(x / 2)`` reductions.

    Mirrors ``Fun-ASR/model.py``::

        olens = 1 + (speech_lengths - 3 + 2 * 1) // 2   # == ceil(x / 2)
        olens = 1 + (olens    - 3 + 2 * 1) // 2
        fake_token_len = (olens - 1) // 2 + 1

    where ``speech_lengths`` is the post-LFR frame count. Each stage is
    ``1 + (x - 1) // 2 == (x + 1) // 2 == ceil(x / 2)``.
    """
    out = lfr_frames
    for _ in range(_LOW_FRAME_RATE_STAGES):
        out = (out + 1) // 2  # ceil(out / 2)
    return out


def fun_asr_audio_token_lengths(input_lengths: Any) -> torch.Tensor:
    """Return Fun-ASR adaptor output lengths (audio token counts) for mel-frame lengths.

    Composes the LFR frontend (``ceil(T / 6)``) with the three-stage
    low-frame-rate adaptor (``ceil(x / 2)`` x3), matching the
    ``fake_token_len`` computation that sizes the audio-placeholder span spliced
    into the Qwen3 input embedding sequence in ``Fun-ASR/model.py``.
    """
    if not isinstance(input_lengths, torch.Tensor):
        input_lengths = torch.tensor(input_lengths)
    lfr_lengths = (input_lengths + _LFR_N - 1) // _LFR_N  # ceil(T / 6)
    tokens = lfr_lengths
    for _ in range(_LOW_FRAME_RATE_STAGES):
        tokens = (tokens + 1) // 2  # ceil(x / 2)
    return tokens


def fun_asr_num_audio_tokens(num_mel_frames: int) -> int:
    """Scalar wrapper for scheduler request construction."""
    return int(fun_asr_audio_token_lengths(num_mel_frames).item())


__all__ = [
    "fun_asr_lfr_length",
    "fun_asr_low_frame_rate_length",
    "fun_asr_audio_token_lengths",
    "fun_asr_num_audio_tokens",
]
