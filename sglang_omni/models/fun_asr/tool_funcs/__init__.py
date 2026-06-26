# SPDX-License-Identifier: Apache-2.0
"""Shared utility functions for Fun-ASR (audio length math, etc.)."""

from .audio_lengths import (
    fun_asr_audio_token_lengths,
    fun_asr_lfr_length,
    fun_asr_low_frame_rate_length,
    fun_asr_num_audio_tokens,
)

__all__ = [
    "fun_asr_audio_token_lengths",
    "fun_asr_lfr_length",
    "fun_asr_low_frame_rate_length",
    "fun_asr_num_audio_tokens",
]
