# SPDX-License-Identifier: Apache-2.0
"""GPU log-mel for the Qwen3-ASR encoder stream.

Cache-miss request building used to run the checkpoint feature extractor
on the host (STFT + mel) and then H2D the fbank. That CPU FFT is the unique-input
c=8–32 limiter. This module keeps the same numerics as the extractor's
``_torch_extract_fbank_features`` but runs them on the encoder stream from a
pinned waveform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

_DEFAULT_N_FFT = 400


@dataclass
class AudioFrontend:
    n_fft: int
    hop_length: int
    n_mels: int
    dither: float
    # Stored as HuggingFace does: (n_freq, n_mels). log_mel does filters.T @ mag.
    mel_filters: torch.Tensor
    hann_window: torch.Tensor | None = None

    def materialize(self, device: torch.device, dtype: torch.dtype) -> None:
        """Keep filters and the Hann window on ``waveform``'s device."""
        if self.mel_filters.device != device or self.mel_filters.dtype != dtype:
            self.mel_filters = self.mel_filters.to(device=device, dtype=dtype)
        if (
            self.hann_window is None
            or self.hann_window.device != device
            or self.hann_window.dtype != dtype
        ):
            self.hann_window = torch.hann_window(
                self.n_fft, device=device, dtype=dtype
            )


def bind_audio_frontend(model: Any, extractor: Any) -> AudioFrontend:
    """Attach hop length, mel filters, and Hann window from the checkpoint extractor.

    Fails at bind time so a missing extractor cannot reach the first
    unique-input request.
    """
    if extractor is None:
        raise ValueError("Qwen3-ASR GPU mel requires a feature extractor")
    try:
        hop_length = int(extractor.hop_length)
        filters = extractor.mel_filters
    except (AttributeError, TypeError) as exc:
        raise ValueError(
            "Qwen3-ASR feature extractor is missing hop_length or mel_filters"
        ) from exc
    if hop_length <= 0:
        raise ValueError(
            f"Qwen3-ASR feature extractor has an invalid hop length: {hop_length}"
        )
    if filters is None:
        raise ValueError("Qwen3-ASR feature extractor has no mel_filters")
    n_fft = int(getattr(extractor, "n_fft", _DEFAULT_N_FFT))
    if n_fft <= 0:
        raise ValueError(f"Qwen3-ASR feature extractor has an invalid n_fft: {n_fft}")
    mel_filters = torch.as_tensor(filters, dtype=torch.float32)
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = mel_filters.device
    else:
        mel_filters = mel_filters.to(device)
    frontend = AudioFrontend(
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=int(getattr(extractor, "feature_size", mel_filters.shape[-1])),
        dither=float(getattr(extractor, "dither", 0.0)),
        mel_filters=mel_filters,
        hann_window=torch.hann_window(
            n_fft, device=mel_filters.device, dtype=torch.float32
        ),
    )
    model._audio_frontend = frontend
    return frontend


def log_mel_spectrogram(
    waveform: torch.Tensor,
    frontend: AudioFrontend,
) -> torch.Tensor:
    """Return log-mel ``[n_mels, n_frames]`` on ``waveform``'s device.

    Matches the checkpoint feature extractor's fbank for a 1-D float32
    waveform: ``stft[..., :-1]``, Slaney mel, log10, max-8, (x+4)/4.
    """
    wave = waveform.reshape(-1).to(dtype=torch.float32)
    frontend.materialize(wave.device, wave.dtype)
    if frontend.dither != 0.0:
        wave = wave + frontend.dither * torch.randn(
            wave.shape, dtype=wave.dtype, device=wave.device
        )
    stft = torch.stft(
        wave,
        frontend.n_fft,
        frontend.hop_length,
        window=frontend.hann_window,
        return_complex=True,
    )
    # note (guozhihao-224): stft[..., :-1] drops the Nyquist bin, so frames
    # == samples // hop_length. the request builder estimates tokens from hop
    # length instead of running the CPU extractor, and must not pad or
    # truncate to a 30s window (transformers#26241).
    magnitudes = stft[..., :-1].abs() ** 2
    mel_spec = frontend.mel_filters.T @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0
