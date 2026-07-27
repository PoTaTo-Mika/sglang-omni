# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-compatible causal AudioVAEs used by VoxCPM1.5 and VoxCPM2."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

from sglang_omni.models.voxcpm_common.streaming_vae import (
    BatchedStreamingVAEDecoder,
)


def _get(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


class CausalConv1d(nn.Conv1d):
    def __init__(
        self, *args: Any, padding: int = 0, output_padding: int = 0, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.__output_padding = output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(
            F.pad(x, (self.__padding * 2 - self.__output_padding, 0))
        )


class CausalTransposeConv1d(nn.ConvTranspose1d):
    def __init__(
        self, *args: Any, padding: int = 0, output_padding: int = 0, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.__output_padding = output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trim = self.__padding * 2 - self.__output_padding
        output = super().forward(x)
        return output[..., :-trim] if trim else output


def _conv(*args: Any, **kwargs: Any) -> nn.Conv1d:
    return weight_norm(CausalConv1d(*args, **kwargs))


def _transpose(*args: Any, **kwargs: Any) -> nn.ConvTranspose1d:
    return weight_norm(CausalTransposeConv1d(*args, **kwargs))


def _snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).square()
    return x.reshape(shape)


class Snake1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _snake(x, self.alpha)


class CausalResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int, groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(dim),
            _conv(
                dim,
                dim,
                kernel_size=7,
                dilation=dilation,
                padding=3 * dilation,
                groups=groups,
            ),
            Snake1d(dim),
            _conv(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CausalEncoderBlock(nn.Module):
    def __init__(self, output_dim: int, stride: int, *, groups: int, v2: bool) -> None:
        super().__init__()
        input_dim = output_dim // 2
        self.block = nn.Sequential(
            *(
                CausalResidualUnit(input_dim, dilation, groups)
                for dilation in (1, 3, 9)
            ),
            Snake1d(input_dim),
            _conv(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2 if v2 else 0,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CausalEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        latent_dim: int,
        rates: list[int],
        *,
        depthwise: bool,
        v2: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [_conv(1, dim, kernel_size=7, padding=3)]
        for stride in rates:
            dim *= 2
            layers.append(
                CausalEncoderBlock(
                    dim,
                    stride,
                    groups=dim // 2 if depthwise else 1,
                    v2=v2,
                )
            )
        self.block = nn.Sequential(*layers)
        self.fc_mu = _conv(dim, latent_dim, kernel_size=3, padding=1)
        self.fc_logvar = _conv(dim, latent_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.block(x)
        return {
            "hidden_state": hidden,
            "mu": self.fc_mu(hidden),
            "logvar": self.fc_logvar(hidden),
        }


class NoiseBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = _conv(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype)
        return x + noise * self.linear(x)


class CausalDecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int,
        *,
        groups: int,
        use_noise_block: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            Snake1d(input_dim),
            _transpose(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        ]
        if use_noise_block:
            layers.append(NoiseBlock(output_dim))
        layers.extend(
            CausalResidualUnit(output_dim, dilation, groups) for dilation in (1, 3, 9)
        )
        self.block = nn.Sequential(*layers)
        self.input_channels = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SampleRateConditionLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        buckets: int,
        *,
        cond_type: str,
        cond_dim: int,
        out_layer: bool,
    ) -> None:
        super().__init__()
        self.cond_type = cond_type
        output_input = input_dim
        if cond_type in {"scale_bias", "scale_bias_init"}:
            self.scale_embed = nn.Embedding(buckets, input_dim)
            self.bias_embed = nn.Embedding(buckets, input_dim)
            if cond_type == "scale_bias":
                nn.init.ones_(self.scale_embed.weight)
                nn.init.zeros_(self.bias_embed.weight)
            else:
                nn.init.normal_(self.scale_embed.weight, mean=1)
                nn.init.normal_(self.bias_embed.weight)
        elif cond_type == "add":
            self.cond_embed = nn.Embedding(buckets, input_dim)
        elif cond_type == "concat":
            if not out_layer:
                raise ValueError("concat sample-rate conditioning needs out_layer")
            self.cond_embed = nn.Embedding(buckets, cond_dim)
            output_input += cond_dim
        else:
            raise ValueError(f"invalid AudioVAE conditioning type: {cond_type}")
        self.out_layer = (
            nn.Sequential(Snake1d(output_input), _conv(output_input, input_dim, 1))
            if out_layer
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if self.cond_type in {"scale_bias", "scale_bias_init"}:
            x = x * self.scale_embed(condition).unsqueeze(-1) + self.bias_embed(
                condition
            ).unsqueeze(-1)
        elif self.cond_type == "add":
            x = x + self.cond_embed(condition).unsqueeze(-1)
        else:
            cond = self.cond_embed(condition).unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = torch.cat((x, cond), 1)
        return self.out_layer(x)


class CausalDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        channels: int,
        rates: list[int],
        *,
        depthwise: bool,
        use_noise_block: bool,
        sr_boundaries: list[int] | None,
        cond_type: str,
        cond_dim: int,
        cond_out_layer: bool,
    ) -> None:
        super().__init__()
        if depthwise:
            layers: list[nn.Module] = [
                _conv(
                    latent_dim,
                    latent_dim,
                    kernel_size=7,
                    padding=3,
                    groups=latent_dim,
                ),
                _conv(latent_dim, channels, kernel_size=1),
            ]
        else:
            layers = [_conv(latent_dim, channels, kernel_size=7, padding=3)]
        for index, stride in enumerate(rates):
            input_dim = channels // 2**index
            output_dim = channels // 2 ** (index + 1)
            layers.append(
                CausalDecoderBlock(
                    input_dim,
                    output_dim,
                    stride,
                    groups=output_dim if depthwise else 1,
                    use_noise_block=use_noise_block,
                )
            )
        layers.extend(
            (
                Snake1d(output_dim),
                _conv(output_dim, 1, kernel_size=7, padding=3),
                nn.Tanh(),
            )
        )
        if sr_boundaries is None:
            self.sr_bin_boundaries = None
            self.sr_bin_buckets = None
            self.model: nn.Module = nn.Sequential(*layers)
            self.sr_cond_model = None
            self.default_sr_idx = None
        else:
            self.model = nn.ModuleList(layers)
            self.register_buffer(
                "sr_bin_boundaries",
                torch.tensor(sr_boundaries, dtype=torch.int32),
            )
            buckets = len(sr_boundaries) + 1
            self.sr_bin_buckets = buckets
            self.default_sr_idx = len(sr_boundaries)
            self.sr_cond_model = nn.ModuleList(
                [
                    SampleRateConditionLayer(
                        layer.input_channels,
                        buckets,
                        cond_type=cond_type,
                        cond_dim=cond_dim,
                        out_layer=cond_out_layer,
                    )
                    if isinstance(layer, CausalDecoderBlock)
                    else None
                    for layer in layers
                ]
            )

    def forward(
        self, x: torch.Tensor, sr_cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.sr_bin_boundaries is None:
            return self.model(x)
        if sr_cond is None:
            sr_cond = torch.full(
                (x.shape[0],),
                self.default_sr_idx,
                dtype=torch.long,
                device=x.device,
            )
        for layer, conditioner in zip(self.model, self.sr_cond_model):
            if conditioner is not None:
                x = conditioner(x, sr_cond)
            x = layer(x)
        return x


class AudioVAE(nn.Module):
    """V1 AudioVAE. Defaults match the official VoxCPM1.5 checkpoint."""

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        super().__init__()
        values = dict(kwargs)
        encoder_dim = int(values.get("encoder_dim", _get(config, "encoder_dim", 128)))
        encoder_rates = list(
            values.get("encoder_rates", _get(config, "encoder_rates", [2, 5, 8, 8]))
        )
        self.latent_dim = int(values.get("latent_dim", _get(config, "latent_dim", 64)))
        decoder_dim = int(values.get("decoder_dim", _get(config, "decoder_dim", 1536)))
        decoder_rates = list(
            values.get("decoder_rates", _get(config, "decoder_rates", [8, 8, 5, 2]))
        )
        depthwise = bool(values.get("depthwise", _get(config, "depthwise", True)))
        use_noise = bool(
            values.get("use_noise_block", _get(config, "use_noise_block", False))
        )
        self.sample_rate = int(
            values.get("sample_rate", _get(config, "sample_rate", 16000))
        )
        self.out_sample_rate = self.sample_rate
        self.encoder_chunk_size = math.prod(encoder_rates)
        self.hop_length = self.encoder_chunk_size
        self.decoder_chunk_size = math.prod(decoder_rates)
        self.chunk_size = self.decoder_chunk_size
        self.encoder = CausalEncoder(
            encoder_dim,
            self.latent_dim,
            encoder_rates,
            depthwise=depthwise,
            v2=False,
        )
        self.decoder = CausalDecoder(
            self.latent_dim,
            decoder_dim,
            decoder_rates,
            depthwise=depthwise,
            use_noise_block=use_noise,
            sr_boundaries=None,
            cond_type="scale_bias",
            cond_dim=128,
            cond_out_layer=False,
        )

    def preprocess(self, audio: torch.Tensor, sample_rate: int | None) -> torch.Tensor:
        if sample_rate is not None and int(sample_rate) != self.sample_rate:
            raise ValueError(f"AudioVAE expects {self.sample_rate} Hz audio")
        pad = (-audio.shape[-1]) % self.encoder_chunk_size
        return F.pad(audio, (0, pad))

    @torch.inference_mode()
    def encode(
        self, audio: torch.Tensor, sample_rate: int | None = None
    ) -> torch.Tensor:
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        return self.encoder(self.preprocess(audio, sample_rate))["mu"]

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def streaming_decoder(self) -> BatchedStreamingVAEDecoder:
        return BatchedStreamingVAEDecoder(self, CausalConv1d, CausalTransposeConv1d)


class AudioVAEV2(AudioVAE):
    """V2 AudioVAE with 48 kHz sample-rate-conditioned decoder."""

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        nn.Module.__init__(self)
        values = dict(kwargs)
        encoder_dim = int(values.get("encoder_dim", _get(config, "encoder_dim", 128)))
        encoder_rates = list(
            values.get("encoder_rates", _get(config, "encoder_rates", [2, 5, 8, 8]))
        )
        self.latent_dim = int(values.get("latent_dim", _get(config, "latent_dim", 64)))
        decoder_dim = int(values.get("decoder_dim", _get(config, "decoder_dim", 2048)))
        decoder_rates = list(
            values.get(
                "decoder_rates", _get(config, "decoder_rates", [8, 6, 5, 2, 2, 2])
            )
        )
        depthwise = bool(values.get("depthwise", _get(config, "depthwise", True)))
        use_noise = bool(
            values.get("use_noise_block", _get(config, "use_noise_block", False))
        )
        self.sample_rate = int(
            values.get("sample_rate", _get(config, "sample_rate", 16000))
        )
        self.out_sample_rate = int(
            values.get("out_sample_rate", _get(config, "out_sample_rate", 48000))
        )
        self.sr_bin_boundaries = values.get(
            "sr_bin_boundaries",
            _get(config, "sr_bin_boundaries", [20000, 30000, 40000]),
        )
        self.encoder_chunk_size = math.prod(encoder_rates)
        self.hop_length = self.encoder_chunk_size
        self.decoder_chunk_size = math.prod(decoder_rates)
        self.chunk_size = self.decoder_chunk_size
        self.encoder = CausalEncoder(
            encoder_dim,
            self.latent_dim,
            encoder_rates,
            depthwise=depthwise,
            v2=True,
        )
        self.decoder = CausalDecoder(
            self.latent_dim,
            decoder_dim,
            decoder_rates,
            depthwise=depthwise,
            use_noise_block=use_noise,
            sr_boundaries=self.sr_bin_boundaries,
            cond_type=str(
                values.get("cond_type", _get(config, "cond_type", "scale_bias"))
            ),
            cond_dim=int(values.get("cond_dim", _get(config, "cond_dim", 128))),
            cond_out_layer=bool(
                values.get("cond_out_layer", _get(config, "cond_out_layer", False))
            ),
        )

    @torch.inference_mode()
    def decode(
        self, latents: torch.Tensor, sr_cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.decoder(latents, sr_cond)


def create_audio_vae(config: Any, *, version: str) -> AudioVAE:
    """Construct the checkpoint-native VAE from a top-level official config."""

    vae_config = _get(config, "audio_vae_config", None)
    if version == "v1":
        return AudioVAE(config=vae_config)
    if version == "v2":
        return AudioVAEV2(config=vae_config)
    raise ValueError(f"unsupported VoxCPM version: {version!r}")


__all__ = [
    "AudioVAE",
    "AudioVAEV2",
    "CausalConv1d",
    "CausalTransposeConv1d",
    "create_audio_vae",
]
