# SPDX-License-Identifier: Apache-2.0
"""Shared SGLang-native model core for VoxCPM and VoxCPM2."""

from __future__ import annotations

import copy
import math
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.voxcpm_common.state_pool import VoxCPMRequestStatePool

try:
    from sglang_omni.vendor.sglang.layers import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RMSNorm,
        RadixAttention,
        RowParallelLinear,
        VocabParallelEmbedding,
    )
except ModuleNotFoundError as error:
    if error.name != "sglang":
        raise

    class _FallbackLinear(nn.Linear):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
            return super().forward(x), None

    class QKVParallelLinear(_FallbackLinear):  # type: ignore[no-redef]
        def __init__(
            self,
            hidden: int,
            head_dim: int,
            heads: int,
            kv_heads: int,
            **kwargs: Any,
        ) -> None:
            self.num_heads, self.num_kv_heads = heads, kv_heads
            super().__init__(
                hidden,
                (heads + 2 * kv_heads) * head_dim,
                bias=bool(kwargs.get("bias", False)),
            )

    class MergedColumnParallelLinear(_FallbackLinear):  # type: ignore[no-redef]
        def __init__(self, in_features: int, outputs: list[int], **kwargs: Any) -> None:
            super().__init__(
                in_features, sum(outputs), bias=bool(kwargs.get("bias", False))
            )

    class RowParallelLinear(_FallbackLinear):  # type: ignore[no-redef]
        def __init__(self, in_features: int, out_features: int, **kwargs: Any) -> None:
            super().__init__(
                in_features, out_features, bias=bool(kwargs.get("bias", False))
            )

    class VocabParallelEmbedding(nn.Embedding):  # type: ignore[no-redef]
        def __init__(self, vocab: int, hidden: int, **_: Any) -> None:
            super().__init__(vocab, hidden)

    class RMSNorm(nn.Module):  # type: ignore[no-redef]
        def __init__(self, hidden: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            normalized = x.float() * torch.rsqrt(
                x.float().square().mean(dim=-1, keepdim=True) + self.eps
            )
            return (normalized * self.weight.float()).to(x.dtype)

    class RadixAttention(nn.Module):  # type: ignore[no-redef]
        """CPU-only import/smoke fallback; it does not implement a radix cache."""

        def __init__(
            self,
            heads: int,
            head_dim: int,
            scaling: float,
            *,
            num_kv_heads: int,
            **_: Any,
        ) -> None:
            super().__init__()
            self.heads, self.kv_heads, self.head_dim = heads, num_kv_heads, head_dim
            self.scaling = scaling

        def forward(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, _: Any
        ) -> torch.Tensor:
            tokens = q.shape[0]
            q = q.view(tokens, self.heads, self.head_dim).transpose(0, 1)[None]
            k = k.view(tokens, self.kv_heads, self.head_dim).transpose(0, 1)[None]
            v = v.view(tokens, self.kv_heads, self.head_dim).transpose(0, 1)[None]
            if self.kv_heads != self.heads:
                repeat = self.heads // self.kv_heads
                k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
            output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return output[0].transpose(0, 1).reshape(tokens, -1)


VoxCPMVersion = Literal["v1", "v2"]


def cfg_get(config: Any, name: str, default: Any = None) -> Any:
    value = (
        config.get(name, default)
        if isinstance(config, dict)
        else getattr(config, name, default)
    )
    return default if value is None else value


def clone_config(config: Any, **updates: Any) -> Any:
    if hasattr(config, "model_copy"):
        result = config.model_copy(deep=True)
    else:
        result = copy.deepcopy(config)
    for name, value in updates.items():
        if isinstance(result, dict):
            result[name] = value
        else:
            setattr(result, name, value)
    return result


def _rope_field(rope_scaling: Any, name: str, default: Any) -> Any:
    if rope_scaling is None:
        return default
    return cfg_get(rope_scaling, name, default)


class MiniCPMLongRoPE(nn.Module):
    """LongRoPE preserving the checkpoint-specific V1/V2 cache math."""

    def __init__(self, config: Any, *, version: VoxCPMVersion) -> None:
        super().__init__()
        heads = int(cfg_get(config, "num_attention_heads"))
        self.dim = int(
            cfg_get(config, "kv_channels", 0) or cfg_get(config, "hidden_size") // heads
        )
        self.max_position = int(cfg_get(config, "max_position_embeddings", 32768))
        self.base = float(cfg_get(config, "rope_theta", 10000.0))
        scaling = cfg_get(config, "rope_scaling")
        original = int(
            _rope_field(scaling, "original_max_position_embeddings", self.max_position)
        )
        short = _rope_field(scaling, "short_factor", [1.0] * (self.dim // 2))
        long = _rope_field(scaling, "long_factor", [1.0] * (self.dim // 2))
        if len(short) != self.dim // 2 or len(long) != self.dim // 2:
            raise ValueError("LongRoPE factors must contain head_dim / 2 entries")
        scale = self.max_position / original
        amplitude = math.sqrt(1 + math.log(scale) / math.log(original))
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        factors = torch.tensor(
            long if self.max_position > original else short, dtype=torch.float32
        )
        positions = torch.arange(self.max_position, dtype=torch.float32)
        if version == "v1":
            # V1 casts inv_freq to the target cache dtype before multiplication.
            freqs = torch.outer(positions, 1.0 / factors) * inv_freq.float()
        else:
            # Kept separate intentionally: official V2 first materializes the
            # factor-adjusted outer product and then multiplies inv_freq.
            freqs = torch.mul(torch.outer(positions, 1.0 / factors), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos() * amplitude, persistent=False)
        self.register_buffer("sin_cached", emb.sin() * amplitude, persistent=False)

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.numel() and int(positions.max()) >= self.max_position:
            raise ValueError("position exceeds configured LongRoPE cache")
        cos = self.cos_cached[positions].unsqueeze(1)
        sin = self.sin_cached[positions].unsqueeze(1)

        def apply(x: torch.Tensor) -> torch.Tensor:
            shape, dtype = x.shape, x.dtype
            x = x.reshape(positions.numel(), -1, self.dim).float()
            first, second = x.chunk(2, dim=-1)
            rotated = torch.cat((-second, first), dim=-1)
            return (x * cos + rotated * sin).to(dtype).reshape(shape)

        return apply(query), apply(key)


class CausalMiniCPMAttention(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        layer_id: int,
        version: VoxCPMVersion,
        use_rope: bool,
        prefix: str,
        quant_config: Any = None,
    ) -> None:
        super().__init__()
        hidden = int(cfg_get(config, "hidden_size"))
        total_heads = int(cfg_get(config, "num_attention_heads"))
        total_kv_heads = int(cfg_get(config, "num_key_value_heads", total_heads))
        self.head_dim = int(cfg_get(config, "kv_channels", 0) or hidden // total_heads)
        self.total_heads = total_heads
        self.total_kv_heads = total_kv_heads
        self.qkv_proj = QKVParallelLinear(
            hidden,
            self.head_dim,
            total_heads,
            total_kv_heads,
            bias=bool(cfg_get(config, "attention_bias", False)),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            total_heads * self.head_dim,
            hidden,
            bias=bool(cfg_get(config, "attention_bias", False)),
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        try:
            from sglang.srt.distributed import get_tensor_model_parallel_world_size

            tp_size = int(get_tensor_model_parallel_world_size())
        except (ModuleNotFoundError, AssertionError):
            # CPU config/tests may have SGLang installed without an initialized
            # tensor-parallel process group.
            tp_size = 1
        if total_heads % tp_size or total_kv_heads % tp_size:
            raise ValueError("MiniCPM attention heads must be divisible by TP size")
        self.num_heads = total_heads // tp_size
        self.num_kv_heads = total_kv_heads // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.rotary_emb = MiniCPMLongRoPE(config, version=version) if use_rope else None
        self.q_norm = (
            RMSNorm(self.head_dim, eps=float(cfg_get(config, "rms_norm_eps", 1e-6)))
            if bool(cfg_get(config, "apply_qk_norm", False))
            else None
        )
        self.k_norm = (
            RMSNorm(self.head_dim, eps=float(cfg_get(config, "rms_norm_eps", 1e-6)))
            if self.q_norm is not None
            else None
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: Any,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        if self.q_norm is not None:
            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view_as(q)
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view_as(k)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(output)
        return output


class CausalMiniCPMMLP(nn.Module):
    def __init__(self, config: Any, *, prefix: str, quant_config: Any = None) -> None:
        super().__init__()
        hidden = int(cfg_get(config, "hidden_size"))
        intermediate = int(cfg_get(config, "intermediate_size"))
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden,
            [intermediate, intermediate],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate,
            hidden,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        output, _ = self.down_proj(F.silu(gate) * up)
        return output


class CausalMiniCPMLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        layer_id: int,
        version: VoxCPMVersion,
        use_rope: bool,
        prefix: str,
        quant_config: Any = None,
    ) -> None:
        super().__init__()
        hidden = int(cfg_get(config, "hidden_size"))
        eps = float(cfg_get(config, "rms_norm_eps", 1e-6))
        self.self_attn = CausalMiniCPMAttention(
            config,
            layer_id=layer_id,
            version=version,
            use_rope=use_rope,
            prefix=f"{prefix}.self_attn",
            quant_config=quant_config,
        )
        self.mlp = CausalMiniCPMMLP(
            config, prefix=f"{prefix}.mlp", quant_config=quant_config
        )
        self.input_layernorm = RMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, forward_batch: Any
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            positions, hidden_states, forward_batch
        )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class CausalMiniCPMModel(nn.Module):
    """MiniCPM4 LM whose attention is entirely SGLang RadixAttention."""

    def __init__(
        self,
        config: Any,
        *,
        version: VoxCPMVersion,
        layer_offset: int,
        use_rope: bool = True,
        prefix: str,
        quant_config: Any = None,
    ) -> None:
        super().__init__()
        self.config = config
        vocab_size = int(cfg_get(config, "vocab_size", 0))
        hidden = int(cfg_get(config, "hidden_size"))
        self.embed_tokens: nn.Module = (
            VocabParallelEmbedding(
                vocab_size,
                hidden,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
            if vocab_size > 0
            else nn.Identity()
        )
        self.layers = nn.ModuleList(
            CausalMiniCPMLayer(
                config,
                layer_id=layer_offset + index,
                version=version,
                use_rope=use_rope,
                prefix=f"{prefix}.layers.{index}",
                quant_config=quant_config,
            )
            for index in range(int(cfg_get(config, "num_hidden_layers")))
        )
        self.norm = RMSNorm(hidden, eps=float(cfg_get(config, "rms_norm_eps", 1e-6)))

    def forward(
        self,
        input_embeds: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
    ) -> torch.Tensor:
        hidden_states = input_embeds
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, forward_batch)
        return self.norm(hidden_states)


class LocalMiniCPMModel(nn.Module):
    """Bidirectional MiniCPM block for LocEnc/DiT (outside radix KV cache)."""

    def __init__(self, config: Any, *, version: VoxCPMVersion) -> None:
        super().__init__()
        hidden = int(cfg_get(config, "hidden_size"))
        heads = int(cfg_get(config, "num_attention_heads"))
        kv_heads = int(cfg_get(config, "num_key_value_heads", heads))
        head_dim = int(cfg_get(config, "kv_channels", 0) or hidden // heads)
        intermediate = int(cfg_get(config, "intermediate_size"))
        eps = float(cfg_get(config, "rms_norm_eps", 1e-6))
        self.layers = nn.ModuleList(
            _LocalLayer(
                config,
                hidden,
                heads,
                kv_heads,
                head_dim,
                intermediate,
                eps,
                version=version,
            )
            for _ in range(int(cfg_get(config, "num_hidden_layers")))
        )
        self.norm = RMSNorm(hidden, eps=eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class _LocalAttention(nn.Module):
    def __init__(
        self,
        config: Any,
        hidden: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        *,
        version: VoxCPMVersion,
    ) -> None:
        super().__init__()
        if heads % kv_heads:
            raise ValueError("local attention heads must be divisible by KV heads")
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.rotary_emb = MiniCPMLongRoPE(config, version=version)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        positions = torch.arange(length, device=x.device, dtype=torch.long).repeat(
            batch
        )
        query, key = self.rotary_emb(
            positions,
            query.reshape(batch * length, -1),
            key.reshape(batch * length, -1),
        )
        query = query.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.kv_heads != self.heads:
            repeats = self.heads // self.kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        value = F.scaled_dot_product_attention(query, key, value)
        return self.o_proj(value.transpose(1, 2).reshape(batch, length, -1))


class _LocalLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        hidden: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        intermediate: int,
        eps: float,
        *,
        version: VoxCPMVersion,
    ) -> None:
        super().__init__()
        self.self_attn = _LocalAttention(
            config,
            hidden,
            heads,
            kv_heads,
            head_dim,
            version=version,
        )
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.mlp.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.mlp.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.input_layernorm = RMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        normed = self.post_attention_layernorm(x)
        return x + self.mlp.down_proj(
            F.silu(self.mlp.gate_proj(normed)) * self.mlp.up_proj(normed)
        )


class VoxCPMLocEnc(nn.Module):
    def __init__(self, config: Any, feat_dim: int, *, version: VoxCPMVersion) -> None:
        super().__init__()
        hidden = int(cfg_get(config, "hidden_size"))
        self.special_token = nn.Parameter(torch.empty(1, 1, 1, hidden))
        self.in_proj = nn.Linear(feat_dim, hidden)
        self.encoder = LocalMiniCPMModel(config, version=version)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 3:
            raise ValueError("features must have shape [tokens, patch, feat_dim]")
        tokens = feat.shape[0]
        special = self.special_token[0].expand(tokens, 1, -1)
        hidden = torch.cat((special, self.in_proj(feat)), dim=1)
        return self.encoder(hidden)[:, 0]


class ScalarQuantizationLayer(nn.Module):
    def __init__(self, hidden: int, latent_dim: int, scale: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden, latent_dim)
        self.out_proj = nn.Linear(latent_dim, hidden)
        self.scale = scale

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        quantized = (
            torch.round(torch.tanh(self.in_proj(hidden)) * self.scale) / self.scale
        )
        return self.out_proj(quantized)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("sinusoidal embedding dimension must be even")
        self.dim = dim

    def forward(self, value: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        value = value.reshape(-1)
        half = self.dim // 2
        frequencies = torch.exp(
            torch.arange(half, device=value.device, dtype=value.dtype)
            * -(math.log(10000.0) / (half - 1))
        )
        phase = scale * value[:, None] * frequencies[None]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class _TimestepEmbedding(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(hidden, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(x)))


class VoxCPMLocDiT(nn.Module):
    def __init__(self, config: Any, feat_dim: int, *, version: VoxCPMVersion) -> None:
        super().__init__()
        hidden = int(cfg_get(config, "hidden_size"))
        self.version = version
        self.in_proj = nn.Linear(feat_dim, hidden)
        self.cond_proj = nn.Linear(feat_dim, hidden)
        self.out_proj = nn.Linear(hidden, feat_dim)
        self.time_embeddings = SinusoidalPosEmb(hidden)
        self.time_mlp = _TimestepEmbedding(hidden)
        self.delta_time_mlp = _TimestepEmbedding(hidden)
        self.decoder = LocalMiniCPMModel(config, version=version)

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        time: torch.Tensor,
        cond: torch.Tensor,
        delta_time: torch.Tensor,
    ) -> torch.Tensor:
        x_hidden = self.in_proj(x.transpose(1, 2))
        cond_hidden = self.cond_proj(cond.transpose(1, 2))
        time_hidden = self.time_mlp(self.time_embeddings(time).to(x_hidden.dtype))
        time_hidden += self.delta_time_mlp(
            self.time_embeddings(delta_time).to(x_hidden.dtype)
        )
        mu = mu.view(x.shape[0], -1, x_hidden.shape[-1])
        if self.version == "v1":
            if mu.shape[1] != 1:
                raise ValueError("V1 DiT expects one conditioning token")
            prefix = (mu[:, 0] + time_hidden)[:, None]
        else:
            prefix = torch.cat((mu, time_hidden[:, None]), dim=1)
        hidden = torch.cat((prefix, cond_hidden, x_hidden), dim=1)
        hidden = self.decoder(hidden)
        start = prefix.shape[1] + cond_hidden.shape[1]
        return self.out_proj(hidden[:, start:]).transpose(1, 2).contiguous()


class UnifiedCFM(nn.Module):
    def __init__(
        self,
        estimator: VoxCPMLocDiT,
        *,
        feat_dim: int,
        patch_size: int,
        inference_timesteps: int,
        mean_mode: bool,
        version: VoxCPMVersion,
    ) -> None:
        super().__init__()
        if inference_timesteps < 1:
            raise ValueError("inference_timesteps must be positive")
        self.estimator = estimator
        self.feat_dim = feat_dim
        self.patch_size = patch_size
        self.inference_timesteps = inference_timesteps
        self.mean_mode = mean_mode
        self.version = version

    def forward(
        self,
        mu: torch.Tensor,
        cond: torch.Tensor,
        temperature: torch.Tensor,
        cfg_value: torch.Tensor,
        z_noise: torch.Tensor | None = None,
        *,
        inference_timesteps: int | None = None,
    ) -> torch.Tensor:
        batch = mu.shape[0]
        noise = (
            torch.randn(
                batch,
                self.feat_dim,
                self.patch_size,
                device=mu.device,
                dtype=mu.dtype,
            )
            if z_noise is None
            else z_noise
        )
        x = noise * temperature[:, None, None]
        solver_steps = int(inference_timesteps or self.inference_timesteps)
        if solver_steps < 1:
            raise ValueError("inference_timesteps must be positive")
        t_span = torch.linspace(
            1, 0, solver_steps + 1, device=mu.device, dtype=mu.dtype
        )
        t_span = t_span + (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        time, dt = t_span[0], t_span[0] - t_span[1]
        zero_steps = max(1, int(len(t_span) * 0.04))
        for step in range(1, len(t_span)):
            if step > zero_steps:
                x_in = torch.cat((x, x), dim=0)
                mu_in = torch.cat((mu, torch.zeros_like(mu)), dim=0)
                cond_in = torch.cat((cond, cond), dim=0)
                time_in = time.expand(2 * batch)
                dt_in = dt.expand(2 * batch)
                if not self.mean_mode:
                    dt_in = torch.zeros_like(dt_in)
                positive, negative = self.estimator(
                    x_in, mu_in, time_in, cond_in, dt_in
                ).chunk(2)
                pos_flat, neg_flat = positive.flatten(1), negative.flatten(1)
                optimal = (pos_flat * neg_flat).sum(1) / (
                    neg_flat.square().sum(1) + 1e-8
                )
                optimal = optimal[:, None, None]
                velocity = negative * optimal + cfg_value[:, None, None] * (
                    positive - negative * optimal
                )
            else:
                velocity = 0.0
            x = x - dt * velocity
            time = time - dt
            if step < len(t_span) - 1:
                dt = time - t_span[step + 1]
        return x


class VoxCPMCore(nn.Module):
    """Shared generation core; wrappers provide the SGLang public forward."""

    def __init__(
        self,
        config: Any,
        *,
        version: VoxCPMVersion,
        inference_timesteps: int,
        max_requests: int,
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config, self.version = config, version
        lm_cfg = cfg_get(config, "lm_config")
        if bool(cfg_get(lm_cfg, "use_mup", False)):
            raise ValueError("MiniCPM4 MuP inference is not supported")
        self.hidden_size = int(cfg_get(lm_cfg, "hidden_size"))
        self.feat_dim = int(cfg_get(config, "feat_dim", 64))
        self.patch_size = int(
            cfg_get(config, "patch_size", 2 if version == "v1" else 4)
        )
        base_layers = int(cfg_get(lm_cfg, "num_hidden_layers"))
        self.base_lm = CausalMiniCPMModel(
            lm_cfg,
            version=version,
            layer_offset=0,
            prefix=f"{prefix}.base_lm".strip("."),
            quant_config=quant_config,
        )
        residual_cfg = clone_config(
            lm_cfg,
            num_hidden_layers=int(cfg_get(config, "residual_lm_num_layers")),
            vocab_size=0,
        )
        self.residual_lm = CausalMiniCPMModel(
            residual_cfg,
            version=version,
            layer_offset=base_layers,
            use_rope=not (
                version == "v2" and bool(cfg_get(config, "residual_lm_no_rope", False))
            ),
            prefix=f"{prefix}.residual_lm".strip("."),
            quant_config=quant_config,
        )
        enc_cfg_src = cfg_get(config, "encoder_config")
        enc_cfg = clone_config(
            lm_cfg,
            hidden_size=int(cfg_get(enc_cfg_src, "hidden_dim")),
            intermediate_size=int(cfg_get(enc_cfg_src, "ffn_dim")),
            num_attention_heads=int(cfg_get(enc_cfg_src, "num_heads")),
            num_key_value_heads=int(cfg_get(lm_cfg, "num_key_value_heads")),
            num_hidden_layers=int(cfg_get(enc_cfg_src, "num_layers")),
            kv_channels=cfg_get(enc_cfg_src, "kv_channels"),
            vocab_size=0,
        )
        dit_cfg_src = cfg_get(config, "dit_config")
        dit_hidden = int(cfg_get(dit_cfg_src, "hidden_dim"))
        dit_cfg = clone_config(
            lm_cfg,
            hidden_size=dit_hidden,
            intermediate_size=int(cfg_get(dit_cfg_src, "ffn_dim")),
            num_attention_heads=int(cfg_get(dit_cfg_src, "num_heads")),
            num_key_value_heads=int(cfg_get(lm_cfg, "num_key_value_heads")),
            num_hidden_layers=int(cfg_get(dit_cfg_src, "num_layers")),
            kv_channels=cfg_get(dit_cfg_src, "kv_channels"),
            vocab_size=0,
        )
        self.feat_encoder = VoxCPMLocEnc(enc_cfg, self.feat_dim, version=version)
        self.enc_to_lm_proj = nn.Linear(
            int(cfg_get(enc_cfg_src, "hidden_dim")), self.hidden_size
        )
        self.fsq_layer = ScalarQuantizationLayer(
            self.hidden_size,
            int(cfg_get(config, "scalar_quantization_latent_dim")),
            int(cfg_get(config, "scalar_quantization_scale", 9)),
        )
        self.lm_to_dit_proj = nn.Linear(self.hidden_size, dit_hidden)
        self.res_to_dit_proj = nn.Linear(self.hidden_size, dit_hidden)
        self.fusion_concat_proj = (
            nn.Linear(2 * self.hidden_size, self.hidden_size)
            if version == "v2"
            else None
        )
        self.stop_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(self.hidden_size, 2, bias=False)
        estimator = VoxCPMLocDiT(dit_cfg, self.feat_dim, version=version)
        self.feat_decoder = UnifiedCFM(
            estimator,
            feat_dim=self.feat_dim,
            patch_size=self.patch_size,
            inference_timesteps=inference_timesteps,
            mean_mode=bool(
                cfg_get(
                    dit_cfg_src,
                    "mean_mode",
                    cfg_get(config, "dit_mean_mode", False),
                )
            ),
            version=version,
        )
        weight = next(self.base_lm.embed_tokens.parameters())
        self.state_pool = VoxCPMRequestStatePool(
            max_requests,
            self.hidden_size,
            self.patch_size,
            self.feat_dim,
            device=weight.device,
            dtype=weight.dtype,
        )
        self.vae: nn.Module | None = None
        self.vae_streaming_decoder: Any = None

    def acquire(self, rid: str) -> int:
        return self.state_pool.acquire(rid)

    acquire_row = acquire

    def reset(self, rid: str) -> None:
        self.state_pool.reset(rid)

    reset_request = reset
    release_row = reset

    def row_for(self, rid: str) -> int | None:
        return self.state_pool.row_for(rid)

    def forward_backbone(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        *,
        input_embeds: torch.Tensor | None,
        feat: torch.Tensor | None,
        feat_mask: torch.Tensor | None,
        row_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        decode = bool(forward_batch.forward_mode.is_decode())
        if decode:
            rows = input_ids.reshape(-1) if row_ids is None else row_ids.reshape(-1)
            rows = rows.to(device=self.state_pool.feedback.device, dtype=torch.long)
            base_inputs = self.state_pool.feedback_for(rows)
            feat = self.state_pool.latents.index_select(0, rows)
            feat_mask = torch.ones(rows.numel(), device=feat.device, dtype=torch.bool)
        else:
            if input_embeds is None:
                raise ValueError("VoxCPM prefill requires continuous input_embeds")
            base_inputs = input_embeds
            if feat is None:
                feat = torch.zeros(
                    base_inputs.shape[0],
                    self.patch_size,
                    self.feat_dim,
                    device=base_inputs.device,
                    dtype=base_inputs.dtype,
                )
            if feat_mask is None:
                feat_mask = torch.zeros(
                    feat.shape[0], device=feat.device, dtype=torch.bool
                )
        if feat.shape != (base_inputs.shape[0], self.patch_size, self.feat_dim):
            raise ValueError("feat must have shape [tokens, patch_size, feat_dim]")
        feat_embeds = self.enc_to_lm_proj(self.feat_encoder(feat))
        feat_embeds = feat_embeds.masked_fill(~feat_mask[:, None], 0)
        base_outputs = self.base_lm(base_inputs, positions, forward_batch)
        base_outputs = torch.where(
            feat_mask[:, None], self.fsq_layer(base_outputs), base_outputs
        )
        if self.version == "v1":
            residual_inputs = base_outputs + feat_embeds
        else:
            residual_inputs = self.fusion_concat_proj(
                torch.cat((base_outputs, feat_embeds), dim=-1)
            )
        residual_outputs = self.residual_lm(residual_inputs, positions, forward_batch)
        sample_indices = self._sample_indices(forward_batch, base_outputs.device)
        sampled_feat = feat.index_select(0, sample_indices)
        sampled = torch.cat(
            (
                base_outputs.index_select(0, sample_indices),
                residual_outputs.index_select(0, sample_indices),
            ),
            dim=-1,
        )
        if row_ids is not None:
            rows = row_ids.reshape(-1).to(self.state_pool.latents.device, torch.long)
            if rows.numel() != sampled.shape[0]:
                raise ValueError("row_ids must match the number of sampled requests")
            self.state_pool.latents.index_copy_(
                0, rows, sampled_feat.to(self.state_pool.latents)
            )
        return sampled

    @staticmethod
    def _sample_indices(forward_batch: Any, device: torch.device) -> torch.Tensor:
        if not bool(forward_batch.forward_mode.is_extend()):
            batch = int(
                getattr(forward_batch, "batch_size", 0) or len(forward_batch.seq_lens)
            )
            return torch.arange(batch, device=device)
        lengths = getattr(forward_batch, "extend_seq_lens", None)
        if lengths is None:
            return torch.tensor([forward_batch.input_ids.numel() - 1], device=device)
        return torch.cumsum(lengths.to(device=device, dtype=torch.long), dim=0) - 1

    @torch.no_grad()
    def generate_batch(
        self,
        hidden_states: torch.Tensor,
        rows: torch.Tensor,
        *,
        temperature: torch.Tensor | None = None,
        cfg_value: torch.Tensor | None = None,
        z_noise: torch.Tensor | None = None,
        inference_timesteps: torch.Tensor | None = None,
        decode_vae: bool = False,
    ) -> dict[str, torch.Tensor]:
        rows = rows.to(self.state_pool.feedback.device, torch.long).reshape(-1)
        if hidden_states.shape != (rows.numel(), 2 * self.hidden_size):
            raise ValueError("hidden_states must be [batch, 2 * hidden_size]")
        lm_hidden, residual_hidden = hidden_states.chunk(2, dim=-1)
        mu_left = self.lm_to_dit_proj(lm_hidden)
        mu_right = self.res_to_dit_proj(residual_hidden)
        mu = (
            mu_left + mu_right
            if self.version == "v1"
            else torch.cat((mu_left, mu_right), dim=-1)
        )
        temperature = (
            self.state_pool.temperature.index_select(0, rows)
            if temperature is None
            else temperature.to(mu)
        )
        cfg_value = (
            self.state_pool.cfg_value.index_select(0, rows)
            if cfg_value is None
            else cfg_value.to(mu)
        )
        cond = self.state_pool.latents.index_select(0, rows).transpose(1, 2)
        if inference_timesteps is None:
            decoded = self.feat_decoder(mu, cond, temperature, cfg_value, z_noise)
        else:
            step_counts = inference_timesteps.to(
                device=mu.device, dtype=torch.long
            ).reshape(-1)
            if step_counts.numel() != rows.numel() or bool((step_counts < 1).any()):
                raise ValueError(
                    "inference_timesteps must be positive and match row count"
                )
            decoded = torch.empty(
                rows.numel(),
                self.feat_dim,
                self.patch_size,
                device=mu.device,
                dtype=mu.dtype,
            )
            for step_count in torch.unique(step_counts).tolist():
                indices = (step_counts == int(step_count)).nonzero(as_tuple=True)[0]
                group_noise = (
                    None if z_noise is None else z_noise.index_select(0, indices)
                )
                group = self.feat_decoder(
                    mu.index_select(0, indices),
                    cond.index_select(0, indices),
                    temperature.index_select(0, indices),
                    cfg_value.index_select(0, indices),
                    group_noise,
                    inference_timesteps=int(step_count),
                )
                decoded.index_copy_(0, indices, group)
        latents = decoded.transpose(1, 2)
        stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(-1)
        feedback = self.enc_to_lm_proj(self.feat_encoder(latents))
        self.state_pool.update(
            rows, feedback=feedback, latents=latents, stop_flag=stop_flag
        )
        output = {"latents": latents, "stop_flag": stop_flag}
        if decode_vae:
            output["vae_chunk"] = self.decode_vae_chunks(latents, rows)
        return output

    def attach_vae(self, vae: nn.Module) -> None:
        self.vae = vae
        streaming = getattr(vae, "streaming_decoder", None)
        self.vae_streaming_decoder = streaming() if callable(streaming) else None

    @torch.no_grad()
    def decode_vae_chunks(
        self, latents: torch.Tensor, rows: torch.Tensor
    ) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("no AudioVAE has been attached")
        decoder = self.vae_streaming_decoder
        if decoder is not None and hasattr(decoder, "decode_chunks"):
            request_ids = [f"row:{int(row)}" for row in rows.tolist()]
            return decoder.decode_chunks(
                latents.float().transpose(1, 2), request_ids, [None] * len(request_ids)
            )
        if not hasattr(self.vae, "decode"):
            raise TypeError(
                "attached AudioVAE has neither decode nor streaming decode_chunks"
            )
        return self.vae.decode(latents.float().transpose(1, 2))
