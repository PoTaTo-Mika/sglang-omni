# SPDX-License-Identifier: Apache-2.0
"""Batched MeanFlow acoustic tail for the dots.tts SGLang engine.

SGLang owns the Qwen2 AR backbone; every step of that backbone additionally
needs one MeanFlow DiT sample plus one patch-encoder step per request. Both are
small transformers with their own KV state, so running them one request at a
time would make the engine launch-bound. This module keeps that state in
slot-indexed pools sized once for ``num_slots x patch_capacity`` and runs
both networks over the whole running batch in a single call per decode step.

The recurrence mirrors ``DotsTtsCore``'s streaming decode: the DiT reads a flow
sequence that interleaves one projected backbone hidden row with
``latent_patch_size`` projected latent rows per audio patch. Everything older
than the last complete unit lives in the DiT KV cache, so only a
``unit_len + hidden_patch_size`` row window has to be kept verbatim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang_omni.models.dots_tts.components.backbone.dit_inference import (
    DiTInferenceContext,
    KvPrefillStep,
    NextPatchStep,
)
from sglang_omni.models.dots_tts.components.backbone.encoder_inference import (
    SemanticEncoderDecodeStep,
)
from sglang_omni.models.dots_tts.components.backbone.inference_utils import (
    build_rotary_cos_sin,
    fuse_qkv_projection,
    project_attention,
)

logger = logging.getLogger(__name__)

# note (chenyang): the tail attends over ~10 query rows per layer, where cuDNN's
# SDPA plan lookup costs ~700us of host time for ~10us of device work and makes
# the whole engine launch-bound. The efficient/math kernels have no such
# per-call planning cost.
_TAIL_SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


@dataclass(frozen=True)
class DotsTtsTailSpec:
    """Static shapes the tail pools are sized from."""

    nfe: int
    patch_capacity: int
    num_slots: int
    hidden_patch_size: int
    latent_patch_size: int
    latent_dim: int
    fm_hidden_size: int

    def __post_init__(self) -> None:
        for name in ("nfe", "patch_capacity", "num_slots"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"dots.tts tail {name} must be positive")

    @property
    def unit_len(self) -> int:
        return self.hidden_patch_size + self.latent_patch_size

    @property
    def window_len(self) -> int:
        return self.unit_len + self.hidden_patch_size

    @property
    def fm_capacity(self) -> int:
        # note (chenyang): one hidden row plus latent_patch_size latent rows per
        # audio patch, and the first patch is preceded by a bare hidden row.
        return self.patch_capacity * self.unit_len + self.hidden_patch_size

    @property
    def dit_cache_tokens(self) -> int:
        return self.fm_capacity - self.window_len + self.unit_len


def batched_causal_update_mask(
    *,
    capacity_tokens: int,
    valid_persistent: torch.Tensor,
    prev_len: int,
    current_len: int,
) -> torch.Tensor:
    """``build_causal_update_mask`` with a per-row valid prefix length.

    Returns ``[rows, 1, prev_len + current_len, capacity_tokens + q_len]``: the
    fresh query block attends causally inside ``prev_len`` and fully across the
    tail, plus every cached key below that row's ``valid_persistent``.
    """
    q_len = int(prev_len) + int(current_len)
    device = valid_persistent.device
    total_kv = int(capacity_tokens) + q_len
    kv_idx = torch.arange(total_kv, device=device)
    q_idx = torch.arange(q_len, device=device).reshape(q_len, 1)
    tail_idx = kv_idx.reshape(1, total_kv) - int(capacity_tokens)
    prev_query = q_idx < int(prev_len)
    prev_causal = (tail_idx >= 0) & (tail_idx < int(prev_len)) & (tail_idx <= q_idx)
    tail = torch.where(prev_query, prev_causal, tail_idx >= 0)
    past = kv_idx.reshape(1, 1, total_kv) < valid_persistent.reshape(-1, 1, 1)
    return (past | tail.unsqueeze(0)).unsqueeze(1)


def batched_rotary_cos_sin(
    rotary: Any,
    *,
    start_pos: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary cos/sin for ``[rows, seq_len]`` positions starting per row."""
    offsets = torch.arange(int(seq_len), device=start_pos.device, dtype=torch.float32)
    positions = start_pos.reshape(-1, 1).to(torch.float32) + offsets
    embedding = rotary(positions)
    return embedding.cos(), embedding.sin()


class DotsTtsAcousticTail:
    """Slot-pooled MeanFlow DiT + patch encoder shared by the whole batch."""

    def __init__(
        self,
        *,
        dit_context: DiTInferenceContext,
        patch_encoder: nn.Module,
        spec: DotsTtsTailSpec,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.spec = spec
        self.device = device
        self.dtype = dtype

        self.dit = dit_context.dit
        self._next_patch_step = NextPatchStep(dit_context, attn_backend="sdpa").eval()
        self._kv_prefill_step = KvPrefillStep(dit_context).eval()

        self._encoder = patch_encoder
        for layer in patch_encoder.encoder.layers:
            fuse_qkv_projection(layer.attn)
        self._encoder_step = SemanticEncoderDecodeStep(patch_encoder).eval()

        dit_attn = self.dit.blocks[0].attn
        self._dit_layers = len(self.dit.blocks)
        self._dit_rotary = dit_attn.rotary
        self._mods_width = int(self.dit.fused_adaln[-1].out_features)

        encoder_attn = patch_encoder.encoder.layers[0].attn
        self._encoder_layers = len(patch_encoder.encoder.layers)
        self._encoder_rotary = encoder_attn.rotary if encoder_attn.rotary_bias else None
        self._encoder_block = int(patch_encoder.out_ds_rate)
        self._encoder_heads = int(encoder_attn.num_heads)
        self._encoder_head_dim = int(encoder_attn.head_dim)

        self._dit_k = torch.zeros(
            (
                spec.nfe,
                self._dit_layers,
                spec.num_slots,
                int(dit_attn.num_heads),
                spec.dit_cache_tokens,
                int(dit_attn.head_dim),
            ),
            device=device,
            dtype=dtype,
        )
        self._dit_v = torch.zeros_like(self._dit_k)
        self._encoder_k = torch.zeros(
            (
                self._encoder_layers,
                spec.num_slots,
                self._encoder_heads,
                spec.patch_capacity * self._encoder_block,
                self._encoder_head_dim,
            ),
            device=device,
            dtype=dtype,
        )
        self._encoder_v = torch.zeros_like(self._encoder_k)
        self._encoder_conv_tail = torch.zeros(
            (
                spec.num_slots,
                int(patch_encoder.ds_proj.in_channels),
                int(patch_encoder.ds_proj.left_padding),
            ),
            device=device,
            dtype=dtype,
        )
        self._window = torch.zeros(
            (spec.num_slots, spec.window_len, spec.fm_hidden_size),
            device=device,
            dtype=dtype,
        )
        self._all_mods = torch.zeros(
            (spec.nfe, spec.num_slots, self._mods_width),
            device=device,
            dtype=dtype,
        )
        self._times = torch.linspace(0.0, 1.0, spec.nfe + 1, device=device, dtype=dtype)
        self._layer_index = torch.arange(self._dit_layers, device=device)
        self._encoder_layer_index = torch.arange(self._encoder_layers, device=device)

        self._fm_seq_len = [0] * spec.num_slots
        self._encoder_seq_len = [0] * spec.num_slots
        self._generators: list[torch.Generator | None] = [None] * spec.num_slots
        self._free_slots = list(reversed(range(spec.num_slots)))

        logger.info(
            "dots.tts acoustic tail pools: slots=%s nfe=%s patch_capacity=%s "
            "dit_cache_tokens=%s pool_bytes=%s",
            spec.num_slots,
            spec.nfe,
            spec.patch_capacity,
            spec.dit_cache_tokens,
            2
            * (self._dit_k.numel() + self._encoder_k.numel())
            * self._dit_k.element_size(),
        )

    # ------------------------------------------------------------------
    # Slot lifecycle
    # ------------------------------------------------------------------

    def acquire_slot(self) -> int:
        if not self._free_slots:
            raise RuntimeError(
                "dots.tts acoustic tail ran out of slots; max_running_requests "
                f"must not exceed {self.spec.num_slots}"
            )
        slot = self._free_slots.pop()
        self._fm_seq_len[slot] = 0
        self._encoder_seq_len[slot] = 0
        self._generators[slot] = None
        self._encoder_conv_tail[slot].zero_()
        self._window[slot].zero_()
        return slot

    def release_slot(self, slot: int) -> None:
        slot = int(slot)
        if slot in self._free_slots:
            return
        self._fm_seq_len[slot] = 0
        self._encoder_seq_len[slot] = 0
        self._generators[slot] = None
        self._free_slots.append(slot)

    def set_slot_seed(self, slot: int, seed: int) -> None:
        """Give one slot its own noise stream so batching cannot perturb it."""
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        self._generators[int(slot)] = generator

    def fm_seq_len(self, slot: int) -> int:
        return self._fm_seq_len[int(slot)]

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_prompt_patches(
        self, slot: int, prompt_latents: torch.Tensor
    ) -> torch.Tensor:
        """Backbone embeddings for one request's prompt audio spans.

        ``prompt_latents`` is ``[1, patches * latent_patch_size, latent_dim]`` in
        whatever normalization the patch encoder expects. Seeds the slot's
        patch-encoder KV cache so decode steps continue the same sequence.
        """
        encoder = self._encoder
        step_inputs = encoder.in_proj(encoder._downsample(prompt_latents))
        tokens = int(step_inputs.size(1))
        if tokens > int(self._encoder_k.size(3)):
            raise ValueError(
                "dots.tts prompt audio exceeds the patch-encoder cache: "
                f"tokens={tokens} capacity={int(self._encoder_k.size(3))}"
            )

        rotary_cos = rotary_sin = None
        if self._encoder_rotary is not None:
            rotary_cos, rotary_sin = build_rotary_cos_sin(
                self._encoder_rotary,
                start_pos=0,
                seq_len=tokens,
                device=prompt_latents.device,
                batched=True,
            )

        x = step_inputs
        with sdpa_kernel(_TAIL_SDPA_BACKENDS):
            for layer_idx, layer in enumerate(encoder.encoder.layers):
                attn_out, key, value = project_attention(
                    layer.attn,
                    layer.attn_norm(x),
                    num_heads=self._encoder_heads,
                    head_dim=self._encoder_head_dim,
                    rotary_cos=rotary_cos,
                    rotary_sin=rotary_sin,
                    is_causal=True,
                    dropout_p=0.0,
                )
                self._encoder_k[layer_idx, slot, :, :tokens].copy_(key[0])
                self._encoder_v[layer_idx, slot, :, :tokens].copy_(value[0])
                x = x + attn_out
                x = x + layer.ffn(layer.ffn_norm(x))

        left_padding = int(encoder.ds_proj.left_padding)
        self._encoder_conv_tail[slot].copy_(
            prompt_latents.transpose(1, 2)[0, :, -left_padding:]
        )
        self._encoder_seq_len[slot] = tokens
        return encoder._project_embeddings(x)[0]

    @torch.no_grad()
    def seed_fm_history(
        self,
        slot: int,
        *,
        fm_rows: torch.Tensor,
        all_mods: torch.Tensor,
    ) -> None:
        """Install one request's prompt flow sequence and its AdaLN modulations.

        ``fm_rows`` is ``[unit_len * patches, fm_hidden_size]``: the projected
        prompt history without the trailing hidden row, which the first tail
        step appends. Everything but the last unit is folded into the DiT KV
        cache right away.
        """
        spec = self.spec
        total = int(fm_rows.size(0))
        if total <= 0 or total % spec.unit_len != 0:
            raise ValueError(
                "dots.tts prompt flow history must be unit-aligned and non-empty: "
                f"rows={total} unit_len={spec.unit_len}"
            )
        if total + spec.hidden_patch_size > spec.fm_capacity:
            raise ValueError(
                "dots.tts prompt flow history exceeds the tail capacity: "
                f"rows={total} fm_capacity={spec.fm_capacity}"
            )

        persistent = total - spec.unit_len
        self._all_mods[:, slot].copy_(all_mods)
        self._window[slot, : spec.unit_len].copy_(fm_rows[persistent:])
        self._window[slot, spec.unit_len :].zero_()
        self._fm_seq_len[slot] = total
        if persistent == 0:
            return

        rotary_cos, rotary_sin = build_rotary_cos_sin(
            self._dit_rotary,
            start_pos=0,
            seq_len=persistent,
            device=fm_rows.device,
            batched=True,
        )
        prefix = fm_rows[:persistent].unsqueeze(0)
        with sdpa_kernel(_TAIL_SDPA_BACKENDS):
            for ode_idx in range(spec.nfe):
                new_k, new_v = self._kv_prefill_step(
                    prefix,
                    all_mods=all_mods[ode_idx : ode_idx + 1],
                    rotary_cos=rotary_cos,
                    rotary_sin=rotary_sin,
                )
                self._dit_k[ode_idx, :, slot, :, :persistent].copy_(new_k[:, 0])
                self._dit_v[ode_idx, :, slot, :, :persistent].copy_(new_v[:, 0])

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_patches(
        self,
        slots: list[int],
        *,
        fm_hidden_rows: torch.Tensor,
        latent_proj: nn.Module,
    ) -> torch.Tensor:
        """Append one hidden row per slot, then sample one latent patch each.

        ``fm_hidden_rows`` is ``[rows, fm_hidden_size]`` already in flow space.
        Returns the normalized latent patches ``[rows, latent_patch_size,
        latent_dim]`` and advances every slot's flow history by one unit.
        """
        spec = self.spec
        slot_index = torch.tensor(slots, device=self.device, dtype=torch.long)
        for slot in slots:
            self._fm_seq_len[slot] += spec.hidden_patch_size
        self._window[slot_index, spec.unit_len :] = fm_hidden_rows.unsqueeze(1).to(
            dtype=self.dtype
        )

        persistent = [self._fm_seq_len[slot] - spec.window_len for slot in slots]
        if min(persistent) < 0:
            raise RuntimeError(
                "dots.tts tail step ran before the prompt flow history was seeded"
            )
        capacity = max(persistent)
        if capacity + spec.unit_len > spec.dit_cache_tokens:
            raise RuntimeError(
                "dots.tts flow history exceeded the DiT cache: "
                f"persistent={capacity} capacity={spec.dit_cache_tokens}"
            )
        persistent_index = torch.tensor(
            persistent, device=self.device, dtype=torch.long
        )

        latent = self._run_meanflow(
            slots=slots,
            slot_index=slot_index,
            persistent_index=persistent_index,
            capacity=capacity,
        )

        carry = self._window[slot_index, spec.unit_len :]
        self._window[slot_index, : spec.hidden_patch_size] = carry
        self._window[slot_index, spec.hidden_patch_size : spec.unit_len] = latent_proj(
            latent
        ).to(dtype=self.dtype)
        for slot in slots:
            self._fm_seq_len[slot] += spec.latent_patch_size
        return latent

    def _run_meanflow(
        self,
        *,
        slots: list[int],
        slot_index: torch.Tensor,
        persistent_index: torch.Tensor,
        capacity: int,
    ) -> torch.Tensor:
        spec = self.spec
        rows = int(slot_index.numel())
        prev_unit = self._window[slot_index, : spec.unit_len]
        current_hidden = self._window[slot_index, spec.unit_len :]
        sdpa_mask = batched_causal_update_mask(
            capacity_tokens=capacity,
            valid_persistent=persistent_index,
            prev_len=spec.unit_len,
            current_len=spec.unit_len,
        )
        rotary_cos, rotary_sin = batched_rotary_cos_sin(
            self._dit_rotary,
            start_pos=persistent_index,
            seq_len=2 * spec.unit_len,
        )
        mods = self._all_mods.index_select(1, slot_index)
        token_index = persistent_index.reshape(1, rows, 1) + torch.arange(
            spec.unit_len, device=self.device
        ).reshape(1, 1, spec.unit_len)
        layer_index = self._layer_index.reshape(self._dit_layers, 1, 1)
        batch_index = slot_index.reshape(1, rows, 1)

        # note (chenyang): every ODE step reads only cache rows below its own
        # valid prefix and writes rows above it, so one gather up front is
        # equivalent to gathering inside the loop and costs one kernel instead
        # of two per step.
        cache_k = self._dit_k[:, :, :, :, :capacity].index_select(2, slot_index)
        cache_v = self._dit_v[:, :, :, :, :capacity].index_select(2, slot_index)

        latent = self._sample_noise(slots)
        with sdpa_kernel(_TAIL_SDPA_BACKENDS):
            for ode_idx in range(spec.nfe):
                velocity, new_k, new_v = self._next_patch_step(
                    latent,
                    prev_unit=prev_unit,
                    current_hidden=current_hidden,
                    all_mods=mods[ode_idx],
                    cache_k=cache_k[ode_idx],
                    cache_v=cache_v[ode_idx],
                    rotary_cos=rotary_cos,
                    rotary_sin=rotary_sin,
                    sdpa_mask=sdpa_mask,
                )
                duration = (self._times[ode_idx + 1] - self._times[ode_idx]).expand(
                    rows
                )
                latent = (latent + duration.view(-1, 1, 1) * velocity).clone()
                self._dit_k[ode_idx][layer_index, batch_index, :, token_index] = (
                    new_k.permute(0, 1, 3, 2, 4)
                )
                self._dit_v[ode_idx][layer_index, batch_index, :, token_index] = (
                    new_v.permute(0, 1, 3, 2, 4)
                )
        return latent

    def _sample_noise(self, slots: list[int]) -> torch.Tensor:
        spec = self.spec
        generators = [self._generators[slot] for slot in slots]
        if all(generator is None for generator in generators):
            return torch.randn(
                (len(slots), spec.latent_patch_size, spec.latent_dim),
                device=self.device,
                dtype=self.dtype,
            )
        return torch.cat(
            [
                torch.randn(
                    (1, spec.latent_patch_size, spec.latent_dim),
                    generator=generator,
                    device=self.device,
                    dtype=self.dtype,
                )
                for generator in generators
            ],
            dim=0,
        )

    @torch.no_grad()
    def encode_feedback(
        self, slots: list[int], latent_patches: torch.Tensor
    ) -> torch.Tensor:
        """Backbone input embedding for each freshly sampled latent patch."""
        slot_index = torch.tensor(slots, device=self.device, dtype=torch.long)
        rows = len(slots)
        block = self._encoder_block
        starts = [self._encoder_seq_len[slot] for slot in slots]
        capacity = max(starts)
        if capacity + block > int(self._encoder_k.size(3)):
            raise RuntimeError(
                "dots.tts patch-encoder cache overflow: "
                f"tokens={capacity + block} capacity={int(self._encoder_k.size(3))}"
            )
        start_index = torch.tensor(starts, device=self.device, dtype=torch.long)

        sdpa_mask = batched_causal_update_mask(
            capacity_tokens=capacity,
            valid_persistent=start_index,
            prev_len=block,
            current_len=0,
        )
        if self._encoder_rotary is None:
            empty = torch.empty((0,), device=self.device)
            rotary_cos, rotary_sin = empty, empty
        else:
            rotary_cos, rotary_sin = batched_rotary_cos_sin(
                self._encoder_rotary,
                start_pos=start_index,
                seq_len=block,
            )

        cache_k = self._encoder_k[:, :, :, :capacity].index_select(1, slot_index)
        cache_v = self._encoder_v[:, :, :, :capacity].index_select(1, slot_index)
        layer_caches = tuple(
            (cache_k[layer_idx], cache_v[layer_idx])
            for layer_idx in range(self._encoder_layers)
        )
        with sdpa_kernel(_TAIL_SDPA_BACKENDS):
            embedding, conv_tail, new_k, new_v = self._encoder_step(
                latent_patches.to(dtype=self.dtype),
                self._encoder_conv_tail.index_select(0, slot_index),
                layer_caches,
                sdpa_mask,
                rotary_cos,
                rotary_sin,
            )

        token_index = start_index.reshape(1, rows, 1) + torch.arange(
            block, device=self.device
        ).reshape(1, 1, block)
        layer_index = self._encoder_layer_index.reshape(self._encoder_layers, 1, 1)
        batch_index = slot_index.reshape(1, rows, 1)
        self._encoder_k[layer_index, batch_index, :, token_index] = new_k.permute(
            0, 1, 3, 2, 4
        )
        self._encoder_v[layer_index, batch_index, :, token_index] = new_v.permute(
            0, 1, 3, 2, 4
        )
        self._encoder_conv_tail[slot_index] = conv_tail
        for slot in slots:
            self._encoder_seq_len[slot] += block
        return embedding.reshape(rows, -1)


__all__ = [
    "DotsTtsAcousticTail",
    "DotsTtsTailSpec",
    "batched_causal_update_mask",
    "batched_rotary_cos_sin",
]
