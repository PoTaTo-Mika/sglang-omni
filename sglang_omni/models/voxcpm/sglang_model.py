# SPDX-License-Identifier: Apache-2.0
"""SGLang-native VoxCPM (V1) synthesis model."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from sglang_omni.models.voxcpm_common import (
    VoxCPMCore,
    load_audiovae_checkpoint,
    load_audiovae_from_model_path,
    load_voxcpm_weights,
)

try:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang_omni.vendor.sglang.core import ForwardBatch
except ModuleNotFoundError as error:
    if error.name != "sglang":
        raise
    ForwardBatch = Any  # type: ignore[misc,assignment]

    class LogitsProcessorOutput:  # type: ignore[no-redef]
        def __init__(
            self, *, next_token_logits: torch.Tensor, hidden_states: torch.Tensor
        ) -> None:
            self.next_token_logits = next_token_logits
            self.hidden_states = hidden_states


def _max_requests(default: int = 1) -> int:
    try:
        from sglang.srt.server_args import get_global_server_args

        return int(get_global_server_args().max_running_requests)
    except Exception:
        return default


class VoxCPMSGLangModel(nn.Module):
    """V1 model with radix-cached base and residual MiniCPM4 LMs."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: Any,
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        timesteps = int(getattr(config, "inference_timesteps", 10))
        self.model = VoxCPMCore(
            config,
            version="v1",
            inference_timesteps=timesteps,
            max_requests=_max_requests(),
            quant_config=quant_config,
            prefix=prefix,
        )
        self.config = config
        self._load_runtime_audiovae_if_available()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.base_lm.embed_tokens

    def acquire(self, rid: str) -> int:
        return self.model.acquire(rid)

    acquire_row = acquire

    def reset(self, rid: str) -> None:
        self.model.reset(rid)

    reset_request = reset
    release_row = reset

    def row_for(self, rid: str) -> int | None:
        return self.model.row_for(rid)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        *,
        feat: torch.Tensor | None = None,
        feat_mask: torch.Tensor | None = None,
        row_ids: torch.Tensor | None = None,
        **_: Any,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model.forward_backbone(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            feat=feat,
            feat_mask=feat_mask,
            row_ids=row_ids,
        )
        return LogitsProcessorOutput(
            next_token_logits=hidden_states.new_empty((hidden_states.shape[0], 1)),
            hidden_states=hidden_states,
        )

    @torch.no_grad()
    def generate_batch(
        self,
        hidden_states: torch.Tensor,
        rows: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        return self.model.generate_batch(hidden_states, rows, **kwargs)

    def attach_vae(self, vae: nn.Module) -> None:
        self.model.attach_vae(vae)

    def _load_runtime_audiovae_if_available(self) -> None:
        try:
            from sglang.srt.server_args import get_global_server_args

            model_path = get_global_server_args().model_path
        except Exception:
            return
        if model_path and os.path.isfile(os.path.join(model_path, "audiovae.pth")):
            self.load_audiovae(model_path)

    def load_audiovae(
        self,
        vae_or_model_path: nn.Module | str,
        model_path: str | None = None,
        *,
        strict: bool = True,
    ) -> nn.Module:
        if isinstance(vae_or_model_path, nn.Module):
            if model_path is None:
                raise ValueError("model_path is required when passing a VAE instance")
            vae = load_audiovae_checkpoint(vae_or_model_path, model_path, strict=strict)
        else:
            if model_path is not None:
                raise ValueError("pass either a model path or (vae, model_path)")
            vae = load_audiovae_from_model_path(
                self.config,
                vae_or_model_path,
                version="v1",
                strict=strict,
            )
        device = next(self.model.parameters()).device
        vae.to(device=device, dtype=torch.float32)
        self.attach_vae(vae)
        return vae

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return load_voxcpm_weights(self, weights)


EntryClass = VoxCPMSGLangModel
