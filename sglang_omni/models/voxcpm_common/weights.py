# SPDX-License-Identifier: Apache-2.0
"""Official VoxCPM checkpoint loading helpers."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

try:
    from sglang.srt.model_loader.weight_utils import default_weight_loader
except ModuleNotFoundError as error:
    if error.name != "sglang":
        raise

    def default_weight_loader(
        parameter: nn.Parameter,
        tensor: torch.Tensor,
        shard_id: str | int | None = None,
    ) -> None:
        if shard_id is not None:
            raise RuntimeError("packed official weights require SGLang to be installed")
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


_STACKED_MAPPING = (
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)


def load_voxcpm_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    strict: bool = False,
) -> set[str]:
    """Load official safetensors names into packed SGLang parameters.

    Official checkpoints name the root modules directly (``base_lm.*``,
    ``residual_lm.*``); wrappers in this repository keep them under ``model``.
    Both forms, plus an optional ``module.`` prefix, are accepted.
    """

    params = dict(module.named_parameters())
    loaded: set[str] = set()
    unexpected: list[str] = []
    for source_name, tensor in weights:
        name = source_name.removeprefix("module.")
        candidates = (name, f"model.{name}")
        target_name = next((item for item in candidates if item in params), None)
        if target_name is not None:
            try:
                _load(params[target_name], tensor)
            except (AssertionError, RuntimeError) as exc:
                raise RuntimeError(
                    "VoxCPM weight shape mismatch: "
                    f"{source_name} {tuple(tensor.shape)} -> "
                    f"{target_name} {tuple(params[target_name].shape)}"
                ) from exc
            loaded.add(target_name)
            continue

        packed = False
        for packed_name, shard_name, shard_id in _STACKED_MAPPING:
            marker = f".{shard_name}."
            if marker not in name:
                continue
            mapped = name.replace(marker, f".{packed_name}.")
            target_name = next(
                (item for item in (mapped, f"model.{mapped}") if item in params),
                None,
            )
            if target_name is None:
                # LocEnc and DiT intentionally retain unpacked nn.Linear names.
                continue
            loader = getattr(
                params[target_name], "weight_loader", default_weight_loader
            )
            loader(params[target_name], tensor, shard_id)
            loaded.add(target_name)
            packed = True
            break
        if packed:
            continue
        if (
            name.endswith("rotary_emb.inv_freq")
            or name.endswith(".cos_cached")
            or name.endswith(".sin_cached")
        ):
            continue
        # Some official archives include tied/output heads unused by synthesis.
        if name.startswith(("lm_head.", "base_lm.lm_head.")):
            continue
        unexpected.append(source_name)

    if strict:
        missing = set(params) - loaded
        if missing or unexpected:
            raise RuntimeError(
                "VoxCPM checkpoint mismatch: "
                f"missing={sorted(missing)[:20]}, unexpected={unexpected[:20]}"
            )
    return loaded


def load_audiovae_checkpoint(
    vae: nn.Module,
    model_path: str,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Any:
    """Load ``audiovae.pth`` and return ``vae`` for convenient attachment."""

    checkpoint_path = (
        model_path
        if os.path.basename(model_path) == "audiovae.pth"
        else os.path.join(model_path, "audiovae.pth")
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"AudioVAE checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=True
        )
    except TypeError:
        # PyTorch < 2.0 has no weights_only argument. Official audiovae.pth is
        # a trusted project artifact, but callers should not pass arbitrary files.
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError("audiovae.pth must contain a state-dict mapping")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("audiovae.pth['state_dict'] must be a mapping")
    vae.load_state_dict(state_dict, strict=strict)
    return vae


def load_audiovae_from_model_path(
    config: Any,
    model_path: str,
    *,
    version: str,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Instantiate V1/V2 AudioVAE from config and load its official pth."""

    from sglang_omni.models.voxcpm_common.audio_vae import create_audio_vae

    # SGLang commonly constructs model modules under a bf16 default dtype,
    # while the official VAE is intentionally fp32.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        vae = create_audio_vae(config, version=version)
    finally:
        torch.set_default_dtype(previous_dtype)
    return load_audiovae_checkpoint(
        vae, model_path, map_location=map_location, strict=strict
    )


def _load(parameter: nn.Parameter, tensor: torch.Tensor) -> None:
    loader = getattr(parameter, "weight_loader", default_weight_loader)
    loader(parameter, tensor)
