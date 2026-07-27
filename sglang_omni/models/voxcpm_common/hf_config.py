# SPDX-License-Identifier: Apache-2.0
"""Transformers config adapters for official VoxCPM checkpoints."""

from __future__ import annotations

from typing import Any, ClassVar

from transformers import AutoConfig, PretrainedConfig


def _config_object(value: Any) -> Any:
    if isinstance(value, dict):
        return PretrainedConfig.from_dict(
            {key: _config_object(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_config_object(item) for item in value]
    return value


class _VoxCPMConfigBase(PretrainedConfig):
    architectures: list[str] | None = None
    architecture: str | None = None
    default_architecture: ClassVar[str] = ""

    def __init__(self, **kwargs: Any) -> None:
        raw_architectures = kwargs.pop("architectures", None)
        raw_architecture = kwargs.pop("architecture", None)
        lm_config = kwargs.get("lm_config")
        if isinstance(lm_config, dict):
            for field_name in (
                "hidden_size",
                "num_attention_heads",
                "num_key_value_heads",
                "num_hidden_layers",
                "max_position_embeddings",
                "vocab_size",
                "rms_norm_eps",
                "rope_theta",
            ):
                if field_name in lm_config:
                    kwargs.setdefault(field_name, lm_config[field_name])
            kwargs.setdefault(
                "head_dim",
                lm_config.get(
                    "kv_channels",
                    int(lm_config["hidden_size"])
                    // int(lm_config["num_attention_heads"]),
                ),
            )
        nested = {key: _config_object(value) for key, value in kwargs.items()}
        super().__init__(**nested)
        self.architecture = raw_architecture or self.model_type
        self.architectures = list(raw_architectures or [self.default_architecture])

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> Any:
        values = dict(config_dict)
        values.setdefault("architectures", [cls.default_architecture])
        lm_config = values.get("lm_config")
        if isinstance(lm_config, dict):
            for field_name in (
                "hidden_size",
                "num_attention_heads",
                "num_key_value_heads",
                "num_hidden_layers",
                "max_position_embeddings",
                "vocab_size",
                "rms_norm_eps",
                "rope_theta",
            ):
                if field_name in lm_config:
                    values.setdefault(field_name, lm_config[field_name])
            values.setdefault(
                "head_dim",
                lm_config.get(
                    "kv_channels",
                    int(lm_config["hidden_size"])
                    // int(lm_config["num_attention_heads"]),
                ),
            )
        return super().from_dict(values, **kwargs)


class VoxCPMConfig(_VoxCPMConfigBase):
    model_type = "voxcpm"
    default_architecture = "VoxCPMSGLangModel"


class VoxCPM2Config(_VoxCPMConfigBase):
    model_type = "voxcpm2"
    default_architecture = "VoxCPM2SGLangModel"


def register_voxcpm_configs() -> None:
    for model_type, config_cls in (
        ("voxcpm", VoxCPMConfig),
        ("voxcpm2", VoxCPM2Config),
    ):
        try:
            AutoConfig.register(model_type, config_cls)
        except ValueError:
            # Idempotent in worker processes and compatible with an upstream
            # Transformers release that may learn the same model type.
            pass


__all__ = ["VoxCPM2Config", "VoxCPMConfig", "register_voxcpm_configs"]
