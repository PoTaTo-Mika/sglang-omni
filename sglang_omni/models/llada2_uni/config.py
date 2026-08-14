# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for LLaDA2-Uni (Diffusion LLM)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from sglang_omni.config import (
    PipelineConfig,
    SequenceParallelPolicy,
    SGLangServerArgsConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
)

_PKG = "sglang_omni.models.llada2_uni"

PREPROCESSING_STAGE = "preprocessing"
IMAGE_STAGE = "image_encoder"
THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"
IMAGE_DECODE_STAGE = "image_decode"

DEFAULT_THINKER_MAX_NEW_TOKENS = 2048
LLADA2_IMAGE_DECODER_ATTENTION_HEADS = 30
LLADA2_IMAGE_DECODER_SP_POLICY = SequenceParallelPolicy(
    attention_heads=LLADA2_IMAGE_DECODER_ATTENTION_HEADS,
    requires_power_of_two=True,
    default_backend="diffusers",
    required_backend="sglang",
    allowed_backends=frozenset({"diffusers", "sglang"}),
    backend_aliases=(
        ("native", "diffusers"),
        ("local", "diffusers"),
        ("omni", "diffusers"),
    ),
    attention_backend_backends=frozenset({"sglang"}),
)


@dataclass(frozen=True)
class ImageDecoderRuntimeSettings:
    backend: str
    attention_backend: str
    png_compress_level: int


def normalize_image_decoder_backend(backend: str) -> str:
    return LLADA2_IMAGE_DECODER_SP_POLICY.normalize_backend(backend)


def resolve_image_decoder_runtime_settings(
    *,
    backend: str | None,
    attention_backend: str | None,
    png_compress_level: int | None,
    sp_size: int,
) -> ImageDecoderRuntimeSettings:
    """Resolve LLaDA2 image-decoder defaults in the model-owned config layer."""
    if sp_size < 1:
        raise ValueError("Image decoder SP size must be positive")
    normalized_backend = (
        normalize_image_decoder_backend(backend) if backend is not None else None
    )
    resolved_backend = LLADA2_IMAGE_DECODER_SP_POLICY.resolve_backend(
        normalized_backend,
        sp_size=sp_size,
    )
    if resolved_backend not in {"diffusers", "sglang"}:
        raise ValueError(f"Unsupported image decoder backend: {backend!r}")

    if attention_backend is None:
        resolved_attention_backend = (
            "fa" if resolved_backend == "sglang" else "torch_sdpa"
        )
    else:
        resolved_attention_backend = attention_backend.strip().lower()
        if not resolved_attention_backend:
            raise ValueError("Image decoder attention backend must not be empty")

    resolved_png_compress_level = (
        (1 if resolved_backend == "sglang" else 6)
        if png_compress_level is None
        else int(png_compress_level)
    )
    if not 0 <= resolved_png_compress_level <= 9:
        raise ValueError("png_compress_level must be between 0 and 9")
    return ImageDecoderRuntimeSettings(
        backend=resolved_backend,
        attention_backend=resolved_attention_backend,
        png_compress_level=resolved_png_compress_level,
    )


class LLaDA2UniPipelineConfig(PipelineConfig):
    """4-stage DLLM pipeline: preprocessing → image_encoder → thinker → decode."""

    architecture: ClassVar[str] = "LLaDA2MoeModelLM"

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {THINKER_STAGE: THINKER_STAGE}

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={"thinker_max_seq_len": 8192},
            runtime_arg_map={"max_seq_len": "thinker_max_seq_len"},
            next=IMAGE_STAGE,
        ),
        StageConfig(
            name=IMAGE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_image_encoder_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            next=THINKER_STAGE,
        ),
        StageConfig(
            name=THINKER_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_dllm_thinker_executor_from_config",
            factory_args={"thinker_max_seq_len": 8192},
            gpu=0,
            next=DECODE_STAGE,
        ),
        StageConfig(
            name=DECODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
        ),
    ]


class LLaDA2UniOmniPipelineConfig(PipelineConfig):
    """5-stage pipeline: preprocessing → image_encoder → thinker → [decode, image_decode].

    The thinker fans out to both decode (text) and image_decode stages.
    The image_decode stage handles text-only requests by detecting no VQ tokens
    and returning ``{"skipped": True}``.
    """

    architecture: ClassVar[str] = "LLaDA2MoeModelLM"
    total_gpu_memory_fraction_dict: ClassVar[dict[str, float]] = {
        IMAGE_STAGE: 0.1,
        THINKER_STAGE: 0.7,
        IMAGE_DECODE_STAGE: 0.2,
    }

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {THINKER_STAGE: THINKER_STAGE}

    @classmethod
    def sequence_parallel_policy(
        cls, *, stage_name: str
    ) -> SequenceParallelPolicy | None:
        if stage_name == IMAGE_DECODE_STAGE:
            return LLADA2_IMAGE_DECODER_SP_POLICY
        return super().sequence_parallel_policy(stage_name=stage_name)

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={"thinker_max_seq_len": 8192},
            runtime_arg_map={"max_seq_len": "thinker_max_seq_len"},
            next=IMAGE_STAGE,
        ),
        StageConfig(
            name=IMAGE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_image_encoder_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(
                    total_gpu_memory_fraction=total_gpu_memory_fraction_dict[
                        IMAGE_STAGE
                    ]
                ),
            ),
            next=THINKER_STAGE,
        ),
        StageConfig(
            name=THINKER_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_dllm_thinker_executor_from_config",
            factory_args={"thinker_max_seq_len": 8192},
            gpu=0,
            next=[THINKER_STAGE, DECODE_STAGE, IMAGE_DECODE_STAGE],
            route_fn=f"{_PKG}.routing.thinker_next",
            runtime=StageRuntimeConfig(
                sglang_server_args=SGLangServerArgsConfig(
                    mem_fraction_static=0.75,
                ),
                resources=StageResourceConfig(
                    total_gpu_memory_fraction=total_gpu_memory_fraction_dict[
                        THINKER_STAGE
                    ]
                ),
            ),
        ),
        StageConfig(
            name=DECODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
        ),
        StageConfig(
            name=IMAGE_DECODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_image_decode_executor",
            factory_args={
                "device": "cuda",
                "dtype": None,
                "resolution_multiplier": 2,
            },
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(
                    total_gpu_memory_fraction=total_gpu_memory_fraction_dict[
                        IMAGE_DECODE_STAGE
                    ]
                ),
            ),
            terminal=True,
        ),
    ]


EntryClass = LLaDA2UniOmniPipelineConfig

Variants = {
    "text": LLaDA2UniPipelineConfig,
    "omni": LLaDA2UniOmniPipelineConfig,
}
