# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import typer

import sglang_omni.models.llada2_uni.config as llada_config
from sglang_omni.cli.serve import apply_parallelism_cli_overrides
from sglang_omni.config import ParallelismConfig
from sglang_omni.config.runtime import resolve_stage_static_factory_args
from sglang_omni.models.llada2_uni.components.decoder_model import (
    ZImageParallelConfig,
    _validate_zimage_ulysses_degree,
)
from sglang_omni.models.llada2_uni.config import (
    IMAGE_DECODE_STAGE,
    LLaDA2UniOmniPipelineConfig,
    Variants,
)


def _image_decode_stage(config):
    return next(stage for stage in config.stages if stage.name == IMAGE_DECODE_STAGE)


def test_zimage_parallel_config_is_constructible() -> None:
    config = ZImageParallelConfig(
        backend="sglang",
        sp_size=2,
        ulysses_degree=2,
    )

    assert config.normalized_backend == "sglang"
    assert config.sglang_ulysses_degree == 2


def test_zimage_parallel_config_rejects_non_power_of_two_sp() -> None:
    with pytest.raises(ValueError, match="power of two"):
        ZImageParallelConfig(
            backend="sglang",
            sp_size=3,
            ulysses_degree=3,
        )


def test_zimage_runtime_rejects_ulysses_degree_incompatible_with_heads() -> None:
    with pytest.raises(ValueError, match="30 attention heads"):
        _validate_zimage_ulysses_degree(
            num_attention_heads=30,
            ulysses_degree=4,
        )


def test_image_sp_variants_are_not_exposed() -> None:
    assert "image-sp2" not in Variants
    assert "image-sp4" not in Variants
    assert "relay_backend" not in LLaDA2UniOmniPipelineConfig.model_fields


def test_llada2_default_thinker_remains_single_gpu() -> None:
    config = LLaDA2UniOmniPipelineConfig(model_path="dummy")

    thinker = next(stage for stage in config.stages if stage.name == "thinker")

    assert thinker.gpu == 0
    assert thinker.tp_size == 1


def test_llada2_exposes_one_image_decoder_parallel_policy() -> None:
    config = LLaDA2UniOmniPipelineConfig(model_path="dummy")

    policy = type(config).sequence_parallel_policy(stage_name=IMAGE_DECODE_STAGE)

    assert policy.attention_heads == 30
    assert policy.requires_power_of_two is True
    assert policy.required_backend == "sglang"


def test_image_decoder_runtime_defaults_are_resolved_by_model_config() -> None:
    resolver = getattr(llada_config, "resolve_image_decoder_runtime_settings", None)
    assert resolver is not None

    settings = resolver(
        backend=None,
        attention_backend=None,
        png_compress_level=None,
        sp_size=4,
    )

    assert settings.backend == "sglang"
    assert settings.attention_backend == "fa"
    assert settings.png_compress_level == 1


def test_image_decoder_cli_does_not_materialize_model_runtime_defaults() -> None:
    config = LLaDA2UniOmniPipelineConfig(model_path="dummy")

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        talker_gpu=None,
        code2wav_gpu=None,
        image_decoder_sp_size=4,
        image_decoder_gpus="0,1,2,3",
    )

    resolved = resolve_stage_static_factory_args(_image_decode_stage(config), config)
    assert "backend" not in resolved
    assert "attention_backend" not in resolved
    assert "png_compress_level" not in resolved


def test_image_decoder_cli_accepts_model_owned_backend_alias() -> None:
    config = LLaDA2UniOmniPipelineConfig(
        model_path="dummy",
        runtime_overrides={IMAGE_DECODE_STAGE: {"backend": "native"}},
    )

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        talker_gpu=None,
        code2wav_gpu=None,
        image_decoder_gpus="0",
    )

    policy = type(config).sequence_parallel_policy(stage_name=IMAGE_DECODE_STAGE)
    assert policy.resolve_backend("native", sp_size=1) == "diffusers"
    assert config.runtime_overrides[IMAGE_DECODE_STAGE]["backend"] == "native"


def test_image_decoder_parallelism_cli_override_is_disjoint_and_parameterized() -> None:
    config = LLaDA2UniOmniPipelineConfig(model_path="dummy")

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=2,
        thinker_gpus="0,1",
        talker_gpu=None,
        code2wav_gpu=None,
        image_decoder_sp_size=2,
        image_decoder_gpus="2,3",
        image_decoder_ring_degree=2,
        image_decoder_backend="sglang",
    )

    thinker = next(stage for stage in config.stages if stage.name == "thinker")
    decoder = _image_decode_stage(config)
    assert thinker.gpu == [0, 1]
    assert thinker.tp_size == 2
    assert thinker.parallelism.tp == 2
    assert decoder.process == "pipeline"
    assert decoder.gpu == [2, 3]
    assert decoder.parallelism.sp == 2
    assert decoder.parallelism.ulysses_degree == 1
    assert decoder.parallelism.ring_degree == 2
    resolved = resolve_stage_static_factory_args(decoder, config)
    assert resolved["backend"] == "sglang"
    assert "attention_backend" not in resolved


def test_image_decoder_cli_rejects_non_power_of_two_sp() -> None:
    config = LLaDA2UniOmniPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="power of two"):
        apply_parallelism_cli_overrides(
            config,
            thinker_tp_size=None,
            thinker_gpus=None,
            talker_gpu=None,
            code2wav_gpu=None,
            image_decoder_sp_size=3,
            image_decoder_gpus="0,1,2",
            image_decoder_backend="sglang",
        )


def test_image_decoder_backend_only_override_uses_default_sp_decomposition() -> None:
    config = LLaDA2UniOmniPipelineConfig(model_path="dummy")

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        talker_gpu=None,
        code2wav_gpu=None,
        image_decoder_backend="sglang",
    )

    decoder = _image_decode_stage(config)
    assert decoder.parallelism.sp == 1


def test_image_decoder_backend_only_override_preserves_yaml_sp_decomposition() -> None:
    config = LLaDA2UniOmniPipelineConfig(model_path="dummy")
    decoder = _image_decode_stage(config)
    decoder.gpu = [0, 1, 2, 3]
    decoder.parallelism = ParallelismConfig(
        sp=4,
        ulysses_degree=1,
        ring_degree=4,
    )

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        talker_gpu=None,
        code2wav_gpu=None,
        image_decoder_backend="sglang",
    )

    decoder = _image_decode_stage(config)
    assert decoder.parallelism.ulysses_degree == 1
    assert decoder.parallelism.ring_degree == 4


def test_image_decoder_cli_preserves_unrelated_stage_and_backend_tuning() -> None:
    config = LLaDA2UniOmniPipelineConfig(model_path="dummy")
    decoder = _image_decode_stage(config)
    decoder.process = "custom_image_process"
    decoder.factory_args.update(
        {
            "attention_backend": "sage_attn",
            "png_compress_level": 4,
        }
    )

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        talker_gpu=None,
        code2wav_gpu=None,
        image_decoder_sp_size=4,
        image_decoder_gpus="0,1,2,3",
    )

    decoder = _image_decode_stage(config)
    assert decoder.process == "custom_image_process"
    assert decoder.factory_args["attention_backend"] == "sage_attn"
    assert decoder.factory_args["png_compress_level"] == 4


def test_image_decoder_cli_overrides_survive_runtime_overrides_overlay() -> None:
    config = LLaDA2UniOmniPipelineConfig(
        model_path="dummy",
        runtime_overrides={
            IMAGE_DECODE_STAGE: {
                "backend": "diffusers",
                "attention_backend": "sage_attn",
                "png_compress_level": 4,
            }
        },
    )

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        talker_gpu=None,
        code2wav_gpu=None,
        image_decoder_sp_size=4,
        image_decoder_gpus="0,1,2,3",
        image_decoder_backend="sglang",
        image_decoder_attention_backend="fa",
    )

    resolved = resolve_stage_static_factory_args(_image_decode_stage(config), config)
    assert resolved["backend"] == "sglang"
    assert resolved["attention_backend"] == "fa"
    assert resolved["png_compress_level"] == 4


def test_image_decoder_cli_rejects_empty_attention_backend() -> None:
    config = LLaDA2UniOmniPipelineConfig(model_path="dummy")
    before = config.model_dump()

    with pytest.raises(typer.BadParameter, match="must not be empty"):
        apply_parallelism_cli_overrides(
            config,
            thinker_tp_size=None,
            thinker_gpus=None,
            talker_gpu=None,
            code2wav_gpu=None,
            image_decoder_sp_size=4,
            image_decoder_gpus="0,1,2,3",
            image_decoder_backend="sglang",
            image_decoder_attention_backend="  ",
        )

    assert config.model_dump() == before


def test_image_decoder_sp4_uses_head_compatible_default_decomposition() -> None:
    config = LLaDA2UniOmniPipelineConfig(model_path="dummy")

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        talker_gpu=None,
        code2wav_gpu=None,
        image_decoder_sp_size=4,
        image_decoder_gpus="0,1,2,3",
    )

    decoder = _image_decode_stage(config)
    assert decoder.parallelism.sp == 4
    assert decoder.parallelism.ulysses_degree == 2
    assert decoder.parallelism.ring_degree == 2


def test_image_decoder_cli_rejects_ulysses_degree_incompatible_with_heads() -> None:
    config = LLaDA2UniOmniPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="30 attention heads"):
        apply_parallelism_cli_overrides(
            config,
            thinker_tp_size=None,
            thinker_gpus=None,
            talker_gpu=None,
            code2wav_gpu=None,
            image_decoder_sp_size=4,
            image_decoder_gpus="0,1,2,3",
            image_decoder_ulysses_degree=4,
        )


@pytest.mark.parametrize(
    "degree_override",
    [
        {"image_decoder_ulysses_degree": 0},
        {"image_decoder_ring_degree": 0},
    ],
)
def test_image_decoder_cli_rejects_zero_parallel_degree(
    degree_override: dict[str, int],
) -> None:
    config = LLaDA2UniOmniPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="must be >= 1"):
        apply_parallelism_cli_overrides(
            config,
            thinker_tp_size=None,
            thinker_gpus=None,
            talker_gpu=None,
            code2wav_gpu=None,
            image_decoder_sp_size=4,
            image_decoder_gpus="0,1,2,3",
            **degree_override,
        )
