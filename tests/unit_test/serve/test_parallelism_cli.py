# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import typer

from sglang_omni.cli.serve import apply_parallelism_cli_overrides
from sglang_omni.config import PipelineConfig, SequenceParallelPolicy, StageConfig

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


class _GenericImagePipelineConfig(PipelineConfig):
    @classmethod
    def sequence_parallel_policy(cls, *, stage_name: str) -> SequenceParallelPolicy:
        return SequenceParallelPolicy(
            attention_heads=12 if stage_name == "image_decode" else None
        )


class _InvalidImagePipelineConfig(PipelineConfig):
    @classmethod
    def sequence_parallel_policy(cls, *, stage_name: str) -> SequenceParallelPolicy:
        return SequenceParallelPolicy(
            attention_heads=0 if stage_name == "image_decode" else None
        )


class _AliasedBackendPipelineConfig(PipelineConfig):
    @classmethod
    def sequence_parallel_policy(cls, *, stage_name: str) -> SequenceParallelPolicy:
        if stage_name != "image_decode":
            return SequenceParallelPolicy()
        return SequenceParallelPolicy(
            attention_heads=12,
            required_backend="native",
            allowed_backends=frozenset({"diffusers"}),
            backend_aliases=(("native", "diffusers"),),
            attention_backend_backends=frozenset({"native"}),
        )


def _image_pipeline_config(config_cls: type[PipelineConfig]) -> PipelineConfig:
    return config_cls(
        model_path="dummy",
        stages=[
            StageConfig(
                name="image_decode",
                process="image_decoder",
                factory=_FACTORY,
                gpu=0,
                terminal=True,
            )
        ],
    )


def test_pipeline_without_model_policy_rejects_sequence_parallelism() -> None:
    config = _image_pipeline_config(PipelineConfig)

    with pytest.raises(typer.BadParameter, match="does not support sequence parallel"):
        apply_parallelism_cli_overrides(
            config,
            thinker_tp_size=None,
            thinker_gpus=None,
            talker_gpu=None,
            code2wav_gpu=None,
            image_decoder_sp_size=2,
            image_decoder_gpus="0,1",
        )


def test_image_decoder_gpu_override_preserves_existing_tp() -> None:
    config = _image_pipeline_config(_GenericImagePipelineConfig)
    decoder = config.stages[0]
    decoder.tp_size = 2
    decoder.parallelism = decoder.parallelism.model_copy(update={"tp": 2})
    decoder.gpu = [0, 1]

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        talker_gpu=None,
        code2wav_gpu=None,
        image_decoder_gpus="2,3",
    )

    assert decoder.tp_size == 2
    assert decoder.parallelism.tp == 2
    assert decoder.gpu == [2, 3]


def test_image_decoder_sp_rejects_existing_tp_without_mutating_it() -> None:
    config = _image_pipeline_config(_GenericImagePipelineConfig)
    decoder = config.stages[0]
    decoder.tp_size = 2
    decoder.parallelism = decoder.parallelism.model_copy(update={"tp": 2})
    decoder.gpu = [0, 1]

    with pytest.raises(typer.BadParameter, match="cannot enable TP and SP"):
        apply_parallelism_cli_overrides(
            config,
            thinker_tp_size=None,
            thinker_gpus=None,
            talker_gpu=None,
            code2wav_gpu=None,
            image_decoder_sp_size=2,
            image_decoder_gpus="2,3",
        )

    assert decoder.tp_size == 2
    assert decoder.parallelism.tp == 2
    assert decoder.gpu == [0, 1]


def test_image_decoder_defaults_use_pipeline_attention_head_metadata() -> None:
    config = _image_pipeline_config(_GenericImagePipelineConfig)

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        talker_gpu=None,
        code2wav_gpu=None,
        image_decoder_sp_size=4,
        image_decoder_gpus="0,1,2,3",
    )

    decoder = config.stages[0]
    assert decoder.parallelism.sp == 4
    assert decoder.parallelism.ulysses_degree == 4
    assert decoder.parallelism.ring_degree == 1


def test_image_decoder_does_not_apply_model_specific_sp_size_rules() -> None:
    config = _image_pipeline_config(_GenericImagePipelineConfig)

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        talker_gpu=None,
        code2wav_gpu=None,
        image_decoder_sp_size=3,
        image_decoder_gpus="0,1,2",
    )

    decoder = config.stages[0]
    assert decoder.parallelism.sp == 3
    assert decoder.parallelism.ulysses_degree == 3
    assert decoder.parallelism.ring_degree == 1


def test_image_decoder_sp1_rejects_nontrivial_decomposition() -> None:
    config = _image_pipeline_config(_GenericImagePipelineConfig)
    before = config.model_dump()

    with pytest.raises(typer.BadParameter, match="must equal"):
        apply_parallelism_cli_overrides(
            config,
            thinker_tp_size=None,
            thinker_gpus=None,
            talker_gpu=None,
            code2wav_gpu=None,
            image_decoder_sp_size=1,
            image_decoder_gpus="0",
            image_decoder_ulysses_degree=2,
            image_decoder_ring_degree=2,
        )

    assert config.model_dump() == before


def test_image_decoder_rejects_invalid_attention_head_metadata() -> None:
    config = _image_pipeline_config(_InvalidImagePipelineConfig)

    with pytest.raises(typer.BadParameter, match="attention head count must be >= 1"):
        apply_parallelism_cli_overrides(
            config,
            thinker_tp_size=None,
            thinker_gpus=None,
            talker_gpu=None,
            code2wav_gpu=None,
            image_decoder_sp_size=4,
            image_decoder_gpus="0,1,2,3",
        )


def test_image_decoder_policy_canonicalizes_required_backend_and_options() -> None:
    config = _image_pipeline_config(_AliasedBackendPipelineConfig)

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        talker_gpu=None,
        code2wav_gpu=None,
        image_decoder_sp_size=2,
        image_decoder_gpus="0,1",
    )

    policy = type(config).sequence_parallel_policy(stage_name="image_decode")
    assert policy.resolve_backend(None, sp_size=2) == "diffusers"
    assert policy.supports_attention_backend("diffusers")
