# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import sglang_omni.config as config_module

from sglang_omni.config import (
    PipelineConfig,
    PlacementConfig,
    SGLangServerArgsConfig,
    SequenceParallelPolicy,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
)

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


class _SequenceParallelPipelineConfig(PipelineConfig):
    @classmethod
    def sequence_parallel_policy(
        cls, *, stage_name: str
    ) -> SequenceParallelPolicy | None:
        if stage_name == "stage":
            return SequenceParallelPolicy(
                attention_heads=6,
                requires_power_of_two=True,
            )
        return None


def _stage(**kwargs) -> StageConfig:
    data = {
        "name": "stage",
        "process": "pipeline",
        "factory": _FACTORY,
        "terminal": True,
    }
    data.update(kwargs)
    return StageConfig(**data)


def test_stage_runtime_schema_accepts_typed_runtime_values() -> None:
    stage = _stage(
        runtime=StageRuntimeConfig(
            resources=StageResourceConfig(total_gpu_memory_fraction=0.25),
            max_seq_len=8192,
            video_fps=2.0,
            sglang_server_args=SGLangServerArgsConfig(mem_fraction_static=0.7),
        ),
        runtime_arg_map={"max_seq_len": "thinker_max_seq_len"},
    )

    assert stage.runtime.resources.total_gpu_memory_fraction == 0.25
    assert stage.runtime.sglang_server_args.mem_fraction_static == 0.7
    assert stage.runtime_arg_map["max_seq_len"] == "thinker_max_seq_len"


def test_invalid_total_gpu_memory_fraction_raises() -> None:
    with pytest.raises(ValueError, match="total_gpu_memory_fraction"):
        StageResourceConfig(total_gpu_memory_fraction=0.0)


def test_invalid_sglang_mem_fraction_static_raises() -> None:
    with pytest.raises(ValueError, match="mem_fraction_static"):
        SGLangServerArgsConfig(mem_fraction_static=1.0)


def test_invalid_stage_runtime_values_raise() -> None:
    with pytest.raises(ValueError, match="max_seq_len"):
        StageRuntimeConfig(max_seq_len=0)
    with pytest.raises(ValueError, match="video_fps"):
        StageRuntimeConfig(video_fps=-1.0)


def test_stage_rejects_terminal_with_next() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                _stage(name="source", next="sink", terminal=True),
                _stage(name="sink"),
            ],
        )


def test_tp_size_syncs_into_parallelism_tp() -> None:
    stage = _stage(tp_size=2, gpu=[0, 1])

    assert stage.tp_size == 2
    assert stage.parallelism.tp == 2
    assert hasattr(config_module, "ParallelismConfig")
    assert not hasattr(config_module, "ResolvedStageParallelism")
    assert not hasattr(config_module, "resolve_stage_parallelism")


def test_parallelism_tp_syncs_back_to_tp_size() -> None:
    stage = _stage(
        gpu=[0, 1],
        parallelism=config_module.ParallelismConfig(tp=2),
    )

    assert stage.tp_size == 2
    assert stage.parallelism.tp == 2


def test_stage_rejects_conflicting_tp_aliases() -> None:
    with pytest.raises(ValueError, match="tp_size.*parallelism.tp"):
        _stage(
            tp_size=2,
            gpu=[0, 1, 2, 3],
            parallelism=config_module.ParallelismConfig(tp=4),
        )


def test_stage_accepts_sequence_parallel_settings_in_parallelism_config() -> None:
    parallelism_config = config_module.ParallelismConfig(
        sp=4,
        ulysses_degree=2,
        ring_degree=2,
    )

    stage = _stage(
        gpu=[0, 1, 2, 3],
        parallelism=parallelism_config,
    )

    assert stage.tp_size == 1
    assert stage.parallelism == parallelism_config


def test_stage_schema_accepts_non_power_of_two_sp() -> None:
    stage = _stage(
        gpu=[0, 1, 2],
        parallelism={"sp": 3, "ulysses_degree": 3},
    )

    assert stage.parallelism.sp == 3
    assert stage.parallelism.ulysses_degree == 3
    assert stage.parallelism.ring_degree == 1


def test_stage_rejects_combined_tp_and_sp() -> None:
    with pytest.raises(ValueError, match="cannot enable TP and SP"):
        _stage(
            tp_size=2,
            gpu=[0, 1, 2, 3],
            parallelism={"tp": 2, "sp": 2},
        )


def test_pipeline_rejects_sequence_parallelism_without_model_policy() -> None:
    with pytest.raises(ValueError, match="does not support sequence parallelism"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                _stage(
                    gpu=[0, 1],
                    parallelism={"sp": 2},
                )
            ],
        )


def test_pipeline_applies_model_sequence_parallel_constraints() -> None:
    with pytest.raises(ValueError, match="power of two"):
        _SequenceParallelPipelineConfig(
            model_path="dummy",
            stages=[
                _stage(
                    gpu=[0, 1, 2],
                    parallelism={"sp": 3},
                )
            ],
        )

    with pytest.raises(ValueError, match="must divide.*6 attention heads"):
        _SequenceParallelPipelineConfig(
            model_path="dummy",
            stages=[
                _stage(
                    gpu=[0, 1, 2, 3],
                    parallelism={"sp": 4, "ulysses_degree": 4},
                )
            ],
        )


def test_pipeline_accepts_model_supported_sequence_parallelism() -> None:
    config = _SequenceParallelPipelineConfig(
        model_path="dummy",
        stages=[
            _stage(
                gpu=[0, 1, 2, 3],
                parallelism={
                    "sp": 4,
                    "ulysses_degree": 2,
                    "ring_degree": 2,
                },
            )
        ],
    )

    assert config.stages[0].parallelism.sp == 4


def test_pipeline_sequence_parallelism_requires_one_gpu_per_rank() -> None:
    with pytest.raises(ValueError, match="one GPU id per SP rank"):
        _SequenceParallelPipelineConfig(
            model_path="dummy",
            stages=[
                _stage(
                    gpu=0,
                    parallelism={"sp": 2},
                )
            ],
        )


def test_pipeline_accepts_placement_config() -> None:
    config = PipelineConfig(
        model_path="dummy",
        placement=PlacementConfig(max_total_gpu_memory_fraction_per_gpu=0.95),
        stages=[_stage()],
    )

    assert config.placement.max_total_gpu_memory_fraction_per_gpu == 0.95


def test_invalid_placement_limit_raises() -> None:
    with pytest.raises(ValueError, match="max_total_gpu_memory_fraction_per_gpu"):
        PlacementConfig(max_total_gpu_memory_fraction_per_gpu=1.1)
