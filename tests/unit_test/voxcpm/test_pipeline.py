# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.models.voxcpm.config import VoxCPMPipelineConfig
from sglang_omni.models.voxcpm2.config import VoxCPM2PipelineConfig
from sglang_omni.models.voxcpm_common.request_builders import (
    build_generation_kwargs,
    build_voxcpm_state,
)
from sglang_omni.models.voxcpm_common.engine_builder import VoxCPMEngineBuilder
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.protocol import CreateSpeechRequest
from sglang_omni.utils.hf import try_resolve_arch_from_raw_config


def _payload(
    *,
    text: str = "你好",
    ref_audio: str | None = None,
    ref_text: str | None = None,
    params: dict | None = None,
) -> StagePayload:
    tts_params = {}
    if ref_audio is not None:
        tts_params["ref_audio"] = ref_audio
    if ref_text is not None:
        tts_params["ref_text"] = ref_text
    return StagePayload(
        request_id="r0",
        request=OmniRequest(
            inputs={"text": text},
            params=params or {},
            metadata={"tts_params": tts_params},
        ),
        data={},
    )


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("voxcpm", VoxCPMPipelineConfig),
        ("VoxCPMForConditionalGeneration", VoxCPMPipelineConfig),
        ("voxcpm2", VoxCPM2PipelineConfig),
        ("VoxCPM2TalkerForConditionalGeneration", VoxCPM2PipelineConfig),
    ],
)
def test_registry_resolves_voxcpm_architectures(architecture, expected):
    assert PIPELINE_CONFIG_REGISTRY.get_config(architecture) is expected


@pytest.mark.parametrize(
    ("config_cls", "variant"),
    [(VoxCPMPipelineConfig, "voxcpm"), (VoxCPM2PipelineConfig, "voxcpm2")],
)
def test_pipeline_wiring(config_cls, variant):
    config = config_cls(model_path="checkpoint")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
    ]
    assert config.stages[0].next == "tts_engine"
    assert config.stages[1].terminal
    assert config.stages[0].factory_args["variant"] == variant
    assert config.stages[1].factory_args["variant"] == variant
    assert config_cls.mem_fraction_role_to_stage() == {"talker": "tts_engine"}
    assert config_cls.generation_sglang_role_to_stage() == {"generation": "tts_engine"}


def test_raw_model_type_resolution(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"voxcpm2"}')
    assert try_resolve_arch_from_raw_config(str(checkpoint)) == "voxcpm2"


def test_engine_builder_injects_missing_voxcpm_model_type(tmp_path):
    config = {
        "architecture": "voxcpm",
        "lm_config": {
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
            "max_position_embeddings": 128,
            "vocab_size": 32,
        },
    }
    (tmp_path / "config.json").write_text(__import__("json").dumps(config))
    builder = VoxCPMEngineBuilder(variant="voxcpm")
    builder.pre_infra_setup(str(tmp_path))
    assert builder.config.model_type == "voxcpm"
    assert builder.config.architectures == ["VoxCPMSGLangModel"]
    assert builder.context_length == 128


def test_voxcpm1_reference_allows_optional_transcript():
    without_text = build_voxcpm_state(
        _payload(ref_audio="data:audio/wav;base64,AAAA"), variant="voxcpm"
    )
    with_text = build_voxcpm_state(
        _payload(
            ref_audio="data:audio/wav;base64,AAAA",
            ref_text="reference",
        ),
        variant="voxcpm",
    )
    assert without_text.reference_mode == "continuation"
    assert without_text.reference_text is None
    assert with_text.reference_text == "reference"


def test_voxcpm2_reference_modes():
    isolated = build_voxcpm_state(
        _payload(ref_audio="data:audio/wav;base64,AAAA"), variant="voxcpm2"
    )
    assert isolated.reference_mode == "isolated"
    continuation = build_voxcpm_state(
        _payload(
            ref_audio="data:audio/wav;base64,AAAA",
            ref_text="reference",
        ),
        variant="voxcpm2",
    )
    assert continuation.reference_mode == "continuation"


def test_generation_parameters_and_validation():
    values = build_generation_kwargs(
        {"max_new_tokens": 100},
        tts_params={
            "cfg_value": 2.5,
            "inference_timesteps": 12,
            "min_len": 3,
            "streaming_prefix_len": 4,
            "seed": 7,
        },
        variant="voxcpm2",
    )
    assert values == {
        "max_new_tokens": 100,
        "min_len": 3,
        "inference_timesteps": 12,
        "cfg_value": 2.5,
        "temperature": 1.0,
        "streaming_prefix_len": 4,
        "seed": 7,
    }
    with pytest.raises(ValueError, match="min_len"):
        build_generation_kwargs(
            {"max_new_tokens": 2},
            tts_params={"min_len": 2},
            variant="voxcpm",
        )


def test_speech_protocol_accepts_voxcpm_parameters():
    request = CreateSpeechRequest(
        input="hello",
        cfg_value=2.5,
        inference_timesteps=12,
        min_len=2,
        streaming_prefix_len=3,
    )
    assert request.cfg_value == 2.5
    assert request.inference_timesteps == 12
