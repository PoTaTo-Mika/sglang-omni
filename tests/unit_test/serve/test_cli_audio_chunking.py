# SPDX-License-Identifier: Apache-2.0
"""Audio chunking is plain configuration: dotted overrides reach it."""

from __future__ import annotations

import pytest

from sglang_omni.config import PipelineConfig
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    apply_typed_stage_kwargs,
    resolve_factory_signature_args,
    resolve_stage_factory_arg_defaults,
    resolve_stage_factory_kwargs,
    resolve_stage_typed_kwargs,
)
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
from sglang_omni.models.whisper_asr.config import WhisperASRPipelineConfig
from sglang_omni.utils.imports import import_string


def _asr_factory_kwargs(config: PipelineConfig) -> dict[str, object]:
    stage = next(s for s in config.stages if s.name == "asr")
    factory = import_string(stage.factory_path)
    args = apply_typed_stage_kwargs(
        factory,
        resolve_stage_factory_kwargs(stage, config),
        resolve_stage_typed_kwargs(stage),
        stage_name=stage.name,
    )
    return resolve_factory_signature_args(
        factory,
        args,
        defaults=resolve_stage_factory_arg_defaults(stage, config),
    )


def test_chunk_length_is_mirrored_into_the_qwen3_asr_stage():
    config = Qwen3ASRPipelineConfig(model_path="dummy")
    assert _asr_factory_kwargs(config)["max_audio_clip_s"] == 30.0

    # A stale mirrored value from a model_dump round-trip must not win.
    dump = config.model_dump()
    dump["stages"][0]["factory"]["max_audio_clip_s"] = 15.0
    rebuilt = Qwen3ASRPipelineConfig(**dump)
    assert _asr_factory_kwargs(rebuilt)["max_audio_clip_s"] == 30.0


def test_dotted_override_reaches_field_and_stage():
    manager = ConfigManager(Qwen3ASRPipelineConfig(model_path="dummy"))
    # The value is a string, as the command line delivers it.
    merged = manager.merge_config({"audio_chunking.max_audio_clip_s": "120"})
    assert merged.audio_chunking.max_audio_clip_s == 120.0
    assert _asr_factory_kwargs(merged)["max_audio_clip_s"] == 120.0


def test_dotted_override_sets_the_concurrency_cap():
    manager = ConfigManager(Qwen3ASRPipelineConfig(model_path="dummy"))
    merged = manager.merge_config({"audio_chunking.max_concurrent_chunks": "64"})
    assert merged.audio_chunking.max_concurrent_chunks == 64


def test_dotted_override_sets_the_total_audio_limit():
    manager = ConfigManager(Qwen3ASRPipelineConfig(model_path="dummy"))
    merged = manager.merge_config({"audio_chunking.max_total_audio_s": "7200"})
    assert merged.audio_chunking.max_total_audio_s == 7200.0


def test_clip_length_past_the_native_limit_is_rejected():
    manager = ConfigManager(WhisperASRPipelineConfig(model_path="dummy"))
    with pytest.raises(ValueError, match="native clip limit"):
        manager.merge_config({"audio_chunking.max_audio_clip_s": "60"})


@pytest.mark.parametrize(
    "path",
    [
        "audio_chunking.allow_audio_chunking",
        "audio_chunking.max_native_clip_s",
        "audio_chunking.min_tail_s",
        "audio_chunking.condition_on_previous_text",
    ],
)
def test_model_owned_fields_are_not_reachable_paths(path):
    # The model-owned side of the contract lives on ClassVars: these are not
    # fields, so the path does not even compile -- same as overriding
    # ``architecture``.
    manager = ConfigManager(Qwen3ASRPipelineConfig(model_path="dummy"))
    with pytest.raises(ValueError, match=path.split(".", 1)[1]):
        manager.merge_config({path: "1"})
