# SPDX-License-Identifier: Apache-2.0
"""CLI override tests for the long-audio chunking flags."""

from __future__ import annotations

import pytest
import typer

from sglang_omni.cli.serve import apply_audio_chunking_cli_overrides
from sglang_omni.config import PipelineConfig
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    resolve_factory_signature_args,
    resolve_stage_factory_arg_defaults,
    resolve_stage_static_factory_args,
)
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
from sglang_omni.models.whisper_asr.config import WhisperASRPipelineConfig
from sglang_omni.utils.imports import import_string


def _asr_stage_args(config: PipelineConfig) -> dict[str, object]:
    stage = next(s for s in config.stages if s.name == "asr")
    return resolve_factory_signature_args(
        import_string(stage.factory),
        resolve_stage_static_factory_args(stage, config),
        defaults=resolve_stage_factory_arg_defaults(stage, config),
    )


def test_chunk_length_is_mirrored_into_the_qwen3_asr_stage():
    config = Qwen3ASRPipelineConfig(model_path="dummy")
    assert _asr_stage_args(config)["max_audio_clip_s"] == 60.0

    # A stale mirrored value from a model_dump round-trip must not win.
    dump = config.model_dump()
    dump["stages"][0]["factory_args"]["max_audio_clip_s"] = 30.0
    rebuilt = Qwen3ASRPipelineConfig(**dump)
    assert _asr_stage_args(rebuilt)["max_audio_clip_s"] == 60.0


def test_dotted_override_reaches_field_and_stage():
    manager = ConfigManager(Qwen3ASRPipelineConfig(model_path="dummy"))
    # The value is a string, as parse_extra_args delivers it.
    merged = manager.merge_config({"audio_chunking.max_audio_clip_s": "120"})
    assert merged.audio_chunking.max_audio_clip_s == 120.0
    assert _asr_stage_args(merged)["max_audio_clip_s"] == 120.0


def test_cli_sets_both_chunking_values():
    config = Qwen3ASRPipelineConfig(model_path="dummy")
    apply_audio_chunking_cli_overrides(
        config, max_audio_clip_s=1200.0, max_concurrent_chunks=64
    )
    assert config.audio_chunking.max_audio_clip_s == 1200.0
    assert config.audio_chunking.max_concurrent_chunks == 64
    assert config.audio_chunking.allow_audio_chunking is True
    assert _asr_stage_args(config)["max_audio_clip_s"] == 1200.0


def test_clip_length_is_bounded_by_the_native_limit():
    config = WhisperASRPipelineConfig(model_path="dummy")
    apply_audio_chunking_cli_overrides(
        config, max_audio_clip_s=15.0, max_concurrent_chunks=None
    )
    assert config.audio_chunking.max_audio_clip_s == 15.0
    assert "max_audio_clip_s" not in _asr_stage_args(config)

    with pytest.raises(typer.BadParameter, match="max_native_clip_s"):
        apply_audio_chunking_cli_overrides(
            config, max_audio_clip_s=60.0, max_concurrent_chunks=None
        )


def test_pipelines_without_chunking_are_rejected():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter, match="supports audio chunking"):
        apply_audio_chunking_cli_overrides(
            config, max_audio_clip_s=None, max_concurrent_chunks=16
        )


def test_omitting_both_flags_is_a_noop():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    before = config.audio_chunking
    apply_audio_chunking_cli_overrides(
        config, max_audio_clip_s=None, max_concurrent_chunks=None
    )
    assert config.audio_chunking is before
