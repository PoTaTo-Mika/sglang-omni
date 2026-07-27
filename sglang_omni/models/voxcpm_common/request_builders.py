# SPDX-License-Identifier: Apache-2.0
"""OpenAI speech payload lowering shared by VoxCPM and VoxCPM2."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.voxcpm_common.payload_types import VoxCPMState
from sglang_omni.proto import StagePayload


def build_voxcpm_state(payload: StagePayload, *, variant: str) -> VoxCPMState:
    inputs = payload.request.inputs or {}
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}

    text, references = _normalize_inputs(inputs)
    if not text.strip():
        raise ValueError(f"{variant} requires non-empty input text")
    if len(references) > 1:
        raise ValueError(f"{variant} accepts at most one reference audio")

    reference = references[0] if references else None
    if reference is None and tts_params.get("ref_audio") is not None:
        reference = _reference_from_value(tts_params["ref_audio"])
    reference_text = (
        reference.get("text") if reference is not None else None
    ) or tts_params.get("ref_text")

    if reference is None:
        reference_mode = "none"
        reference_audio = None
    else:
        reference_audio = _normalize_reference_audio(reference)
        if variant == "voxcpm":
            reference_mode = "continuation"
        else:
            reference_mode = (
                "continuation"
                if isinstance(reference_text, str) and reference_text.strip()
                else "isolated"
            )

    return VoxCPMState(
        variant=variant,
        target_text=text,
        reference_text=(
            reference_text.strip()
            if isinstance(reference_text, str) and reference_text.strip()
            else None
        ),
        reference_audio=reference_audio,
        reference_mode=reference_mode,
        generation_kwargs=build_generation_kwargs(
            params, tts_params=tts_params, variant=variant
        ),
        # The terminal adapter resolves this from the loaded AudioVAE config.
        sample_rate=0,
    )


def build_generation_kwargs(
    params: dict[str, Any], *, tts_params: dict[str, Any], variant: str
) -> dict[str, Any]:
    defaults = {
        "max_new_tokens": 4096 if variant == "voxcpm" else 8192,
        "min_len": 2,
        "inference_timesteps": 10,
        "cfg_value": 2.0,
        "temperature": 1.0,
        "streaming_prefix_len": 3,
    }
    values = dict(defaults)
    aliases = {
        "max_len": "max_new_tokens",
        "max_tokens": "max_new_tokens",
    }
    for source in (tts_params, params):
        for field in defaults:
            if source.get(field) is not None:
                values[field] = source[field]
        for alias, field in aliases.items():
            if source.get(alias) is not None:
                values[field] = source[alias]
        if source.get("seed") is not None:
            values["seed"] = source["seed"]

    values["max_new_tokens"] = int(values["max_new_tokens"])
    values["min_len"] = int(values["min_len"])
    values["inference_timesteps"] = int(values["inference_timesteps"])
    values["streaming_prefix_len"] = int(values["streaming_prefix_len"])
    values["cfg_value"] = float(values["cfg_value"])
    values["temperature"] = float(values["temperature"])
    if "seed" in values:
        values["seed"] = int(values["seed"])
    _validate_generation_kwargs(values)
    return values


def _normalize_inputs(inputs: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(inputs, str):
        return inputs, []
    if not isinstance(inputs, dict):
        return str(inputs) if inputs is not None else "", []
    references = inputs.get("references") or []
    if not isinstance(references, list):
        raise ValueError("VoxCPM references must be a list")
    if any(not isinstance(reference, dict) for reference in references):
        raise ValueError("VoxCPM references must be objects")
    return str(inputs.get("text", inputs.get("input", ""))), [
        dict(reference) for reference in references
    ]


def _reference_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.startswith("data:"):
        header, separator, data = value.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("VoxCPM ref_audio data URI must be base64 encoded")
        media_type = header.removeprefix("data:").split(";", 1)[0] or "audio/wav"
        return {"data": data, "media_type": media_type}
    return {"audio_path": str(value)}


def _normalize_reference_audio(reference: dict[str, Any]) -> dict[str, Any]:
    if reference.get("audio_path") is not None:
        return {"audio_path": str(reference["audio_path"])}
    for key in ("ref_audio", "audio"):
        if reference.get(key) is not None:
            return _reference_from_value(reference[key])
    if reference.get("bytes") is not None:
        return {"bytes": bytes(reference["bytes"])}
    data = reference.get("base64") or reference.get("data")
    if data is not None:
        return {
            "data": str(data),
            "media_type": str(reference.get("media_type") or "audio/wav"),
        }
    raise ValueError("VoxCPM reference has no audio payload")


def _validate_generation_kwargs(values: dict[str, Any]) -> None:
    if values["max_new_tokens"] < 1:
        raise ValueError("VoxCPM max_new_tokens must be positive")
    if values["min_len"] < 0 or values["min_len"] >= values["max_new_tokens"]:
        raise ValueError("VoxCPM min_len must be >= 0 and below max_new_tokens")
    if values["inference_timesteps"] < 1:
        raise ValueError("VoxCPM inference_timesteps must be positive")
    if values["cfg_value"] <= 0:
        raise ValueError("VoxCPM cfg_value must be positive")
    if values["temperature"] < 0:
        raise ValueError("VoxCPM temperature must be non-negative")
    if values["streaming_prefix_len"] < 1:
        raise ValueError("VoxCPM streaming_prefix_len must be positive")


__all__ = ["build_generation_kwargs", "build_voxcpm_state"]
