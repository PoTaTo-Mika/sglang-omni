# SPDX-License-Identifier: Apache-2.0
"""Prompt construction and SGLang scheduler adapters for VoxCPM."""

from __future__ import annotations

import base64
import collections
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from sglang_omni.models.voxcpm_common.payload_types import VoxCPMState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.utils.audio_payload import audio_waveform_payload

_MAX_AUDIO_TEXT_RATIO = 6
_MAX_AUDIO_TEXT_SLACK = 10

try:
    from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
except ModuleNotFoundError as error:
    if error.name != "sglang":
        raise
    from sglang_omni.scheduling.types import ARRequestData

    @dataclass
    class SGLangARRequestData(ARRequestData):  # type: ignore[no-redef]
        """CPU import fallback; serving replaces this with the backend class."""

        req: Any = None
        synced: bool = False
        generation_steps: int = 0
        suppress_tokens: list[int] | None = None
        top_p: float = 1.0
        top_k: int = -1
        repetition_penalty: float = 1.0
        input_embeds_are_projected: bool = False
        prefill_input_embeds: torch.Tensor | None = None
        decode_input_embeds: list[torch.Tensor] = field(default_factory=list)
        stage_payload: Any = None
        pending_feedback_queue: Any = field(default_factory=collections.deque)


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    value = (
        config.get(name, default)
        if isinstance(config, dict)
        else getattr(config, name, default)
    )
    return default if value is None else value


class CJKSplitTokenizer:
    """AutoTokenizer adapter that splits pure multi-CJK vocabulary entries."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        vocabulary = tokenizer.get_vocab()
        self.multichar_cjk = {
            token
            for token in vocabulary
            if len(token.replace("▁", "")) >= 2
            and all(_is_cjk(char) for char in token.replace("▁", ""))
        }

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenizer.tokenize(text)
        split: list[str] = []
        for token in tokens:
            clean = token.replace("▁", "")
            if token in self.multichar_cjk or clean in self.multichar_cjk:
                split.extend(list(clean))
            else:
                split.append(token)
        ids = self.tokenizer.convert_tokens_to_ids(split)
        return [int(item) for item in ids]

    def __len__(self) -> int:
        return len(self.tokenizer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)


def _is_cjk(char: str) -> bool:
    value = ord(char)
    return (
        0x3400 <= value <= 0x4DBF
        or 0x4E00 <= value <= 0x9FFF
        or 0xF900 <= value <= 0xFAFF
        or 0x3040 <= value <= 0x30FF
        or 0xAC00 <= value <= 0xD7AF
    )


def load_voxcpm_tokenizer(checkpoint_dir: str) -> CJKSplitTokenizer:
    from transformers import AutoTokenizer

    return CJKSplitTokenizer(
        AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    )


@dataclass
class VoxCPMSGLangRequestData(SGLangARRequestData):
    enforce_request_limits: bool = True
    state: VoxCPMState | None = None
    feat: torch.Tensor | None = None
    feat_mask: torch.Tensor | None = None
    initial_decode_context: torch.Tensor | None = None
    row_id: int | None = None
    continue_token_id: int = 0
    stop_token_id: int = 0
    engine_start_s: float = 0.0
    generated_latents: list[torch.Tensor] = field(default_factory=list)
    generated_audio: list[torch.Tensor] = field(default_factory=list)
    stop_step: int | None = None
    finish_kind: str | None = None
    is_streaming: bool = False


def _audio_source(reference: Any) -> Any:
    if not isinstance(reference, dict):
        return reference
    if reference.get("audio_path") is not None:
        return reference["audio_path"]
    if reference.get("bytes") is not None:
        return bytes(reference["bytes"])
    if reference.get("data") is not None:
        return base64.b64decode(reference["data"], validate=True)
    raise ValueError("VoxCPM reference audio is empty")


def _encode_reference(
    reference: Any,
    *,
    vae: Any,
    device: torch.device,
    patch_size: int,
    feat_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from sglang_omni.utils.audio import load_audio

    waveform = load_audio(
        _audio_source(reference),
        source_name="VoxCPM reference",
        target_sample_rate=int(vae.sample_rate),
        mono=True,
    )
    wav = torch.as_tensor(waveform, dtype=torch.float32, device=device).reshape(1, -1)
    align = int(patch_size) * int(vae.encoder_chunk_size)
    left_pad = (-int(wav.shape[-1])) % align
    if left_pad:
        wav = torch.nn.functional.pad(wav, (left_pad, 0))
    with torch.inference_mode():
        flat = (
            vae.encode(wav, int(vae.sample_rate))
            .transpose(1, 2)
            .reshape(-1, feat_dim)
            .float()
        )
    return flat.reshape(-1, patch_size, feat_dim), flat


def build_prompt_tensors(
    *,
    variant: str,
    tokenizer: CJKSplitTokenizer,
    target_text: str,
    reference_text: str | None,
    reference_mode: str,
    reference_latents: torch.Tensor | None,
    patch_size: int,
    feat_dim: int,
    audio_start_token_id: int = 101,
    ref_audio_start_token_id: int = 103,
    ref_audio_end_token_id: int = 104,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return exact text ids, aligned feature rows, masks and decoder context."""
    continuation = reference_mode == "continuation" and reference_latents is not None
    text = (reference_text or "") + target_text if continuation else target_text
    text_ids = tokenizer.encode(text) + [int(audio_start_token_id)]
    zeros = torch.zeros(len(text_ids), patch_size, feat_dim, dtype=torch.float32)
    mask = torch.zeros(len(text_ids), dtype=torch.bool)
    decode_context = None

    if reference_latents is None:
        ids = text_ids
        feat = zeros
    elif variant == "voxcpm2" and reference_mode == "isolated":
        count = int(reference_latents.shape[0])
        ids = (
            [int(ref_audio_start_token_id)]
            + [0] * count
            + [int(ref_audio_end_token_id)]
            + text_ids
        )
        padding = torch.zeros(1, patch_size, feat_dim, dtype=torch.float32)
        feat = torch.cat((padding, reference_latents.cpu(), padding, zeros), 0)
        mask = torch.cat(
            (
                torch.zeros(1, dtype=torch.bool),
                torch.ones(count, dtype=torch.bool),
                torch.zeros(1 + len(text_ids), dtype=torch.bool),
            )
        )
    else:
        count = int(reference_latents.shape[0])
        ids = text_ids + [0] * count
        feat = torch.cat((zeros, reference_latents.cpu()), 0)
        mask = torch.cat(
            (
                torch.zeros(len(text_ids), dtype=torch.bool),
                torch.ones(count, dtype=torch.bool),
            )
        )
        context_frames = 4 if variant == "voxcpm" else 12
        decode_context = reference_latents.reshape(-1, feat_dim)[
            -context_frames:
        ].contiguous()

    if not (len(ids) == feat.shape[0] == mask.shape[0]):
        raise RuntimeError("VoxCPM prompt ids/features/mask length mismatch")
    return torch.tensor(ids, dtype=torch.long), feat, mask, decode_context


def validate_prompt_length(
    prompt_len: int, max_new_tokens: int, max_model_len: int
) -> None:
    if prompt_len > max_model_len:
        raise ValueError(f"VoxCPM prompt is too long: {prompt_len} > {max_model_len}")
    if prompt_len + max_new_tokens > max_model_len:
        raise ValueError(
            "VoxCPM prompt plus max_new_tokens exceeds model context: "
            f"{prompt_len} + {max_new_tokens} > {max_model_len}"
        )


def cap_generation_length(target_token_count: int, max_new_tokens: int) -> int:
    """Apply the official VoxCPM text-relative generation safety bound."""
    if target_token_count < 1:
        raise ValueError("VoxCPM target_token_count must be positive")
    return min(
        int(max_new_tokens),
        target_token_count * _MAX_AUDIO_TEXT_RATIO + _MAX_AUDIO_TEXT_SLACK,
    )


def make_voxcpm_scheduler_adapters(
    *,
    model: Any,
    tokenizer: CJKSplitTokenizer,
    vae: Any,
    variant: str,
    reset_request: Callable[[str], None],
):
    def request_builder(payload: StagePayload) -> VoxCPMSGLangRequestData:
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        state = VoxCPMState.from_dict(payload.data)
        kwargs = state.generation_kwargs
        patch_size = int(model.model.patch_size)
        feat_dim = int(model.model.feat_dim)
        device = next(model.parameters()).device
        reference_latents = None
        flat_reference = None
        if state.reference_audio is not None:
            reference_latents, flat_reference = _encode_reference(
                state.reference_audio,
                vae=vae,
                device=device,
                patch_size=patch_size,
                feat_dim=feat_dim,
            )
        config = getattr(model, "config", None)
        input_ids, feat, feat_mask, decode_context = build_prompt_tensors(
            variant=variant,
            tokenizer=tokenizer,
            target_text=state.target_text,
            reference_text=state.reference_text,
            reference_mode=state.reference_mode,
            reference_latents=reference_latents,
            patch_size=patch_size,
            feat_dim=feat_dim,
            audio_start_token_id=int(_config_value(config, "audio_start_token", 101)),
            ref_audio_start_token_id=int(
                _config_value(config, "ref_audio_start_token", 103)
            ),
            ref_audio_end_token_id=int(
                _config_value(config, "ref_audio_end_token", 104)
            ),
        )
        del flat_reference

        stop_id = int(getattr(tokenizer, "eos_token_id", None) or 2)
        continue_id = int(_config_value(config, "audio_start_token", 101))
        max_new_tokens = cap_generation_length(
            len(tokenizer.encode(state.target_text)),
            int(kwargs["max_new_tokens"]),
        )
        kwargs["max_new_tokens"] = max_new_tokens
        max_model_len = int(_config_value(config, "max_position_embeddings", 32768))
        validate_prompt_length(int(input_ids.shape[0]), max_new_tokens, max_model_len)
        sampling = SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            stop_token_ids=[stop_id],
        )
        sampling.normalize(None)
        vocab_size = int(
            _config_value(
                _config_value(config, "lm_config", None), "vocab_size", len(tokenizer)
            )
        )
        sampling.verify(vocab_size)
        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids.tolist(),
            sampling_params=sampling,
            eos_token_ids={stop_id},
            vocab_size=vocab_size,
            extra_key=f"voxcpm-continuous:{payload.request_id}",
        )
        req.tokenizer = None
        req._input_embeds_are_projected = True
        data = VoxCPMSGLangRequestData(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            output_ids=req.output_ids,
            req=req,
            state=state,
            feat=feat,
            feat_mask=feat_mask,
            initial_decode_context=decode_context,
            continue_token_id=continue_id,
            stop_token_id=stop_id,
            engine_start_s=time.perf_counter(),
            input_embeds_are_projected=True,
            is_streaming=bool((payload.request.params or {}).get("stream", False)),
        )
        data.stage_payload = payload
        return data

    def result_adapter(data: VoxCPMSGLangRequestData) -> StagePayload:
        request_id = data.stage_payload.request_id
        try:
            payload = data.stage_payload
            state = data.state
            waveform = (
                torch.cat(data.generated_audio, dim=-1)
                if data.generated_audio
                else torch.empty(0, dtype=torch.float32)
            )
            sample_rate = int(vae.out_sample_rate)
            state.audio_waveform = waveform
            state.sample_rate = sample_rate
            state.prompt_tokens = int(data.input_ids.shape[0])
            state.completion_tokens = len(data.generated_latents)
            state.engine_time_s = time.perf_counter() - data.engine_start_s
            payload.data = state.to_dict()
            payload.data.update(
                audio_waveform_payload(
                    waveform,
                    sample_rate=sample_rate,
                    modality="audio",
                    source_hint="VoxCPM",
                )
            )
            usage = build_usage(state)
            if usage is not None:
                payload.data["usage"] = usage
            payload.data["finish_reason"] = data.finish_kind or "stop"
            return payload
        finally:
            reset_request(request_id)

    return request_builder, result_adapter


__all__ = [
    "CJKSplitTokenizer",
    "VoxCPMSGLangRequestData",
    "build_prompt_tensors",
    "load_voxcpm_tokenizer",
    "make_voxcpm_scheduler_adapters",
    "validate_prompt_length",
]
