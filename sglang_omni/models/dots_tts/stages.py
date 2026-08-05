# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the dots.tts pipeline."""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sglang_omni.models.dots_tts.components.model_config import ModelConfig
from sglang_omni.models.dots_tts.payload_types import (
    DotsTtsState,
    load_dots_tts_state,
    store_dots_tts_state,
)
from sglang_omni.models.dots_tts.prompt_builder import DOTS_TTS_MAX_SEQUENCE_LENGTH
from sglang_omni.models.dots_tts.request_builders import preprocess_dots_tts_payload
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.checkpoint import resolve_checkpoint

CONFIG_FILENAME = "config.json"
DEFAULT_PRECISION = "bfloat16"
DEFAULT_ODE_METHOD = "euler"
DEFAULT_EOS_THRESHOLD = 0.8


def create_preprocessing_executor(max_concurrency: int = 1) -> SimpleScheduler:
    return SimpleScheduler(preprocess_dots_tts_payload, max_concurrency=max_concurrency)


def create_reference_encode_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = DEFAULT_PRECISION,
    max_sequence_length: int = DOTS_TTS_MAX_SEQUENCE_LENGTH,
    max_concurrency: int = 1,
    ref_audio_cache_max_items: int = 64,
    ref_audio_cache_max_bytes: int = 512 * 1024 * 1024,
) -> SimpleScheduler:
    # note (chenyang): reference encoding for dots.tts is CPU-only waveform IO;
    # device/dtype are accepted so the stage keeps the shared TTS factory
    # signature and can be placed alongside the engine without config churn.
    del device, gpu_id, dtype

    from transformers import AutoTokenizer

    from sglang_omni.models.dots_tts.reference_encode import DotsTtsReferenceEncoder

    checkpoint_dir = resolve_checkpoint(model_path)
    model_config = _load_dots_tts_config(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    encoder = DotsTtsReferenceEncoder(
        tokenizer=tokenizer,
        sample_rate=int(model_config.vocoder.sample_rate),
        samples_per_patch=_samples_per_patch(model_config),
        model_identity=checkpoint_dir,
        max_sequence_length=max_sequence_length,
        cache_max_items=ref_audio_cache_max_items,
        cache_max_bytes=ref_audio_cache_max_bytes,
    )
    return SimpleScheduler(encoder.encode_payload, max_concurrency=max_concurrency)


def create_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = DEFAULT_PRECISION,
    ode_method: str = DEFAULT_ODE_METHOD,
    eos_threshold: float = DEFAULT_EOS_THRESHOLD,
    max_sequence_length: int = DOTS_TTS_MAX_SEQUENCE_LENGTH,
    optimize: bool = False,
    max_concurrency: int = 16,
) -> SimpleScheduler:
    resolved_device = _resolve_device(device, gpu_id)
    if bool(optimize) and int(max_concurrency) > 1:
        # note (chenyang): optimized StaticCache workspaces are process-global; keep
        # the eager path for concurrent SimpleScheduler workers until the AR loop
        # moves onto OmniScheduler / SGLang KV.
        raise ValueError(
            "dots.tts tts_engine max_concurrency > 1 requires optimize=false"
        )
    model = _load_model(
        resolve_checkpoint(model_path),
        resolved_device,
        dtype,
        bool(optimize),
        int(max_sequence_length),
    )
    patch_size = int(model.config.patch_size)

    def _generate(payload: StagePayload) -> StagePayload:
        state = load_dots_tts_state(payload)
        generate_inputs = _build_generate_inputs(state, device=resolved_device)
        if state.seed is not None:
            # note (chenyang): the flow-matching solver draws its noise from the
            # global RNG, so a seeded request is only reproducible while this
            # stage stays single-threaded (SimpleScheduler max_concurrency=1).
            torch.manual_seed(int(state.seed))

        started = time.perf_counter()
        latents = model.generate_latents(
            generate_inputs,
            precision=dtype,
            ode_method=ode_method,
            num_steps=state.num_steps,
            guidance_scale=state.guidance_scale,
            speaker_scale=state.speaker_scale,
            eos_threshold=eos_threshold,
        )
        state.engine_time_s = time.perf_counter() - started
        state.generated_latents = latents.detach().to(device="cpu", dtype=torch.float32)
        state.completion_tokens = int(latents.size(1)) // patch_size
        state.finish_reason = "stop"
        state.prompt_audio = None
        state.generation_schedule_ids = None
        return store_dots_tts_state(payload, state)

    return SimpleScheduler(_generate, max_concurrency=int(max_concurrency))


def create_audio_decode_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = DEFAULT_PRECISION,
    max_sequence_length: int = DOTS_TTS_MAX_SEQUENCE_LENGTH,
    optimize: bool = False,
    keep_latents: bool = False,
    max_batch_size: int = 1,
    max_batch_wait_ms: int = 0,
) -> SimpleScheduler:
    from sglang_omni.models.dots_tts.audio_decode import DotsTtsBatchVocoder
    from sglang_omni.models.dots_tts.components.vocoder.vocoder_inference import (
        VocoderInference,
    )

    resolved_device = _resolve_device(device, gpu_id)
    # Shares the lru_cache entry with the colocated tts_engine stage so the
    # 5 GB checkpoint is loaded once per process.
    model = _load_model(
        resolve_checkpoint(model_path),
        resolved_device,
        dtype,
        bool(optimize),
        int(max_sequence_length),
    )
    vocoder = DotsTtsBatchVocoder(
        VocoderInference(model.vocoder),
        sample_rate=int(model.config.vocoder.sample_rate),
        device=torch.device(resolved_device),
        keep_latents=keep_latents,
    )
    return vocoder.build_scheduler(
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


@lru_cache(maxsize=None)
def _load_model(
    checkpoint_dir: str,
    device: str,
    precision: str,
    optimize: bool,
    max_sequence_length: int,
) -> Any:
    from sglang_omni.models.dots_tts.components.model import DotsTtsModel
    from sglang_omni.models.dots_tts.components.utils.util import get_dtype

    model = DotsTtsModel.from_pretrained(checkpoint_dir)
    # Only the core network runs in low precision upstream; the vocoder and the
    # speaker encoder stay float32.
    model.core.to(dtype=get_dtype(precision))
    model = model.to(torch.device(device)).eval()
    model.set_optimize(optimize, max_sequence_length=max_sequence_length)
    return model


def _load_dots_tts_config(checkpoint_dir: str) -> ModelConfig:
    config_path = Path(checkpoint_dir) / CONFIG_FILENAME
    return ModelConfig.model_validate(
        json.loads(config_path.read_text(encoding="utf-8"))
    )


def _samples_per_patch(model_config: ModelConfig) -> int:
    hop_size = int(np.prod(model_config.vocoder.downsample_rates))
    return int(model_config.patch_size) * hop_size


def _resolve_device(device: str, gpu_id: int | None) -> str:
    if gpu_id is not None:
        return f"cuda:{int(gpu_id)}"
    return device


def _build_generate_inputs(
    state: DotsTtsState, *, device: str
) -> dict[str, torch.Tensor | str]:
    schedule_ids = state.generation_schedule_ids
    if not schedule_ids:
        raise RuntimeError(
            "dots.tts tts_engine requires a generation schedule from reference_encode"
        )
    prompt_audio = state.prompt_audio
    if prompt_audio is None:
        raise RuntimeError(
            "dots.tts tts_engine requires prompt audio from reference_encode"
        )
    if not state.prompt_text:
        raise RuntimeError(
            "dots.tts tts_engine requires the reference transcript for continuation"
        )

    return {
        "generation_schedule": torch.tensor(
            schedule_ids, dtype=torch.long, device=device
        ).unsqueeze(0),
        "prompt_audio": prompt_audio.to(device=device, dtype=torch.float32).reshape(
            1, -1
        ),
        "prompt_text": state.prompt_text,
        "text": state.text,
    }


__all__ = [
    "create_audio_decode_executor",
    "create_preprocessing_executor",
    "create_reference_encode_executor",
    "create_tts_engine_executor",
]
