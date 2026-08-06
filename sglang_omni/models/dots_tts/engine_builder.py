# SPDX-License-Identifier: Apache-2.0
"""dots.tts SGLang engine builder."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

logger = logging.getLogger(__name__)


def _coerce_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


class DotsTtsEngineBuilder(TtsEngineBuilder):
    """Builds the OmniScheduler stage that owns the dots.tts AR backbone."""

    model_name = "dots.tts"

    def __init__(
        self,
        *,
        context_length: int,
        num_steps: int,
        eos_threshold: float,
        max_audio_patches: int,
        total_gpu_memory_fraction: float | None = None,
    ) -> None:
        from sglang_omni.models.dots_tts.hf_config import DOTS_TTS_MODEL_ARCH_OVERRIDE

        if int(num_steps) <= 0:
            raise ValueError(f"dots.tts num_steps must be positive, got {num_steps}")
        if not 0.0 < float(eos_threshold) < 1.0:
            raise ValueError(
                f"dots.tts eos_threshold must be in (0, 1), got {eos_threshold}"
            )
        if int(max_audio_patches) <= 0:
            raise ValueError(
                f"dots.tts max_audio_patches must be positive, got {max_audio_patches}"
            )

        self.model_arch_override = DOTS_TTS_MODEL_ARCH_OVERRIDE
        self.context_length = int(context_length)
        self.num_steps = int(num_steps)
        self.eos_threshold = float(eos_threshold)
        self.max_audio_patches = int(max_audio_patches)
        self.total_gpu_memory_fraction = total_gpu_memory_fraction
        self.special_tokens: Any = None
        self._model_runner: Any = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from sglang_omni.models.dots_tts.hf_config import register_dots_tts_hf_config

        register_dots_tts_hf_config(checkpoint_dir)

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            "max_running_requests": 8,
            "dtype": dtype,
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "max_prefill_tokens": min(int(self.context_length), 8192),
            "sampling_backend": "pytorch",
            "trust_remote_code": False,
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        # note (chenyang): context_length reaches ServerArgs through
        # build_sglang_server_args, so an override here would be applied twice.
        overrides.pop("context_length", None)
        overrides["tp_size"] = 1

        if "disable_radix_cache" in overrides and not _coerce_bool(
            overrides["disable_radix_cache"],
            name="server_args_overrides.disable_radix_cache",
        ):
            raise ValueError(
                "dots.tts prefix/radix cache is not currently supported: prompt "
                "audio spans carry continuous embeddings that token ids cannot key"
            )
        overrides["disable_radix_cache"] = True

        if (
            "chunked_prefill_size" in overrides
            and int(overrides.get("chunked_prefill_size") or 0) != 0
        ):
            raise ValueError(
                "dots.tts requires chunked_prefill_size=0 because the acoustic "
                "tail seeds its flow history from one whole prefill forward"
            )
        overrides["chunked_prefill_size"] = 0
        if bool(overrides.get("enable_torch_compile", False)):
            raise ValueError("dots.tts torch.compile is not currently supported")

    def infra_kwargs(self) -> dict[str, Any]:
        return {"total_gpu_memory_fraction": self.total_gpu_memory_fraction}

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        from transformers import AutoTokenizer

        from sglang_omni.models.dots_tts.engine_io import load_dots_tts_special_tokens

        model = model_worker.model_runner.model
        model.eval()
        model.init_tail(
            checkpoint_dir,
            nfe=self.num_steps,
            max_audio_patches=self.max_audio_patches,
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
        self.special_tokens = load_dots_tts_special_tokens(tokenizer)
        logger.info(
            "dots.tts SGLang startup: gpu_id=%s device=%s num_steps=%s "
            "eos_threshold=%s max_audio_patches=%s max_running_requests=%s "
            "radix_cache=%s",
            gpu_id,
            device,
            self.num_steps,
            self.eos_threshold,
            self.max_audio_patches,
            int(server_args.max_running_requests),
            not bool(server_args.disable_radix_cache),
        )

    def get_model_buffer_bs(self, model: Any) -> int | None:
        return int(model._decode_input_embedding.num_embeddings)

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang_omni.models.dots_tts.model_runner import DotsTTSModelRunner

        self._model_runner = DotsTTSModelRunner(
            model_worker,
            output_proc,
            nfe=self.num_steps,
            eos_threshold=self.eos_threshold,
        )
        return self._model_runner

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        from sglang_omni.models.dots_tts.engine_io import (
            make_dots_tts_scheduler_adapters,
        )

        return make_dots_tts_scheduler_adapters(
            model=model,
            special_tokens=self.special_tokens,
            reset_request=self._model_runner.reset_request,
            num_steps=self.num_steps,
            max_audio_patches=self.max_audio_patches,
        )

    def make_abort_callback(self) -> Any | None:
        return self._model_runner.reset_request


__all__ = ["DotsTtsEngineBuilder"]
