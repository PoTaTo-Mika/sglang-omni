# SPDX-License-Identifier: Apache-2.0
"""dots.tts SGLang engine builder."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from sglang_omni.models.dots_tts.hf_config import DOTS_TTS_MODEL_FILENAME
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

logger = logging.getLogger(__name__)

_DTYPE_BYTES = 2
# Headroom inside the SGLang share for its own activations and bookkeeping.
_SGLANG_SLACK_BYTES = 2 * 2**30
_DEFAULT_STAGE_SHARE = 0.84


def _total_gpu_bytes(gpu_id: int) -> int:
    return int(torch.cuda.get_device_properties(int(gpu_id)).total_memory)


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
        max_running_requests: int = 16,
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
        if int(max_running_requests) <= 0:
            raise ValueError(
                "dots.tts max_running_requests must be positive, got "
                f"{max_running_requests}"
            )

        self.model_arch_override = DOTS_TTS_MODEL_ARCH_OVERRIDE
        self.context_length = int(context_length)
        self.num_steps = int(num_steps)
        self.eos_threshold = float(eos_threshold)
        self.max_audio_patches = int(max_audio_patches)
        self.max_running_requests = int(max_running_requests)
        self.total_gpu_memory_fraction = total_gpu_memory_fraction
        self.special_tokens: Any = None
        self._model_runner: Any = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from sglang_omni.models.dots_tts.hf_config import (
            load_dots_tts_llm_config,
            load_dots_tts_model_config,
            register_dots_tts_hf_config,
        )

        register_dots_tts_hf_config(checkpoint_dir)
        self._model_config = load_dots_tts_model_config(checkpoint_dir)
        llm_config = load_dots_tts_llm_config(checkpoint_dir)
        head_dim = int(llm_config.hidden_size) // int(llm_config.num_attention_heads)
        self._kv_bytes_per_token = (
            2
            * _DTYPE_BYTES
            * int(llm_config.num_hidden_layers)
            * int(llm_config.num_key_value_heads)
            * head_dim
        )
        self._weight_bytes = sum(
            path.stat().st_size
            for path in (Path(checkpoint_dir) / DOTS_TTS_MODEL_FILENAME,)
        )

    def _sglang_mem_fraction(self) -> float:
        """Share of the card SGLang may take for weights plus backbone KV.

        note (chenyang): higgs and ming hand their whole stage fraction to
        ``mem_fraction_static`` because weights and KV are all their SGLang
        stage allocates. dots has a second consumer in the same stage — the
        acoustic tail pools, which SGLang knows nothing about — so the stage's
        declared share is split here and the remainder is checked against the
        pools in :meth:`setup_model`.
        """
        kv_bytes = (
            self.max_running_requests * self.context_length * self._kv_bytes_per_token
        )
        needed = self._weight_bytes + kv_bytes + _SGLANG_SLACK_BYTES
        fraction = needed / float(_total_gpu_bytes(self.gpu_id))
        stage_share = self.total_gpu_memory_fraction or _DEFAULT_STAGE_SHARE
        if fraction >= stage_share:
            raise ValueError(
                "dots.tts tts_engine needs "
                f"{fraction:.2f} of the card for weights + backbone KV alone, "
                f"which does not fit the declared stage share {stage_share:.2f}"
            )
        return round(fraction, 3)

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        # note (chenyang): the backbone only needs max_running_requests *
        # context_length tokens of KV (a few hundred MB), while the acoustic
        # tail pools are allocated after SGLang's and need the rest of the card.
        # Without this cap SGLang's auto budget takes ~60 GB of KV it will never
        # use and the tail pools only just fit.
        return {
            "mem_fraction_static": self._sglang_mem_fraction(),
            "max_running_requests": self.max_running_requests,
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

        # note (chenyang): --max-running-requests reaches this stage as a server
        # arg and also sizes the tail pools, so adopt it here or the startup
        # budget check would validate a slot count the engine never uses.
        requested = overrides.get("max_running_requests")
        if requested is not None and int(requested) != self.max_running_requests:
            logger.warning(
                "dots.tts max_running_requests=%s overrides the stage value %s; "
                "resizing the acoustic tail pools to match",
                int(requested),
                self.max_running_requests,
            )
            self.max_running_requests = int(requested)
            overrides["mem_fraction_static"] = self._sglang_mem_fraction()

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
        self._check_stage_budget(gpu_id)
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

    def _check_stage_budget(self, gpu_id: int) -> None:
        """Fail fast when the tail pools do not fit the declared stage share."""
        from sglang_omni.models.dots_tts.tail import tail_pool_bytes_per_slot

        config = self._model_config
        per_slot = tail_pool_bytes_per_slot(
            nfe=self.num_steps,
            patch_capacity=self.max_audio_patches + 1,
            unit_len=1 + int(config.patch_size),
            dit_layers=int(config.DiT.num_layers),
            dit_heads=int(config.DiT.num_heads),
            dit_head_dim=int(config.DiT.hidden_size) // int(config.DiT.num_heads),
            encoder_layers=int(config.PatchEncoder.num_layers),
            encoder_heads=int(config.PatchEncoder.num_heads),
            encoder_head_dim=(
                int(config.PatchEncoder.hidden_size)
                // int(config.PatchEncoder.num_heads)
            ),
            encoder_block=int(config.patch_size) // 2,
        )
        total = _total_gpu_bytes(gpu_id)
        stage_share = self.total_gpu_memory_fraction or _DEFAULT_STAGE_SHARE
        sglang_bytes = int(self._sglang_mem_fraction() * total)
        pool_budget = int(stage_share * total) - sglang_bytes
        needed = per_slot * self.max_running_requests
        gib = 2**30
        if needed > pool_budget:
            affordable = max(0, pool_budget // per_slot)
            raise ValueError(
                "dots.tts acoustic tail pools do not fit the declared stage "
                f"share: max_running_requests={self.max_running_requests} needs "
                f"{needed / gib:.1f} GiB but only {pool_budget / gib:.1f} GiB is "
                f"left after SGLang's {sglang_bytes / gib:.1f} GiB. Lower "
                f"max_running_requests to {affordable}, lower max_audio_patches, "
                "or raise the stage's total_gpu_memory_fraction."
            )
        logger.info(
            "dots.tts stage budget: share=%.2f (%.1f GiB) sglang=%.1f GiB "
            "tail_pools=%.1f GiB/%.1f GiB slack=%.1f GiB",
            stage_share,
            stage_share * total / gib,
            sglang_bytes / gib,
            needed / gib,
            pool_budget / gib,
            (pool_budget - needed) / gib,
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
