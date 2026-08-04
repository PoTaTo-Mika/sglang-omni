# SPDX-License-Identifier: Apache-2.0
"""OmniScheduler/SGLang engine builder shared by both VoxCPM generations."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


def _get(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    value = (
        config.get(name, default)
        if isinstance(config, dict)
        else getattr(config, name, default)
    )
    return default if value is None else value


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.lower().strip()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be boolean")


class VoxCPMEngineBuilder(TtsEngineBuilder):
    model_name = "VoxCPM"
    context_length = 32768

    def __init__(
        self, *, variant: str, total_gpu_memory_fraction: float | None = None
    ) -> None:
        if variant not in {"voxcpm", "voxcpm2"}:
            raise ValueError(f"unsupported VoxCPM variant: {variant}")
        self.variant = variant
        self.model_name = "VoxCPM2" if variant == "voxcpm2" else "VoxCPM1.5"
        self.model_arch_override = (
            "VoxCPM2SGLangModel" if variant == "voxcpm2" else "VoxCPMSGLangModel"
        )
        self.total_gpu_memory_fraction = total_gpu_memory_fraction
        self.config: Any = None
        self.tokenizer: Any = None
        self.vae: Any = None
        self._runner: Any = None
        self.decrypted_config_file: str | None = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from sglang_omni.models.voxcpm_common.hf_config import (
            VoxCPM2Config,
            VoxCPMConfig,
            register_voxcpm_configs,
        )

        register_voxcpm_configs()
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
        raw_config["model_type"] = self.variant
        raw_config["architectures"] = [self.model_arch_override]
        config_cls = VoxCPM2Config if self.variant == "voxcpm2" else VoxCPMConfig
        self.config = config_cls.from_dict(raw_config)
        self.decrypted_config_file = os.path.join(
            tempfile.gettempdir(),
            f"{self.variant}_sglang_config_{abs(hash(checkpoint_dir))}.json",
        )
        with open(self.decrypted_config_file, "w", encoding="utf-8") as config_file:
            json.dump(raw_config, config_file)
        lm_config = _get(self.config, "lm_config")
        self.context_length = int(
            _get(
                lm_config,
                "max_position_embeddings",
                _get(self.config, "max_position_embeddings", 32768),
            )
        )

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        defaults = {
            "max_running_requests": 8,
            "dtype": dtype,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "chunked_prefill_size": 0,
            "max_prefill_tokens": min(self.context_length, 8192),
            "sampling_backend": "pytorch",
            "trust_remote_code": False,
            "decrypted_config_file": self.decrypted_config_file,
        }
        if self.total_gpu_memory_fraction is not None:
            defaults["mem_fraction_static"] = float(self.total_gpu_memory_fraction)
        return defaults

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        if "disable_radix_cache" in overrides and not _as_bool(
            overrides["disable_radix_cache"], "disable_radix_cache"
        ):
            raise ValueError(
                "VoxCPM continuous latent prompts require disable_radix_cache=true"
            )
        if int(overrides.get("chunked_prefill_size", 0) or 0) != 0:
            raise ValueError("VoxCPM does not support chunked prefill")
        if not _as_bool(
            overrides.get("disable_overlap_schedule", True),
            "disable_overlap_schedule",
        ):
            raise ValueError("VoxCPM requires overlap scheduling to be disabled")
        if _as_bool(
            overrides.get("enable_torch_compile", False), "enable_torch_compile"
        ):
            raise ValueError("VoxCPM torch.compile is not supported")
        overrides["disable_radix_cache"] = True
        overrides["chunked_prefill_size"] = 0
        overrides["disable_overlap_schedule"] = True
        overrides["enable_torch_compile"] = False
        overrides["tp_size"] = 1

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del gpu_id, server_args
        from sglang_omni.models.voxcpm_common.audio_vae import AudioVAE, AudioVAEV2
        from sglang_omni.models.voxcpm_common.engine_io import load_voxcpm_tokenizer

        model = model_worker.model_runner.model
        model.eval()
        self.tokenizer = load_voxcpm_tokenizer(checkpoint_dir)
        vae_config = _get(self.config, "audio_vae_config")
        self.vae = (
            AudioVAEV2(config=vae_config)
            if self.variant == "voxcpm2"
            else AudioVAE(config=vae_config)
        )
        # The causal vocoder is numerically sensitive and official weights are fp32.
        self.vae.to(device=device, dtype=__import__("torch").float32).eval()
        model.load_audiovae(self.vae, checkpoint_dir, strict=True)
        if device.startswith("cuda"):
            torch = __import__("torch")
            with torch.inference_mode():
                self.vae.decode(
                    torch.zeros(
                        1,
                        int(self.vae.latent_dim),
                        int(model.model.patch_size),
                        device=device,
                        dtype=torch.float32,
                    )
                )
            torch.cuda.synchronize()

    def get_model_buffer_bs(self, model: Any) -> int | None:
        return int(model.model.state_pool.capacity)

    def post_cuda_graph_setup(self, model: Any, server_args: Any) -> None:
        if os.environ.get("SGLANG_VOXCPM_DISABLE_DIFFUSION_GRAPH") == "1":
            return
        requested = list(server_args.cuda_graph_bs)
        max_bs = max(requested)
        diffusion_buckets = [
            bs for bs in requested if bs <= 8 or bs % 16 == 0 or bs == max_bs
        ]
        model.init_diffusion_graphs(diffusion_buckets)

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang_omni.models.voxcpm_common.model_runner import VoxCPMModelRunner

        self._runner = VoxCPMModelRunner(
            model_worker, output_proc, variant=self.variant
        )
        return self._runner

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        from sglang_omni.models.voxcpm_common.engine_io import (
            make_voxcpm_scheduler_adapters,
        )

        return make_voxcpm_scheduler_adapters(
            model=model,
            tokenizer=self.tokenizer,
            vae=self.vae,
            variant=self.variant,
            reset_request=self._runner.reset_request,
        )

    def make_abort_callback(self) -> Any:
        return self._runner.reset_request

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "enable_async_decode": (
                os.environ.get("SGLANG_VOXCPM_ENABLE_ASYNC_DECODE") == "1"
            ),
            "async_decode_min_batch_size": 16,
        }

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        model_runner.set_stream_outbox(scheduler.outbox)


__all__ = ["VoxCPMEngineBuilder"]
