"""Backend wrapper for LLaDA2-Uni ZImage DiT decoding.

The upstream diffusers ``ZImageTransformer2DModel`` has already been fixed.
The optional ``sglang`` backend wraps sglang Diffusion's optimized ZImage
implementation behind the same forward API.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn

from sglang_omni.models.llada2_uni.config import normalize_image_decoder_backend
from safetensors.torch import load_file

SEQ_MULTI_OF = 32


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _create_coordinate_grid(size, start=None, device=None):
    if start is None:
        start = (0 for _ in size)
    axes = [
        torch.arange(x0, x0 + span, dtype=torch.int32, device=device)
        for x0, span in zip(start, size)
    ]
    grids = torch.meshgrid(axes, indexing="ij")
    return torch.stack(grids, dim=-1)


@dataclass(frozen=True)
class ZImageParallelConfig:
    """Runtime choices for ZImage DiT backend loading."""

    backend: str = "diffusers"
    tp_size: int = 1
    sp_size: int = 1
    ulysses_degree: int | None = None
    ring_degree: int = 1
    attention_backend: str = "torch_sdpa"
    enable_compile: bool = False
    compile_mode: str = "max-autotune-no-cudagraphs"
    invert_timestep: bool = True
    negate_output: bool = True

    def __post_init__(self) -> None:
        if self.tp_size < 1 or self.sp_size < 1:
            raise ValueError("ZImage TP and SP sizes must be positive")
        if self.sp_size & (self.sp_size - 1):
            raise ValueError("ZImage SP size must be a power of two")
        if self.ring_degree < 1:
            raise ValueError("ZImage ring degree must be positive")
        if self.sp_size != self.sglang_ulysses_degree * self.ring_degree:
            raise ValueError(
                "ZImage sp_size must equal ulysses_degree * ring_degree: "
                f"{self.sp_size} != {self.sglang_ulysses_degree} * "
                f"{self.ring_degree}"
            )

    @property
    def normalized_backend(self) -> str:
        return normalize_image_decoder_backend(self.backend)

    @property
    def sglang_ulysses_degree(self) -> int:
        return self.ulysses_degree if self.ulysses_degree is not None else self.sp_size


_SGLANG_DIFFUSION_RUNTIME_READY = False


def _validate_zimage_ulysses_degree(
    *, num_attention_heads: int, ulysses_degree: int
) -> None:
    if num_attention_heads < 1:
        raise ValueError("ZImage attention head count must be positive")
    if ulysses_degree < 1:
        raise ValueError("ZImage Ulysses degree must be positive")
    if num_attention_heads % ulysses_degree != 0:
        raise ValueError(
            f"ZImage Ulysses degree {ulysses_degree} must divide the model's "
            f"{num_attention_heads} attention heads"
        )


def _torch_dtype_to_sglang_precision(dtype: torch.dtype) -> str:
    precision_by_dtype = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    try:
        return precision_by_dtype[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported SGLang ZImage dtype: {dtype}") from exc


def _configure_sglang_device_environment(device: torch.device) -> int:
    """Bind SGLang diffusion's local process to the stage-assigned CUDA device."""
    if device.type != "cuda" or device.index is None:
        raise ValueError(
            "SGLang ZImage backend requires an explicitly indexed CUDA device"
        )
    os.environ["LOCAL_RANK"] = str(device.index)
    return device.index


def _build_sglang_zimage_dit_config(cfg: dict[str, Any]):
    from sglang.multimodal_gen.configs.models.dits.zimage import (
        ZImageArchConfig,
        ZImageDitConfig,
    )

    arch_config = ZImageArchConfig(
        all_patch_size=tuple(cfg["all_patch_size"]),
        all_f_patch_size=tuple(cfg["all_f_patch_size"]),
        in_channels=cfg["in_channels"],
        dim=cfg["dim"],
        num_layers=cfg["n_layers"],
        n_refiner_layers=cfg["n_refiner_layers"],
        num_attention_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        norm_eps=cfg["norm_eps"],
        qk_norm=cfg["qk_norm"],
        cap_feat_dim=cfg["cap_feat_dim"],
        rope_theta=cfg["rope_theta"],
        t_scale=cfg["t_scale"],
        axes_dims=tuple(cfg["axes_dims"]),
        axes_lens=tuple(cfg["axes_lens"]),
    )
    return ZImageDitConfig(arch_config=arch_config)


def _ensure_sglang_diffusion_runtime(
    model_path: str,
    cfg: dict[str, Any],
    dtype: torch.dtype,
    parallel_config: ZImageParallelConfig,
    device: torch.device,
) -> Any:
    """Initialize sglang diffusion runtime for the ZImage backend."""

    global _SGLANG_DIFFUSION_RUNTIME_READY

    _validate_zimage_ulysses_degree(
        num_attention_heads=int(cfg["n_heads"]),
        ulysses_degree=parallel_config.sglang_ulysses_degree,
    )

    from sglang.multimodal_gen.configs.pipeline_configs.zimage import (
        ZImagePipelineConfig,
    )
    from sglang.multimodal_gen.runtime.distributed import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        get_global_server_args,
        set_global_server_args,
    )
    from sglang.multimodal_gen.utils import set_mixed_precision_policy

    dit_config = _build_sglang_zimage_dit_config(cfg)
    set_mixed_precision_policy(
        param_dtype=dtype,
        reduce_dtype=dtype,
        output_dtype=dtype,
    )

    if _SGLANG_DIFFUSION_RUNTIME_READY:
        return dit_config

    tp_size = parallel_config.tp_size
    sp_size = parallel_config.sp_size
    ulysses_degree = parallel_config.sglang_ulysses_degree
    ring_degree = parallel_config.ring_degree
    expected_world_size = tp_size * sp_size
    base_gpu_id = _configure_sglang_device_environment(device)

    world_size = int(os.environ.get("WORLD_SIZE", str(expected_world_size)))
    if expected_world_size == 1:
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        distributed_init_method = f"tcp://127.0.0.1:{_free_tcp_port()}"
    else:
        if world_size != expected_world_size:
            raise RuntimeError(
                "SGLang ZImage parallel runtime requires torchrun with one "
                "process per rank: "
                f"WORLD_SIZE={world_size}, expected={expected_world_size}."
            )
        distributed_init_method = "env://"

    pipeline_config = ZImagePipelineConfig(dit_config=dit_config)
    pipeline_config.dit_precision = _torch_dtype_to_sglang_precision(dtype)

    try:
        get_global_server_args()
    except ValueError:
        server_args = ServerArgs(
            model_path=model_path,
            trust_remote_code=True,
            attention_backend=parallel_config.attention_backend,
            num_gpus=world_size,
            tp_size=tp_size,
            sp_degree=sp_size,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            enable_cfg_parallel=False,
            pipeline_config=pipeline_config,
            dit_cpu_offload=False,
            dit_layerwise_offload=False,
            text_encoder_cpu_offload=False,
            image_encoder_cpu_offload=False,
            use_fsdp_inference=False,
            base_gpu_id=base_gpu_id,
        )
        set_global_server_args(server_args)

    if not model_parallel_is_initialized():
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=tp_size,
            sp_size=sp_size,
            cfg_degree=1,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            distributed_init_method=distributed_init_method,
        )

    _SGLANG_DIFFUSION_RUNTIME_READY = True
    return dit_config


class _SGLangZImageModelAdapter(nn.Module):
    """Expose sglang diffusion's ZImage model with the diffusers forward API."""

    def __init__(self, model: nn.Module, parallel_config: ZImageParallelConfig):
        super().__init__()
        self.model = model
        self.parallel_config = parallel_config
        self._freqs_cache_key: tuple[Any, ...] | None = None
        self._freqs_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    @staticmethod
    def _normalize_image_latent(image: torch.Tensor) -> torch.Tensor:
        while image.dim() > 4 and image.shape[0] == 1:
            image = image.squeeze(0)
        if image.dim() != 4:
            raise ValueError(
                "ZImage expects each sample latent to have shape [C, F, H, W], "
                f"got {tuple(image.shape)}"
            )
        return image

    def _get_freqs_cis(
        self,
        image: torch.Tensor,
        cap_feat: torch.Tensor,
        patch_size: int,
        f_patch_size: int,
    ):
        device = image.device
        cap_ori_len = int(cap_feat.size(0))
        cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
        _, frames, height, width = image.size()

        p_h = p_w = patch_size
        p_f = f_patch_size
        f_tokens = frames // p_f
        h_tokens = height // p_h
        w_tokens = width // p_w
        image_ori_len = f_tokens * h_tokens * w_tokens
        image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

        cache_key = (
            cap_ori_len,
            cap_padding_len,
            frames,
            height,
            width,
            patch_size,
            f_patch_size,
            device.type,
            device.index,
        )
        if self._freqs_cache_key == cache_key and self._freqs_cache is not None:
            return self._freqs_cache

        cap_pos_ids = _create_coordinate_grid(
            size=(cap_ori_len + cap_padding_len, 1, 1),
            start=(1, 0, 0),
            device=device,
        ).flatten(0, 2)
        image_ori_pos_ids = _create_coordinate_grid(
            size=(f_tokens, h_tokens, w_tokens),
            start=(cap_ori_len + cap_padding_len + 1, 0, 0),
            device=device,
        ).flatten(0, 2)
        image_padding_pos_ids = (
            _create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
            .flatten(0, 2)
            .repeat(image_padding_len, 1)
        )
        image_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
        freqs_cis = (
            self.model.rotary_emb(cap_pos_ids),
            self.model.rotary_emb(image_pos_ids),
        )
        self._freqs_cache_key = cache_key
        self._freqs_cache = freqs_cis
        return freqs_cis

    @staticmethod
    def _sp_world_size() -> int:
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_sp_world_size,
        )

        return get_sp_world_size()

    @staticmethod
    def _shard_sequence_for_sp(
        x: torch.Tensor, freqs_cis: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_sp_parallel_rank,
            get_sp_world_size,
        )

        sp_world_size = get_sp_world_size()
        if sp_world_size <= 1:
            return x, freqs_cis
        seq_len = x.shape[1]
        if seq_len % sp_world_size != 0:
            raise RuntimeError(
                "ZImage SP requires padded sequence length divisible by SP size: "
                f"seq_len={seq_len}, sp_size={sp_world_size}"
            )
        rank = get_sp_parallel_rank()
        chunk = seq_len // sp_world_size
        start = rank * chunk
        end = start + chunk
        return x[:, start:end, :].contiguous(), (
            freqs_cis[0][start:end].contiguous(),
            freqs_cis[1][start:end].contiguous(),
        )

    def _run_model_sp(
        self,
        image: torch.Tensor,
        cap_feat: torch.Tensor,
        timestep: torch.Tensor,
        patch_size: int,
        f_patch_size: int,
    ) -> torch.Tensor:
        """Run the partial ZImage DiT path with explicit sequence sharding."""
        from sglang.multimodal_gen.runtime.distributed.communication_op import (
            sequence_model_parallel_all_gather,
        )

        model = self.model
        key = f"{patch_size}-{f_patch_size}"
        freqs_cis = self._get_freqs_cis(image, cap_feat, patch_size, f_patch_size)

        t = model.t_embedder(1000.0 - timestep)
        adaln_input = t.type_as(image)

        x, cap, x_size, x_valid_lens, cap_valid_lens = model.patchify_and_embed(
            [image], [cap_feat], patch_size, f_patch_size
        )
        x, _ = model.all_x_embedder[key](x)
        x = model._replace_padding_with_token(x, x_valid_lens, model.x_pad_token)
        x, x_freqs_cis = self._shard_sequence_for_sp(x, freqs_cis[1])
        for layer in model.noise_refiner:
            x = layer(x, x_freqs_cis, adaln_input)
        x = sequence_model_parallel_all_gather(x, dim=1)

        cap, _ = model.cap_embedder(cap)
        cap = model._replace_padding_with_token(
            cap, cap_valid_lens, model.cap_pad_token
        )
        cap_freqs_cis = freqs_cis[0]
        for layer in model.context_refiner:
            cap = layer(
                cap,
                cap_freqs_cis,
                skip_sequence_parallel_override=True,
            )

        unified = torch.cat([x, cap], dim=1)
        unified_freqs_cis = (
            torch.cat([freqs_cis[1][0], freqs_cis[0][0]], dim=0),
            torch.cat([freqs_cis[1][1], freqs_cis[0][1]], dim=0),
        )
        unified, unified_freqs_cis = self._shard_sequence_for_sp(
            unified, unified_freqs_cis
        )
        for layer in model.layers:
            unified = layer(unified, unified_freqs_cis, adaln_input)
        unified = sequence_model_parallel_all_gather(unified, dim=1)

        unified = model.all_final_layer[key](unified, adaln_input)
        outputs = list(unified.unbind(dim=0))
        return -model.unpatchify(outputs, x_size, patch_size, f_patch_size)[0]

    def forward(
        self,
        x,
        t,
        cap_feats,
        return_dict: bool = True,
        patch_size: int = 2,
        f_patch_size: int = 1,
        **_: Any,
    ):
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )

        outputs = []
        for idx, (image, cap_feat) in enumerate(zip(x, cap_feats)):
            image = self._normalize_image_latent(image)
            if t.dim() == 0:
                t_i = t
            elif t.shape[0] == 1:
                t_i = t[0]
            else:
                t_i = t[idx]
            timestep = t_i.float()
            timestep = (
                (1.0 - timestep) * 1000.0
                if self.parallel_config.invert_timestep
                else timestep * 1000.0
            )
            timestep = timestep.reshape(1).to(device=image.device)
            # This adapter does not enable step-index-dependent sparse attention
            # or cache policies. A batch index is not a denoising-step index.
            with set_forward_context(
                current_timestep=0, attn_metadata=None, forward_batch=None
            ):
                if self._sp_world_size() > 1:
                    pred = self._run_model_sp(
                        image, cap_feat, timestep, patch_size, f_patch_size
                    )
                else:
                    pred = self.model(
                        hidden_states=image.unsqueeze(0),
                        encoder_hidden_states=[cap_feat],
                        timestep=timestep,
                        patch_size=patch_size,
                        f_patch_size=f_patch_size,
                        freqs_cis=self._get_freqs_cis(
                            image, cap_feat, patch_size, f_patch_size
                        ),
                    )
            pred = self._normalize_image_latent(pred)
            outputs.append(-pred if self.parallel_config.negate_output else pred)
        if not return_dict:
            return (outputs,)

        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        return Transformer2DModelOutput(sample=outputs)


class ZImageTransformer2DModelWrapper(nn.Module):
    """Backend-selecting wrapper for LLaDA-Uni ZImage DiT decoding."""

    SUPPORTED_BACKENDS = {"diffusers", "sglang"}

    def __init__(
        self,
        model_path: str,
        decoder_dir: str,
        cfg: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        backend: str | None = None,
        parallel_config: ZImageParallelConfig | None = None,
    ) -> None:
        super().__init__()
        if parallel_config is None:
            parallel_config = ZImageParallelConfig(backend=backend or "diffusers")
        elif backend is not None and backend != parallel_config.backend:
            parallel_config = replace(parallel_config, backend=backend)
        self.parallel_config = parallel_config
        self.backend = parallel_config.normalized_backend
        if self.backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported ZImage backend: {self.backend!r}. "
                f"Supported backends: {sorted(self.SUPPORTED_BACKENDS)}"
            )
        self.model = self._load_model(model_path, decoder_dir, cfg, device, dtype)

    def _load_model(
        self,
        model_path: str,
        decoder_dir: str,
        cfg: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        if self.backend == "diffusers":
            return self._load_diffusers_model(decoder_dir, cfg, device, dtype)
        return self._load_sglang_model(model_path, decoder_dir, cfg, device, dtype)

    def _maybe_compile(self, model: nn.Module) -> nn.Module:
        if not self.parallel_config.enable_compile:
            return model
        if self.backend == "sglang":
            return model
        return torch.compile(
            model,
            mode=self.parallel_config.compile_mode,
            fullgraph=False,
        )

    def _load_diffusers_model(
        self,
        decoder_dir: str,
        cfg: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        from diffusers.models.transformers.transformer_z_image import (
            ZImageTransformer2DModel as DiffusersZImageTransformer2DModel,
        )

        kwargs = dict(
            all_patch_size=tuple(cfg["all_patch_size"]),
            all_f_patch_size=tuple(cfg["all_f_patch_size"]),
            in_channels=cfg["in_channels"],
            dim=cfg["dim"],
            n_layers=cfg["n_layers"],
            n_refiner_layers=cfg["n_refiner_layers"],
            n_heads=cfg["n_heads"],
            n_kv_heads=cfg["n_kv_heads"],
            norm_eps=cfg["norm_eps"],
            qk_norm=cfg["qk_norm"],
            cap_feat_dim=cfg["cap_feat_dim"],
            rope_theta=cfg["rope_theta"],
            t_scale=cfg["t_scale"],
            axes_dims=list(cfg["axes_dims"]),
            axes_lens=list(cfg["axes_lens"]),
        )
        with torch.device("meta"):
            model = DiffusersZImageTransformer2DModel(**kwargs)
        ckpt = os.path.join(decoder_dir, "model.safetensors")
        state_dict = load_file(ckpt, device=str(device))
        state_dict = {
            key.replace("semantic_embedder.", "cap_embedder."): value
            for key, value in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=False,
            assign=True,
        )
        if missing or unexpected:
            raise RuntimeError(
                "Failed to load diffusers ZImage checkpoint: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        del state_dict
        model = model.to(device=device, dtype=dtype).eval()
        return self._maybe_compile(model)

    def _load_sglang_model(
        self,
        model_path: str,
        decoder_dir: str,
        cfg: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        from sglang.multimodal_gen.runtime.loader.fsdp_load import (
            load_model_from_full_model_state_dict,
        )
        from sglang.multimodal_gen.runtime.loader.utils import (
            get_param_names_mapping,
            set_default_torch_dtype,
        )
        from sglang.multimodal_gen.runtime.loader.weight_utils import (
            safetensors_weights_iterator,
        )
        from sglang.multimodal_gen.runtime.models.dits.zimage import (
            ZImageTransformer2DModel as SGLangZImageTransformer2DModel,
        )

        dit_config = _ensure_sglang_diffusion_runtime(
            model_path, cfg, dtype, self.parallel_config, device
        )
        with set_default_torch_dtype(dtype), torch.device("meta"):
            model = SGLangZImageTransformer2DModel(config=dit_config, hf_config=cfg)

        mapping = dict(model.param_names_mapping)
        mapping[r"semantic_embedder\.(.*)$"] = r"cap_embedder.\1"
        mapping_fn = get_param_names_mapping(mapping)
        load_model_from_full_model_state_dict(
            model=model,
            full_sd_iterator=safetensors_weights_iterator(
                [os.path.join(decoder_dir, "model.safetensors")]
            ),
            device=device,
            param_dtype=dtype,
            strict=True,
            cpu_offload=False,
            param_names_mapping=mapping_fn,
        )
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return _SGLangZImageModelAdapter(model, self.parallel_config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
