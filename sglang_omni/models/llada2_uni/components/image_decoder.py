"""LLaDA2 image decoder: SigVQ + ZImage Diffusion + VAE.

Converts discrete VQ token IDs produced by the thinker into a PIL image.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from safetensors.torch import load_file
from torchvision.transforms.functional import to_pil_image

from sglang_omni.models.llada2_uni.components.decoder_model import (
    ZImageTransformer2DModel,
)
from sglang_omni.models.llada2_uni.components.sigvq import SigVQ
from sglang_omni.models.llada2_uni.components.transport import Sampler, create_transport
from sglang_omni.models.weight_loader import resolve_model_path

logger = logging.getLogger(__name__)


def _create_decoder_model_fn(
    model, cap_pos, cap_neg, cfg_scale, patch_size, f_patch_size, dtype
):
    n = len(cap_pos)
    doubled = cap_pos + cap_neg

    def fn(x, t, **kw):
        t_t = (
            torch.tensor([t], device=x.device, dtype=torch.float32)
            if not isinstance(t, torch.Tensor)
            else t.float()
        )
        if t_t.dim() == 0:
            t_t = t_t.unsqueeze(0)
        if t_t.shape[0] == 1 and x.shape[0] > 1:
            t_t = t_t.expand(x.shape[0])
        if cfg_scale > 0:
            out = model(
                x=list(x.to(dtype).repeat(2, 1, 1, 1, 1).unbind(0)),
                t=t_t.repeat(2),
                cap_feats=doubled,
                patch_size=patch_size,
                f_patch_size=f_patch_size,
                return_dict=False,
            )
            pos, neg = out[0][:n], out[0][n:]
            res = []
            for p, ng in zip(pos, neg):
                p, ng = p.float(), ng.float()
                pred = p + cfg_scale * (p - ng)
                on, nn_ = torch.linalg.vector_norm(p), torch.linalg.vector_norm(pred)
                if nn_ > on:
                    pred *= on / nn_
                res.append(pred)
            return torch.stack(res)
        out = model(
            x=list(x.to(dtype).unbind(0)),
            t=t_t,
            cap_feats=cap_pos,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
            return_dict=False,
        )
        return torch.stack([o.float() for o in out[0]])

    return fn


class LLaDA2ImageDecoder:
    """3-stage image decoder: SigVQ -> Diffusion ODE -> VAE.

    Args:
        model_path: Hugging Face model ID or root model directory (parent of
            decoder/, vae/, image_tokenizer/).
        device: Torch device string.
        dtype: Model dtype (default: bfloat16).
        decode_mode: ``"normal"`` for standard 50-step decoder,
            ``"decoder-turbo"`` for distilled 8-step decoder. Used as the default
            when :meth:`decode` is called without an explicit ``decode_mode``.
        num_steps: Default number of ODE sampling steps.
        resolution_multiplier: Default upscale factor (2 = 1024px from 512px tokens).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        decode_mode: str = "normal",
        num_steps: int = 50,
        resolution_multiplier: int = 2,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model_path = str(resolve_model_path(model_path))
        self.decode_mode = decode_mode
        self.num_steps = num_steps
        self.resolution_multiplier = resolution_multiplier

        self._sigvq: SigVQ | None = None
        self._diff_model: ZImageTransformer2DModel | None = None
        self._diff_model_mode: str | None = None
        self._vae: AutoencoderKL | None = None
        self._diff_config: dict | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_sigvq(self):
        if self._sigvq is not None:
            return
        sigvq_path = os.path.join(
            self.model_path, "image_tokenizer", "sigvq_embedding.pt"
        )
        self._sigvq = SigVQ(vocab_size=16384, inner_dim=4096).to(
            self.device, dtype=self.dtype
        )
        self._sigvq.load_state_dict(
            torch.load(sigvq_path, map_location=self.device, weights_only=True)
        )
        self._sigvq.eval()
        logger.info("SigVQ loaded from %s", sigvq_path)

    def _ensure_diff_model(self, decode_mode: str):
        if self._diff_model is not None and self._diff_model_mode == decode_mode:
            return
        if self._diff_model is not None:
            logger.info(
                "Switching diffusion model: %s -> %s (releasing GPU memory)",
                self._diff_model_mode,
                decode_mode,
            )
            del self._diff_model
            self._diff_model = None
            self._diff_config = None
            self._diff_model_mode = None
            torch.cuda.empty_cache()

        if decode_mode == "decoder-turbo":
            decoder_dir = os.path.join(self.model_path, "decoder-turbo")
        else:
            decoder_dir = os.path.join(self.model_path, "decoder")

        config_path = os.path.join(decoder_dir, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        # Override axes_lens and cap_feat_dim to match SigVQ output
        cfg["axes_lens"] = [32768, 1024, 1024]
        cfg["cap_feat_dim"] = 4096
        self._diff_config = cfg

        with torch.device("meta"):
            self._diff_model = ZImageTransformer2DModel(**cfg)
        ckpt = os.path.join(decoder_dir, "model.safetensors")
        self._diff_model.load_state_dict(
            load_file(ckpt, device=str(self.device)), assign=True
        )
        self._diff_model = self._diff_model.to(dtype=self.dtype).eval()
        self._diff_model_mode = decode_mode
        logger.info(
            "Diffusion model loaded from %s (%s mode)", decoder_dir, decode_mode
        )

    def _ensure_vae(self):
        if self._vae is not None:
            return
        vae_dir = os.path.join(self.model_path, "vae")
        self._vae = (
            AutoencoderKL.from_pretrained(vae_dir, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        logger.info("VAE loaded from %s", vae_dir)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def decode(
        self,
        token_ids: list[int],
        h: int,
        w: int,
        *,
        decode_mode: str | None = None,
        num_steps: int | None = None,
        resolution_multiplier: int | None = None,
        seed: int | None = None,
    ):
        """Decode VQ token IDs into a PIL Image.

        Args:
            token_ids: List of VQ token IDs (without the +157184 offset).
            h: Semantic grid height (image_pixels // 16).
            w: Semantic grid width (image_pixels // 16).
            decode_mode: Override instance default; switches diffusion weights
                between ``decoder/`` and ``decoder-turbo/`` (single-slot reload).
            num_steps: Override default ODE step count.
            resolution_multiplier: Override default upscale factor.
            seed: If set, draws initial noise with a deterministic generator.
                If ``None``, each call gets fresh randomness from the global RNG.

        Returns:
            PIL.Image.Image
        """
        mode = decode_mode if decode_mode is not None else self.decode_mode
        steps = num_steps if num_steps is not None else self.num_steps
        rmul = (
            resolution_multiplier
            if resolution_multiplier is not None
            else self.resolution_multiplier
        )

        # Stage 1: SigVQ -> semantic features
        self._ensure_sigvq()
        th = h * 16 * rmul
        tw = w * 16 * rmul
        tok = torch.tensor(token_ids).view(1, 1, h, w).float().to(self.device)
        up = F.interpolate(tok, scale_factor=2, mode="nearest").long().view(1, -1)
        cap_pos = [self._sigvq(up).squeeze(0)]
        cap_neg = [torch.zeros_like(cap_pos[0])]

        # Stage 2: Diffusion ODE sampling
        self._ensure_diff_model(mode)
        cfg = self._diff_config
        noise_shape = [1, 16, 1, 2 * (th // 16), 2 * (tw // 16)]
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            z = torch.randn(noise_shape, device=self.device, generator=generator)
        else:
            generator = None
            z = torch.randn(noise_shape, device=self.device)
        model_fn = _create_decoder_model_fn(
            self._diff_model,
            cap_pos,
            cap_neg,
            cfg_scale=0.0 if mode == "decoder-turbo" else 1.0,
            patch_size=cfg.get("all_patch_size", (2,))[0],
            f_patch_size=cfg.get("all_f_patch_size", (1,))[0],
            dtype=self.dtype,
        )

        sampler = Sampler(create_transport("Linear", "velocity", None))
        sample_fn = sampler.sample_ode(
            sampling_method="euler",
            num_steps=steps,
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
            time_shifting_factor=6,
            stochast_ratio=1.0 if mode == "decoder-turbo" else 0.0,
            generator=generator,
        )
        samples = sample_fn(z, model_fn)[-1].squeeze(2)

        # Stage 3: VAE decode
        self._ensure_vae()
        s = samples.to(self.dtype)
        s = (s / self._vae.config.scaling_factor) + self._vae.config.shift_factor
        px = ((self._vae.decode(s, return_dict=False)[0] + 1) / 2).clamp_(0, 1)

        return to_pil_image(px[0].float())

    @torch.inference_mode()
    def decode_to_bytes(
        self,
        token_ids: list[int],
        h: int,
        w: int,
        format: str = "PNG",
        **decode_kwargs: Any,
    ) -> bytes:
        """Decode VQ token IDs into image bytes.

        Args:
            token_ids: List of VQ token IDs (without the +157184 offset).
            h: Semantic grid height.
            w: Semantic grid width.
            format: PIL image format (PNG or JPEG).
            **decode_kwargs: Forwarded to :meth:`decode` (decode_mode, num_steps,
                resolution_multiplier, seed).

        Returns:
            Image bytes.
        """
        import io

        image = self.decode(token_ids, h, w, **decode_kwargs)
        buf = io.BytesIO()
        image.save(buf, format=format)
        return buf.getvalue()
