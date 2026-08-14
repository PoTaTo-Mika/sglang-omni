# SPDX-License-Identifier: Apache-2.0
"""Preprocessor for LLaDA2-Uni: tokenize text and prepare image inputs."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from sglang_omni.models.llada2_uni.components.common import (
    load_llada2_tokenizer,
    resolve_local_model_dir,
)
from sglang_omni.models.llada2_uni.config import (
    DEFAULT_THINKER_MAX_NEW_TOKENS,
    IMAGE_STAGE,
)
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.preprocessing.image import (
    compute_image_cache_key,
    ensure_image_list_async,
)
from sglang_omni.proto import StagePayload

# LLaDA2-Uni chat template tokens
ROLE_HUMAN = "<role>HUMAN</role>"
ROLE_ASSISTANT = "<role>ASSISTANT</role>"
ROLE_SYSTEM = "<role>SYSTEM</role>"

# System prompts per task kind
SYSTEM_PROMPT_CHAT = "You are a multimodal understanding assistant."
SYSTEM_PROMPT_T2I = "You are a text-to-image generation assistant."
SYSTEM_PROMPT_T2I_THINKING = (
    "You are a text-to-image generation assistant with a thinking process."
)
EDIT_SYSTEM_PROMPT = "You are an image editing assistant."

# Token used for classifier-free guidance unconditional prompts
UNCOND_TEXT = "<uncondition>"


# Image special token strings
SOI_TOKEN = "<|image|>"  # id=156901
EOI_TOKEN = "<|/image|>"  # id=156902
BOI_TOKEN = "<boi>"  # id=156904

# Default image generation dimensions (pixels, before the //2 resolution
# multiplier applied by the model).  The image header encodes grid
# dimensions in 16-px semantic-patch units:
#   grid_h = image_h // 2 // 16,  grid_w = image_w // 2 // 16
DEFAULT_T2I_IMAGE_H = 1024
DEFAULT_T2I_IMAGE_W = 1024

IMAGE_TOKEN_OFFSET = 157184  # VQ codebook indices are offset by this value
DUMMY_IMAGE_TOKEN_ID = IMAGE_TOKEN_OFFSET  # <IMAGE0>, used as placeholder

# Pixel budgets for image resize (single-image / multi-image)
SINGLE_IMAGE_MIN_PIXELS = 128 * 128
SINGLE_IMAGE_MAX_PIXELS = 800 * 800
MULTI_IMAGE_MIN_PIXELS = 128 * 128
MULTI_IMAGE_MAX_PIXELS = 448 * 448

logger = logging.getLogger(__name__)


def _resolve_edit_cfg_scales(
    image_generation: dict[str, Any],
) -> tuple[float, float]:
    """Resolve edit CFG scales while preserving the legacy cfg_scale API."""
    if "cfg_text_scale" in image_generation:
        cfg_text_scale = float(image_generation["cfg_text_scale"])
    elif "cfg_scale" in image_generation:
        legacy_cfg_scale = float(image_generation["cfg_scale"])
        cfg_text_scale = 0.0 if legacy_cfg_scale == 1.0 else legacy_cfg_scale
    else:
        cfg_text_scale = 4.0

    cfg_image_scale = float(image_generation.get("cfg_image_scale", 0.0))
    return cfg_text_scale, cfg_image_scale


def align_cfg_unconditional_input_ids(
    tokenizer: Any,
    conditional_input_ids: list[int],
    unconditional_input_ids: list[int],
) -> tuple[list[int], int]:
    """Left-pad a CFG companion with mask tokens to physical alignment."""
    if len(unconditional_input_ids) > len(conditional_input_ids):
        raise ValueError(
            "CFG unconditional input cannot be longer than conditional input: "
            f"cond={len(conditional_input_ids)}, "
            f"uncond={len(unconditional_input_ids)}"
        )
    left_pad_length = len(conditional_input_ids) - len(unconditional_input_ids)
    if left_pad_length == 0:
        return list(unconditional_input_ids), 0

    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError("LLaDA2 tokenizer has no mask_token_id for CFG padding")
    return (
        [int(mask_token_id)] * left_pad_length + list(unconditional_input_ids),
        left_pad_length,
    )


def validate_prompt_seq_len(
    input_ids: torch.Tensor,
    *,
    max_seq_len: int | None,
    max_new_tokens: int = DEFAULT_THINKER_MAX_NEW_TOKENS,
    request_id: str | None = None,
) -> None:
    if max_seq_len is None:
        return
    prompt_len = int(input_ids.numel())
    if prompt_len >= max_seq_len:
        logger.info(
            "rejecting request %s: prompt %d tokens >= max_seq_len %d",
            request_id,
            prompt_len,
            max_seq_len,
        )
        raise ValueError(
            f"The input ({prompt_len} tokens) is longer than the model's "
            f"context length ({max_seq_len} tokens)."
        )
    total_tokens = prompt_len + int(max_new_tokens)
    if total_tokens > max_seq_len:
        logger.info(
            "rejecting request %s: prompt %d + max_new_tokens %d = %d tokens "
            ">= max_seq_len %d",
            request_id,
            prompt_len,
            int(max_new_tokens),
            total_tokens,
            max_seq_len,
        )
        raise ValueError(
            f"Requested token count exceeds the model's maximum context length "
            f"of {max_seq_len} tokens. You requested a total of {total_tokens} "
            f"tokens: {prompt_len} tokens from the input messages and "
            f"{int(max_new_tokens)} tokens for the completion. Please reduce "
            f"the number of tokens in the input messages or the completion to "
            f"fit within the limit."
        )


def _compute_target_dims(
    height: int,
    width: int,
    min_pixels: int,
    max_pixels: int,
    factor: int,
) -> tuple[int, int]:
    """Scale dimensions to fit within [min_pixels, max_pixels], aligned to factor."""
    new_h = max(round(height / factor) * factor, factor)
    new_w = max(round(width / factor) * factor, factor)

    if new_h * new_w > max_pixels:
        scale = math.sqrt(max_pixels / (height * width))
        new_h = max(math.floor(height * scale / factor) * factor, factor)
        new_w = max(math.floor(width * scale / factor) * factor, factor)
    elif new_h * new_w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        new_h = math.ceil(height * scale / factor) * factor
        new_w = math.ceil(width * scale / factor) * factor

    return new_h, new_w


def _resize_and_center_crop(
    img: Image.Image,
    target_h: int,
    target_w: int,
    factor: int,
) -> Image.Image:
    """Resize a PIL Image to cover the target area, then center-crop to a factor-aligned size."""
    width, height = img.size
    scale = max(target_h / height, target_w / width)
    resize_h = int(round(height * scale))
    resize_w = int(round(width * scale))
    img = img.resize((resize_w, resize_h), resample=Image.BICUBIC)

    crop_h = max((resize_h // factor) * factor, target_h)
    crop_w = max((resize_w // factor) * factor, target_w)
    top = (resize_h - crop_h) // 2
    left = (resize_w - crop_w) // 2
    return img.crop((left, top, left + crop_w, top + crop_h))


def _resize_images(
    images: list[Image.Image],
    factor: int,
) -> list[Image.Image]:
    """Resize PIL Images to fit within pixel budgets, preserving aspect ratio."""
    if len(images) == 1:
        min_pixels, max_pixels = SINGLE_IMAGE_MIN_PIXELS, SINGLE_IMAGE_MAX_PIXELS
    else:
        min_pixels, max_pixels = MULTI_IMAGE_MIN_PIXELS, MULTI_IMAGE_MAX_PIXELS

    result = []
    for img in images:
        width, height = img.size
        target_h, target_w = _compute_target_dims(
            height, width, min_pixels, max_pixels, factor
        )
        result.append(_resize_and_center_crop(img, target_h, target_w, factor))
    return result


# ---------------------------------------------------------------------------
# Edit-mode image preprocessing (var_center_crop for generation-pipeline
# resolution alignment, not understanding-mode smart_resize)
# ---------------------------------------------------------------------------


def _center_crop(pil_image: Image.Image, crop_size: tuple[int, int]) -> Image.Image:
    """Center-crop a PIL Image to the given (width, height)."""
    cw, ch = crop_size
    w, h = pil_image.size
    left = max(0, (w - cw) // 2)
    top = max(0, (h - ch) // 2)
    return pil_image.crop((left, top, left + cw, top + ch)).resize(
        (cw, ch), Image.LANCZOS
    )


def _generate_crop_size_list(
    num_patches: int, patch_size: int, max_ratio: float = 4.0
) -> list[tuple[int, int]]:
    """Generate candidate crop sizes for *num_patches* patches at *patch_size*."""
    assert max_ratio >= 1.0
    crop_size_list: list[tuple[int, int]] = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


def _var_center_crop(
    pil_image: Image.Image,
    crop_size_list: list[tuple[int, int]],
    random_top_k: int = 1,
) -> Image.Image:
    """Select a crop size from *crop_size_list* and center-crop the image.

    At inference time (*random_top_k=1*), picks the largest crop that
    preserves the most of the image (sorted by remaining-area ratio,
    then picks from the top-k).
    """
    import random

    w, h = pil_image.size
    rem_percent = [
        min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list
    ]
    crop_size = random.choice(
        sorted(
            ((x, y) for x, y in zip(rem_percent, crop_size_list)),
            reverse=True,
        )[:random_top_k]
    )[1]
    return _center_crop(pil_image, crop_size)


def preprocess_image_edit(images: list[Image.Image], factor: int) -> list[Image.Image]:
    """Crop and preprocess PIL images for image editing.

    Uses ``_var_center_crop`` with a 512px patch budget, matching the
    generation pipeline's resolution constraints.  Editing must encode
    the source image at a resolution aligned with the decoder's expected
    token grid.
    """
    crop_size_list = _generate_crop_size_list((512 // factor) ** 2, factor)
    return [_var_center_crop(img, crop_size_list) for img in images]


def edit_image_pixel_values(
    images: list[Image.Image],
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    image_mean: list[float],
    image_std: list[float],
    rescale_factor: float,
) -> dict[str, torch.Tensor]:
    """Patchify edit images with the model reference's float32 math order.

    Keeping rescale and normalization in float32 avoids source-VQ changes near
    codebook boundaries that can otherwise degrade edit fidelity.
    """
    import torchvision.transforms.v2.functional as tvF

    all_patches: list[torch.Tensor] = []
    all_grids: list[list[int]] = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        t = tvF.to_dtype(tvF.to_image(img), dtype=torch.float32, scale=False)
        h, w = t.shape[-2:]
        t = t * rescale_factor  # Keep the rescale operation in float32.
        mean = torch.tensor(image_mean, dtype=t.dtype).view(-1, 1, 1)
        std = torch.tensor(image_std, dtype=t.dtype).view(-1, 1, 1)
        t = (t - mean) / std
        if t.ndim == 3:
            t = t.unsqueeze(0)
        if t.shape[0] % temporal_patch_size != 0:
            reps = t[-1:].repeat(temporal_patch_size - 1, 1, 1, 1)
            t = torch.cat([t, reps], dim=0)
        grid_t = t.shape[0] // temporal_patch_size
        grid_h = h // patch_size
        grid_w = w // patch_size
        ch = t.shape[1]
        t = t.unsqueeze(0).view(
            1,
            grid_t,
            temporal_patch_size,
            ch,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        t = t.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flat = t.reshape(
            1,
            grid_t * grid_h * grid_w,
            ch * temporal_patch_size * patch_size * patch_size,
        )
        all_patches.append(flat.squeeze(0))
        all_grids.append([grid_t, grid_h, grid_w])
    return {
        "pixel_values": torch.cat(all_patches, dim=0),
        "image_grid_thw": torch.tensor(all_grids, dtype=torch.long),
    }


class LLaDA2Preprocessor:
    """Preprocessor for LLaDA2-Uni model (text + image)."""

    def __init__(self, model_path: str, max_seq_len: int | None = None):
        self._max_seq_len = max_seq_len
        self._model_dir = resolve_local_model_dir(model_path)
        self._tokenizer = load_llada2_tokenizer(model_path)

        # Load the Open Source Qwen2VLImageProcessor (do_resize=False, crop handles sizing)
        from transformers import Qwen2VLImageProcessor

        tokenizer_path = str(Path(self._model_dir) / "image_tokenizer")

        try:
            # TODO: Remove this compatibility override once the upstream processor config
            # is corrected from merge_size=2 to merge_size=1. Do not patch the checkpoint
            # files in place.
            self._image_processor = Qwen2VLImageProcessor.from_pretrained(
                tokenizer_path,
                local_files_only=True,
                do_resize=False,  # Disable resize, use manual crop instead
                merge_size=1,
            )
        except (OSError, ValueError, RuntimeError):
            if Path(model_path).exists():
                raise
            self._image_processor = Qwen2VLImageProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=False,
                subfolder="image_tokenizer",
                do_resize=False,
            )
            self._model_dir = str(
                resolve_model_path(model_path, local_files_only=False)
            )

        # Match LLaDA2.0-Uni's standalone HF implementation exactly: rescale first,
        # then normalize, instead of Transformers' fused equivalent.
        def rescale_and_normalize(
            images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
        ):
            if do_rescale:
                images = images * rescale_factor
            if do_normalize:
                mean = torch.tensor(
                    image_mean, dtype=images.dtype, device=images.device
                ).view(-1, 1, 1)
                std = torch.tensor(
                    image_std, dtype=images.dtype, device=images.device
                ).view(-1, 1, 1)
                images = (images - mean) / std
            return images

        self._image_processor.rescale_and_normalize = rescale_and_normalize
        self._merge_size = self._image_processor.merge_size
        self._factor = self._image_processor.patch_size * self._merge_size

        # Cache special token IDs
        self._eoi_id = self._tokenizer.convert_tokens_to_ids(EOI_TOKEN)
        self._boi_id = self._tokenizer.convert_tokens_to_ids(BOI_TOKEN)
        self._soi_id = self._tokenizer.convert_tokens_to_ids(SOI_TOKEN)

    async def __call__(self, payload: StagePayload) -> StagePayload:
        request = payload.request
        raw_inputs = request.inputs
        if isinstance(raw_inputs, list):
            messages = raw_inputs
            raw_images, image_counts_per_msg = self._extract_raw_images(messages)
        else:
            messages = raw_inputs.get("messages", [])
            raw_images = raw_inputs.get("images")
            if raw_images is None:
                raw_images, image_counts_per_msg = self._extract_raw_images(messages)
            else:
                image_counts_per_msg = None

        self._validate_messages(messages)

        # Detect task kind early so it influences prompt construction
        request_metadata = payload.request.metadata if payload.request else {}
        task_kind = "chat"
        if (
            isinstance(request_metadata, dict)
            and "image_generation" in request_metadata
            and isinstance(request_metadata["image_generation"], dict)
        ):
            # image_generation + input images → edit; image_generation + no images → t2i
            has_images = bool(raw_images)
            task_kind = "edit" if has_images else "t2i"

        if task_kind == "edit":
            self._require_edit_instruction(messages)

        image_cache_key = compute_image_cache_key(raw_images)
        images = await ensure_image_list_async(raw_images) if raw_images else []

        # --- Edit dispatch: build the edit payload and return early ---
        if task_kind == "edit":
            return self._build_edit_payload(payload, messages, images, request_metadata)

        encoder_inputs: dict[str, dict[str, Any]] = {}
        image_token_counts: list[int] = []
        image_parts_by_msg: dict[int, list[str]] = {}

        if images:
            cropped = _resize_images(images, self._factor)
            img_result = self._image_processor(images=cropped, return_tensors="pt")
            pixel_values = img_result["pixel_values"]
            image_grid_thw = img_result["image_grid_thw"]
            image_enc_inputs: dict[str, Any] = {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
            if image_cache_key:
                image_enc_inputs["cache_key"] = image_cache_key
            encoder_inputs[IMAGE_STAGE] = image_enc_inputs

            if image_counts_per_msg is None:
                last_user_idx = max(len(messages) - 1, 0)
                for i, m in enumerate(messages):
                    if m.get("role", "user") == "user":
                        last_user_idx = i
                image_counts_per_msg = [(last_user_idx, len(images))]

            img_idx = 0
            for msg_idx, count in image_counts_per_msg:
                parts: list[str] = []
                for _ in range(count):
                    t, h, w = image_grid_thw[img_idx].tolist()
                    h_token = f"<|reserved_token_{h}|>"
                    w_token = f"<|reserved_token_{w}|>"
                    # LLaDA2.0-Uni's VQ-VAE produces one token per patch (no
                    # PatchMerger), so the placeholder count matches the
                    # raw patch count regardless of merge_size.
                    num_image_tokens = t * h * w
                    img_header = f"{SOI_TOKEN}{h_token}{w_token}{BOI_TOKEN}"
                    image_token_counts.append(num_image_tokens)
                    parts.extend([img_header, EOI_TOKEN])
                    img_idx += 1
                image_parts_by_msg[msg_idx] = parts
        else:
            encoder_inputs[IMAGE_STAGE] = {"_skip": True, "_result": {}}

        # Build prompt text (task_kind selects the system prompt)
        ig = (
            request_metadata.get("image_generation", {})
            if isinstance(request_metadata, dict)
            else {}
        )
        prompt_task_kind = task_kind
        if task_kind == "t2i" and isinstance(ig, dict) and ig.get("mode") == "thinking":
            prompt_task_kind = "t2i_thinking"
        text_prompt = self._build_prompt(
            messages, image_parts_by_msg=image_parts_by_msg, task_kind=prompt_task_kind
        )
        input_ids = self._tokenizer.encode(text_prompt, add_special_tokens=False)

        if image_token_counts:
            input_ids = self._insert_image_placeholders(input_ids, image_token_counts)

        # For t2i tasks, append the image generation header tokens after
        # the <role>ASSISTANT</role> tag so the model knows where to
        # produce VQ image tokens.  This mirrors the model's own
        # generate_image() which appends <soi><h><w><boi>.
        stream_state: dict[str, Any] = {}
        if task_kind == "t2i":
            ig = (
                request_metadata.get("image_generation", {})
                if isinstance(request_metadata, dict)
                else {}
            )
            image_h = ig.get("image_h", DEFAULT_T2I_IMAGE_H)
            image_w = ig.get("image_w", DEFAULT_T2I_IMAGE_W)
            grid_h = image_h // 2 // 16
            grid_w = image_w // 2 // 16
            mode = ig.get("mode", "normal") if isinstance(ig, dict) else "normal"
            cfg_scale = ig.get("cfg_scale", 1.0) if isinstance(ig, dict) else 1.0
            cfg_rescale = ig.get("cfg_rescale", 0.7) if isinstance(ig, dict) else 0.7

            stream_state["image_info"] = [{"grid_h": grid_h, "grid_w": grid_w}]
            dllm_steps = ig.get("dllm_steps") if isinstance(ig, dict) else None
            if dllm_steps is not None:
                stream_state["dllm_steps"] = int(dllm_steps)

            if mode == "thinking":
                # Thinking mode Phase 1: model generates reasoning text
                # until <boi>, then Phase 2 generates VQ tokens.
                # Do NOT append image header — model produces <boi> itself.
                stream_state["thinking_mode"] = True
                stream_state["thinking_phase"] = 1
                stream_state["cfg_scale"] = cfg_scale
                stream_state["cfg_rescale"] = cfg_rescale
            else:
                # Normal mode: append image header immediately.
                header_ids = self._build_t2i_header_ids(grid_h, grid_w)
                input_ids.extend(header_ids)

                # Build unconditional input_ids for CFG when cfg_scale > 1.0
                if cfg_scale > 1.0:
                    uncond_input_ids = self._build_t2i_uncond_input_ids(grid_h, grid_w)
                    uncond_input_ids, uncond_left_pad_len = (
                        align_cfg_unconditional_input_ids(
                            self._tokenizer, input_ids, uncond_input_ids
                        )
                    )
                    stream_state["uncond_input_ids"] = uncond_input_ids
                    stream_state["uncond_left_pad_len"] = uncond_left_pad_len
                    stream_state["cfg_scale"] = cfg_scale
                    stream_state["cfg_rescale"] = cfg_rescale

        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)

        validate_prompt_seq_len(
            input_ids_tensor,
            max_seq_len=self._max_seq_len,
            max_new_tokens=request.params.get(
                "max_new_tokens", DEFAULT_THINKER_MAX_NEW_TOKENS
            ),
            request_id=payload.request_id,
        )

        prompt = {"input_ids": input_ids_tensor}

        state = LLaDA2UniPipelineState(
            prompt=prompt,
            encoder_inputs=encoder_inputs,
            request_metadata=request_metadata,
            task_kind=task_kind,
            stream_state=stream_state,
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
            timing=payload.timing,
        )

    @staticmethod
    def _extract_raw_images(
        messages: list[dict[str, Any]],
    ) -> tuple[list[Any], list[tuple[int, int]]]:
        """Return (images, image_counts_per_msg) with per-message image counts."""
        raw_images: list[Any] = []
        image_counts_per_msg: list[tuple[int, int]] = []
        for msg_idx, msg in enumerate(messages):
            msg_count = 0
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {})
                        if isinstance(url, dict):
                            url = url.get("url", "")
                        if url:
                            raw_images.append(url)
                            msg_count += 1
                    elif item.get("type") == "image":
                        img = item.get("image", "")
                        if img:
                            raw_images.append(img)
                            msg_count += 1
            if msg_count > 0:
                image_counts_per_msg.append((msg_idx, msg_count))
        return raw_images, image_counts_per_msg

    @staticmethod
    def _validate_messages(messages: list[dict[str, Any]]) -> None:
        if not isinstance(messages, list):
            raise ValueError("Preprocessing expects a list of chat messages")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dict with role/content")

    def _build_prompt(
        self,
        messages: list[dict[str, Any]],
        image_parts_by_msg: dict[int, list[str]] | None = None,
        task_kind: str = "chat",
    ) -> str:
        """Build LLaDA2-Uni chat format prompt.

        Image blocks are inserted at the start of their originating message's
        content via *image_parts_by_msg* (message index -> header/footer tokens).

        *task_kind* selects the system prompt:
          - "chat": multimodal understanding assistant
          - "edit": image editing assistant
          - "t2i": text-to-image generation assistant
          - "t2i_thinking": text-to-image generation with thinking
        """
        if task_kind == "t2i":
            system_prompt = SYSTEM_PROMPT_T2I
        elif task_kind == "t2i_thinking":
            system_prompt = SYSTEM_PROMPT_T2I_THINKING
        elif task_kind == "edit":
            system_prompt = EDIT_SYSTEM_PROMPT
        else:
            system_prompt = SYSTEM_PROMPT_CHAT

        parts: list[str] = []

        parts.append(f"{ROLE_SYSTEM} {system_prompt} ")

        for msg_idx, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                continue

            role_tag = ROLE_HUMAN if role == "user" else ROLE_ASSISTANT

            img_prefix = ""
            if image_parts_by_msg and msg_idx in image_parts_by_msg:
                img_prefix = "".join(image_parts_by_msg[msg_idx])

            if isinstance(content, str):
                parts.append(f"{role_tag}{img_prefix}{content}")
            elif isinstance(content, list):
                text_parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "text")
                        if item_type == "text":
                            text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                parts.append(f"{role_tag}{img_prefix}{''.join(text_parts)}")
            else:
                parts.append(f"{role_tag}{img_prefix}{content}")

        parts.append(ROLE_ASSISTANT)
        return "".join(parts)

    def _build_t2i_header_ids(self, grid_h: int, grid_w: int) -> list[int]:
        """Build image generation header token IDs for t2i requests.

        Returns the token sequence ``<soi><|reserved_token_{h}|><|reserved_token_{w}|><boi>``
        which tells the model to generate image tokens after this header.
        The tokens are built via the tokenizer to avoid BPE splitting issues.

        Args:
            grid_h: Grid height in 16-px semantic-patch units.
            grid_w: Grid width in 16-px semantic-patch units.
        """
        header_ids = [self._soi_id]
        h_token = f"<|reserved_token_{grid_h}|>"
        w_token = f"<|reserved_token_{grid_w}|>"
        header_ids.extend(self._tokenizer.encode(h_token, add_special_tokens=False))
        header_ids.extend(self._tokenizer.encode(w_token, add_special_tokens=False))
        header_ids.append(self._boi_id)
        return header_ids

    # ------------------------------------------------------------------
    # T2I helpers
    # ------------------------------------------------------------------

    def _build_t2i_uncond_input_ids(self, grid_h: int, grid_w: int) -> list[int]:
        """Build unconditional input_ids for T2I CFG.

        Replaces the user text with ``<uncondition>`` while keeping
        the image generation header.
        """
        uncond_prompt = (
            f"{ROLE_SYSTEM} {SYSTEM_PROMPT_T2I} "
            f"{ROLE_HUMAN}{UNCOND_TEXT}"
            f"{ROLE_ASSISTANT}"
        )
        uncond_ids = self._tokenizer.encode(uncond_prompt, add_special_tokens=False)
        # Append the same image header as the conditional prompt
        header_ids = self._build_t2i_header_ids(grid_h, grid_w)
        uncond_ids.extend(header_ids)
        return uncond_ids

    # ------------------------------------------------------------------
    # Edit helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_user_prompt_text(
        messages: list[dict[str, Any]],
    ) -> str:
        """Concatenate text from the last user message (string or content list)."""
        for msg in reversed(messages):
            if msg.get("role") not in (None, "user"):
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type", "text") == "text":
                        texts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        texts.append(item)
                return "".join(texts)
        return ""

    @classmethod
    def _require_edit_instruction(
        cls,
        messages: list[dict[str, Any]],
    ) -> str:
        instruction_text = cls._extract_user_prompt_text(messages)
        if not instruction_text.strip():
            raise ValueError("Image editing requires a non-empty instruction")
        return instruction_text

    def _populate_edit_cfg_stream_state(
        self,
        *,
        stream_state: dict[str, Any],
        image_generation: dict[str, Any],
        conditional_input_ids: list[int],
        instruction_text: str,
        src_image_block: str,
        grid_h: int,
        grid_w: int,
        num_image_tokens: int,
    ) -> None:
        cfg_text_scale, cfg_image_scale = _resolve_edit_cfg_scales(image_generation)
        if cfg_text_scale <= 0.0 and cfg_image_scale <= 0.0:
            return

        uncond_input_ids = self._build_edit_uncond_input_ids(
            src_image_block,
            grid_h,
            grid_w,
            [num_image_tokens],
        )
        uncond_input_ids, uncond_left_pad_len = align_cfg_unconditional_input_ids(
            self._tokenizer,
            conditional_input_ids,
            uncond_input_ids,
        )
        stream_state["uncond_input_ids"] = uncond_input_ids
        stream_state["uncond_left_pad_len"] = uncond_left_pad_len
        stream_state["cfg_scale"] = cfg_text_scale
        stream_state["cfg_rescale"] = float(image_generation.get("cfg_rescale", 0.7))

        if cfg_image_scale <= 0.0:
            return

        no_img_input_ids = self._build_edit_no_img_input_ids(
            instruction_text,
            grid_h,
            grid_w,
        )
        no_img_input_ids, no_img_left_pad_len = align_cfg_unconditional_input_ids(
            self._tokenizer,
            conditional_input_ids,
            no_img_input_ids,
        )
        stream_state["uncond_img_input_ids"] = no_img_input_ids
        stream_state["uncond_img_left_pad_len"] = no_img_left_pad_len
        stream_state["cfg_image_scale"] = cfg_image_scale

    def _build_edit_payload(
        self,
        payload: StagePayload,
        messages: list[dict[str, Any]],
        images: list[Image.Image],
        request_metadata: dict[str, Any],
    ) -> StagePayload:
        """Build an image-edit payload.

        Prompt structure (mirrors HF ``modeling_llada2uni_moe.py:edit_image``):
            <sys-edit> <soi><h><w><boi> <src_VQ_placeholders> <|/image|>
            {instruction} <role>ASSISTANT</role> <soi><h><w><boi>

        The first ``<boi>`` block holds VQ placeholders that
        ``merge_image_tokens_for_thinker`` will replace with the encoder's
        actual VQ ids. The trailing ``<soi>...<boi>`` (no body) signals the
        dLLM to emit the edited image's VQ tokens after the assistant tag.

        Output grid matches the input image (HF behavior).
        """

        # Use the first image only (HF edit takes a single source image).
        instruction_text = self._require_edit_instruction(messages)
        src_image = images[0]

        # Encode source image using edit-specific preprocessing (var_center_crop
        # with 512px patch budget, matching the generation pipeline).
        cropped = preprocess_image_edit([src_image], self._factor)
        image_processor = self._image_processor
        img_result = edit_image_pixel_values(
            cropped,
            patch_size=image_processor.patch_size,
            temporal_patch_size=image_processor.temporal_patch_size,
            merge_size=image_processor.merge_size,
            image_mean=image_processor.image_mean,
            image_std=image_processor.image_std,
            rescale_factor=image_processor.rescale_factor,
        )
        pixel_values = img_result["pixel_values"]
        image_grid_thw = img_result["image_grid_thw"]
        encoder_inputs: dict[str, dict[str, Any]] = {
            IMAGE_STAGE: {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
        }

        t, h_raw, w_raw = image_grid_thw[0].tolist()
        # LLaDA2-Uni VQ-VAE: one token per patch (no PatchMerger).
        num_image_tokens = t * h_raw * w_raw
        # Decoder grid: post-merge dimensions for image_info
        grid_h = (t * h_raw) // self._merge_size
        grid_w = w_raw // self._merge_size

        # Source image block in the prompt: <soi><h><w><boi>[VQ placeholders]<eoi>
        h_token = f"<|reserved_token_{h_raw}|>"
        w_token = f"<|reserved_token_{w_raw}|>"
        src_image_block = f"{SOI_TOKEN}{h_token}{w_token}{BOI_TOKEN}{EOI_TOKEN}"

        # Build the full edit prompt
        text_prompt = (
            f"{ROLE_SYSTEM} {EDIT_SYSTEM_PROMPT} "
            f"{ROLE_HUMAN}"
            f"{src_image_block}"
            f"{instruction_text}"
            f"{ROLE_ASSISTANT}"
        )
        input_ids = self._tokenizer.encode(text_prompt, add_special_tokens=False)

        # Insert VQ placeholders into the source image block
        input_ids = self._insert_image_placeholders(input_ids, [num_image_tokens])

        # Append the output image header (<soi><h><w><boi>) so the model
        # knows where to emit the edited image's VQ tokens.
        header_ids = self._build_t2i_header_ids(grid_h, grid_w)
        input_ids.extend(header_ids)

        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)

        validate_prompt_seq_len(
            input_ids_tensor,
            max_seq_len=self._max_seq_len,
            max_new_tokens=(
                request_metadata.get("params", {}).get(
                    "max_new_tokens", DEFAULT_THINKER_MAX_NEW_TOKENS
                )
                if isinstance(request_metadata, dict)
                else DEFAULT_THINKER_MAX_NEW_TOKENS
            ),
            request_id=payload.request_id,
        )

        prompt = {"input_ids": input_ids_tensor}

        stream_state: dict[str, Any] = {}
        stream_state["image_info"] = [{"grid_h": grid_h, "grid_w": grid_w}]

        ig_meta = request_metadata.get("image_generation", {})
        dllm_steps = ig_meta.get("dllm_steps") if isinstance(ig_meta, dict) else None
        if dllm_steps is not None:
            stream_state["dllm_steps"] = int(dllm_steps)

        self._populate_edit_cfg_stream_state(
            stream_state=stream_state,
            image_generation=ig_meta if isinstance(ig_meta, dict) else {},
            conditional_input_ids=input_ids,
            instruction_text=instruction_text,
            src_image_block=src_image_block,
            grid_h=grid_h,
            grid_w=grid_w,
            num_image_tokens=num_image_tokens,
        )

        state = LLaDA2UniPipelineState(
            prompt=prompt,
            encoder_inputs=encoder_inputs,
            request_metadata=request_metadata,
            task_kind="edit",
            stream_state=stream_state,
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
            timing=payload.timing,
        )

    def _build_edit_uncond_input_ids(
        self,
        src_image_block: str,
        grid_h: int,
        grid_w: int,
        image_token_counts: list[int],
    ) -> list[int]:
        """Build unconditional input_ids for edit CFG.

        Mirrors the conditional edit prompt but replaces the instruction
        text with ``<uncondition>``, keeping the source image VQ
        placeholders intact.
        """
        uncond_prompt = (
            f"{ROLE_SYSTEM} {EDIT_SYSTEM_PROMPT} "
            f"{ROLE_HUMAN}"
            f"{src_image_block}"
            f"{UNCOND_TEXT}"
            f"{ROLE_ASSISTANT}"
        )
        uncond_ids = self._tokenizer.encode(uncond_prompt, add_special_tokens=False)
        # Insert VQ placeholders (same positions as the conditional prompt)
        uncond_ids = self._insert_image_placeholders(uncond_ids, image_token_counts)
        # Append the output image header
        header_ids = self._build_t2i_header_ids(grid_h, grid_w)
        uncond_ids.extend(header_ids)
        return uncond_ids

    def _build_edit_no_img_input_ids(
        self,
        instruction_text: str,
        grid_h: int,
        grid_w: int,
    ) -> list[int]:
        """Build the no-image (drop source image) branch for editing CFG.

        Mirrors the conditional edit prompt but removes the source image block,
        keeping the instruction intact (HF ``uncond_img`` / ``ui``).
        """
        no_img_prompt = (
            f"{ROLE_SYSTEM} {EDIT_SYSTEM_PROMPT} "
            f"{ROLE_HUMAN}"
            f"{SOI_TOKEN}"
            f"{instruction_text}"
            f"{ROLE_ASSISTANT}"
        )
        no_img_ids = self._tokenizer.encode(no_img_prompt, add_special_tokens=False)
        header_ids = self._build_t2i_header_ids(grid_h, grid_w)
        no_img_ids.extend(header_ids)
        return no_img_ids

    def _insert_image_placeholders(
        self,
        input_ids: list[int],
        image_token_counts: list[int],
    ) -> list[int]:
        new_ids: list[int] = []
        cursor = 0
        search_start = 0

        for image_idx, num_tokens in enumerate(image_token_counts):
            boi_idx = next(
                (
                    i
                    for i in range(search_start, len(input_ids))
                    if input_ids[i] == self._boi_id
                ),
                None,
            )
            if boi_idx is None:
                raise ValueError(
                    f"Expected image block {image_idx} but no matching <boi> token was found"
                )

            eoi_idx = next(
                (
                    i
                    for i in range(boi_idx + 1, len(input_ids))
                    if input_ids[i] == self._eoi_id
                ),
                None,
            )
            if eoi_idx is None:
                raise ValueError(
                    f"No <eoi> token found after <boi> for image block {image_idx}"
                )

            new_ids.extend(input_ids[cursor : boi_idx + 1])
            new_ids.extend([DUMMY_IMAGE_TOKEN_ID] * num_tokens)
            cursor = eoi_idx
            search_start = eoi_idx + 1

        new_ids.extend(input_ids[cursor:])
        return new_ids
