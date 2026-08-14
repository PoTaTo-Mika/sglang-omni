"""Triton helpers for LLaDA2-Uni LowConfidence decode.

The standard no-CFG fast path uses two kernels for one denoise block update:
(1) per-row/per-vocab-tile argmax and softmax denominator partials; (2) final
row reductions plus block-level LowConfidence keep/update in place.  This avoids
materializing a full ``[block_size, vocab_size]`` softmax and removes the small
PyTorch op chain from the decode step.

CFG helpers cover the large-vocab argmax/confidence reduction only. Guidance
and optional ``cfg_rescale`` remain in the PyTorch caller.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _partial_argmax_sumexp_kernel(
    logits_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_arg_ptr,
    stride_n: tl.constexpr,
    stride_v: tl.constexpr,
    num_tiles: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    FORCE_IMAGE_ONLY: tl.constexpr,
    IMAGE_TOKEN_OFFSET: tl.constexpr,
    ALLOWED_TOKEN_IDS: tl.constexpr,
    NUM_ALLOWED_TOKENS: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    valid = offs < V
    if FORCE_IMAGE_ONLY:
        valid = valid & (offs >= IMAGE_TOKEN_OFFSET)
        for i in tl.static_range(0, NUM_ALLOWED_TOKENS):
            valid = valid | (offs == ALLOWED_TOKEN_IDS[i])

    vals = tl.load(
        logits_ptr + row * stride_n + offs * stride_v,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    local_max = tl.max(vals, axis=0)
    has_valid = tl.max(valid.to(tl.int32), axis=0) > 0
    safe_max = tl.where(has_valid, local_max, 0.0)
    shifted = tl.where(valid, vals - safe_max, -float("inf"))
    local_sum = tl.where(has_valid, tl.sum(tl.exp(shifted), axis=0), 0.0)
    local_arg = tile * BLOCK_V + tl.argmax(vals, axis=0)

    out = row * num_tiles + tile
    tl.store(partial_max_ptr + out, local_max)
    tl.store(partial_sum_ptr + out, local_sum)
    tl.store(partial_arg_ptr + out, local_arg)


@triton.jit
def _partial_cfg_argmax_sumexp_kernel(
    cond_ptr,
    uncond_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_arg_ptr,
    cond_stride_n: tl.constexpr,
    cond_stride_v: tl.constexpr,
    uncond_stride_n: tl.constexpr,
    uncond_stride_v: tl.constexpr,
    num_tiles: tl.constexpr,
    V: tl.constexpr,
    CFG_SCALE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    FORCE_IMAGE_ONLY: tl.constexpr,
    IMAGE_TOKEN_OFFSET: tl.constexpr,
    ALLOWED_TOKEN_IDS: tl.constexpr,
    NUM_ALLOWED_TOKENS: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    valid = offs < V
    if FORCE_IMAGE_ONLY:
        valid = valid & (offs >= IMAGE_TOKEN_OFFSET)
        for i in tl.static_range(0, NUM_ALLOWED_TOKENS):
            valid = valid | (offs == ALLOWED_TOKEN_IDS[i])

    cond = tl.load(
        cond_ptr + row * cond_stride_n + offs * cond_stride_v,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    uncond = tl.load(
        uncond_ptr + row * uncond_stride_n + offs * uncond_stride_v,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    vals = uncond + CFG_SCALE * (cond - uncond)
    vals = tl.where(valid, vals, -float("inf"))

    local_max = tl.max(vals, axis=0)
    has_valid = tl.max(valid.to(tl.int32), axis=0) > 0
    safe_max = tl.where(has_valid, local_max, 0.0)
    shifted = tl.where(valid, vals - safe_max, -float("inf"))
    local_sum = tl.where(has_valid, tl.sum(tl.exp(shifted), axis=0), 0.0)
    local_arg = tile * BLOCK_V + tl.argmax(vals, axis=0)

    out = row * num_tiles + tile
    tl.store(partial_max_ptr + out, local_max)
    tl.store(partial_sum_ptr + out, local_sum)
    tl.store(partial_arg_ptr + out, local_arg)


@triton.jit
def _finalize_argmax_conf_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    partial_arg_ptr,
    token_ids_ptr,
    probs_ptr,
    num_tiles: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_T)
    valid = offs < num_tiles
    base = row * num_tiles

    local_max = tl.load(partial_max_ptr + base + offs, mask=valid, other=-float("inf"))
    local_sum = tl.load(partial_sum_ptr + base + offs, mask=valid, other=0.0)

    global_max = tl.max(local_max, axis=0)
    total_sum = tl.sum(local_sum * tl.exp(local_max - global_max), axis=0)
    best_tile = tl.argmax(local_max, axis=0)
    token_id = tl.load(partial_arg_ptr + base + best_tile)
    prob = 1.0 / total_sum

    tl.store(token_ids_ptr + row, token_id)
    tl.store(probs_ptr + row, prob)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _normalize_allowed_token_ids(
    allowed_token_ids: tuple[int, ...], vocab_size: int
) -> tuple[int, ...]:
    normalized = tuple(sorted({int(token_id) for token_id in allowed_token_ids}))
    if any(token_id < 0 or token_id >= vocab_size for token_id in normalized):
        raise ValueError("allowed token IDs must be within the model vocabulary")
    return normalized


def _alloc_partials(rows: int, num_tiles: int, device: torch.device):
    partial_max = torch.empty((rows, num_tiles), device=device, dtype=torch.float32)
    partial_sum = torch.empty((rows, num_tiles), device=device, dtype=torch.float32)
    partial_arg = torch.empty((rows, num_tiles), device=device, dtype=torch.int64)
    token_ids = torch.empty((rows,), device=device, dtype=torch.int64)
    probs = torch.empty((rows,), device=device, dtype=torch.float32)
    return partial_max, partial_sum, partial_arg, token_ids, probs


def argmax_confidence_triton(
    logits: torch.Tensor,
    *,
    force_image_only: bool = False,
    image_token_offset: int = 0,
    allowed_token_ids: tuple[int, ...] = (),
    block_v: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``argmax(logits)`` and winning softmax probability per row."""
    assert logits.is_cuda
    assert logits.dim() == 2
    logits = logits.contiguous()
    rows, vocab = logits.shape
    allowed_token_ids = _normalize_allowed_token_ids(allowed_token_ids, vocab)
    num_tiles = triton.cdiv(vocab, block_v)
    block_t = _next_power_of_2(num_tiles)
    partial_max, partial_sum, partial_arg, token_ids, probs = _alloc_partials(
        rows, num_tiles, logits.device
    )

    _partial_argmax_sumexp_kernel[(rows, num_tiles)](
        logits,
        partial_max,
        partial_sum,
        partial_arg,
        logits.stride(0),
        logits.stride(1),
        num_tiles,
        vocab,
        BLOCK_V=block_v,
        FORCE_IMAGE_ONLY=force_image_only,
        IMAGE_TOKEN_OFFSET=image_token_offset,
        ALLOWED_TOKEN_IDS=allowed_token_ids,
        NUM_ALLOWED_TOKENS=len(allowed_token_ids),
        num_warps=4,
    )
    _finalize_argmax_conf_kernel[(rows,)](
        partial_max,
        partial_sum,
        partial_arg,
        token_ids,
        probs,
        num_tiles,
        BLOCK_T=block_t,
        num_warps=1,
    )
    return token_ids, probs


def cfg_argmax_confidence_triton(
    cond_logits: torch.Tensor,
    uncond_logits: torch.Tensor,
    *,
    cfg_scale: float,
    force_image_only: bool = False,
    image_token_offset: int = 0,
    allowed_token_ids: tuple[int, ...] = (),
    block_v: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Argmax/confidence for ``uncond + cfg_scale * (cond - uncond)``.

    This fast path intentionally covers scale-only CFG.  ``cfg_rescale`` needs
    additional row-wise variance reductions, so callers should keep using the
    PyTorch fallback when rescale is enabled.
    """
    assert cond_logits.is_cuda and uncond_logits.is_cuda
    assert cond_logits.shape == uncond_logits.shape
    assert cond_logits.dim() == 2
    cond_logits = cond_logits.contiguous()
    uncond_logits = uncond_logits.contiguous()
    rows, vocab = cond_logits.shape
    allowed_token_ids = _normalize_allowed_token_ids(allowed_token_ids, vocab)
    num_tiles = triton.cdiv(vocab, block_v)
    block_t = _next_power_of_2(num_tiles)
    partial_max, partial_sum, partial_arg, token_ids, probs = _alloc_partials(
        rows, num_tiles, cond_logits.device
    )

    _partial_cfg_argmax_sumexp_kernel[(rows, num_tiles)](
        cond_logits,
        uncond_logits,
        partial_max,
        partial_sum,
        partial_arg,
        cond_logits.stride(0),
        cond_logits.stride(1),
        uncond_logits.stride(0),
        uncond_logits.stride(1),
        num_tiles,
        vocab,
        CFG_SCALE=float(cfg_scale),
        BLOCK_V=block_v,
        FORCE_IMAGE_ONLY=force_image_only,
        IMAGE_TOKEN_OFFSET=image_token_offset,
        ALLOWED_TOKEN_IDS=allowed_token_ids,
        NUM_ALLOWED_TOKENS=len(allowed_token_ids),
        num_warps=4,
    )
    _finalize_argmax_conf_kernel[(rows,)](
        partial_max,
        partial_sum,
        partial_arg,
        token_ids,
        probs,
        num_tiles,
        BLOCK_T=block_t,
        num_warps=1,
    )
    return token_ids, probs


@triton.jit
def _finalize_keep_update_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    partial_arg_ptr,
    input_ids_ptr,
    tpf_ptr,
    input_start: tl.constexpr,
    rows_n: tl.constexpr,
    num_tiles: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_B: tl.constexpr,
    MASK_ID: tl.constexpr,
    THRESHOLD: tl.constexpr,
    NUM_TO_TRANSFER: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_B)
    tiles = tl.arange(0, BLOCK_T)
    valid_rows = rows < rows_n
    valid_tiles = tiles < num_tiles

    pm = tl.load(
        partial_max_ptr + rows[:, None] * num_tiles + tiles[None, :],
        mask=valid_rows[:, None] & valid_tiles[None, :],
        other=-float("inf"),
    )
    ps = tl.load(
        partial_sum_ptr + rows[:, None] * num_tiles + tiles[None, :],
        mask=valid_rows[:, None] & valid_tiles[None, :],
        other=0.0,
    )

    global_max = tl.max(pm, axis=1)  # [BLOCK_B]
    total_sum = tl.sum(ps * tl.exp(pm - global_max[:, None]), axis=1)
    best_tile = tl.argmax(pm, axis=1)  # [BLOCK_B]
    token_ids = tl.load(
        partial_arg_ptr + rows * num_tiles + best_tile,
        mask=valid_rows,
        other=0,
    )
    probs = 1.0 / total_sum

    old_ids = tl.load(input_ids_ptr + input_start + rows, mask=valid_rows, other=0)
    masked = valid_rows & (old_ids == MASK_ID)
    conf = tl.where(masked, probs, -float("inf"))

    high_conf = conf > THRESHOLD
    high_count = tl.sum(high_conf.to(tl.int32), axis=0)

    active = masked
    topk_keep = rows < 0  # all false
    for _ in tl.static_range(NUM_TO_TRANSFER):
        cur = tl.where(active, conf, -float("inf"))
        val = tl.max(cur, axis=0)
        idx = tl.argmax(cur, axis=0)
        picked = (rows == idx) & (val != -float("inf"))
        topk_keep = topk_keep | picked
        active = active & (rows != idx)

    use_high = high_count >= NUM_TO_TRANSFER
    keep = tl.where(use_high, high_conf, topk_keep) & masked
    tl.store(input_ids_ptr + input_start + rows, token_ids, mask=keep)
    tpf = tl.sum(keep.to(tl.int32), axis=0)
    tl.store(tpf_ptr, tpf)


def low_confidence_update_triton(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    input_start: int,
    mask_id: int,
    threshold: float,
    num_to_transfer: int,
    force_image_only: bool = False,
    image_token_offset: int = 0,
    allowed_token_ids: tuple[int, ...] = (),
    block_v: int = 1024,
) -> int:
    """Run one standard LowConfidence block update with two Triton kernels.

    Kernel 1 computes per-row/per-vocab-tile partial maxima and exp sums.
    Kernel 2 reduces those partials, applies the LowConfidence keep rule over
    the block, updates ``input_ids`` in place, and writes the step tpf.
    """
    assert logits.is_cuda and input_ids.is_cuda
    assert logits.dim() == 2
    logits = logits.contiguous()
    rows, vocab = logits.shape
    allowed_token_ids = _normalize_allowed_token_ids(allowed_token_ids, vocab)
    assert rows > 0
    assert 1 <= num_to_transfer <= rows

    num_tiles = triton.cdiv(vocab, block_v)
    block_t = _next_power_of_2(num_tiles)
    partial_max = torch.empty(
        (rows, num_tiles), device=logits.device, dtype=torch.float32
    )
    partial_sum = torch.empty(
        (rows, num_tiles), device=logits.device, dtype=torch.float32
    )
    partial_arg = torch.empty(
        (rows, num_tiles), device=logits.device, dtype=torch.int64
    )
    tpf = torch.empty((), device=logits.device, dtype=torch.int32)

    _partial_argmax_sumexp_kernel[(rows, num_tiles)](
        logits,
        partial_max,
        partial_sum,
        partial_arg,
        logits.stride(0),
        logits.stride(1),
        num_tiles,
        vocab,
        BLOCK_V=block_v,
        FORCE_IMAGE_ONLY=force_image_only,
        IMAGE_TOKEN_OFFSET=image_token_offset,
        ALLOWED_TOKEN_IDS=allowed_token_ids,
        NUM_ALLOWED_TOKENS=len(allowed_token_ids),
        num_warps=4,
    )
    block_b = _next_power_of_2(rows)
    _finalize_keep_update_kernel[(1,)](
        partial_max,
        partial_sum,
        partial_arg,
        input_ids,
        tpf,
        input_start=input_start,
        rows_n=rows,
        num_tiles=num_tiles,
        BLOCK_T=block_t,
        BLOCK_B=block_b,
        MASK_ID=mask_id,
        THRESHOLD=float(threshold),
        NUM_TO_TRANSFER=int(num_to_transfer),
        num_warps=8,
    )
    return int(tpf.item())
