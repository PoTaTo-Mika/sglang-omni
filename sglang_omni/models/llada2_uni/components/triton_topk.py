"""Fused Triton kernel for the LLaDA2 group-limited top-k router.

Reference PyTorch implementation is `LLaDA2MoeSparseMoeBlock._group_limited_topk`
in `components/thinker.py`. This module provides `grouped_topk_triton(scores,
num_experts_per_tok, n_group, topk_group)` which returns the same (topk_weights,
topk_ids) pair using one Triton kernel per token instead of five separate
PyTorch launches (view → top2 → sum → topk_group → mask → scatter → topk).

Numerical parity vs. PyTorch:
  * The kernel selects the same experts (barring exact-tie corner cases; sigmoid
    scores make exact ties astronomically unlikely).
  * Tie-break rule: on equal scores, the smaller expert index wins (matches
    Triton's `tl.argmax` and PyTorch's `torch.argmax` semantics).
  * `topk_weights` returned here is gathered from the bias-added routing scores.
    In the caller, this value is immediately overwritten with a bias-free
    `torch.gather(scores, ...)`, so we only need `topk_ids` to be equivalent.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_topk_kernel(
    scores_ptr,  # [num_tokens, E] float32, contiguous
    topk_weights_ptr,  # [num_tokens, TOP_K] float32
    topk_ids_ptr,  # [num_tokens, TOP_K] int64
    stride_scores_n,
    stride_scores_e,
    stride_w_n,
    stride_w_k,
    stride_i_n,
    stride_i_k,
    E: tl.constexpr,
    N_GROUP: tl.constexpr,
    EPG: tl.constexpr,  # experts per group = E // N_GROUP
    TOPK_GROUP: tl.constexpr,  # groups kept per token
    TOP_K: tl.constexpr,  # experts kept per token
    NEG_INF: tl.constexpr,
):
    token = tl.program_id(0)

    expert_offs = tl.arange(0, E)
    row_ptr = scores_ptr + token * stride_scores_n
    scores = tl.load(row_ptr + expert_offs * stride_scores_e)

    # -------- step 1: per-group top-2 sum -----------------------------------
    # Reshape [E] → [N_GROUP, EPG]; the two argmaxes per row give top-2.
    scores_2d = tl.reshape(scores, (N_GROUP, EPG))
    epg_offs = tl.arange(0, EPG)

    max1_val = tl.max(scores_2d, axis=1)  # [N_GROUP]
    max1_idx = tl.argmax(scores_2d, axis=1)  # [N_GROUP]
    is_max1 = epg_offs[None, :] == max1_idx[:, None]
    scores_2d_wo_max1 = tl.where(is_max1, NEG_INF, scores_2d)
    max2_val = tl.max(scores_2d_wo_max1, axis=1)  # [N_GROUP]
    group_scores = max1_val + max2_val  # [N_GROUP]

    # -------- step 2: pick top TOPK_GROUP groups (order unimportant) --------
    # Note: creating int1 tensors via comparisons is more portable across
    # Triton versions than tl.full(..., dtype=tl.int1) directly.
    group_offs = tl.arange(0, N_GROUP)
    group_active = group_offs >= 0  # all-True [N_GROUP] int1
    group_selected = group_offs < 0  # all-False [N_GROUP] int1
    for _ in tl.static_range(TOPK_GROUP):
        masked = tl.where(group_active, group_scores, NEG_INF)
        pick = tl.argmax(masked, axis=0)  # scalar int32
        is_pick = group_offs == pick
        group_selected = group_selected | is_pick
        group_active = group_active & (is_pick == 0)

    # -------- step 3: broadcast group mask to expert mask [E] ---------------
    # expert e = g*EPG + i belongs to group g; broadcast group_selected along i
    # then flatten. reshape is row-major so this matches scores.view(N,G,EPG).
    expert_selected_2d = tl.broadcast_to(group_selected[:, None], (N_GROUP, EPG))
    expert_selected = tl.reshape(expert_selected_2d, (E,))
    masked_scores = tl.where(expert_selected, scores, NEG_INF)

    # -------- step 4: iterative top-K over active experts -------------------
    # Iterative max-and-mask: k iterations, each picks the current argmax and
    # masks it out. Output order is decreasing score (stricter than PyTorch's
    # `torch.topk(sorted=False)` but downstream treats it as an unordered set).
    out_w_ptr = topk_weights_ptr + token * stride_w_n
    out_i_ptr = topk_ids_ptr + token * stride_i_n
    active = expert_offs >= 0  # all-True [E] int1
    for k in tl.static_range(TOP_K):
        cur = tl.where(active, masked_scores, NEG_INF)
        val = tl.max(cur, axis=0)
        idx = tl.argmax(cur, axis=0)
        tl.store(out_w_ptr + k * stride_w_k, val)
        tl.store(out_i_ptr + k * stride_i_k, idx.to(tl.int64))
        active = active & (expert_offs != idx)


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def grouped_topk_triton(
    scores: torch.Tensor,
    num_experts_per_tok: int,
    n_group: int,
    topk_group: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton drop-in for `LLaDA2MoeSparseMoeBlock._group_limited_topk`.

    Args:
        scores: [num_tokens, num_experts] float32 (bias-added routing scores).
        num_experts_per_tok: top-k experts to activate (=TOP_K).
        n_group: number of expert groups.
        topk_group: number of groups kept per token.

    Returns:
        topk_weights: [num_tokens, num_experts_per_tok] float32 — values from
            the routing scores at the selected expert indices, in decreasing
            score order.
        topk_ids: [num_tokens, num_experts_per_tok] int64 — selected expert
            indices, in same order as topk_weights.

    Notes:
        Selects the same set of experts as `_group_limited_topk` (reference
        PyTorch implementation). Downstream code overwrites `topk_weights`
        with a bias-free gather so only `topk_ids` needs to match the
        reference set.
    """
    assert scores.is_cuda, "grouped_topk_triton needs a CUDA tensor"
    assert scores.dim() == 2, "scores must be [num_tokens, num_experts]"
    num_tokens, num_experts = scores.shape

    # Config validation. These constraints are hard requirements — Triton
    # constexpr shapes must be powers of two, and TOPK_GROUP > N_GROUP would
    # silently no-op on redundant picks instead of failing loudly.
    assert (
        num_experts % n_group == 0
    ), f"num_experts ({num_experts}) must be divisible by n_group ({n_group})"
    experts_per_group = num_experts // n_group
    assert _is_pow2(num_experts), (
        f"num_experts ({num_experts}) must be a power of two (Triton tl.arange "
        "constraint)"
    )
    assert _is_pow2(n_group) and _is_pow2(experts_per_group), (
        f"n_group ({n_group}) and experts_per_group ({experts_per_group}) "
        "must both be powers of two"
    )
    assert (
        1 <= topk_group <= n_group
    ), f"topk_group ({topk_group}) must be in [1, n_group={n_group}]"
    max_active = topk_group * experts_per_group
    assert 1 <= num_experts_per_tok <= max_active, (
        f"num_experts_per_tok ({num_experts_per_tok}) must be in "
        f"[1, topk_group*experts_per_group={max_active}]"
    )

    if scores.dtype != torch.float32:
        scores = scores.to(torch.float32)
    scores = scores.contiguous()

    topk_weights = torch.empty(
        (num_tokens, num_experts_per_tok), dtype=torch.float32, device=scores.device
    )
    topk_ids = torch.empty(
        (num_tokens, num_experts_per_tok), dtype=torch.int64, device=scores.device
    )

    # Empty batch: skip kernel launch (grid=(0,) is legal but wasteful).
    if num_tokens == 0:
        return topk_weights, topk_ids

    _grouped_topk_kernel[(num_tokens,)](
        scores,
        topk_weights,
        topk_ids,
        scores.stride(0),
        scores.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        E=num_experts,
        N_GROUP=n_group,
        EPG=experts_per_group,
        TOPK_GROUP=topk_group,
        TOP_K=num_experts_per_tok,
        NEG_INF=float("-inf"),
    )
    return topk_weights, topk_ids
