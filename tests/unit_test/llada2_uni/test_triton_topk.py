# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("sglang")

from sglang_omni.models.llada2_uni.components import triton_topk
from sglang_omni.models.llada2_uni.components.thinker import LLaDA2MoeSparseMoeBlock

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton top-k tests require CUDA"
)


def _reference_grouped_topk(
    scores: torch.Tensor,
    *,
    num_experts_per_tok: int,
    n_group: int,
    topk_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_experts = scores.shape
    experts_per_group = num_experts // n_group
    group_scores = (
        scores.view(num_tokens, n_group, experts_per_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_ids = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, group_ids, True)
    expert_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, n_group, experts_per_group)
        .reshape(num_tokens, num_experts)
    )
    masked_scores = scores.masked_fill(~expert_mask, -torch.inf)
    return torch.topk(
        masked_scores,
        k=num_experts_per_tok,
        dim=-1,
        sorted=False,
    )


@pytest.mark.parametrize(
    ("num_tokens", "num_experts", "n_group", "topk_group", "top_k"),
    [(19, 64, 8, 2, 4), (257, 256, 8, 4, 8)],
)
def test_grouped_topk_matches_pytorch_selected_experts(
    num_tokens: int,
    num_experts: int,
    n_group: int,
    topk_group: int,
    top_k: int,
) -> None:
    storage = torch.rand(
        (num_tokens, num_experts * 2), device="cuda", dtype=torch.float32
    )
    scores = storage[:, ::2]
    assert not scores.is_contiguous()
    _, expected_ids = _reference_grouped_topk(
        scores,
        num_experts_per_tok=top_k,
        n_group=n_group,
        topk_group=topk_group,
    )

    actual_weights, actual_ids = triton_topk.grouped_topk_triton(
        scores,
        num_experts_per_tok=top_k,
        n_group=n_group,
        topk_group=topk_group,
    )

    assert torch.equal(
        actual_ids.sort(dim=-1).values,
        expected_ids.sort(dim=-1).values,
    )
    torch.testing.assert_close(actual_weights, scores.gather(1, actual_ids))


def test_grouped_topk_caller_falls_back_after_triton_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = LLaDA2MoeSparseMoeBlock.__new__(LLaDA2MoeSparseMoeBlock)
    torch.nn.Module.__init__(block)
    block.num_experts = 64
    block.num_experts_per_tok = 4
    block.n_group = 8
    block.topk_group = 2
    scores = torch.rand((11, 64), device="cuda", dtype=torch.float32)
    expected_weights, expected_ids = block._group_limited_topk_fallback(scores)

    def fail_triton(*args, **kwargs):
        raise RuntimeError("injected Triton failure")

    monkeypatch.setattr(triton_topk, "grouped_topk_triton", fail_triton)

    actual_weights, actual_ids = block._group_limited_topk(scores)

    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(actual_weights, expected_weights)
