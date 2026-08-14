# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("sglang")

from sglang_omni.models.llada2_uni.components import thinker as thinker_module
from sglang_omni.models.llada2_uni.components.thinker import (
    LLaDA2MoeSparseMoeBlock,
)
from sglang_omni.scheduling.dllm_scheduler import DllmScheduler


class _RoutedExperts(nn.Module):
    def forward(self, hidden_states, topk_output):
        return hidden_states * 2


class _SharedExperts(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 3


def _make_sparse_block() -> LLaDA2MoeSparseMoeBlock:
    block = LLaDA2MoeSparseMoeBlock.__new__(LLaDA2MoeSparseMoeBlock)
    nn.Module.__init__(block)
    block.experts = _RoutedExperts()
    block.shared_experts = _SharedExperts()
    block.alt_stream = None

    def route(hidden_states):
        num_tokens = hidden_states.shape[0]
        return (
            torch.ones((num_tokens, 1), device=hidden_states.device),
            torch.zeros(
                (num_tokens, 1), device=hidden_states.device, dtype=torch.int64
            ),
            torch.zeros((num_tokens, 1), device=hidden_states.device),
        )

    block._route = route
    return block


def test_sparse_moe_reduces_routed_and_shared_partials_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = _make_sparse_block()
    hidden_states = torch.randn((6, 8))
    reduced_inputs: list[torch.Tensor] = []

    monkeypatch.setattr(
        thinker_module,
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )

    def fake_all_reduce(value: torch.Tensor) -> torch.Tensor:
        reduced_inputs.append(value.clone())
        return value + 1

    monkeypatch.setattr(
        thinker_module,
        "tensor_model_parallel_all_reduce",
        fake_all_reduce,
    )

    output = block._forward_single_stream(hidden_states, hidden_states.clone())

    assert len(reduced_inputs) == 1
    torch.testing.assert_close(reduced_inputs[0], hidden_states * 5)
    torch.testing.assert_close(output, hidden_states * 5 + 1)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Dual-stream parity requires CUDA"
)
def test_sparse_moe_dual_stream_matches_single_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = _make_sparse_block().cuda()
    block.alt_stream = torch.cuda.Stream()
    hidden_states = torch.randn((37, 16), device="cuda")
    monkeypatch.setattr(
        thinker_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    monkeypatch.setattr(thinker_module, "get_is_capture_mode", lambda: False)
    single_stream = block(hidden_states)
    monkeypatch.setattr(thinker_module, "get_is_capture_mode", lambda: True)
    dual_stream = block(hidden_states)
    torch.cuda.synchronize()

    torch.testing.assert_close(dual_stream, single_stream)


@pytest.mark.parametrize(
    ("tp_size", "expected_fanout"),
    [(1, False), (4, True)],
)
def test_dllm_scheduler_declares_tp_work_fanout(
    tp_size: int,
    expected_fanout: bool,
) -> None:
    scheduler = DllmScheduler(
        tp_worker=SimpleNamespace(tp_rank=0),
        tree_cache=object(),
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=object(),
        server_args=SimpleNamespace(tp_size=tp_size, chunked_prefill_size=128),
        model_config=object(),
        dllm_config=SimpleNamespace(block_size=128),
        request_builder=lambda payload: payload,
        result_adapter=lambda result: result,
    )

    assert scheduler.tp_size == tp_size
    assert scheduler.requires_tp_work_fanout is expected_fanout
