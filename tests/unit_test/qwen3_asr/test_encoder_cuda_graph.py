# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang_omni.models.qwen3_asr.audio_lengths import qwen3_asr_num_audio_tokens
from sglang_omni.models.qwen3_asr.encoder_cuda_graph import (
    Qwen3ASREncoderCudaGraphRunner,
    _bucket_conv_chunks,
    _bucket_packed_tokens,
    _capture_cu_seqlens,
    _gather_indices,
    _pad_cu_seqlens,
    _plan_static_shapes,
    _prepare_conv_chunks,
)
from sglang_omni.models.qwen3_asr.sglang_model import Qwen3ASRForConditionalGeneration


def test_bucket_conv_chunks_rounds_up_within_cap() -> None:
    assert _bucket_conv_chunks(1, 256) == 1
    assert _bucket_conv_chunks(3, 256) == 4
    assert _bucket_conv_chunks(9, 256) == 16
    assert _bucket_conv_chunks(257, 256) is None


def test_bucket_packed_tokens_uses_step() -> None:
    assert _bucket_packed_tokens(1) == 64
    assert _bucket_packed_tokens(64) == 64
    assert _bucket_packed_tokens(65) == 128
    assert _bucket_packed_tokens(8193) is None


def test_plan_static_shapes_adds_dummy_segment_for_token_padding() -> None:
    planned = _plan_static_shapes(
        chunk_count=1,
        token_count=65,
        segment_count=1,
        max_chunks=256,
        max_segments=64,
    )
    assert planned is not None
    c_bucket, n_bucket, s_bucket = planned
    assert c_bucket == 1
    assert n_bucket == 128
    assert s_bucket == 2


def test_plan_static_shapes_rejects_over_cap() -> None:
    assert (
        _plan_static_shapes(
            chunk_count=300,
            token_count=8,
            segment_count=1,
            max_chunks=256,
            max_segments=64,
        )
        is None
    )


def test_pad_cu_seqlens_absorbs_token_padding_in_dummy_segments() -> None:
    cu = torch.tensor([0, 65], dtype=torch.int32)
    padded = _pad_cu_seqlens(cu, s_bucket=2, n_bucket=128)
    assert padded.tolist() == [0, 65, 128]


def test_capture_cu_seqlens_has_no_empty_segments() -> None:
    cu = _capture_cu_seqlens(2, 64, torch.device("cpu"))
    assert cu.tolist() == [0, 63, 64]
    lengths = (cu[1:] - cu[:-1]).tolist()
    assert min(lengths) >= 1
    assert sum(lengths) == 64
    single = _capture_cu_seqlens(1, 64, torch.device("cpu"))
    assert single.tolist() == [0, 64]


def test_gather_indices_pads_with_dummy_source() -> None:
    mask = torch.tensor([[True, True, False], [True, False, False]])
    idx = _gather_indices(mask, n_bucket=8, dummy_src=99)
    assert idx[:3].tolist() == [0, 1, 3]
    assert idx[3:].tolist() == [99, 99, 99, 99, 99]


def test_prepare_conv_chunks_pads_short_tails_to_window() -> None:
    n_window = 100
    chunk_width = n_window * 2
    n_mels = 8
    frames = 50
    packed = torch.randn(n_mels, frames)
    lens = torch.tensor([frames], dtype=torch.long)
    padded, mask, aftercnn = _prepare_conv_chunks(packed, lens, n_window=n_window)
    assert tuple(padded.shape) == (1, 1, n_mels, chunk_width)
    assert mask.shape[0] == 1
    assert int(aftercnn.item()) == qwen3_asr_num_audio_tokens(frames)
    assert int(mask.sum().item()) == qwen3_asr_num_audio_tokens(frames)


class _EagerTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))
        self.dtype = torch.float32
        self.calls = 0

    def forward(self, input_features, feature_lens=None):
        del feature_lens
        self.calls += 1
        n_tokens = input_features.shape[-1]
        return SimpleNamespace(last_hidden_state=torch.ones(n_tokens, 4))


def _model_with(runner) -> Qwen3ASRForConditionalGeneration:
    model = object.__new__(Qwen3ASRForConditionalGeneration)
    nn.Module.__init__(model)
    model.audio_tower = _EagerTower()
    model._encoder_graph_runner = runner
    return model


def _item(num_frames: int, n_mels: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        feature=torch.randn(1, n_mels, num_frames),
        feature_attention_mask=torch.ones(1, num_frames, dtype=torch.long),
    )


def test_get_audio_feature_routes_through_graph_runner() -> None:
    observed = {}

    class _Runner:
        def run(self, input_features, feature_lens):
            observed["feat"] = tuple(input_features.shape)
            observed["lens"] = feature_lens.tolist()
            n_tokens = int(feature_lens.sum().item())
            return torch.ones(n_tokens, 4)

    model = _model_with(_Runner())
    out = model.get_audio_feature([_item(12)])
    assert observed["feat"] == (8, 12)
    assert observed["lens"] == [12]
    assert tuple(out.shape) == (12, 4)
    assert model.audio_tower.calls == 0


def test_get_audio_feature_falls_back_to_eager_when_runner_declines() -> None:
    class _DecliningRunner:
        def run(self, input_features, feature_lens):
            del input_features, feature_lens
            return None

    model = _model_with(_DecliningRunner())
    out = model.get_audio_feature([_item(12)])
    assert tuple(out.shape) == (12, 4)
    assert torch.equal(out, torch.ones(12, 4))
    assert model.audio_tower.calls == 1


def _pending_runner() -> Qwen3ASREncoderCudaGraphRunner:
    runner = object.__new__(Qwen3ASREncoderCudaGraphRunner)
    runner._lock = threading.Lock()
    runner._pending = {}
    runner._graphs = {}
    runner._failed = set()
    runner._min_free_bytes = 1
    runner._pool = None
    runner._device = "cuda:0"
    return runner


def test_capture_one_pending_noop_when_empty() -> None:
    runner = _pending_runner()
    assert runner.has_pending() is False
    assert runner.capture_one_pending() is False


def test_capture_one_pending_skips_already_cached_key() -> None:
    runner = _pending_runner()
    key = (1, 64, 1)
    runner._pending[key] = (80, 10)
    runner._graphs[key] = object()
    assert runner.capture_one_pending() is False
    assert runner.has_pending() is False


def test_capture_one_pending_records_vram_failure() -> None:
    runner = _pending_runner()
    key = (2, 128, 2)
    runner._pending[key] = (80, 10)
    runner._enough_free_vram = lambda: (False, 0)
    assert runner.capture_one_pending() is False
    assert key in runner._failed
    assert runner.has_pending() is False


def test_capture_one_pending_invokes_capture() -> None:
    runner = _pending_runner()
    key = (4, 256, 4)
    runner._pending[key] = (80, 10)
    runner._pending[(8, 512, 8)] = (80, 10)
    captured = {}

    def _fake_capture(c_bucket, n_bucket, s_bucket, n_mels, t_cnn):
        captured["args"] = (c_bucket, n_bucket, s_bucket, n_mels, t_cnn)
        return SimpleNamespace()

    runner._enough_free_vram = lambda: (True, 1)
    runner._capture = _fake_capture
    with patch("torch.cuda.device", return_value=nullcontext()):
        assert runner.capture_one_pending() is True
    assert captured["args"] == (4, 256, 4, 80, 10)
    assert key in runner._graphs
    assert runner.has_pending() is True


def test_gpu_is_idle_follows_default_stream_query() -> None:
    runner = _pending_runner()
    runner._device = torch.device("cuda:0")
    busy = SimpleNamespace(query=lambda: False)
    idle = SimpleNamespace(query=lambda: True)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.default_stream", return_value=busy),
    ):
        assert runner.gpu_is_idle() is False
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.default_stream", return_value=idle),
    ):
        assert runner.gpu_is_idle() is True
