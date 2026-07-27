# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.voxcpm_common import core
from sglang_omni.models.voxcpm_common.audio_vae import AudioVAE, AudioVAEV2
from sglang_omni.models.voxcpm_common.core import (
    CausalMiniCPMAttention,
    MiniCPMLongRoPE,
)
from sglang_omni.models.voxcpm_common.state_pool import VoxCPMRequestStatePool
from sglang_omni.models.voxcpm_common.weights import load_voxcpm_weights


def test_state_pool_isolates_and_recycles_request_rows():
    pool = VoxCPMRequestStatePool(
        2, 4, 2, 3, device=torch.device("cpu"), dtype=torch.float32
    )
    first = pool.acquire("first")
    second = pool.acquire("second")
    assert first != second
    pool.configure([first, second], temperature=torch.tensor([0.5, 0.9]))
    pool.update(
        torch.tensor([first]),
        feedback=torch.ones(1, 4),
        latents=torch.ones(1, 2, 3),
        stop_flag=torch.ones(1, dtype=torch.long),
    )
    assert pool.feedback[second].count_nonzero() == 0
    assert pool.steps.tolist() == [1, 0]

    pool.reset("first")
    recycled = pool.acquire("third")
    assert recycled == first
    assert pool.feedback[recycled].count_nonzero() == 0
    assert pool.temperature[recycled] == 1
    assert pool.row_for("first") is None


def test_state_pool_exhaustion_and_idempotent_reset():
    pool = VoxCPMRequestStatePool(
        1, 2, 1, 2, device=torch.device("cpu"), dtype=torch.float32
    )
    assert pool.acquire("request") == pool.acquire("request")
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.acquire("other")
    pool.reset("missing")
    pool.reset("request")
    assert pool.acquire("other") == 0


def test_long_rope_version_shapes_and_bounds():
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        max_position_embeddings=8,
        rope_theta=10000.0,
        rope_scaling={
            "original_max_position_embeddings": 4,
            "short_factor": [1.0, 1.0],
            "long_factor": [2.0, 2.0],
        },
    )
    positions = torch.tensor([0, 1, 2])
    query = torch.randn(3, 8)
    key = torch.randn(3, 8)
    for version in ("v1", "v2"):
        rope = MiniCPMLongRoPE(config, version=version)
        output_query, output_key = rope(positions, query, key)
        assert output_query.shape == query.shape
        assert output_key.shape == key.shape
        with pytest.raises(ValueError, match="position exceeds"):
            rope(torch.tensor([8]), query[:1], key[:1])


def test_voxcpm2_residual_attention_can_disable_rope(monkeypatch):
    class _LayerStub(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(core, "QKVParallelLinear", _LayerStub)
    monkeypatch.setattr(core, "RowParallelLinear", _LayerStub)
    monkeypatch.setattr(core, "RadixAttention", _LayerStub)
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        kv_channels=4,
        max_position_embeddings=8,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )
    with_rope = CausalMiniCPMAttention(
        config,
        layer_id=0,
        version="v2",
        use_rope=True,
        prefix="with_rope",
    )
    without_rope = CausalMiniCPMAttention(
        config,
        layer_id=1,
        version="v2",
        use_rope=False,
        prefix="without_rope",
    )
    assert isinstance(with_rope.rotary_emb, MiniCPMLongRoPE)
    assert without_rope.rotary_emb is None


def test_weight_loader_accepts_official_root_names():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.proj = nn.Linear(2, 2, bias=False)

    module = Wrapper()
    loaded = load_voxcpm_weights(module, [("proj.weight", torch.full((2, 2), 3.0))])
    assert loaded == {"model.proj.weight"}
    torch.testing.assert_close(module.model.proj.weight, torch.full((2, 2), 3.0))


def test_audio_vae_sample_rates_come_from_model_config():
    common = {
        "encoder_dim": 32,
        "encoder_rates": [2],
        "latent_dim": 4,
        "decoder_dim": 32,
        "decoder_rates": [2],
    }
    assert AudioVAE(config={**common, "sample_rate": 44100}).out_sample_rate == 44100
    v2 = AudioVAEV2(
        config={
            **common,
            "sample_rate": 16000,
            "out_sample_rate": 32000,
            "sr_bin_boundaries": [12000, 24000],
        }
    )
    assert v2.sample_rate == 16000
    assert v2.out_sample_rate == 32000
