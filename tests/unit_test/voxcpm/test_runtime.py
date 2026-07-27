# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.voxcpm_common.audio_vae import (
    CausalConv1d,
    CausalTransposeConv1d,
)
from sglang_omni.models.voxcpm_common.engine_io import (
    CJKSplitTokenizer,
    build_prompt_tensors,
    validate_prompt_length,
)
from sglang_omni.models.voxcpm_common.model_runner import (
    VoxCPMModelRunner,
    derive_step_seed,
)
from sglang_omni.models.voxcpm_common.streaming_vae import (
    BatchedStreamingVAEDecoder,
)


class _Tokenizer:
    eos_token_id = 2

    def get_vocab(self):
        return {"你好": 7, "你": 8, "好": 9, "x": 10}

    def tokenize(self, text):
        return ["你好"] if text == "你好" else list(text)

    def convert_tokens_to_ids(self, tokens):
        vocab = self.get_vocab()
        return [vocab[token] for token in tokens]

    def __len__(self):
        return 16


def test_cjk_multichar_tokens_are_split():
    assert CJKSplitTokenizer(_Tokenizer()).encode("你好") == [8, 9]


def test_voxcpm2_isolated_reference_prompt_layout():
    ids, features, mask, context = build_prompt_tensors(
        variant="voxcpm2",
        tokenizer=CJKSplitTokenizer(_Tokenizer()),
        target_text="x",
        reference_text=None,
        reference_mode="isolated",
        reference_latents=torch.ones(2, 4, 3),
        patch_size=4,
        feat_dim=3,
    )
    assert ids.tolist() == [103, 0, 0, 104, 10, 101]
    assert mask.tolist() == [False, True, True, False, False, False]
    assert features.shape == (6, 4, 3)
    assert context is None


def test_continuation_retains_version_decode_history():
    latents = torch.arange(16 * 3, dtype=torch.float32).reshape(4, 4, 3)
    ids, _, mask, context = build_prompt_tensors(
        variant="voxcpm2",
        tokenizer=CJKSplitTokenizer(_Tokenizer()),
        target_text="x",
        reference_text="x",
        reference_mode="continuation",
        reference_latents=latents,
        patch_size=4,
        feat_dim=3,
    )
    assert len(ids) == len(mask) == 7
    assert context.shape == (12, 3)
    torch.testing.assert_close(context, latents.reshape(-1, 3)[-12:])


def test_prompt_length_checks_prompt_and_generation_budget():
    validate_prompt_length(8, 4, 12)
    with pytest.raises(ValueError, match="prompt is too long"):
        validate_prompt_length(13, 1, 12)
    with pytest.raises(ValueError, match="max_new_tokens"):
        validate_prompt_length(9, 4, 12)


def test_step_seed_is_request_deterministic_and_step_distinct():
    assert derive_step_seed(7, 3) == derive_step_seed(7, 3)
    assert derive_step_seed(7, 3) != derive_step_seed(7, 4)
    assert derive_step_seed(7, 3) != derive_step_seed(8, 3)


class _TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        conv = CausalConv1d(1, 1, 3, padding=1, bias=False)
        nn.init.constant_(conv.weight, 1.0)
        self.decoder = nn.Sequential(conv)

    def decode(self, latents):
        return self.decoder(latents)


def test_streaming_vae_keeps_request_state_isolated():
    vae = _TinyDecoder()
    decoder = BatchedStreamingVAEDecoder(vae, CausalConv1d, CausalTransposeConv1d)
    first = decoder.decode_chunks(torch.ones(2, 1, 2), ["a", "b"])
    second = decoder.decode_chunks(torch.tensor([[[2.0]], [[3.0]]]), ["a", "b"])
    assert first.shape == (2, 1, 2)
    assert second.shape == (2, 1, 1)
    assert second[0].item() != second[1].item()
    decoder.release("a")
    restarted = decoder.decode_chunks(torch.ones(1, 1, 1), ["a"])
    assert restarted.shape == (1, 1, 1)


def test_streaming_vae_chunks_match_full_causal_decode():
    vae = _TinyDecoder()
    decoder = BatchedStreamingVAEDecoder(vae, CausalConv1d, CausalTransposeConv1d)
    latents = torch.arange(1, 6, dtype=torch.float32).reshape(1, 1, 5)
    expected = vae.decode(latents)
    actual = torch.cat(
        (
            decoder.decode_chunks(latents[..., :2], ["request"]),
            decoder.decode_chunks(latents[..., 2:], ["request"]),
        ),
        -1,
    )
    torch.testing.assert_close(actual, expected)


def test_runner_reset_releases_request_and_decoder_state():
    released = []
    model_resets = []
    runner = object.__new__(VoxCPMModelRunner)
    runner._states = {"request": SimpleNamespace(row=0)}
    runner._decoder = SimpleNamespace(release=released.append)
    runner.model = SimpleNamespace(reset_request=model_resets.append)

    runner.reset_request("request")
    runner.reset_request("request")

    assert runner._states == {}
    assert released == ["request", "request"]
    assert model_resets == ["request", "request"]


def test_runner_passes_mixed_cfm_step_counts_without_global_mutation():
    captured = {}

    class _Model:
        def generate_batch(self, hidden, rows, **kwargs):
            captured.update(kwargs)
            return {
                "latents": hidden.new_zeros((2, 1, 1)),
                "stop_flag": torch.zeros(2, dtype=torch.long),
            }

    runner = object.__new__(VoxCPMModelRunner)
    runner.model = _Model()
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                state=SimpleNamespace(generation_kwargs={"inference_timesteps": steps})
            )
        )
        for steps in (2, 5)
    ]
    output = runner._generate_grouped(
        hidden=torch.ones(2, 3),
        rows=torch.tensor([0, 1]),
        temperatures=torch.ones(2),
        cfg_values=torch.ones(2),
        noise=torch.ones(2, 1, 1),
        requests=requests,
    )

    assert output["latents"].shape == (2, 1, 1)
    assert captured["inference_timesteps"].tolist() == [2, 5]
