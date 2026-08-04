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
    cap_generation_length,
    validate_prompt_length,
)
from sglang_omni.models.voxcpm_common.model_runner import (
    VoxCPMModelRunner,
    _RequestRuntime,
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


def test_generation_length_uses_official_text_relative_bound():
    assert cap_generation_length(20, 2048) == 130
    assert cap_generation_length(20, 64) == 64
    with pytest.raises(ValueError, match="target_token_count"):
        cap_generation_length(0, 64)


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


def test_async_lookahead_routes_done_rows_to_padding_and_trims_input():
    runner = object.__new__(VoxCPMModelRunner)
    runner._states = {
        "done": SimpleNamespace(row=0),
        "active": SimpleNamespace(row=1),
    }
    pool = SimpleNamespace(
        done=torch.tensor([True, False, False]),
        padding_row=2,
    )
    runner.model = SimpleNamespace(model=SimpleNamespace(state_pool=pool))
    forward_batch = SimpleNamespace(input_ids=torch.tensor([9, 9, 9, 9]))
    requests = [
        SimpleNamespace(request_id="done"),
        SimpleNamespace(request_id="active"),
    ]

    runner.before_decode(
        forward_batch,
        schedule_batch=None,
        requests=requests,
        is_lookahead=True,
    )

    assert forward_batch.input_ids.tolist() == [2, 1]
    assert forward_batch.voxcpm_rows.tolist() == [2, 1]


def test_runner_reuses_packed_collect_staging():
    runner = object.__new__(VoxCPMModelRunner)
    runner.model = SimpleNamespace(
        model=SimpleNamespace(state_pool=SimpleNamespace(capacity=4))
    )
    runner._states = {
        rid: SimpleNamespace(decoder_initialized=False) for rid in ("first", "second")
    }
    runner._collect_device_staging = None
    runner._decoder = SimpleNamespace(
        decode_chunks=lambda latents, request_ids, contexts: torch.tensor(
            [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]
        )
    )
    requests = [
        SimpleNamespace(
            request_id=rid,
            data=SimpleNamespace(initial_decode_context=None),
        )
        for rid in ("first", "second")
    ]

    first, latent_shape, audio_width = runner._decode_pack_gpu(
        latents=torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
        stop_flags=torch.tensor([0, 1]),
        requests=requests,
    )
    staging_ptr = runner._collect_device_staging.data_ptr()
    second, _, _ = runner._decode_pack_gpu(
        latents=torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]]),
        stop_flags=torch.tensor([1, 0]),
        requests=requests,
    )

    assert latent_shape == (1, 2)
    assert audio_width == 3
    assert runner._collect_device_staging.data_ptr() == staging_ptr
    assert first.data_ptr() == second.data_ptr()
    assert second.tolist() == [
        [5.0, 6.0, 10.0, 11.0, 12.0, 1.0],
        [7.0, 8.0, 20.0, 21.0, 22.0, 0.0],
    ]


def test_runner_buffers_three_vae_patches_and_flushes_on_stop():
    decode_inputs = []

    def decode_chunks(latents, request_ids, contexts):
        decode_inputs.append((latents.clone(), request_ids, contexts))
        return latents.squeeze(1)

    runner = object.__new__(VoxCPMModelRunner)
    runner.model = SimpleNamespace(
        model=SimpleNamespace(state_pool=SimpleNamespace(capacity=4))
    )
    runner._states = {"request": _RequestRuntime(row=0)}
    runner._latent_collect_staging = None
    runner._vae_decode_every = 3
    runner._decoder = SimpleNamespace(decode_chunks=decode_chunks)
    data = SimpleNamespace(
        initial_decode_context=None,
        state=SimpleNamespace(
            generation_kwargs={
                "min_len": 0,
                "max_new_tokens": 10,
                "streaming_prefix_len": 1,
            }
        ),
        generated_latents=[],
        generated_audio=[],
        stop_step=None,
        finish_kind=None,
        is_streaming=False,
    )
    request = SimpleNamespace(request_id="request", data=data)

    for step in range(1, 4):
        runner._decode_and_collect_buffered(
            latents=torch.tensor([[[float(step)]]]),
            stop_flags=torch.tensor([0]),
            step_counts=[step],
            requests=[request],
        )

    assert len(decode_inputs) == 1
    assert decode_inputs[0][0].tolist() == [[[1.0, 2.0, 3.0]]]
    assert len(data.generated_latents) == 3
    assert data.generated_audio[0].tolist() == [1.0, 2.0, 3.0]

    runner._decode_and_collect_buffered(
        latents=torch.tensor([[[4.0]]]),
        stop_flags=torch.tensor([1]),
        step_counts=[4],
        requests=[request],
    )

    assert len(decode_inputs) == 2
    assert decode_inputs[1][0].tolist() == [[[4.0]]]
    assert data.generated_audio[1].tolist() == [4.0]
    assert data.stop_step == 3
    assert data.finish_kind == "stop"


def test_async_lookahead_rejects_nondefault_cfm_steps():
    runner = object.__new__(VoxCPMModelRunner)
    runner._vae_decode_every = 1
    runner.model = SimpleNamespace(
        diffusion_graph_max_bs=8,
        model=SimpleNamespace(feat_decoder=SimpleNamespace(inference_timesteps=10)),
    )
    data = SimpleNamespace(
        state=SimpleNamespace(generation_kwargs={"inference_timesteps": 4})
    )
    batch = SimpleNamespace(reqs=[SimpleNamespace(_omni_data=data)])

    assert runner.lookahead_eligible(batch) is False


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


def test_runner_uses_diffusion_graph_for_default_step_count():
    captured = {}

    class _Model:
        model = SimpleNamespace(feat_decoder=SimpleNamespace(inference_timesteps=10))
        diffusion_graph_max_bs = 8

        def generate_batch_graphed(self, hidden, rows, **kwargs):
            captured.update(kwargs)
            return {
                "latents": hidden.new_zeros((2, 1, 1)),
                "stop_flag": torch.zeros(2, dtype=torch.long),
            }

        def generate_batch(self, *_args, **_kwargs):
            raise AssertionError("default step count should use the diffusion graph")

    runner = object.__new__(VoxCPMModelRunner)
    runner.model = _Model()
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                state=SimpleNamespace(generation_kwargs={"inference_timesteps": 10})
            )
        )
        for _ in range(2)
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
    assert captured["z_noise"].shape == (2, 1, 1)
