# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sglang_omni.client.client import Client
from sglang_omni.client.types import CompletionStreamChunk
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.llada2_uni import stages as llada2_stages
from sglang_omni.models.llada2_uni.algorithm.low_confidence_cfg import (
    _get_num_transfer_tokens,
    _slice_cfg_output_ids,
)
from sglang_omni.models.llada2_uni.components import preprocessor as preprocessor_module
from sglang_omni.models.llada2_uni.components.preprocessor import LLaDA2Preprocessor
from sglang_omni.models.llada2_uni.config import LLaDA2UniOmniPipelineConfig
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.models.llada2_uni.request_builders import _thinking_phase1_to_phase2
from sglang_omni.pipeline.coordinator import _compute_timings_ms
from sglang_omni.proto import CompleteMessage, OmniRequest, StagePayload
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import (
    _build_chat_generate_request,
    _is_bad_request_error,
)
from sglang_omni.serve.protocol import ChatCompletionRequest, ImageGenerationParams


def test_thinker_rejects_attention_backend_without_cfg_padding() -> None:
    with pytest.raises(
        ValueError,
        match="LLaDA2-Uni CFG requires attention_backend='flashinfer'",
    ):
        llada2_stages.create_sglang_dllm_thinker_executor_from_config(
            "unused-model",
            server_args_overrides={"attention_backend": "triton"},
        )


def test_example_config_loads_omni_pipeline() -> None:
    config = ConfigManager.from_file("examples/configs/llada2_uni.yaml").config

    assert isinstance(config, LLaDA2UniOmniPipelineConfig)


def test_image_decoder_resolves_hf_model_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from sglang_omni.models.llada2_uni.components import (
        image_decoder as image_decoder_module,
    )

    resolved_path = tmp_path / "model-snapshot"
    resolved_path.mkdir()
    resolved_model_ids: list[str] = []

    def fake_resolve_model_path(model_path: str):
        resolved_model_ids.append(model_path)
        return resolved_path

    monkeypatch.setattr(
        image_decoder_module,
        "resolve_model_path",
        fake_resolve_model_path,
    )

    decoder = image_decoder_module.LLaDA2ImageDecoder(
        "inclusionAI/LLaDA2.0-Uni",
        device="cpu",
    )

    assert resolved_model_ids == ["inclusionAI/LLaDA2.0-Uni"]
    assert decoder.model_path == str(resolved_path)


def test_image_decoder_accepts_existing_local_model_path(tmp_path) -> None:
    from sglang_omni.models.llada2_uni.components.image_decoder import (
        LLaDA2ImageDecoder,
    )

    decoder = LLaDA2ImageDecoder(str(tmp_path), device="cpu")

    assert decoder.model_path == str(tmp_path)


def test_image_decoder_resolves_unindexed_cuda_to_current_stage_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from sglang_omni.models.llada2_uni.components.image_decoder import (
        LLaDA2ImageDecoder,
    )

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    decoder = LLaDA2ImageDecoder(str(tmp_path), device="cuda")

    assert decoder.device == torch.device("cuda:3")


def test_sglang_image_decoder_defaults_to_flash_attention(tmp_path) -> None:
    from sglang_omni.models.llada2_uni.components.image_decoder import (
        LLaDA2ImageDecoder,
    )

    decoder = LLaDA2ImageDecoder(
        str(tmp_path),
        device="cpu",
        backend="sglang",
        attention_backend=None,
    )

    assert decoder.attention_backend == "fa"


@pytest.mark.parametrize(
    ("token_ids", "h", "w", "error"),
    [
        ([1, 2, 3], 2, 2, r"exactly h \* w"),
        ([1, 2, 3, 16384], 2, 2, "between 0 and 16383"),
        ([1], 0, 1, "positive"),
    ],
)
def test_image_decoder_rejects_invalid_vq_inputs_before_model_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    token_ids: list[int],
    h: int,
    w: int,
    error: str,
) -> None:
    from sglang_omni.models.llada2_uni.components.image_decoder import (
        LLaDA2ImageDecoder,
    )

    decoder = LLaDA2ImageDecoder(str(tmp_path), device="cpu")
    monkeypatch.setattr(
        decoder,
        "_ensure_diff_model",
        lambda mode: pytest.fail("invalid input reached model loading"),
    )

    with pytest.raises(ValueError, match=error):
        decoder.decode(token_ids, h, w)


def test_sp_image_decoder_broadcasts_leader_conditioning_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from sglang_omni.models.llada2_uni.components.image_decoder import (
        LLaDA2ImageDecoder,
    )

    decoder = LLaDA2ImageDecoder(
        str(tmp_path),
        device="cpu",
        backend="sglang",
        stage_role="leader",
        sp_size=2,
        ulysses_degree=2,
    )
    monkeypatch.setattr(decoder, "_ensure_diff_model", lambda mode: None)
    monkeypatch.setattr(
        decoder,
        "_ensure_sigvq",
        lambda: (_ for _ in ()).throw(RuntimeError("bad SigVQ checkpoint")),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    statuses: list[int] = []

    def record_broadcast(tensor: torch.Tensor, src: int) -> None:
        assert src == 0
        statuses.append(int(tensor.item()))

    monkeypatch.setattr(torch.distributed, "broadcast", record_broadcast)

    with pytest.raises(RuntimeError, match="bad SigVQ checkpoint"):
        decoder.decode([1], 1, 1)

    assert statuses == [0]


def test_sp_image_decoder_follower_stops_after_leader_conditioning_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from sglang_omni.models.llada2_uni.components.image_decoder import (
        LLaDA2ImageDecoder,
    )

    decoder = LLaDA2ImageDecoder(
        str(tmp_path),
        device="cpu",
        backend="sglang",
        stage_role="follower",
        sp_rank=1,
        sp_size=2,
        ulysses_degree=2,
    )
    monkeypatch.setattr(decoder, "_ensure_diff_model", lambda mode: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    broadcast_shapes: list[tuple[int, ...]] = []

    def fail_status(tensor: torch.Tensor, src: int) -> None:
        assert src == 0
        broadcast_shapes.append(tuple(tensor.shape))
        tensor.zero_()

    monkeypatch.setattr(torch.distributed, "broadcast", fail_status)

    assert decoder.decode([1], 1, 1) is None
    assert broadcast_shapes == [(1,)]


def test_image_decoder_bytes_api_rejects_parallel_follower(tmp_path) -> None:
    from sglang_omni.models.llada2_uni.components.image_decoder import (
        LLaDA2ImageDecoder,
    )

    decoder = LLaDA2ImageDecoder(
        str(tmp_path),
        device="cpu",
        backend="sglang",
        stage_role="follower",
        sp_rank=1,
        sp_size=2,
        ulysses_degree=2,
    )

    with pytest.raises(RuntimeError, match="leader-only"):
        decoder.decode_to_bytes([1], 1, 1)


def test_sglang_decoder_uses_stage_assigned_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.llada2_uni.components.decoder_model import (
        _configure_sglang_device_environment,
    )

    monkeypatch.setenv("LOCAL_RANK", "0")

    base_gpu_id = _configure_sglang_device_environment(torch.device("cuda:3"))

    assert base_gpu_id == 3
    assert os.environ["LOCAL_RANK"] == "3"


def test_sglang_decoder_normalizes_singleton_batch_latent() -> None:
    from sglang_omni.models.llada2_uni.components.decoder_model import (
        _SGLangZImageModelAdapter,
    )

    latent = torch.empty(1, 16, 1, 64, 64)

    normalized = _SGLangZImageModelAdapter._normalize_image_latent(latent)

    assert normalized.shape == (16, 1, 64, 64)


def test_sglang_decoder_context_refiner_skips_sequence_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.multimodal_gen.runtime.distributed import communication_op

    from sglang_omni.models.llada2_uni.components.decoder_model import (
        ZImageParallelConfig,
        _SGLangZImageModelAdapter,
    )

    skip_sp_overrides: list[bool] = []

    class PassthroughLayer(torch.nn.Module):
        def forward(self, hidden_states, *args, **kwargs):
            del args, kwargs
            return hidden_states

    class ContextRefinerLayer(torch.nn.Module):
        def forward(
            self,
            hidden_states,
            freqs_cis,
            *,
            skip_sequence_parallel_override=False,
        ):
            del freqs_cis
            skip_sp_overrides.append(skip_sequence_parallel_override)
            return hidden_states

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.noise_refiner = torch.nn.ModuleList([PassthroughLayer()])
            self.context_refiner = torch.nn.ModuleList([ContextRefinerLayer()])
            self.layers = torch.nn.ModuleList([PassthroughLayer()])
            self.all_x_embedder = {"2-1": lambda hidden: (hidden, None)}
            self.all_final_layer = {
                "2-1": lambda hidden, adaln_input: hidden,
            }
            self.x_pad_token = torch.empty(0)
            self.cap_pad_token = torch.empty(0)

        @staticmethod
        def t_embedder(timestep):
            return timestep[:, None]

        @staticmethod
        def patchify_and_embed(images, cap_feats, patch_size, f_patch_size):
            del images, patch_size, f_patch_size
            return (
                torch.zeros(1, 2, 1),
                cap_feats[0].unsqueeze(0),
                [(1, 1, 1)],
                [2],
                [cap_feats[0].shape[0]],
            )

        @staticmethod
        def _replace_padding_with_token(hidden, valid_lens, pad_token):
            del valid_lens, pad_token
            return hidden

        @staticmethod
        def cap_embedder(hidden):
            return hidden, None

        @staticmethod
        def unpatchify(outputs, x_size, patch_size, f_patch_size):
            del x_size, patch_size, f_patch_size
            return outputs

    adapter = _SGLangZImageModelAdapter(
        Model(),
        ZImageParallelConfig(backend="sglang", sp_size=2, ulysses_degree=2),
    )
    freqs = (torch.zeros(2, 1), torch.zeros(2, 1))
    monkeypatch.setattr(adapter, "_get_freqs_cis", lambda *args: (freqs, freqs))
    monkeypatch.setattr(
        adapter,
        "_shard_sequence_for_sp",
        lambda hidden, hidden_freqs: (hidden, hidden_freqs),
    )
    monkeypatch.setattr(
        communication_op,
        "sequence_model_parallel_all_gather",
        lambda hidden, dim: hidden,
    )

    adapter._run_model_sp(
        image=torch.zeros(1, 1, 1, 1),
        cap_feat=torch.zeros(2, 1),
        timestep=torch.tensor([500.0]),
        patch_size=2,
        f_patch_size=1,
    )

    assert skip_sp_overrides == [True]


def test_sglang_decoder_maps_runtime_precision_from_dtype() -> None:
    from sglang_omni.models.llada2_uni.components.decoder_model import (
        _torch_dtype_to_sglang_precision,
    )

    assert _torch_dtype_to_sglang_precision(torch.bfloat16) == "bf16"
    assert _torch_dtype_to_sglang_precision(torch.float16) == "fp16"
    assert _torch_dtype_to_sglang_precision(torch.float32) == "fp32"


def test_sglang_decoder_return_dict_is_output_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    from sglang_omni.models.llada2_uni.components.decoder_model import (
        ZImageParallelConfig,
        _SGLangZImageModelAdapter,
    )

    class Model(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            del kwargs
            return torch.zeros_like(hidden_states[0]).unsqueeze(0)

    adapter = _SGLangZImageModelAdapter(Model(), ZImageParallelConfig(backend="sglang"))
    monkeypatch.setattr(adapter, "_sp_world_size", lambda: 1)
    monkeypatch.setattr(
        adapter,
        "_get_freqs_cis",
        lambda *args: (torch.empty(0), torch.empty(0)),
    )

    output = adapter(
        x=[torch.zeros(16, 1, 2, 2)],
        t=torch.tensor([0.5]),
        cap_feats=[torch.zeros(1, 4096)],
        return_dict=True,
    )

    assert isinstance(output, Transformer2DModelOutput)
    assert len(output.sample) == 1


def test_sglang_decoder_does_not_use_batch_index_as_timestep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.multimodal_gen.runtime.managers import forward_context

    from sglang_omni.models.llada2_uni.components.decoder_model import (
        ZImageParallelConfig,
        _SGLangZImageModelAdapter,
    )

    class Model(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            del kwargs
            return torch.zeros_like(hidden_states[0]).unsqueeze(0)

    observed_timesteps: list[int] = []

    @contextmanager
    def record_forward_context(*, current_timestep, **kwargs):
        del kwargs
        observed_timesteps.append(current_timestep)
        yield

    adapter = _SGLangZImageModelAdapter(Model(), ZImageParallelConfig(backend="sglang"))
    monkeypatch.setattr(adapter, "_sp_world_size", lambda: 1)
    monkeypatch.setattr(
        adapter,
        "_get_freqs_cis",
        lambda *args: (torch.empty(0), torch.empty(0)),
    )
    monkeypatch.setattr(forward_context, "set_forward_context", record_forward_context)

    adapter(
        x=[torch.zeros(16, 1, 2, 2), torch.zeros(16, 1, 2, 2)],
        t=torch.tensor([0.5, 0.5]),
        cap_feats=[torch.zeros(1, 4096), torch.zeros(1, 4096)],
        return_dict=False,
    )

    assert observed_timesteps == [0, 0]


def test_image_generation_params_validate_decoder_inputs() -> None:
    params = ImageGenerationParams(
        mode="thinking",
        decode_mode="decoder-turbo",
        decoder_steps=8,
        cfg_scale=4.0,
        cfg_rescale=0.7,
        image_h=1024,
        image_w=768,
        dllm_steps=16,
    )

    dumped = params.model_dump(exclude_none=True)
    assert dumped["dllm_steps"] == 16
    with pytest.raises(ValidationError):
        ImageGenerationParams(image_h=1000)
    with pytest.raises(ValidationError):
        ImageGenerationParams(dllm_steps=0)


def test_image_generation_params_preserve_edit_cfg_scales() -> None:
    params = ImageGenerationParams(cfg_text_scale=0.0, cfg_image_scale=2.0)

    dumped = params.model_dump(exclude_none=True)
    assert dumped["cfg_text_scale"] == 0.0
    assert dumped["cfg_image_scale"] == 2.0

    with pytest.raises(ValidationError):
        ImageGenerationParams(cfg_text_scale=-0.1)


def test_image_generation_metadata_preserves_only_explicit_parameters() -> None:
    default_request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "edit"}],
        image_generation={},
    )
    legacy_request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "edit"}],
        image_generation={"cfg_scale": 1.0},
    )

    default_metadata = _build_chat_generate_request(default_request).metadata
    legacy_metadata = _build_chat_generate_request(legacy_request).metadata

    assert default_metadata["image_generation"] == {}
    assert legacy_metadata["image_generation"] == {"cfg_scale": 1.0}


def test_default_image_generation_metadata_dispatches_to_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ChatCompletionRequest(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "image.png"}},
                    {"type": "text", "text": "make it brighter"},
                ],
            }
        ],
        image_generation={},
    )
    generate_request = _build_chat_generate_request(request)
    payload = StagePayload(
        request_id="edit-default",
        request=OmniRequest(
            inputs=[message.model_dump() for message in request.messages],
            metadata=generate_request.metadata,
        ),
        data={},
    )
    preprocessor = object.__new__(LLaDA2Preprocessor)
    observed: dict = {}
    image_sentinel = object()

    async def fake_ensure_image_list(raw_images):
        observed["raw_images"] = raw_images
        return [image_sentinel]

    def fake_build_edit_payload(payload, messages, images, request_metadata):
        observed["images"] = images
        observed["image_generation"] = request_metadata["image_generation"]
        return "edit-payload"

    monkeypatch.setattr(
        preprocessor_module,
        "ensure_image_list_async",
        fake_ensure_image_list,
    )
    preprocessor._build_edit_payload = fake_build_edit_payload

    assert asyncio.run(preprocessor(payload)) == "edit-payload"
    assert observed["raw_images"] == ["image.png"]
    assert observed["images"] == [image_sentinel]
    assert observed["image_generation"] == {}


@pytest.mark.parametrize(
    ("image_generation", "expected"),
    [
        ({}, (4.0, 0.0)),
        ({"cfg_scale": 1.0}, (0.0, 0.0)),
        ({"cfg_scale": 3.0}, (3.0, 0.0)),
        ({"cfg_text_scale": 0.0, "cfg_image_scale": 2.0}, (0.0, 2.0)),
        ({"cfg_text_scale": 4.0, "cfg_image_scale": 2.0}, (4.0, 2.0)),
    ],
)
def test_edit_cfg_scale_resolution(
    image_generation: dict[str, float],
    expected: tuple[float, float],
) -> None:
    assert preprocessor_module._resolve_edit_cfg_scales(image_generation) == expected


@pytest.mark.parametrize(
    ("image_generation", "expected_text_scale", "has_image_branch"),
    [
        ({}, 4.0, False),
        ({"cfg_text_scale": 4.0}, 4.0, False),
        ({"cfg_text_scale": 0.0, "cfg_image_scale": 2.0}, 0.0, True),
        ({"cfg_text_scale": 4.0, "cfg_image_scale": 2.0}, 4.0, True),
    ],
)
def test_edit_cfg_builds_required_companion_branches(
    image_generation: dict[str, float],
    expected_text_scale: float,
    has_image_branch: bool,
) -> None:
    preprocessor = object.__new__(LLaDA2Preprocessor)
    preprocessor._build_edit_uncond_input_ids = lambda *args: [7, 8]
    preprocessor._build_edit_no_img_input_ids = lambda *args: [6]
    stream_state: dict = {}

    preprocessor._populate_edit_cfg_stream_state(
        stream_state=stream_state,
        image_generation=image_generation,
        instruction_text="edit",
        src_image_block="<image>",
        grid_h=2,
        grid_w=2,
        num_image_tokens=4,
    )

    assert stream_state["uncond_input_ids"] == [7, 8]
    assert "uncond_left_pad_len" not in stream_state
    assert "uncond_img_left_pad_len" not in stream_state
    assert stream_state["cfg_scale"] == expected_text_scale
    assert ("uncond_img_input_ids" in stream_state) is has_image_branch
    assert ("cfg_image_scale" in stream_state) is has_image_branch


@pytest.mark.parametrize(
    "image_generation",
    [
        {"cfg_text_scale": 0.0, "cfg_image_scale": 0.0},
        {"cfg_scale": 1.0},
    ],
)
def test_edit_cfg_skips_companions_when_disabled(
    image_generation: dict[str, float],
) -> None:
    preprocessor = object.__new__(LLaDA2Preprocessor)
    stream_state: dict = {}

    preprocessor._populate_edit_cfg_stream_state(
        stream_state=stream_state,
        image_generation=image_generation,
        instruction_text="edit",
        src_image_block="<image>",
        grid_h=2,
        grid_w=2,
        num_image_tokens=4,
    )

    assert stream_state == {}


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user"}],
        [{"role": "user", "content": ""}],
        [{"role": "user", "content": " \t\n"}],
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "image.png"}},
                    {"type": "text", "text": "  "},
                ],
            }
        ],
    ],
)
def test_edit_rejects_empty_instruction_before_image_processing(
    messages: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ChatCompletionRequest(
        messages=messages,
        image_generation={},
    )
    generate_request = _build_chat_generate_request(request)
    payload = StagePayload(
        request_id="edit-empty",
        request=OmniRequest(
            inputs={"messages": messages, "images": ["image.png"]},
            metadata=generate_request.metadata,
        ),
        data={},
    )
    preprocessor = object.__new__(LLaDA2Preprocessor)

    async def fail_if_image_is_loaded(raw_images):
        pytest.fail("empty edit instructions must be rejected before image loading")

    monkeypatch.setattr(
        preprocessor_module,
        "ensure_image_list_async",
        fail_if_image_is_loaded,
    )

    with pytest.raises(ValueError, match="non-empty instruction"):
        asyncio.run(preprocessor(payload))

    assert _is_bad_request_error(
        ValueError("Image editing requires a non-empty instruction")
    )


@pytest.mark.parametrize(
    ("inputs", "metadata"),
    [
        (
            [{"role": "user", "content": "Draw two frames."}],
            {"interleaved_generation": {}, "audios": ["audio.wav"]},
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {"url": "video.mp4"},
                            },
                            {"type": "text", "text": "Draw two frames."},
                        ],
                    }
                ]
            },
            {"interleaved_generation": {}},
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": 123}],
                    }
                ]
            },
            {"interleaved_generation": {}},
        ),
    ],
)
def test_interleaved_preprocessor_rejects_non_text_input(
    inputs: object,
    metadata: dict,
) -> None:
    payload = StagePayload(
        request_id="interleaved-non-text",
        request=OmniRequest(
            inputs=inputs,
            metadata=metadata,
        ),
        data={},
    )
    preprocessor = object.__new__(LLaDA2Preprocessor)

    with pytest.raises(ValueError, match="text-only input"):
        asyncio.run(preprocessor(payload))


@pytest.mark.parametrize("stream", [False, True])
def test_chat_endpoint_rejects_empty_edit_instruction_with_400(stream: bool) -> None:
    class UnexpectedCompletionClient:
        async def completion(self, *args, **kwargs):
            pytest.fail("invalid edit request reached the completion client")

        async def completion_stream(self, *args, **kwargs):
            pytest.fail("invalid edit request reached the streaming client")
            yield

    client = TestClient(
        create_app(UnexpectedCompletionClient(), model_name="llada2-uni")
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "image.png"},
                        }
                    ],
                }
            ],
            "image_generation": {},
            "stream": stream,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Image editing requires a non-empty instruction"
    )


@pytest.mark.parametrize(
    "image_fields",
    [
        {"image_generation": {}},
        {"modalities": ["image"]},
    ],
)
def test_chat_endpoint_rejects_streaming_image_generation_with_400(
    image_fields: dict,
) -> None:
    streaming_calls: list[bool] = []

    class EmptyCompletionClient:
        async def completion_stream(self, *args, **kwargs):
            streaming_calls.append(True)
            if False:
                yield

    client = TestClient(create_app(EmptyCompletionClient(), model_name="llada2-uni"))
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Draw a cat."}],
            "stream": True,
            **image_fields,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Image generation does not support streaming; set stream=false"
    )
    assert streaming_calls == []


def test_chat_endpoint_allows_streaming_text_for_image_input() -> None:
    streaming_calls: list[bool] = []

    class TextCompletionClient:
        async def completion_stream(self, *args, **kwargs):
            streaming_calls.append(True)
            yield CompletionStreamChunk(
                request_id="image-understanding",
                text="A cat.",
                modality="text",
                finish_reason="stop",
            )

    client = TestClient(create_app(TextCompletionClient(), model_name="llada2-uni"))
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "image.png"},
                        },
                        {"type": "text", "text": "Describe the image."},
                    ],
                }
            ],
            "modalities": ["text"],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    assert events[0]["choices"][0]["delta"]["content"] == "A cat."
    assert "data: [DONE]\n\n" in response.text
    assert streaming_calls == [True]


def test_image_result_preserves_main_fields_and_timings() -> None:
    chunk = Client._default_result_builder(
        "req-image",
        {
            "decode": {
                "text": "reasoning",
                "finish_reason": "stop",
                "output_token_logprobs": [{"token": 7}],
                "omni_rollout": {"trace": "ok"},
                "weight_version": "weights-v2",
            },
            "image_decode": {"image": "base64-png"},
            "timings": {"thinker_ms": 12.5, "e2e_ms": 18.0},
        },
    )

    assert chunk.text == "reasoning"
    assert chunk.image == "base64-png"
    assert chunk.modality == "image"
    assert chunk.output_token_logprobs == [{"token": 7}]
    assert chunk.omni_rollout == {"trace": "ok"}
    assert chunk.weight_version == "weights-v2"
    assert chunk.timings == {"thinker_ms": 12.5, "e2e_ms": 18.0}


def test_timing_reentry_and_protocol_round_trip() -> None:
    raw_timing = {
        "thinker.enter": 1_000_000,
        "thinker.done": 4_500_000,
        "thinker.enter.2": 5_000_000,
        "thinker.done.2": 11_250_000,
    }

    assert _compute_timings_ms(raw_timing) == {
        "thinker_ms": 3.5,
        "thinker_pass2_ms": 6.25,
        "e2e_ms": 10.25,
    }

    request = OmniRequest(inputs="prompt", params={}, metadata={})
    payload = StagePayload(
        request_id="req-timing",
        request=request,
        data={"value": 1},
        timing=raw_timing,
    )
    restored = StagePayload.from_dict(payload.to_dict())
    assert restored.timing == raw_timing

    complete = CompleteMessage(
        request_id="req-timing",
        from_stage="thinker",
        success=True,
        result={"text": "ok"},
        timing=raw_timing,
    )
    assert CompleteMessage.from_dict(complete.to_dict()).timing == raw_timing


def test_transfer_schedule_clamps_steps_to_block_length() -> None:
    schedule = _get_num_transfer_tokens(block_length=4, steps=100)

    assert schedule.tolist() == [1, 1, 1, 1]


def test_cfg_output_slice_ignores_unconditional_mask_padding() -> None:
    # The uncond branch has one mask pad at the front. Counting all mask tokens
    # would produce start=1 and leak token 20 from the prompt into output_ids.
    denoised_ids = torch.tensor(
        [
            [10, 11, 30, 31],
            [99, 20, 30, 31],
        ]
    )

    outputs = _slice_cfg_output_ids(
        denoised_ids,
        start_list=[2, 1],
        cond_idx=0,
        is_dllm_prefill=False,
    )

    assert outputs[0].tolist() == [30, 31]
    assert outputs[1].tolist() == [30, 31]
    assert all(
        output.numel() == 0
        for output in _slice_cfg_output_ids(
            denoised_ids, [2, 1], 0, is_dllm_prefill=True
        )
    )


def test_thinking_phase_requires_model_generated_boi() -> None:
    tokenizer = SimpleNamespace(
        boi_token_id=99,
        convert_tokens_to_ids=lambda token: 99,
    )
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[10, 11]])},
        thinker_out={"output_ids": [20, 21], "is_final": True},
        stream_state={
            "thinking_mode": True,
            "thinking_phase": 1,
            "image_info": [{"grid_h": 8, "grid_w": 8}],
        },
        task_kind="t2i",
    )

    with pytest.raises(RuntimeError, match=r"did not produce <boi>.*2 token"):
        _thinking_phase1_to_phase2(state, tokenizer)
