# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import threading
from queue import Queue
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from sglang_omni.client.client import Client
from sglang_omni.client.types import CompletionStreamChunk
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.llada2_uni import stages as llada2_stages
from sglang_omni.models.llada2_uni.algorithm.low_confidence_cfg import (
    LowConfidenceCFG,
    _get_num_transfer_tokens,
    _slice_cfg_output_ids,
)
from sglang_omni.models.llada2_uni.components import preprocessor as preprocessor_module
from sglang_omni.models.llada2_uni.components.preprocessor import (
    LLaDA2Preprocessor,
    align_cfg_unconditional_input_ids,
)
from sglang_omni.models.llada2_uni.config import LLaDA2UniOmniPipelineConfig
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.models.llada2_uni.request_builders import _thinking_phase1_to_phase2
from sglang_omni.pipeline.coordinator import _compute_timings_ms
from sglang_omni.proto import CompleteMessage, OmniRequest, StagePayload
from sglang_omni.scheduling import dllm_scheduler as dllm_scheduler_module
from sglang_omni.scheduling.dllm_scheduler import DllmScheduler
from sglang_omni.scheduling.messages import IncomingMessage
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
    preprocessor._tokenizer = SimpleNamespace(mask_token_id=99)
    preprocessor._build_edit_uncond_input_ids = lambda *args: [7, 8]
    preprocessor._build_edit_no_img_input_ids = lambda *args: [6]
    stream_state: dict = {}

    preprocessor._populate_edit_cfg_stream_state(
        stream_state=stream_state,
        image_generation=image_generation,
        conditional_input_ids=[10, 11, 12],
        instruction_text="edit",
        src_image_block="<image>",
        grid_h=2,
        grid_w=2,
        num_image_tokens=4,
    )

    assert stream_state["uncond_input_ids"] == [99, 7, 8]
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
        conditional_input_ids=[10, 11, 12],
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


def test_cfg_uncond_prompt_is_aligned_before_scheduling() -> None:
    scheduler = object.__new__(DllmScheduler)
    scheduler._waiting_queue = []
    scheduler._cond_to_unconds = {}
    scheduler._uncond_to_cond = {}
    scheduler._uncond_rids = set()
    tokenizer = SimpleNamespace(mask_token_id=99)
    cond = SimpleNamespace(
        rid="cond",
        origin_input_ids=[1, 2, 3, 4],
        sampling_params=SimpleNamespace(max_new_tokens=32),
        vocab_size=100,
        eos_token_ids={9},
        dllm_config=None,
    )

    uncond_ids, left_pad_len = align_cfg_unconditional_input_ids(
        tokenizer, cond.origin_input_ids, [7, 8]
    )
    cond._uncond_left_pad_len = left_pad_len
    scheduler._create_uncond_companion(
        cond,
        uncond_ids,
        left_pad_len,
        "-uncond",
        mark_img=False,
    )

    companion = scheduler._waiting_queue[0]
    assert companion.origin_input_ids == [99, 99, 7, 8]
    assert companion._dllm_left_pad_len == 2

    with pytest.raises(ValueError, match="physically aligned"):
        scheduler._create_uncond_companion(
            cond,
            [7, 8],
            left_pad_len,
            "-invalid",
            mark_img=False,
        )
    with pytest.raises(ValueError, match="cannot be longer"):
        align_cfg_unconditional_input_ids(
            tokenizer, cond.origin_input_ids, [5, 6, 7, 8, 9]
        )


def test_cfg_uncond_positions_match_official_left_padding() -> None:
    scheduler = object.__new__(DllmScheduler)

    def apply_padding(pad_len: int, block_offset: int) -> list[int]:
        positions = torch.cat(
            [
                torch.arange(block_offset, block_offset + 32),
                torch.arange(block_offset, block_offset + 32),
            ]
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DLLM_EXTEND,
            extend_seq_lens_cpu=[32, 32],
            positions=positions,
            seq_lens=torch.tensor([32, 32]),
        )
        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(_dllm_left_pad_len=0),
                SimpleNamespace(_dllm_left_pad_len=pad_len),
            ]
        )

        scheduler._apply_cfg_padding_metadata(forward_batch, batch)
        assert forward_batch.dllm_left_pad_lens.tolist() == [0, pad_len]
        return forward_batch.positions[32:].tolist()

    assert apply_padding(pad_len=2, block_offset=0) == [0, 0, *range(30)]
    assert apply_padding(pad_len=2, block_offset=32) == [*range(30, 62)]
    assert apply_padding(pad_len=40, block_offset=32) == [*[0] * 9, *range(1, 24)]


def test_cfg_phases_follow_conditional_request() -> None:
    scheduler = object.__new__(DllmScheduler)
    scheduler._cond_to_unconds = {"cond": ["cond-uncond"]}
    cond = SimpleNamespace(
        rid="cond",
        dllm_phase="staging_prefill",
        _is_uncond=False,
    )
    uncond = SimpleNamespace(
        rid="cond-uncond",
        dllm_phase="staging_decode",
        _is_uncond=True,
    )

    scheduler._synchronize_cfg_phases([cond, uncond])

    assert uncond.dllm_phase == cond.dllm_phase


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


class _FakeReq:
    def __init__(self, rid: str, *, finishes_on_check: bool = False):
        self.rid = rid
        self.output_ids: list[int] = []
        self.finished_reason = SimpleNamespace(to_json=lambda: {"type": "length"})
        self.req_pool_idx = None
        self._finished = False
        self._finishes_on_check = finishes_on_check

    @property
    def output_ids_through_stop(self) -> list[int]:
        return self.output_ids

    def check_finished(self) -> None:
        if self._finishes_on_check:
            self._finished = True

    def finished(self) -> bool:
        return self._finished

    def is_dllm_prefill(self) -> bool:
        return False


def _new_cfg_scheduler(
    cond: _FakeReq,
    companions: list[_FakeReq],
) -> DllmScheduler:
    scheduler = object.__new__(DllmScheduler)
    scheduler._abort_lock = threading.Lock()
    scheduler._aborted_request_ids = set()
    scheduler._cond_to_unconds = {cond.rid: [companion.rid for companion in companions]}
    scheduler._uncond_to_cond = {companion.rid: cond.rid for companion in companions}
    scheduler._uncond_rids = {companion.rid for companion in companions}
    scheduler._orphaned_uncond_rids = set()
    scheduler._rid_to_req_data = {}
    scheduler._waiting_queue = []
    scheduler._staging_queue = [cond, *companions]
    scheduler.inbox = Queue()
    scheduler.outbox = Queue()
    scheduler.tree_cache = SimpleNamespace(
        cache_unfinished_req=lambda *args, **kwargs: None
    )
    scheduler.req_to_token_pool = SimpleNamespace(free=lambda req: None)
    scheduler._result_adapter = lambda data: data
    return scheduler


def test_cfg_abort_purges_both_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    cond = _FakeReq("cond")
    uncond = _FakeReq("cond-uncond")
    scheduler = _new_cfg_scheduler(cond, [uncond])
    scheduler._rid_to_req_data[cond.rid] = object()
    scheduler._aborted_request_ids.add(cond.rid)
    released: list[str] = []
    monkeypatch.setattr(
        dllm_scheduler_module,
        "release_kv_cache",
        lambda req, tree_cache: released.append(req.rid),
    )

    scheduler._drain_and_purge()

    assert scheduler._waiting_queue == []
    assert scheduler._staging_queue == []
    assert scheduler._rid_to_req_data == {}
    assert scheduler._cond_to_unconds == {}
    assert scheduler._uncond_to_cond == {}
    assert scheduler._uncond_rids == set()
    assert set(released) == {cond.rid, uncond.rid}


def test_cfg_completion_retires_all_companion_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cond = _FakeReq("cond", finishes_on_check=True)
    uncond_text = _FakeReq("cond-uncond")
    uncond_image = _FakeReq("cond-uncond-img")
    scheduler = _new_cfg_scheduler(cond, [uncond_text, uncond_image])
    req_data = SimpleNamespace(output_ids=[], finish_reason=None)
    scheduler._rid_to_req_data[cond.rid] = req_data
    released: list[str] = []
    monkeypatch.setattr(
        dllm_scheduler_module,
        "release_kv_cache",
        lambda req, tree_cache: released.append(req.rid),
    )
    excluded: list[_FakeReq] = []
    batch = SimpleNamespace(
        reqs=[cond, uncond_text, uncond_image],
        filter_batch=lambda *, chunked_req_to_exclude: excluded.extend(
            chunked_req_to_exclude
        ),
    )
    batch_result = SimpleNamespace(next_token_ids=[[7], [7], [7]])

    scheduler._apply_results(batch, batch_result)
    scheduler._post_step(batch)

    assert scheduler._staging_queue == []
    assert scheduler._cond_to_unconds == {}
    assert scheduler._uncond_to_cond == {}
    assert scheduler._uncond_rids == set()
    assert scheduler._orphaned_uncond_rids == set()
    assert set(released) == {cond.rid, uncond_text.rid, uncond_image.rid}
    output = scheduler.outbox.get_nowait()
    assert output.request_id == cond.rid
    assert output.type == "result"
    assert req_data.output_ids == [7]
    assert uncond_text.output_ids == [7]
    assert uncond_image.output_ids == [7]
    assert set(excluded) == {cond, uncond_text, uncond_image}


def test_result_adapter_failure_is_request_scoped() -> None:
    cond = _FakeReq("cond", finishes_on_check=True)
    uncond = _FakeReq("cond-uncond")
    scheduler = _new_cfg_scheduler(cond, [uncond])
    scheduler._rid_to_req_data[cond.rid] = SimpleNamespace(
        output_ids=[], finish_reason=None
    )

    def _fail_adapter(req_data):
        raise RuntimeError("missing <boi>")

    scheduler._result_adapter = _fail_adapter
    batch = SimpleNamespace(reqs=[cond, uncond])
    batch_result = SimpleNamespace(next_token_ids=[[7], [7]])

    scheduler._apply_results(batch, batch_result)

    output = scheduler.outbox.get_nowait()
    assert output.request_id == cond.rid
    assert output.type == "error"
    assert output.data == "missing <boi>"


def test_cfg_abort_from_image_companion_purges_whole_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cond = _FakeReq("cond")
    uncond_text = _FakeReq("cond-uncond")
    uncond_image = _FakeReq("cond-uncond-img")
    scheduler = _new_cfg_scheduler(cond, [uncond_text, uncond_image])
    scheduler._rid_to_req_data[cond.rid] = object()
    scheduler._aborted_request_ids.add(uncond_image.rid)
    released: list[str] = []
    monkeypatch.setattr(
        dllm_scheduler_module,
        "release_kv_cache",
        lambda req, tree_cache: released.append(req.rid),
    )

    scheduler._drain_and_purge()

    assert scheduler._staging_queue == []
    assert scheduler._cond_to_unconds == {}
    assert scheduler._uncond_to_cond == {}
    assert scheduler._uncond_rids == set()
    assert set(released) == {cond.rid, uncond_text.rid, uncond_image.rid}


class _ScheduleReq(_FakeReq):
    def __init__(
        self,
        rid: str,
        *,
        is_uncond: bool = False,
        is_uncond_image: bool = False,
        group_rid: str | None = None,
    ):
        super().__init__(rid)
        self._is_uncond = is_uncond
        self._is_uncond_img = is_uncond_image
        self._cfg_group_rid = group_rid
        self.dllm_phase = "staging_decode"
        self.is_chunked = 0
        self.dllm_block_offset = 0
        self.full_untruncated_fill_ids = [rid]
        self.extend_input_len = 0
        self.origin_input_ids = [1]
        self.last_node = f"node-{rid}"

    def init_next_round_input(self, tree_cache=None) -> None:
        self.dllm_block_offset += 32
        self.full_untruncated_fill_ids = [
            *self.full_untruncated_fill_ids,
            "next-block",
        ]


class _FakeScheduleBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.forward_mode = None
        self.decoding_reqs = None

    @classmethod
    def init_new(cls, *, reqs, **kwargs):
        return cls(reqs)

    def prepare_for_extend(self) -> None:
        return None


class _AcceptingPrefillAdder:
    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.tree_cache = args[1]

    def add_dllm_staging_req(self, req):
        self.can_run_list.append(req)
        req.extend_input_len = 32
        return dllm_scheduler_module.AddReqResult.CONTINUE

    def add_one_req(self, req, **kwargs):
        self.can_run_list.append(req)
        req.extend_input_len = 32
        self.tree_cache.inc_lock_ref(req.last_node)
        return dllm_scheduler_module.AddReqResult.CONTINUE


class _PartiallyAcceptingPrefillAdder(_AcceptingPrefillAdder):
    def add_one_req(self, req, **kwargs):
        if len(self.can_run_list) < 2:
            self.can_run_list.append(req)
            req.extend_input_len = 32
            self.tree_cache.inc_lock_ref(req.last_node)
            return dllm_scheduler_module.AddReqResult.CONTINUE
        return dllm_scheduler_module.AddReqResult.NO_TOKEN

    def add_dllm_staging_req(self, req):
        if len(self.can_run_list) < 2:
            self.can_run_list.append(req)
            req.extend_input_len = 32
            return dllm_scheduler_module.AddReqResult.CONTINUE
        return dllm_scheduler_module.AddReqResult.NO_TOKEN


class _RejectingPrefillAdder(_AcceptingPrefillAdder):
    def add_one_req(self, req, **kwargs):
        return dllm_scheduler_module.AddReqResult.NO_TOKEN

    def add_dllm_staging_req(self, req):
        return dllm_scheduler_module.AddReqResult.NO_TOKEN


class _FakeTreeCache:
    def __init__(self) -> None:
        self.locked_nodes: set[str] = set()

    def inc_lock_ref(self, node: str) -> None:
        self.locked_nodes.add(node)

    def dec_lock_ref(self, node: str) -> None:
        self.locked_nodes.remove(node)


def _new_scheduling_scheduler(
    waiting: list[_ScheduleReq],
    *,
    cond_to_unconds: dict[str, list[str]] | None = None,
) -> DllmScheduler:
    scheduler = object.__new__(DllmScheduler)
    scheduler._waiting_queue = list(waiting)
    scheduler._staging_queue = []
    scheduler._cond_to_unconds = cond_to_unconds or {}
    scheduler._uncond_to_cond = {
        companion_rid: cond_rid
        for cond_rid, companion_rids in scheduler._cond_to_unconds.items()
        for companion_rid in companion_rids
    }
    scheduler.server_args = SimpleNamespace(
        page_size=1,
        max_prefill_tokens=4096,
    )
    scheduler.dllm_config = SimpleNamespace(
        block_size=32,
        max_running_requests=3,
    )
    scheduler._chunked_prefill_size = 32
    scheduler.tree_cache = _FakeTreeCache()
    scheduler.token_to_kv_pool_allocator = object()
    scheduler.req_to_token_pool = object()
    scheduler.model_config = object()
    return scheduler


def test_scheduler_does_not_mix_normal_request_with_cfg_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normal = _ScheduleReq("normal")
    cond = _ScheduleReq("cond", group_rid="cond")
    uncond_text = _ScheduleReq(
        "cond-uncond",
        is_uncond=True,
        group_rid="cond",
    )
    uncond_image = _ScheduleReq(
        "cond-uncond-img",
        is_uncond=True,
        is_uncond_image=True,
        group_rid="cond",
    )
    scheduler = _new_scheduling_scheduler(
        [normal, cond, uncond_text, uncond_image],
        cond_to_unconds={
            cond.rid: [uncond_text.rid, uncond_image.rid],
        },
    )
    monkeypatch.setattr(
        dllm_scheduler_module,
        "PrefillAdder",
        _AcceptingPrefillAdder,
    )
    monkeypatch.setattr(
        dllm_scheduler_module,
        "ScheduleBatch",
        _FakeScheduleBatch,
    )

    batch = scheduler._schedule_next_batch()

    assert [req.rid for req in batch.reqs] == ["normal"]
    assert [req.rid for req in scheduler._waiting_queue] == [
        "cond",
        "cond-uncond",
        "cond-uncond-img",
    ]
    assert scheduler.tree_cache.locked_nodes == {"node-normal"}


def test_scheduler_restores_partially_admitted_staging_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cond = _ScheduleReq("cond", group_rid="cond")
    uncond_text = _ScheduleReq(
        "cond-uncond",
        is_uncond=True,
        group_rid="cond",
    )
    uncond_image = _ScheduleReq(
        "cond-uncond-img",
        is_uncond=True,
        is_uncond_image=True,
        group_rid="cond",
    )
    scheduler = _new_scheduling_scheduler(
        [],
        cond_to_unconds={
            cond.rid: [uncond_text.rid, uncond_image.rid],
        },
    )
    scheduler._staging_queue = [cond, uncond_text, uncond_image]
    monkeypatch.setattr(
        dllm_scheduler_module,
        "PrefillAdder",
        _PartiallyAcceptingPrefillAdder,
    )

    batch = scheduler._schedule_next_batch()

    assert batch is None
    assert scheduler._waiting_queue == []
    assert scheduler._staging_queue == [cond, uncond_text, uncond_image]
    for req in (cond, uncond_text, uncond_image):
        assert req.dllm_block_offset == 0
        assert req.full_untruncated_fill_ids == [req.rid]
        assert req.extend_input_len == 0

    monkeypatch.setattr(
        dllm_scheduler_module,
        "PrefillAdder",
        _AcceptingPrefillAdder,
    )
    monkeypatch.setattr(
        dllm_scheduler_module,
        "ScheduleBatch",
        _FakeScheduleBatch,
    )
    retry_batch = scheduler._schedule_next_batch()

    assert retry_batch.reqs == [cond, uncond_text, uncond_image]
    for req in retry_batch.reqs:
        assert req.dllm_block_offset == 32
        assert req.full_untruncated_fill_ids == [req.rid, "next-block"]
        assert req.extend_input_len == 32


def test_scheduler_keeps_three_way_cfg_group_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cond = _ScheduleReq("cond", group_rid="cond")
    uncond_text = _ScheduleReq(
        "cond-uncond",
        is_uncond=True,
        group_rid="cond",
    )
    uncond_image = _ScheduleReq(
        "cond-uncond-img",
        is_uncond=True,
        is_uncond_image=True,
        group_rid="cond",
    )
    scheduler = _new_scheduling_scheduler(
        [cond, uncond_text, uncond_image],
        cond_to_unconds={
            cond.rid: [uncond_text.rid, uncond_image.rid],
        },
    )
    monkeypatch.setattr(
        dllm_scheduler_module,
        "PrefillAdder",
        _AcceptingPrefillAdder,
    )
    monkeypatch.setattr(
        dllm_scheduler_module,
        "ScheduleBatch",
        _FakeScheduleBatch,
    )

    batch = scheduler._schedule_next_batch()

    assert [req.rid for req in batch.reqs] == [
        "cond",
        "cond-uncond",
        "cond-uncond-img",
    ]


def test_scheduler_validates_prefill_budget_for_complete_cfg_group() -> None:
    request_group = [
        _ScheduleReq("cond", group_rid="cond"),
        _ScheduleReq(
            "cond-uncond",
            is_uncond=True,
            group_rid="cond",
        ),
        _ScheduleReq(
            "cond-uncond-img",
            is_uncond=True,
            is_uncond_image=True,
            group_rid="cond",
        ),
    ]
    for req in request_group:
        req.origin_input_ids = list(range(64))
    scheduler = _new_scheduling_scheduler(request_group)
    scheduler.server_args.page_size = 8
    scheduler.server_args.max_prefill_tokens = 160

    with pytest.raises(RuntimeError, match="at least 161 max_prefill_tokens"):
        scheduler._validate_request_group_capacity(request_group)

    scheduler.server_args.max_prefill_tokens = 161
    scheduler._validate_request_group_capacity(request_group)


def test_scheduler_rejects_oversized_cfg_group_without_crashing_stage() -> None:
    cond = _ScheduleReq("cond", group_rid="cond")
    cond.origin_input_ids = list(range(64))
    cond._uncond_input_ids = list(cond.origin_input_ids)
    cond._uncond_img_input_ids = list(cond.origin_input_ids)
    scheduler = _new_scheduling_scheduler([])
    scheduler.server_args.page_size = 8
    scheduler.server_args.max_prefill_tokens = 160
    scheduler._abort_lock = threading.Lock()
    scheduler._aborted_request_ids = set()
    scheduler._rid_to_req_data = {}
    scheduler._uncond_rids = set()
    scheduler._orphaned_uncond_rids = set()
    scheduler.inbox = Queue()
    scheduler.outbox = Queue()
    scheduler._request_builder = lambda data: SimpleNamespace(req=cond)

    def create_companion(
        cond_req,
        uncond_input_ids,
        left_pad_len,
        rid_suffix,
        mark_img,
    ):
        companion = _ScheduleReq(
            f"{cond_req.rid}{rid_suffix}",
            is_uncond=True,
            is_uncond_image=mark_img,
            group_rid=cond_req.rid,
        )
        companion.origin_input_ids = list(uncond_input_ids)
        scheduler._waiting_queue.append(companion)
        scheduler._cond_to_unconds.setdefault(cond_req.rid, []).append(companion.rid)
        scheduler._uncond_to_cond[companion.rid] = cond_req.rid
        scheduler._uncond_rids.add(companion.rid)

    scheduler._create_uncond_companion = create_companion
    scheduler.inbox.put(
        IncomingMessage(
            request_id=cond.rid,
            type="new_request",
            data={},
        )
    )

    scheduler._drain_and_purge()

    assert scheduler._waiting_queue == []
    assert scheduler._rid_to_req_data == {}
    assert scheduler._cond_to_unconds == {}
    assert scheduler._uncond_to_cond == {}
    assert scheduler._uncond_rids == set()
    error = scheduler.outbox.get_nowait()
    assert error.request_id == cond.rid
    assert error.type == "error"
    assert "at least 161 max_prefill_tokens" in error.data


def test_scheduler_defers_partially_admitted_cfg_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cond = _ScheduleReq("cond", group_rid="cond")
    uncond_text = _ScheduleReq(
        "cond-uncond",
        is_uncond=True,
        group_rid="cond",
    )
    uncond_image = _ScheduleReq(
        "cond-uncond-img",
        is_uncond=True,
        is_uncond_image=True,
        group_rid="cond",
    )
    scheduler = _new_scheduling_scheduler(
        [cond, uncond_text, uncond_image],
        cond_to_unconds={
            cond.rid: [uncond_text.rid, uncond_image.rid],
        },
    )
    monkeypatch.setattr(
        dllm_scheduler_module,
        "PrefillAdder",
        _PartiallyAcceptingPrefillAdder,
    )

    batch = scheduler._schedule_next_batch()

    assert batch is None
    assert scheduler._staging_queue == []
    assert [req.rid for req in scheduler._waiting_queue] == [
        "cond",
        "cond-uncond",
        "cond-uncond-img",
    ]
    assert scheduler.tree_cache.locked_nodes == set()
    for req in (cond, uncond_text, uncond_image):
        assert req.dllm_block_offset == 0
        assert req.full_untruncated_fill_ids == [req.rid]
        assert req.extend_input_len == 0

    monkeypatch.setattr(
        dllm_scheduler_module,
        "PrefillAdder",
        _AcceptingPrefillAdder,
    )
    monkeypatch.setattr(
        dllm_scheduler_module,
        "ScheduleBatch",
        _FakeScheduleBatch,
    )
    retry_batch = scheduler._schedule_next_batch()

    assert retry_batch.reqs == [cond, uncond_text, uncond_image]
    for req in retry_batch.reqs:
        assert req.dllm_block_offset == 32
        assert req.full_untruncated_fill_ids == [req.rid, "next-block"]
        assert req.extend_input_len == 32


@pytest.mark.parametrize("from_staging", [False, True])
def test_scheduler_restores_rejected_cfg_group_before_retry(
    from_staging: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cond = _ScheduleReq("cond", group_rid="cond")
    uncond_text = _ScheduleReq(
        "cond-uncond",
        is_uncond=True,
        group_rid="cond",
    )
    uncond_image = _ScheduleReq(
        "cond-uncond-img",
        is_uncond=True,
        is_uncond_image=True,
        group_rid="cond",
    )
    request_group = [cond, uncond_text, uncond_image]
    scheduler = _new_scheduling_scheduler(
        [] if from_staging else request_group,
        cond_to_unconds={
            cond.rid: [uncond_text.rid, uncond_image.rid],
        },
    )
    if from_staging:
        scheduler._staging_queue = request_group
    monkeypatch.setattr(
        dllm_scheduler_module,
        "PrefillAdder",
        _RejectingPrefillAdder,
    )

    assert scheduler._schedule_next_batch() is None
    for req in request_group:
        assert req.dllm_block_offset == 0
        assert req.full_untruncated_fill_ids == [req.rid]
        assert req.extend_input_len == 0

    monkeypatch.setattr(
        dllm_scheduler_module,
        "PrefillAdder",
        _AcceptingPrefillAdder,
    )
    monkeypatch.setattr(
        dllm_scheduler_module,
        "ScheduleBatch",
        _FakeScheduleBatch,
    )
    retry_batch = scheduler._schedule_next_batch()

    assert retry_batch.reqs == request_group
    for req in retry_batch.reqs:
        assert req.dllm_block_offset == 32
        assert req.full_untruncated_fill_ids == [req.rid, "next-block"]
        assert req.extend_input_len == 32


def test_malformed_cfg_batch_does_not_fall_back_to_standard_decode() -> None:
    algorithm = object.__new__(LowConfidenceCFG)
    algorithm._run_standard = lambda model_runner, forward_batch: "standard"
    forward_batch = SimpleNamespace(
        batch_size=3,
        reqs=[
            _ScheduleReq("cond", group_rid="cond"),
            _ScheduleReq(
                "cond-uncond-img",
                is_uncond=True,
                is_uncond_image=True,
                group_rid="cond",
            ),
            _ScheduleReq("other"),
        ],
    )

    with pytest.raises(RuntimeError, match="Malformed CFG batch"):
        algorithm.run(None, forward_batch)


def test_cfg_batch_rejects_companions_from_another_request_group() -> None:
    algorithm = object.__new__(LowConfidenceCFG)
    algorithm._run_cfg_batch2 = lambda *args: "cfg"
    forward_batch = SimpleNamespace(
        batch_size=2,
        reqs=[
            _ScheduleReq("cond-a", group_rid="cond-a"),
            _ScheduleReq(
                "cond-b-uncond",
                is_uncond=True,
                group_rid="cond-b",
            ),
        ],
    )

    with pytest.raises(RuntimeError, match="Malformed CFG batch"):
        algorithm.run(None, forward_batch)
