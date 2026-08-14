import threading
from queue import Empty

import pytest
import torch

from sglang_omni.cli.serve import apply_parallelism_cli_overrides
from sglang_omni.models.llada2_uni.components.preprocessor import IMAGE_TOKEN_OFFSET
from sglang_omni.models.llada2_uni.config import (
    IMAGE_DECODE_STAGE,
    INTERLEAVED_COLLECT_STAGE,
    THINKER_STAGE,
    LLaDA2UniInterleavedPipelineConfig,
)
from sglang_omni.models.llada2_uni.interleaved import (
    InterleavedCollectorScheduler,
    InterleavedGenerationConfig,
    build_cfg_plan,
    parse_image_header,
)
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.models.llada2_uni.request_builders import (
    _advance_interleaved_state,
    _attach_cfg_branches,
)
from sglang_omni.models.llada2_uni.routing import thinker_next
from sglang_omni.pipeline.coordinator import _compute_timings_ms
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage


class _Tokenizer:
    SOI = 101
    BOI = 102
    EOI = 103
    HEIGHT = 104
    WIDTH = 105
    UNCOND = 106

    _token_to_id = {
        "<|image|>": SOI,
        "<boi>": BOI,
        "<|/image|>": EOI,
        "<uncondition>": UNCOND,
    }

    def convert_tokens_to_ids(self, token):
        return self._token_to_id.get(token, -1)

    def convert_ids_to_tokens(self, token_id):
        return {
            self.HEIGHT: "<|reserved_token_2|>",
            self.WIDTH: "<|reserved_token_2|>",
        }.get(token_id, f"token-{token_id}")

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "<uncondition>":
            return [self.UNCOND]
        return [900]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "|".join(str(token_id) for token_id in token_ids)


def _metadata(**overrides):
    config = {
        "max_frames": 2,
        "max_image_tokens": 16,
        "cfg_scale": 4.0,
        "cfg_text_scale": 7.5,
        "cfg_image_scale": 1.5,
    }
    config.update(overrides)
    return {"interleaved_generation": config}


def _payload(state, request_id="request"):
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="prompt"),
        data=state.to_dict(),
    )


def test_config_and_image_header_validation():
    config = InterleavedGenerationConfig.from_metadata(
        _metadata(max_frames=3, dllm_steps=8)
    )
    assert config.max_frames == 3
    assert config.decoder_steps == 8
    stream_state = config.to_stream_state(prompt_length=10, max_seq_len=8192)
    assert stream_state["interleaved_image_dllm_steps"] == 8
    assert "interleaved_dllm_steps" not in stream_state

    tokenizer = _Tokenizer()
    header = parse_image_header(
        [77, tokenizer.SOI, tokenizer.HEIGHT, tokenizer.WIDTH, tokenizer.BOI],
        tokenizer,
    )
    assert header.soi_position == 1
    assert header.grid_h == 2
    assert header.grid_w == 2
    assert header.image_token_count == 4

    with pytest.raises(ValueError, match="did not end"):
        parse_image_header([tokenizer.SOI, tokenizer.HEIGHT], tokenizer)
    with pytest.raises(ValueError, match="max_frames"):
        InterleavedGenerationConfig.from_metadata(_metadata(max_frames=0))


def test_cfg_plan_matches_reference_first_and_later_frame_conditions():
    tokenizer = _Tokenizer()
    config = InterleavedGenerationConfig.from_metadata(_metadata())
    suffix = [tokenizer.SOI, tokenizer.HEIGHT, tokenizer.WIDTH, tokenizer.BOI]

    first_ids = [10, 11] + suffix
    first = build_cfg_plan(
        full_ids=first_ids,
        header=parse_image_header(first_ids, tokenizer),
        frame_index=0,
        tokenizer=tokenizer,
        config=config,
    )
    assert first.mode == "simple"
    assert first.cfg_scale == 4.0
    assert first.branches["uncond"] == [900] + suffix

    later_ids = [20, tokenizer.EOI, 30, 31] + suffix
    later = build_cfg_plan(
        full_ids=later_ids,
        header=parse_image_header(later_ids, tokenizer),
        frame_index=1,
        tokenizer=tokenizer,
        config=config,
    )
    assert later.mode == "editing"
    assert later.branches["no_text"] == [20, tokenizer.EOI, tokenizer.UNCOND] + suffix
    assert later.branches["no_image"] == [900, 30, 31] + suffix


def test_cfg_plan_skips_companions_when_all_scales_are_disabled() -> None:
    tokenizer = _Tokenizer()
    config = InterleavedGenerationConfig.from_metadata(
        _metadata(cfg_scale=0.0, cfg_text_scale=0.0, cfg_image_scale=0.0)
    )
    full_ids = [10, tokenizer.SOI, tokenizer.HEIGHT, tokenizer.WIDTH, tokenizer.BOI]

    plan = build_cfg_plan(
        full_ids=full_ids,
        header=parse_image_header(full_ids, tokenizer),
        frame_index=0,
        tokenizer=tokenizer,
        config=config,
    )

    assert plan.mode == "none"
    assert plan.branches == {}


def test_cfg_plan_uses_two_branches_when_later_frame_disables_image_guidance() -> None:
    tokenizer = _Tokenizer()
    config = InterleavedGenerationConfig.from_metadata(
        _metadata(cfg_scale=0.0, cfg_text_scale=4.0, cfg_image_scale=0.0)
    )
    suffix = [tokenizer.SOI, tokenizer.HEIGHT, tokenizer.WIDTH, tokenizer.BOI]
    full_ids = [20, tokenizer.EOI, 30, 31] + suffix

    plan = build_cfg_plan(
        full_ids=full_ids,
        header=parse_image_header(full_ids, tokenizer),
        frame_index=1,
        tokenizer=tokenizer,
        config=config,
    )

    assert plan.mode == "simple"
    assert plan.branches["uncond"] == [20, tokenizer.EOI, tokenizer.UNCOND] + suffix
    assert plan.cfg_scale == 4.0


def test_cfg_attachment_aligns_dynamic_and_pre_aligned_branches() -> None:
    tokenizer = _Tokenizer()
    tokenizer.mask_token_id = 99
    request = type("Request", (), {})()

    _attach_cfg_branches(
        request,
        tokenizer=tokenizer,
        conditional_input_ids=[10, 11, 12],
        uncond_input_ids=[7, 8],
        cfg_scale=4.0,
        cfg_rescale=0.7,
        uncond_img_input_ids=[99, 6, 5],
        uncond_img_left_pad_len=1,
        cfg_image_scale=1.5,
    )

    assert request._uncond_input_ids == [99, 7, 8]
    assert request._uncond_left_pad_len == 1
    assert request._uncond_img_input_ids == [99, 6, 5]
    assert request._uncond_img_left_pad_len == 1
    assert request._cfg_scale == 4.0
    assert request._cfg_image_scale == 1.5
    assert request._cfg_rescale == 0.7


def test_interleaved_text_length_finishes_the_current_segment() -> None:
    tokenizer = _Tokenizer()
    config = InterleavedGenerationConfig.from_metadata(_metadata())
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[1, 2]])},
        thinker_out={"output_ids": [70, 71], "is_final": True},
        stream_state=config.to_stream_state(prompt_length=2, max_seq_len=8192),
        request_metadata=_metadata(),
        task_kind="interleaved",
    )

    _advance_interleaved_state(
        state,
        tokenizer,
        completed_phase="text",
        finish_reason="length",
    )

    assert state.stream_state["interleaved_phase"] == "done"
    assert state.stream_state["interleaved_finish_reason"] == "length"
    assert state.stream_state["interleaved_trailing_text"] == "70|71"
    assert state.prompt["input_ids"].flatten().tolist() == [1, 2, 70, 71]
    assert state.thinker_out["output_ids"] == [70, 71]


def test_interleaved_state_machine_fans_out_one_frame_then_reenters_text():
    tokenizer = _Tokenizer()
    config = InterleavedGenerationConfig.from_metadata(_metadata())
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[1, 2]])},
        thinker_out={
            "output_ids": [
                77,
                tokenizer.SOI,
                tokenizer.HEIGHT,
                tokenizer.WIDTH,
                tokenizer.BOI,
            ],
            "is_final": True,
        },
        stream_state=config.to_stream_state(prompt_length=2, max_seq_len=8192),
        request_metadata=_metadata(),
        task_kind="interleaved",
    )

    _advance_interleaved_state(state, tokenizer, completed_phase="text")
    assert state.stream_state["interleaved_phase"] == "image"
    assert state.stream_state["interleaved_cfg_plan"]["mode"] == "simple"
    assert state.stream_state["interleaved_current_frame"]["image_token_count"] == 4

    vq_tokens = [IMAGE_TOKEN_OFFSET + index for index in range(4)]
    state.thinker_out = {"output_ids": vq_tokens[:2], "is_final": True}
    _advance_interleaved_state(state, tokenizer, completed_phase="image")
    assert state.stream_state["interleaved_phase"] == "image"
    assert (
        state.stream_state["interleaved_current_frame"]["remaining_image_tokens"] == 2
    )
    assert thinker_next("request", _payload(state)) == THINKER_STAGE

    state.thinker_out = {
        "output_ids": vq_tokens[2:] + [tokenizer.EOI],
        "is_final": True,
    }
    _advance_interleaved_state(state, tokenizer, completed_phase="image")
    assert state.thinker_out["output_ids"] == vq_tokens

    assert state.stream_state["interleaved_phase"] == "text"
    assert state.stream_state["interleaved_frame_index"] == 1
    assert state.stream_state["interleaved_emit_frame"] is True
    assert state.stream_state["interleaved_needs_reentry"] is True
    assert state.prompt["input_ids"].flatten().tolist()[-5:] == vq_tokens + [
        tokenizer.EOI
    ]

    payload = _payload(state)
    assert thinker_next(payload.request_id, payload) == [
        IMAGE_DECODE_STAGE,
        THINKER_STAGE,
    ]


def _state_at_image_phase() -> tuple[LLaDA2UniPipelineState, _Tokenizer]:
    tokenizer = _Tokenizer()
    config = InterleavedGenerationConfig.from_metadata(_metadata())
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[1, 2]])},
        thinker_out={
            "output_ids": [
                77,
                tokenizer.SOI,
                tokenizer.HEIGHT,
                tokenizer.WIDTH,
                tokenizer.BOI,
            ],
            "is_final": True,
        },
        stream_state=config.to_stream_state(prompt_length=2, max_seq_len=8192),
        request_metadata=_metadata(),
        task_kind="interleaved",
    )
    _advance_interleaved_state(state, tokenizer, completed_phase="text")
    return state, tokenizer


def test_interleaved_text_phase_rejects_image_vocab_tokens() -> None:
    tokenizer = _Tokenizer()
    config = InterleavedGenerationConfig.from_metadata(_metadata())
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[1, 2]])},
        thinker_out={
            "output_ids": [
                IMAGE_TOKEN_OFFSET + 7,
                tokenizer.SOI,
                tokenizer.HEIGHT,
                tokenizer.WIDTH,
                tokenizer.BOI,
            ],
            "is_final": True,
        },
        stream_state=config.to_stream_state(prompt_length=2, max_seq_len=8192),
        request_metadata=_metadata(),
        task_kind="interleaved",
    )

    with pytest.raises(ValueError, match="text phase emitted image token"):
        _advance_interleaved_state(state, tokenizer, completed_phase="text")


def test_interleaved_image_phase_requires_eoi_after_all_vq_tokens() -> None:
    state, tokenizer = _state_at_image_phase()
    vq_tokens = [IMAGE_TOKEN_OFFSET + index for index in range(4)]

    state.thinker_out = {"output_ids": vq_tokens[:2] + [tokenizer.EOI]}
    with pytest.raises(ValueError, match="EOI with 2 VQ tokens remaining"):
        _advance_interleaved_state(state, tokenizer, completed_phase="image")

    state, tokenizer = _state_at_image_phase()
    state.thinker_out = {"output_ids": vq_tokens}
    with pytest.raises(ValueError, match="without EOI"):
        _advance_interleaved_state(state, tokenizer, completed_phase="image")


def test_interleaved_image_phase_rejects_text_token_before_eoi() -> None:
    state, tokenizer = _state_at_image_phase()
    state.thinker_out = {"output_ids": [77, tokenizer.EOI]}

    with pytest.raises(ValueError, match="non-image token before EOI"):
        _advance_interleaved_state(state, tokenizer, completed_phase="image")


def test_collector_abort_tombstones_are_bounded_and_persistent(monkeypatch) -> None:
    collector = InterleavedCollectorScheduler()
    limit = collector._ABORT_TOMBSTONE_LIMIT

    for index in range(limit + 2):
        collector.abort(f"request-{index}")

    assert len(collector._aborted) == limit
    assert "request-0" not in collector._aborted
    aborted_request_id = f"request-{limit + 1}"
    received: list[str] = []
    reached_probe = threading.Event()

    def receive(request_id, payload) -> None:
        del payload
        if request_id == "probe":
            reached_probe.set()
        else:
            received.append(request_id)

    monkeypatch.setattr(collector, "_receive", receive)
    collector.inbox.put(IncomingMessage(aborted_request_id, "new_request"))
    collector.inbox.put(IncomingMessage(aborted_request_id, "new_request"))
    collector.inbox.put(IncomingMessage("probe", "new_request"))

    thread = threading.Thread(target=collector.start, daemon=True)
    thread.start()
    try:
        assert reached_probe.wait(timeout=2.0)
    finally:
        collector.stop()
        thread.join(timeout=2.0)

    assert received == []
    assert aborted_request_id in collector._aborted


@pytest.mark.parametrize("frame_first", [True, False])
def test_collector_joins_final_thinker_state_and_decoded_frames(frame_first):
    state = LLaDA2UniPipelineState(
        stream_state={
            "interleaved_done": True,
            "interleaved_frame_index": 1,
            "interleaved_finish_reason": "stop",
            "interleaved_segments": [{"text": "frame text"}],
            "interleaved_trailing_text": "after",
            "interleaved_usage": {"total_tokens": 10},
        },
        task_kind="interleaved",
    )
    final_payload = _payload(state)
    final_payload.timing = {
        "thinker.enter": 1_000_000,
        "thinker.done": 3_000_000,
    }
    frame_payload = StagePayload(
        request_id="request",
        request=final_payload.request,
        data={
            "kind": "interleaved_frame",
            "frame": {
                "index": 1,
                "text": "frame text",
                "grid_h": 2,
                "grid_w": 2,
                "image": "base64-image",
                "format": "png",
            },
        },
        timing={
            "image_decode.enter": 4_000_000,
            "image_decode.done": 9_000_000,
        },
    )
    collector = InterleavedCollectorScheduler()

    ordered = (
        [frame_payload, final_payload]
        if frame_first
        else [final_payload, frame_payload]
    )
    collector._receive("request", ordered[0])
    with pytest.raises(Empty):
        collector.outbox.get_nowait()
    collector._receive("request", ordered[1])

    result_payload = collector.outbox.get_nowait().data
    result = result_payload.data
    assert result["modality"] == "interleaved"
    assert [item["type"] for item in result["content"]] == [
        "text",
        "image",
        "text",
    ]
    assert result["content"][1]["image"]["data"] == "base64-image"
    assert "images" not in result
    assert "text" not in result
    assert result["finish_reason"] == "stop"
    assert _compute_timings_ms(result_payload.timing) == {
        "thinker_ms": 2.0,
        "image_decode_ms": 5.0,
        "e2e_ms": 8.0,
    }


def test_collector_orders_image_decode_timings_by_frame_index() -> None:
    state = LLaDA2UniPipelineState(
        stream_state={
            "interleaved_done": True,
            "interleaved_frame_index": 2,
            "interleaved_segments": [{"text": "first"}, {"text": "second"}],
        },
        task_kind="interleaved",
    )
    final_payload = _payload(state)
    final_payload.timing = {
        "thinker.enter": 1_000_000,
        "thinker.done": 2_000_000,
    }

    def frame_payload(index: int, enter_ns: int, done_ns: int) -> StagePayload:
        return StagePayload(
            request_id="request",
            request=final_payload.request,
            data={
                "kind": "interleaved_frame",
                "frame": {
                    "index": index,
                    "image": f"base64-image-{index}",
                    "format": "png",
                },
            },
            timing={
                "image_decode.enter": enter_ns,
                "image_decode.done": done_ns,
            },
        )

    collector = InterleavedCollectorScheduler()
    collector._receive("request", frame_payload(2, 10_000_000, 17_000_000))
    collector._receive("request", final_payload)
    collector._receive("request", frame_payload(1, 3_000_000, 8_000_000))

    result_payload = collector.outbox.get_nowait().data
    assert [
        part["image"]["data"]
        for part in result_payload.data["content"]
        if part["type"] == "image"
    ] == ["base64-image-1", "base64-image-2"]
    assert _compute_timings_ms(result_payload.timing) == {
        "thinker_ms": 1.0,
        "image_decode_ms": 5.0,
        "image_decode_pass2_ms": 7.0,
        "e2e_ms": 16.0,
    }


def test_interleaved_pipeline_topologies_keep_thinker_and_decoder_disjoint():
    single = LLaDA2UniInterleavedPipelineConfig(model_path="model")
    stages = {stage.name: stage for stage in single.stages}
    assert stages[THINKER_STAGE].gpu == 0
    assert stages[THINKER_STAGE].stream_to == []
    assert stages[IMAGE_DECODE_STAGE].gpu == 1
    assert stages[IMAGE_DECODE_STAGE].can_accept_stream_before_payload is False
    assert stages[INTERLEAVED_COLLECT_STAGE].terminal is True

    apply_parallelism_cli_overrides(
        single,
        thinker_tp_size=2,
        thinker_gpus="0,1",
        talker_gpu=None,
        code2wav_gpu=None,
        image_decoder_sp_size=2,
        image_decoder_gpus="2,3",
        image_decoder_ring_degree=2,
        image_decoder_backend="sglang",
    )
    stages = {stage.name: stage for stage in single.stages}
    assert stages[THINKER_STAGE].gpu == [0, 1]
    assert stages[THINKER_STAGE].parallelism.tp == 2
    assert stages[IMAGE_DECODE_STAGE].gpu == [2, 3]
    assert stages[IMAGE_DECODE_STAGE].parallelism.sp == 2
    assert set(stages[THINKER_STAGE].gpu).isdisjoint(stages[IMAGE_DECODE_STAGE].gpu)
