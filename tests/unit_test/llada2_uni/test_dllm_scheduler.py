# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from queue import Queue
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from sglang_omni.models.llada2_uni.algorithm.low_confidence_cfg import LowConfidenceCFG
from sglang_omni.models.llada2_uni.components.preprocessor import (
    align_cfg_unconditional_input_ids,
)
from sglang_omni.scheduling import dllm_scheduler as dllm_scheduler_module
from sglang_omni.scheduling.dllm_scheduler import DllmScheduler
from sglang_omni.scheduling.messages import IncomingMessage


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
