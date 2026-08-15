# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import multiprocessing
import queue
import threading
from types import SimpleNamespace

import pytest

from sglang_omni.config import (
    ParallelismConfig,
    PipelineConfig,
    SequenceParallelPolicy,
    StageConfig,
)
from sglang_omni.pipeline import parallel_control
from sglang_omni.pipeline.mp_runner import _build_sp_stage_specs
from sglang_omni.pipeline.parallel_control import (
    ParallelAbortMessage,
    ParallelFollowerControlPlane,
    ParallelLeaderFanout,
    ParallelStageContext,
    RequestDispatchTracker,
)
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.pipeline.tp_control import TPWorkMessage
from sglang_omni.proto import (
    AbortMessage,
    AdminMessage,
    AdminOperation,
    AdminResult,
    AdminResultMessage,
    ShutdownMessage,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.scheduling.types import ParallelSchedulerCapabilities
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeRelay,
    RecordingStageControlPlane,
)


class _SequenceParallelPipelineConfig(PipelineConfig):
    @classmethod
    def sequence_parallel_policy(
        cls, *, stage_name: str
    ) -> SequenceParallelPolicy | None:
        return SequenceParallelPolicy()


async def _deliver_follower_abort(
    stage: Stage,
    control_plane: RecordingStageControlPlane,
    abort_message: object,
) -> None:
    stage._running = True
    listener = asyncio.create_task(stage._abort_listener())
    try:
        await control_plane.abort_inbox.put(abort_message)
        for _ in range(100):
            if control_plane.abort_inbox.empty():
                await asyncio.sleep(0)
                return
            await asyncio.sleep(0)
        pytest.fail("follower abort listener did not consume the abort")
    finally:
        stage._running = False
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)


def _parallel_context(
    role: str,
    *,
    fanout: object | None = None,
) -> ParallelStageContext:
    if role == "leader" and fanout is None:
        fanout = ParallelLeaderFanout(
            "image_decode",
            follower_work_queues=[],
            follower_abort_queues=[],
        )
    return ParallelStageContext(
        kind="sp",
        rank=0 if role == "leader" else 1,
        size=2,
        role=role,
        fanout=fanout,
    )


def test_sp_stage_launch_assigns_roles_and_factory_rank_args() -> None:
    stage = StageConfig(
        name="image_decode",
        factory="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
        gpu=[0, 1],
        parallelism=ParallelismConfig(
            sp=2,
            ulysses_degree=2,
            ring_degree=1,
        ),
        terminal=True,
    )
    config = _SequenceParallelPipelineConfig(model_path="dummy", stages=[stage])

    specs = _build_sp_stage_specs(
        ctx=multiprocessing.get_context("spawn"),
        stage_cfg=stage,
        config=config,
        gpu_ids=[0, 1],
        nccl_port=29600,
        recv_endpoint="ipc:///tmp/image_decode.sock",
        base_factory_args={},
        stage_kwargs={
            "stage_name": "image_decode",
            "factory": stage.factory,
            "next_stages": None,
            "is_terminal": True,
        },
    )

    assert [spec.role for spec in specs] == ["leader", "follower"]
    assert [spec.sp_rank for spec in specs] == [0, 1]
    assert [spec.gpu_id for spec in specs] == [0, 1]
    assert [spec.factory_args["stage_role"] for spec in specs] == [
        "leader",
        "follower",
    ]
    assert all(spec.factory_args["sp_size"] == 2 for spec in specs)
    assert all(spec.factory_args["ulysses_degree"] == 2 for spec in specs)
    assert all(spec.factory_args["ring_degree"] == 1 for spec in specs)
    assert len(specs[0].follower_admin_result_queues) == 1
    assert (
        specs[1].internal_admin_result_queue is specs[0].follower_admin_result_queues[0]
    )
    assert [spec.parallel_kind for spec in specs] == ["sp", "sp"]
    assert [spec.parallel_rank for spec in specs] == [0, 1]
    assert [spec.parallel_size for spec in specs] == [2, 2]


def test_tp_work_message_keeps_legacy_two_argument_constructor() -> None:
    payload = object()

    message = TPWorkMessage(request_id="request", data=payload)

    assert message.request_id == "request"
    assert message.data is payload


def test_parallel_fanout_correlates_work_and_abort_dispatch_ids() -> None:
    work_queue: queue.Queue = queue.Queue()
    abort_queue: queue.Queue = queue.Queue()
    fanout = ParallelLeaderFanout(
        "image_decode",
        follower_work_queues=[work_queue],
        follower_abort_queues=[abort_queue],
    )
    payload = SimpleNamespace(request_id="req-dispatch")

    dispatch_id = fanout.fanout_work(payload)
    asyncio.run(fanout.fanout_abort(AbortMessage(payload.request_id)))

    work_msg = work_queue.get_nowait()
    abort_msg = abort_queue.get_nowait()
    assert dispatch_id == 1
    assert work_msg.dispatch_id == dispatch_id
    assert abort_msg.dispatch_id == dispatch_id


def test_parallel_fanout_reuses_dispatch_id_until_terminal_ack() -> None:
    abort_queue: queue.Queue = queue.Queue()
    fanout = ParallelLeaderFanout(
        "image_decode",
        follower_work_queues=[],
        follower_abort_queues=[abort_queue],
    )
    request_id = "req-duplicate-abort"
    dispatch_id = fanout.fanout_work(SimpleNamespace(request_id=request_id))

    first = asyncio.run(fanout.fanout_abort(AbortMessage(request_id)))
    second = asyncio.run(fanout.fanout_abort(AbortMessage(request_id)))

    assert first == second == dispatch_id
    assert abort_queue.get_nowait().dispatch_id == dispatch_id
    assert abort_queue.get_nowait().dispatch_id == dispatch_id

    fanout.acknowledge_terminal(request_id, dispatch_id)
    with pytest.raises(RuntimeError, match="No active stage work dispatch"):
        asyncio.run(fanout.fanout_abort(AbortMessage(request_id)))


def test_parallel_fanout_queues_sequential_request_id_reuse() -> None:
    fanout = ParallelLeaderFanout(
        "image_decode",
        follower_work_queues=[],
        follower_abort_queues=[],
    )
    request_id = "req-reused"
    first_dispatch_id = fanout.fanout_work(SimpleNamespace(request_id=request_id))
    second_dispatch_id = fanout.fanout_work(SimpleNamespace(request_id=request_id))

    assert second_dispatch_id == first_dispatch_id + 1

    fanout.acknowledge_terminal(request_id, first_dispatch_id)
    assert (
        asyncio.run(fanout.fanout_abort(AbortMessage(request_id))) == second_dispatch_id
    )

    fanout.acknowledge_terminal(request_id, second_dispatch_id)


def test_parallel_stage_context_rejects_leader_without_fanout() -> None:
    context_type = getattr(parallel_control, "ParallelStageContext", None)
    assert context_type is not None
    with pytest.raises(ValueError, match="leader.*fanout"):
        context_type(kind="sp", rank=0, size=2, role="leader")


def test_simple_scheduler_declares_parallel_drain_capabilities() -> None:
    scheduler = SimpleScheduler(lambda payload: payload)

    assert scheduler.parallel_capabilities.fanout_work is True
    assert scheduler.parallel_capabilities.drain_aborted_work is True


def test_parallel_follower_ignores_replayed_abort_when_abort_precedes_work() -> None:
    request_id = "req-abort-before-work"
    allow_work = threading.Event()

    class GatedWorkQueue:
        def __init__(self) -> None:
            self._queue: queue.Queue = queue.Queue()

        def put_nowait(self, item: object) -> None:
            self._queue.put_nowait(item)

        def get(self, timeout: float) -> object:
            if not allow_work.wait(timeout):
                raise queue.Empty
            return self._queue.get(timeout=timeout)

    work_queue = GatedWorkQueue()
    abort_queue: queue.Queue = queue.Queue()
    fanout = ParallelLeaderFanout(
        "image_decode",
        follower_work_queues=[work_queue],
        follower_abort_queues=[abort_queue],
    )
    payload = SimpleNamespace(request_id=request_id, timing={})
    dispatch_id = fanout.fanout_work(payload)
    asyncio.run(fanout.fanout_abort(AbortMessage(request_id)))

    cleanup_calls: list[str] = []
    scheduler = SimpleScheduler(
        lambda payload: payload,
        abort_callback=cleanup_calls.append,
    )
    control_plane = ParallelFollowerControlPlane(
        stage_name="image_decode",
        work_queue=work_queue,
        abort_queue=abort_queue,
    )
    stage = Stage(
        name="image_decode",
        role="follower",
        get_next=lambda request_id, output: None,
        gpu_id=None,
        endpoints={},
        control_plane=control_plane,
        relay=FakeRelay(),
        scheduler=scheduler,
        parallel_context=_parallel_context("follower"),
    )

    async def run() -> None:
        run_task = asyncio.create_task(stage.run())
        for _ in range(100):
            if abort_queue.empty():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("abort drain was not registered before work")

        allow_work.set()
        for _ in range(100):
            if cleanup_calls == [request_id]:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("aborted stage-fanned work did not drain")

        abort_queue.put_nowait(
            ParallelAbortMessage(request_id=request_id, dispatch_id=dispatch_id)
        )
        for _ in range(100):
            if abort_queue.empty():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("replayed abort was not consumed")

        try:
            assert not scheduler._abort_drains.is_pending(request_id, dispatch_id)
        finally:
            work_queue.put_nowait(ShutdownMessage())
            await asyncio.wait_for(run_task, timeout=1.0)

    asyncio.run(run())


def test_parallel_stage_abort_uses_drain_contract_after_work_fanout() -> None:
    calls: list[tuple[str, str]] = []
    scheduler = SimpleNamespace(
        inbox=queue.Queue(),
        parallel_capabilities=ParallelSchedulerCapabilities(
            fanout_work=True,
            drain_aborted_work=True,
        ),
        abort=lambda request_id: calls.append(("abort", request_id)),
        mark_request_aborted_for_drain=lambda request_id, dispatch_id: calls.append(
            ("abort_drain", f"{request_id}:{dispatch_id}")
        ),
        acknowledge_request_terminal=lambda request_id, dispatch_id: None,
    )
    fanout = ParallelLeaderFanout(
        "image_decode",
        follower_work_queues=[],
        follower_abort_queues=[],
    )
    stage = Stage(
        name="image_decode",
        role="leader",
        get_next=lambda request_id, output: None,
        gpu_id=None,
        endpoints={},
        control_plane=RecordingStageControlPlane(),
        relay=FakeRelay(),
        scheduler=scheduler,
        parallel_context=_parallel_context("leader", fanout=fanout),
    )

    asyncio.run(stage._execute(SimpleNamespace(request_id="req-abort", timing={})))
    stage._on_abort("req-abort")

    assert calls == [("abort_drain", "req-abort:1")]


def test_parallel_stage_abort_without_fanned_work_uses_normal_abort() -> None:
    calls: list[tuple[str, str]] = []
    scheduler = SimpleNamespace(
        parallel_capabilities=ParallelSchedulerCapabilities(),
        abort=lambda request_id: calls.append(("abort", request_id)),
    )
    stage = Stage(
        name="image_decode",
        role="leader",
        get_next=lambda request_id, output: None,
        gpu_id=None,
        endpoints={},
        control_plane=RecordingStageControlPlane(),
        relay=FakeRelay(),
        scheduler=scheduler,
        parallel_context=_parallel_context("leader"),
    )

    stage._on_abort("req-unseen")

    assert calls == [("abort", "req-unseen")]


def test_scheduler_owned_abort_uses_propagation_control_path() -> None:
    calls: list[tuple[str, str]] = []
    scheduler = SimpleNamespace(
        parallel_capabilities=ParallelSchedulerCapabilities(
            synchronize_abort=True,
        ),
        abort=lambda request_id: calls.append(("local", request_id)),
        propagate_abort=lambda request_id: calls.append(("propagated", request_id)),
    )
    stage = Stage(
        name="thinker",
        role="leader",
        get_next=lambda request_id, output: None,
        gpu_id=None,
        endpoints={},
        control_plane=RecordingStageControlPlane(),
        relay=FakeRelay(),
        scheduler=scheduler,
        parallel_context=_parallel_context("leader"),
    )

    stage._on_abort("req-omni")

    assert calls == [("propagated", "req-omni")]


def test_abort_drain_after_terminal_enqueue_releases_scheduler_tombstone() -> None:
    cleanup_calls: list[str] = []
    scheduler = SimpleScheduler(
        lambda payload: payload,
        abort_callback=cleanup_calls.append,
    )
    stage = Stage(
        name="image_decode",
        role="leader",
        get_next=lambda request_id, output: None,
        gpu_id=None,
        endpoints={},
        control_plane=RecordingStageControlPlane(),
        relay=FakeRelay(),
        scheduler=scheduler,
        parallel_context=_parallel_context("leader"),
    )
    request_id = "req-terminal-race"
    stage._active_requests.add(request_id)
    asyncio.run(stage._execute(SimpleNamespace(request_id=request_id, timing={})))
    scheduler._emit_result(request_id, object(), scheduler.outbox)

    stage._on_abort(request_id)
    asyncio.run(stage._drain_outbox_external())

    assert not scheduler._abort_drains.is_pending(request_id, dispatch_id=1)
    assert request_id not in scheduler._aborted
    assert cleanup_calls == [request_id]


def test_parallel_follower_ignores_abort_after_terminal_drain() -> None:
    request_id = "req-terminal-before-abort"
    tracker = RequestDispatchTracker()
    tracker.register_work(request_id, dispatch_id=1)
    tracker.finish_terminal(request_id)

    assert tracker.should_ignore_abort(request_id, dispatch_id=1)


def test_parallel_follower_distinguishes_reused_id_abort_dispatch() -> None:
    request_id = "req-reused"
    tracker = RequestDispatchTracker()
    tracker.register_work(request_id, dispatch_id=1)
    tracker.finish_terminal(request_id)

    assert not tracker.should_ignore_abort(request_id, dispatch_id=2)


def test_parallel_follower_ignores_old_abort_after_reused_id_work() -> None:
    request_id = "req-reused"
    tracker = RequestDispatchTracker()
    tracker.register_work(request_id, dispatch_id=1)
    tracker.finish_terminal(request_id)
    tracker.register_work(request_id, dispatch_id=2)

    assert tracker.should_ignore_abort(request_id, dispatch_id=1)


def test_request_dispatch_tracker_retains_completed_watermark() -> None:
    tracker = RequestDispatchTracker()
    for dispatch_id in range(1, 12_001):
        request_id = f"req-{dispatch_id}"
        tracker.register_work(request_id, dispatch_id)
        tracker.finish_terminal(request_id)

    assert tracker.should_ignore_abort("req-1", dispatch_id=1)
    assert not tracker.should_ignore_abort("req-next", dispatch_id=12_001)


def test_request_dispatch_tracker_handles_out_of_order_completion() -> None:
    tracker = RequestDispatchTracker()
    tracker.register_work("req-1", dispatch_id=1)
    tracker.register_work("req-2", dispatch_id=2)

    tracker.finish_terminal("req-2")

    assert tracker.should_ignore_abort("req-2", dispatch_id=2)
    assert not tracker.should_ignore_abort("req-1", dispatch_id=1)

    tracker.finish_terminal("req-1")

    assert tracker.should_ignore_abort("req-1", dispatch_id=1)
    assert tracker.should_ignore_abort("req-2", dispatch_id=2)
    assert not tracker.should_ignore_abort("req-3", dispatch_id=3)


def test_request_dispatch_tracker_compresses_long_out_of_order_range() -> None:
    tracker = RequestDispatchTracker()
    tracker.register_work("req-1", dispatch_id=1)
    for dispatch_id in range(2, 12_001):
        request_id = f"req-{dispatch_id}"
        tracker.register_work(request_id, dispatch_id)
        tracker.finish_terminal(request_id)

    assert tracker._completed_dispatch_ranges == [(2, 12_000)]

    tracker.finish_terminal("req-1")

    assert tracker._completed_dispatch_watermark == 12_000
    assert tracker._completed_dispatch_ranges == []


def test_request_dispatch_tracker_queues_sequential_same_id_dispatches() -> None:
    tracker = RequestDispatchTracker()
    tracker.register_work("req-reused", dispatch_id=1)
    tracker.register_work("req-reused", dispatch_id=2)

    assert tracker.first_active_dispatch_id("req-reused") == 1
    assert tracker.finish_terminal("req-reused") == 1
    assert tracker.first_active_dispatch_id("req-reused") == 2
    assert tracker.finish_terminal("req-reused") == 2


def test_request_dispatch_tracker_preserves_abort_that_precedes_work() -> None:
    tracker = RequestDispatchTracker()
    request_id = "req-abort-before-work"

    tracker.record_abort(request_id, dispatch_id=1)

    assert tracker.register_work(request_id, dispatch_id=1)
    assert tracker.finish_terminal(request_id) == 1
    assert not tracker.has_active_work(request_id)


def test_parallel_stage_requires_complete_rank_metadata() -> None:
    with pytest.raises(ValueError, match="size must be at least 2"):
        ParallelStageContext(
            kind="sp",
            rank=0,
            size=1,
            role="leader",
            fanout=ParallelLeaderFanout(
                "image_decode",
                follower_work_queues=[],
                follower_abort_queues=[],
            ),
        )


def test_parallel_stage_rejects_streaming_scheduler_without_stream_fanout() -> None:
    scheduler = StreamingSimpleScheduler(lambda payload: payload)
    assert scheduler.requires_tp_work_fanout is True

    with pytest.raises(TypeError, match="does not support parallel stages"):
        Stage(
            name="streaming",
            role="leader",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={},
            control_plane=RecordingStageControlPlane(),
            relay=FakeRelay(),
            scheduler=scheduler,
            parallel_context=_parallel_context("leader"),
        )


def test_tp_stage_rejects_legacy_incomplete_scheduler_capability() -> None:
    scheduler = SimpleNamespace(
        inbox=queue.Queue(),
        outbox=queue.Queue(),
        requires_tp_work_fanout=False,
    )
    fanout = ParallelLeaderFanout(
        "thinker",
        follower_work_queues=[],
        follower_abort_queues=[],
    )

    with pytest.raises(TypeError, match="must declare parallel_capabilities"):
        Stage(
            name="thinker",
            role="leader",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={},
            control_plane=RecordingStageControlPlane(),
            relay=FakeRelay(),
            scheduler=scheduler,
            parallel_context=ParallelStageContext(
                kind="tp",
                rank=0,
                size=2,
                role="leader",
                fanout=fanout,
            ),
        )


def test_simple_scheduler_keeps_legacy_tp_fanout_capability() -> None:
    scheduler = SimpleScheduler(lambda payload: payload)

    assert scheduler.requires_tp_work_fanout is True


def test_threaded_scheduler_keeps_legacy_tp_fanout_capability() -> None:
    scheduler = ThreadedSimpleScheduler(lambda payload: payload)

    assert scheduler.requires_tp_work_fanout is True
    scheduler.stop()


def test_parallel_stage_stop_waits_for_scheduler_thread() -> None:
    started = threading.Event()
    allow_exit = threading.Event()

    class Scheduler:
        parallel_capabilities = ParallelSchedulerCapabilities()

        def __init__(self) -> None:
            self.inbox = queue.Queue()
            self.outbox = queue.Queue()

        def start(self) -> None:
            started.set()
            allow_exit.wait()

        def stop(self) -> None:
            pass

        def abort(self, request_id: str) -> None:
            del request_id

    async def run() -> None:
        stage = Stage(
            name="image_decode",
            role="leader",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={},
            control_plane=RecordingStageControlPlane(),
            relay=FakeRelay(),
            scheduler=Scheduler(),
            parallel_context=_parallel_context("leader"),
        )
        await stage.start()
        assert await asyncio.to_thread(started.wait, 1.0)

        stop_task = asyncio.create_task(stage.stop())
        await asyncio.sleep(0.05)
        try:
            assert not stop_task.done()
        finally:
            allow_exit.set()
        await asyncio.wait_for(stop_task, timeout=1.0)
        assert stage._scheduler_thread is None

    asyncio.run(run())


def test_abort_listener_failure_stops_stage() -> None:
    class FaultyControlPlane(RecordingStageControlPlane):
        async def recv_abort(self) -> object:
            raise RuntimeError("abort channel failed")

    async def run() -> None:
        control_plane = FaultyControlPlane()
        scheduler = SimpleNamespace(
            parallel_capabilities=ParallelSchedulerCapabilities(),
            abort=lambda request_id: None,
        )
        stage = Stage(
            name="image_decode",
            role="leader",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={},
            control_plane=control_plane,
            relay=FakeRelay(),
            scheduler=scheduler,
            parallel_context=_parallel_context("leader"),
        )
        stage._running = True
        listener = asyncio.create_task(stage._abort_listener())
        listener.add_done_callback(
            lambda task: stage._on_background_task_done(task, "abort listener")
        )

        with pytest.raises(RuntimeError, match="abort channel failed"):
            await listener
        await asyncio.sleep(0)

        assert not stage._running
        assert isinstance(stage._background_task_error, RuntimeError)
        assert control_plane.closed

    asyncio.run(run())


def test_sp_stage_admin_reports_sp_rank_metadata() -> None:
    class Scheduler:
        parallel_capabilities = ParallelSchedulerCapabilities()

        def admin(self, action, payload):
            return {"success": True, "data": {"action": action, **payload}}

        def abort(self, request_id):
            del request_id

    class Fanout:
        async def collect_admin_results(self, op_id, *, timeout_s):
            del timeout_s
            return [
                AdminResultMessage(
                    AdminResult(
                        op_id=op_id,
                        stage="image_decode",
                        action="model_info",
                        success=True,
                        rank=1,
                        role="follower",
                    )
                )
            ]

    async def run() -> None:
        control_plane = RecordingStageControlPlane()
        stage = Stage(
            name="image_decode",
            role="leader",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={},
            control_plane=control_plane,
            relay=FakeRelay(),
            scheduler=Scheduler(),
            parallel_context=_parallel_context("leader", fanout=Fanout()),
        )

        await stage._on_admin(
            AdminMessage(AdminOperation(op_id="op-1", action="model_info"))
        )

        result = control_plane.completions[0].result
        assert result.rank == 0
        assert result.data["parallel_kind"] == "sp"
        assert result.data["parallel_size"] == 2
        assert result.data["sp_size"] == 2
        assert "tp_size" not in result.data
        assert [item["rank"] for item in result.data["rank_results"]] == [0, 1]

    asyncio.run(run())
