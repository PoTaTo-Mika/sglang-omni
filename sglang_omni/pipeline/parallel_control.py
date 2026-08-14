# SPDX-License-Identifier: Apache-2.0
"""Internal control helpers shared by TP and SP stage ranks.

The external pipeline control plane is owned by rank zero. These helpers mirror
control messages and fan out scheduler inputs to the remaining ranks so every
participant enters collectives in the same order.
"""

from __future__ import annotations

import asyncio
import bisect
import logging
import queue as queue_mod
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

from sglang_omni.proto import (
    AbortMessage,
    AdminMessage,
    AdminResultMessage,
    ProfilerStartMessage,
    ProfilerStopMessage,
    ShutdownMessage,
)

logger = logging.getLogger(__name__)

_WORK_POLL_SECONDS = 0.1


@dataclass
class ParallelWorkMessage:
    """Scheduler input sent from a parallel-stage leader to its followers.

    ``dispatch_id`` is optional only so legacy ``TPWorkMessage(request_id,
    data)`` construction remains source-compatible. Leader fanout always
    assigns it, and the stage rejects delivered work without one.
    """

    request_id: str
    data: Any
    dispatch_id: int | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("Parallel-stage work requires a non-empty request_id")
        if self.dispatch_id is not None and self.dispatch_id < 1:
            raise ValueError("Parallel-stage work dispatch ID must be positive")


@dataclass
class ParallelAbortMessage:
    """Abort correlated with one stage work dispatch."""

    request_id: str
    dispatch_id: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("Parallel-stage abort requires a non-empty request_id")
        if self.dispatch_id < 1:
            raise ValueError("Parallel-stage abort dispatch ID must be positive")


class RequestDispatchTracker:
    """Track stage work dispatches through abort and terminal drain."""

    def __init__(self) -> None:
        self._active_dispatch_ids: dict[str, deque[int]] = {}
        self._pending_abort_dispatch_ids: dict[str, set[int]] = {}
        self._completed_dispatch_watermark = 0
        self._completed_dispatch_ranges: list[tuple[int, int]] = []

    def register_work(self, request_id: str, dispatch_id: int) -> bool:
        """Register work, returning false for an already-finished dispatch."""
        if dispatch_id < 1:
            raise ValueError("Parallel-stage work dispatch ID must be positive")
        if self._is_completed_dispatch(dispatch_id):
            return False
        active_dispatch_ids = self._active_dispatch_ids.setdefault(request_id, deque())
        if dispatch_id in active_dispatch_ids:
            return False
        if active_dispatch_ids and dispatch_id < active_dispatch_ids[-1]:
            raise RuntimeError(
                f"Request {request_id!r} in a parallel stage received out-of-order "
                f"work dispatch {dispatch_id} after {active_dispatch_ids[-1]}"
            )
        active_dispatch_ids.append(dispatch_id)
        return True

    def has_active_work(self, request_id: str) -> bool:
        return bool(self._active_dispatch_ids.get(request_id))

    def first_active_dispatch_id(self, request_id: str) -> int | None:
        active_dispatch_ids = self._active_dispatch_ids.get(request_id)
        return active_dispatch_ids[0] if active_dispatch_ids else None

    def should_ignore_abort(self, request_id: str, dispatch_id: int) -> bool:
        if dispatch_id < 1:
            raise ValueError("Parallel-stage abort dispatch ID must be positive")
        if self._is_completed_dispatch(dispatch_id):
            return True
        active_dispatch_ids = self._active_dispatch_ids.get(request_id)
        if active_dispatch_ids:
            if dispatch_id in active_dispatch_ids:
                return False
            if dispatch_id < active_dispatch_ids[-1]:
                raise RuntimeError(
                    f"Request {request_id!r} in a parallel stage received abort "
                    f"dispatch {dispatch_id} outside active dispatches "
                    f"{list(active_dispatch_ids)}"
                )
        return False

    def record_abort(self, request_id: str, dispatch_id: int | None) -> None:
        if dispatch_id is None:
            return
        if dispatch_id < 1:
            raise ValueError("Parallel-stage abort dispatch ID must be positive")
        self._pending_abort_dispatch_ids.setdefault(request_id, set()).add(dispatch_id)

    def finish_terminal(self, request_id: str) -> int | None:
        """Complete the oldest active dispatch for ``request_id``.

        Scheduler terminal messages carry only a request ID. Schedulers that
        accept sequential reuse of an ID must therefore emit terminal messages
        in dispatch order.
        """
        active_dispatch_ids = self._active_dispatch_ids.get(request_id)
        if not active_dispatch_ids:
            return None
        dispatch_id = active_dispatch_ids.popleft()
        if not active_dispatch_ids:
            self._active_dispatch_ids.pop(request_id, None)
        pending_aborts = self._pending_abort_dispatch_ids.get(request_id)
        if pending_aborts is not None:
            pending_aborts.discard(dispatch_id)
            if not pending_aborts:
                self._pending_abort_dispatch_ids.pop(request_id, None)
        self._record_completed_dispatch(dispatch_id)
        return dispatch_id

    def _record_completed_dispatch(self, dispatch_id: int) -> None:
        if dispatch_id <= self._completed_dispatch_watermark:
            return
        ranges = self._completed_dispatch_ranges
        index = bisect.bisect_left(ranges, (dispatch_id, dispatch_id))
        start = end = dispatch_id
        if index > 0 and ranges[index - 1][1] + 1 >= dispatch_id:
            index -= 1
            start, end = ranges.pop(index)
            end = max(end, dispatch_id)
        while index < len(ranges) and ranges[index][0] <= end + 1:
            next_start, next_end = ranges.pop(index)
            start = min(start, next_start)
            end = max(end, next_end)
        ranges.insert(index, (start, end))

        while ranges and ranges[0][0] == self._completed_dispatch_watermark + 1:
            _, completed_end = ranges.pop(0)
            self._completed_dispatch_watermark = completed_end

    def _is_completed_dispatch(self, dispatch_id: int) -> bool:
        if dispatch_id <= self._completed_dispatch_watermark:
            return True
        ranges = self._completed_dispatch_ranges
        index = bisect.bisect_right(ranges, (dispatch_id, float("inf"))) - 1
        return index >= 0 and ranges[index][0] <= dispatch_id <= ranges[index][1]


class ParallelLeaderFanout:
    """Broadcast leader-owned stage events to TP or SP followers."""

    def __init__(
        self,
        stage_name: str,
        *,
        follower_work_queues: list[Any],
        follower_abort_queues: list[Any],
        follower_admin_result_queues: list[Any] | None = None,
    ) -> None:
        self.stage_name = stage_name
        self._follower_work_queues = list(follower_work_queues)
        self._follower_abort_queues = list(follower_abort_queues)
        self._follower_admin_result_queues = list(follower_admin_result_queues or [])
        self._next_dispatch_id = 1
        self._request_dispatch_ids: dict[str, deque[int]] = {}

    async def fanout_control(
        self,
        msg: (
            ShutdownMessage | ProfilerStartMessage | ProfilerStopMessage | AdminMessage
        ),
    ) -> None:
        for q in self._follower_work_queues:
            q.put_nowait(msg)

    def fanout_work(self, payload: Any) -> int:
        request_id = getattr(payload, "request_id", "")
        if not request_id:
            raise ValueError(
                "Parallel-stage work payload requires a non-empty request_id"
            )
        dispatch_id = self._next_dispatch_id
        self._next_dispatch_id += 1
        self._request_dispatch_ids.setdefault(request_id, deque()).append(dispatch_id)
        msg = ParallelWorkMessage(
            request_id=request_id,
            data=payload,
            dispatch_id=dispatch_id,
        )
        for q in self._follower_work_queues:
            q.put_nowait(msg)
        return dispatch_id

    async def fanout_abort(self, msg: AbortMessage) -> int:
        try:
            dispatch_id = self._request_dispatch_ids[msg.request_id][-1]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"No active stage work dispatch for abort {msg.request_id!r}"
            ) from exc
        follower_msg = ParallelAbortMessage(
            request_id=msg.request_id,
            dispatch_id=dispatch_id,
        )
        for q in self._follower_abort_queues:
            q.put_nowait(follower_msg)
        return dispatch_id

    def acknowledge_terminal(self, request_id: str, dispatch_id: int) -> None:
        dispatch_ids = self._request_dispatch_ids.get(request_id)
        if not dispatch_ids:
            return
        expected_dispatch_id = dispatch_ids[0]
        if dispatch_id != expected_dispatch_id:
            raise RuntimeError(
                f"Request {request_id!r} in a parallel stage reached terminal for "
                f"dispatch {dispatch_id}, expected oldest active dispatch "
                f"{expected_dispatch_id}"
            )
        dispatch_ids.popleft()
        if not dispatch_ids:
            self._request_dispatch_ids.pop(request_id, None)

    async def collect_admin_results(
        self,
        op_id: str,
        *,
        timeout_s: float = 60.0,
    ) -> list[AdminResultMessage]:
        """Collect one admin result from every follower rank."""
        if not self._follower_admin_result_queues:
            return []

        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                None,
                lambda q=q: q.get(timeout=timeout_s),
            )
            for q in self._follower_admin_result_queues
        ]
        raw_results = await asyncio.gather(*tasks)
        results: list[AdminResultMessage] = []
        for msg in raw_results:
            if not isinstance(msg, AdminResultMessage):
                raise ValueError(
                    f"Unexpected parallel follower admin result: {type(msg).__name__}"
                )
            if msg.result.op_id != op_id:
                raise ValueError(
                    "Unexpected parallel follower admin op id: "
                    f"{msg.result.op_id} != {op_id}"
                )
            results.append(msg)
        return results

    def close(self) -> None:
        self._follower_work_queues.clear()
        self._follower_abort_queues.clear()
        self._follower_admin_result_queues.clear()
        self._request_dispatch_ids.clear()


@dataclass(frozen=True)
class ParallelStageContext:
    """Validated rank metadata and leader fanout for one parallel stage."""

    kind: Literal["tp", "sp"]
    rank: int
    size: int
    role: Literal["leader", "follower"]
    fanout: ParallelLeaderFanout | None = None

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("parallel stage size must be at least 2")
        if not 0 <= self.rank < self.size:
            raise ValueError(
                f"parallel stage rank must be in [0, {self.size}), got {self.rank}"
            )
        if self.role == "leader":
            if self.rank != 0:
                raise ValueError("parallel stage leader must use rank 0")
            if self.fanout is None:
                raise ValueError("parallel stage leader requires fanout")
        else:
            if self.rank == 0:
                raise ValueError("parallel stage follower must use rank > 0")
            if self.fanout is not None:
                raise ValueError("parallel stage follower cannot own fanout")


class ParallelFollowerControlPlane:
    """Follower-side control plane backed by multiprocessing queues."""

    def __init__(
        self,
        *,
        stage_name: str,
        recv_endpoint: str = "",
        work_queue: Any,
        abort_queue: Any,
        admin_result_queue: Any | None = None,
    ) -> None:
        self.stage_name = stage_name
        self.recv_endpoint = recv_endpoint
        self._work_queue = work_queue
        self._abort_queue = abort_queue
        self._admin_result_queue = admin_result_queue
        self._closed = False

    async def start(self) -> None:
        logger.info(
            "Parallel follower control plane started for stage %s", self.stage_name
        )

    async def recv(
        self,
    ) -> (
        AdminMessage
        | ShutdownMessage
        | ProfilerStartMessage
        | ProfilerStopMessage
        | ParallelWorkMessage
    ):
        msg = await self._recv_from_queue(self._work_queue)
        if isinstance(
            msg,
            (
                AdminMessage,
                ShutdownMessage,
                ProfilerStartMessage,
                ProfilerStopMessage,
                ParallelWorkMessage,
            ),
        ):
            return msg
        raise ValueError(f"Unexpected parallel follower work message: {type(msg)}")

    async def recv_abort(self) -> AbortMessage | ParallelAbortMessage:
        msg = await self._recv_from_queue(self._abort_queue)
        if isinstance(msg, (AbortMessage, ParallelAbortMessage)):
            return msg
        raise ValueError(f"Unexpected parallel follower abort message: {type(msg)}")

    async def send_admin_result(self, msg: AdminResultMessage) -> None:
        if self._admin_result_queue is None:
            raise RuntimeError(
                f"Parallel follower stage {self.stage_name} has no admin result queue"
            )
        self._admin_result_queue.put_nowait(msg)

    async def _recv_from_queue(self, q: Any) -> Any:
        loop = asyncio.get_running_loop()
        while True:
            if self._closed:
                raise RuntimeError(
                    "Parallel follower control plane closed for stage "
                    f"{self.stage_name}"
                )
            try:
                return await loop.run_in_executor(
                    None,
                    lambda: q.get(timeout=_WORK_POLL_SECONDS),
                )
            except queue_mod.Empty:
                continue

    def close(self) -> None:
        self._closed = True
