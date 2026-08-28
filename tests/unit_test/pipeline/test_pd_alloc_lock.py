# SPDX-License-Identifier: Apache-2.0
"""One lock across the two threads a PD Decode half allocates from.

The allocator's alloc reads the free list, slices it, and writes the
remainder back, with no lock of its own. On a Decode half the scheduler
thread and the comm event loop both call it, and interleaving there hands
the same slots to both.
"""

from __future__ import annotations

import threading

import pytest

from sglang_omni.scheduling.pd_alloc_lock import LockedKVAllocator


class _FakeAllocator:
    """Stands in for the real allocator; only identity matters here."""

    def alloc(self, n):  # pragma: no cover - never called
        return list(range(n))

    def free(self, indices):  # pragma: no cover - never called
        return None


class _RacyAllocator:
    """An allocator with the same read-modify-write shape as upstream's."""

    def __init__(self, size: int) -> None:
        self.free_pages = list(range(size))
        self.page_size = 1
        self.handed_out: list[int] = []

    def alloc(self, need_size: int):
        taken = self.free_pages[:need_size]
        # A thread switch here is what produces the overlap in production.
        threading.current_thread()
        self.free_pages = self.free_pages[need_size:]
        self.handed_out.extend(taken)
        return taken

    def free(self, index) -> None:
        self.free_pages.extend(index)

    def available_size(self) -> int:
        return len(self.free_pages)


def test_two_threads_never_receive_the_same_slot() -> None:
    inner = _RacyAllocator(4096)
    allocator = LockedKVAllocator(inner)
    seen: list[int] = []
    lock = threading.Lock()

    def take() -> None:
        for _ in range(64):
            got = allocator.alloc(4)
            with lock:
                seen.extend(got)

    threads = [threading.Thread(target=take) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seen) == len(set(seen))


def test_everything_else_is_delegated() -> None:
    inner = _RacyAllocator(16)
    allocator = LockedKVAllocator(inner)

    assert allocator.available_size() == 16
    assert allocator.page_size == 1


def test_a_write_reaches_the_wrapped_allocator() -> None:
    """Upstream code sets attributes on the allocator it was handed."""
    inner = _RacyAllocator(16)
    allocator = LockedKVAllocator(inner)

    allocator.page_size = 4

    assert inner.page_size == 4


def test_freeing_returns_the_slots() -> None:
    inner = _RacyAllocator(8)
    allocator = LockedKVAllocator(inner)

    taken = allocator.alloc(8)
    assert allocator.available_size() == 0

    allocator.free(taken)
    assert allocator.available_size() == 8


def test_every_holder_of_the_allocator_gets_the_same_locked_object() -> None:
    """One lock only helps if no caller keeps an unwrapped alias."""
    from types import SimpleNamespace

    from sglang_omni.scheduling.pd_scheduler import (
        OmniDecodeScheduler,
        _serialize_kv_allocation,
        kv_allocator_holders,
    )

    raw = _FakeAllocator()
    scheduler = OmniDecodeScheduler.__new__(OmniDecodeScheduler)
    scheduler.token_to_kv_pool_allocator = raw
    scheduler.tree_cache = SimpleNamespace(token_to_kv_pool_allocator=raw)

    _serialize_kv_allocation(scheduler)

    holders = kv_allocator_holders(scheduler)
    assert set(holders) == {"scheduler", "tree_cache"}
    assert len({id(a) for a in holders.values()}) == 1
    assert all(isinstance(a, LockedKVAllocator) for a in holders.values())


def test_a_holder_that_cannot_be_rebound_is_an_error() -> None:
    """Skipping it silently would leave an alias outside the lock."""
    from types import SimpleNamespace

    from sglang_omni.scheduling.pd_scheduler import (
        OmniDecodeScheduler,
        _serialize_kv_allocation,
    )

    scheduler = OmniDecodeScheduler.__new__(OmniDecodeScheduler)
    scheduler.token_to_kv_pool_allocator = _FakeAllocator()
    scheduler.tree_cache = SimpleNamespace()  # no allocator attribute

    with pytest.raises(RuntimeError, match="would bypass the lock"):
        _serialize_kv_allocation(scheduler)


def test_a_receiver_built_after_the_wrap_gets_the_locked_allocator() -> None:
    """Ordering matters: a receiver built first would hold the raw object."""
    import queue
    from types import SimpleNamespace

    from sglang_omni.scheduling.pd_scheduler import _serialize_kv_allocation
    from sglang_omni.scheduling.pd_utils import DecodeKVReceiver

    inner = _RacyAllocator(64)
    scheduler = SimpleNamespace(
        token_to_kv_pool_allocator=inner,
        tree_cache=SimpleNamespace(token_to_kv_pool_allocator=inner),
    )
    _serialize_kv_allocation(scheduler)

    receiver = DecodeKVReceiver(
        pool_id="thinker_decode:kv",
        allocator=scheduler.token_to_kv_pool_allocator,
        admissions=queue.SimpleQueue(),
        allowed_resume_schemas=frozenset({"v1"}),
    )

    assert isinstance(receiver._allocator, LockedKVAllocator)
