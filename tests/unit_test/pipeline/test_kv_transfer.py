# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

from sglang_omni.comm.data_ref import BackendRef, DataKind, DataLayout, DataRef
from sglang_omni.comm.engine import CommEngine
from sglang_omni.comm.kv_transfer import KVBufferRegion, KVPageDestination, KVPool
from sglang_omni.comm.router import CommRouter
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.proto import (
    DataAckMessage,
    DataReadyMessage,
    KVBufferSpec,
    KVPoolLayout,
    KVTransferPrepareMessage,
    KVTransferReadyMessage,
)
from tests.unit_test.fixtures.pipeline_fakes import RecordingStageControlPlane
from tests.unit_test.pipeline.helpers import make_stage


class _KVOp:
    def __init__(self, metadata: dict[str, Any], *, require_ack: bool) -> None:
        self._metadata = metadata
        self.require_ack = require_ack
        self.acked = False
        self.waited = False
        self.failed: BaseException | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def mark_receiver_done(self) -> None:
        self.acked = True

    def mark_receiver_failed(self, exc: BaseException) -> None:
        self.failed = exc

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        del timeout
        self.waited = True
        if self.failed is not None:
            raise self.failed
        if self.require_ack and not self.acked:
            raise RuntimeError("sender operation completed before receiver ACK")


class _PagedRelay:
    device = "cpu"

    def __init__(self) -> None:
        self.pools: dict[str, KVPool] = {}
        self.put_ops: list[_KVOp] = []
        self.generic_put_called = False

    def register_kv_pool(self, pool: KVPool) -> None:
        self.pools[pool.pool_id] = pool

    def prepare_kv_destination(self, pool_id: str, *, sender_id: str) -> dict[str, Any]:
        del sender_id
        pool = self.pools[pool_id]
        return {
            "transfer_info": {
                "size": sum(
                    buffer.tensor.numel() * buffer.tensor.element_size()
                    for buffer in pool.buffers
                )
            },
            "fake_kv": {"pool_id": pool_id},
        }

    async def put_kv_pages(
        self,
        *,
        source_pool_id: str,
        source_page_indices: tuple[int, ...],
        destination_ref: dict[str, Any],
        destination_page_indices: tuple[int, ...],
        request_id: str,
        receiver_id: str,
    ) -> _KVOp:
        del destination_page_indices, request_id, receiver_id
        destination_pool_id = destination_ref["fake_kv"]["pool_id"]
        size = sum(
            buffer.bytes_per_page * len(source_page_indices)
            for buffer in self.pools[source_pool_id].buffers
        )
        op = _KVOp(
            {
                "transfer_info": {"size": size},
                "fake_kv": {
                    "source_pool_id": source_pool_id,
                    "destination_pool_id": destination_pool_id,
                },
            },
            require_ack=True,
        )
        self.put_ops.append(op)
        return op

    async def get_kv_pages(
        self,
        metadata: dict[str, Any],
        *,
        destination_pool_id: str,
        source_page_indices: tuple[int, ...],
        destination_page_indices: tuple[int, ...],
        request_id: str,
    ) -> _KVOp:
        del request_id
        source = self.pools[metadata["fake_kv"]["source_pool_id"]]
        destination = self.pools[destination_pool_id]
        for source_buffer, destination_buffer in zip(
            source.buffers, destination.buffers, strict=True
        ):
            source_bytes = source_buffer.byte_view().reshape(
                source_buffer.page_count, source_buffer.bytes_per_page
            )
            destination_bytes = destination_buffer.byte_view().reshape(
                destination_buffer.page_count,
                destination_buffer.bytes_per_page,
            )
            for source_index, destination_index in zip(
                source_page_indices, destination_page_indices, strict=True
            ):
                destination_bytes[destination_index].copy_(source_bytes[source_index])
        return _KVOp({"transfer_info": {"size": 0}}, require_ack=False)

    async def put_async(self, *args: Any, **kwargs: Any) -> _KVOp:
        del args, kwargs
        self.generic_put_called = True
        raise AssertionError("KV transfer must not use generic put_async")

    async def get_async(self, *args: Any, **kwargs: Any) -> _KVOp:
        del args, kwargs
        raise AssertionError("KV transfer must not use generic get_async")

    def cleanup(self, request_id: str) -> None:
        del request_id

    def close(self) -> None:
        pass


class _Receiver:
    def __init__(self, page_indices: tuple[int, ...]) -> None:
        self.page_indices = page_indices
        self.committed: list[str] = []
        self.aborted: list[str] = []

    def reserve(self, request: KVTransferPrepareMessage) -> KVPageDestination:
        return KVPageDestination(
            pool_id=request.target_pool_id,
            page_indices=self.page_indices,
        )

    def commit(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination,
    ) -> None:
        del destination
        self.committed.append(request.request_id)

    def abort(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination | None,
        error: BaseException,
    ) -> None:
        del destination, error
        self.aborted.append(request.request_id)


class _Lease:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


class _LinkedControlPlane:
    def __init__(self, engines: dict[str, CommEngine]) -> None:
        self.engines = engines
        self.messages: list[Any] = []

    async def send_to_stage(
        self,
        next_stage: str,
        next_stage_endpoint: str,
        message: Any,
    ) -> None:
        del next_stage_endpoint
        self.messages.append(message)
        target = self.engines[next_stage]
        if isinstance(message, KVTransferPrepareMessage):
            ready = target.prepare_kv_receive(
                message,
                relay=target.inbound_relay(message.from_stage),
            )
            await self.send_to_stage(message.from_stage, "unused", ready)
        elif isinstance(message, KVTransferReadyMessage):
            target.kv_transfer_ready(message)
        elif isinstance(message, DataReadyMessage):
            data_ref = DataRef.from_dict(message.data_ref or {})
            result = await target.read_data(
                relay=target.relay(data_ref.transport),
                request_id=message.request_id,
                data_ref=data_ref,
            )
            assert result is None
            self.engines[message.from_stage].ack_transfer(
                DataAckMessage(
                    request_id=message.request_id,
                    from_stage=message.to_stage,
                    to_stage=message.from_stage,
                    object_id=data_ref.object_id,
                )
            )


def _engine(stage_name: str, peer_name: str, relay: _PagedRelay) -> CommEngine:
    return CommEngine(
        CommRouter(
            stage_name=stage_name,
            gpu_id=None,
            same_process_targets=set(),
            gpu_stage_names=set(),
            stage_gpu_ids={peer_name: ()},
            injected_relay=relay,
            comm_config={"ack_timeout_s": 1.0},
        )
    )


def _pool(pool_id: str, tensor: torch.Tensor) -> KVPool:
    return KVPool(
        pool_id=pool_id,
        layout_id="NHD",
        page_size=1,
        buffers=(
            KVBufferRegion(
                name="layer.0.kv",
                tensor=tensor,
                bytes_per_page=int(tensor[0].numel() * tensor.element_size()),
            ),
        ),
    )


def test_kv_control_messages_round_trip() -> None:
    layout = KVPoolLayout(
        layout_id="NHD",
        page_size=1,
        buffers=(KVBufferSpec("layer.0.kv", bytes_per_page=8),),
    )
    prepare = KVTransferPrepareMessage(
        request_id="request",
        transfer_id="transfer",
        from_stage="prefill",
        to_stage="decode",
        source_pool_id="source",
        target_pool_id="destination",
        source_page_indices=(1, 4),
        source_layout=layout,
        metadata={"sequence_length": 2},
    )
    assert deserialize_message(serialize_message(prepare)) == prepare

    ready = KVTransferReadyMessage(
        request_id="request",
        transfer_id="transfer",
        from_stage="decode",
        to_stage="prefill",
        success=True,
        destination_pool_id="destination",
        destination_page_indices=(3, 5),
        destination_ref={
            "transport": "shm",
            "info": {"transfer_info": {"size": 16}},
            "length": 16,
        },
    )
    assert deserialize_message(serialize_message(ready)) == ready


def test_kv_transfer_reserves_copies_commits_and_releases() -> None:
    async def _run() -> None:
        relay = _PagedRelay()
        source_engine = _engine("source", "destination", relay)
        destination_engine = _engine("destination", "source", relay)
        control_plane = _LinkedControlPlane(
            {"source": source_engine, "destination": destination_engine}
        )

        source_tensor = torch.arange(24, dtype=torch.uint8).reshape(6, 4)
        destination_tensor = torch.zeros_like(source_tensor)
        source_engine.register_kv_pool(_pool("source_pool", source_tensor))
        destination_engine.register_kv_pool(
            _pool("destination_pool", destination_tensor)
        )
        receiver = _Receiver((0, 3))
        destination_engine.register_kv_receiver("destination_pool", receiver)
        lease = _Lease()

        data_ref = await source_engine.send_kv_pages(
            control_plane=control_plane,
            request_id="request",
            transfer_id="transfer",
            source_pool_id="source_pool",
            source_page_indices=(1, 4),
            target_pool_id="destination_pool",
            from_stage="source",
            to_stage="destination",
            target_endpoint="unused",
            lease=lease,
        )

        assert data_ref.object_id == "transfer"
        assert torch.equal(destination_tensor[0], source_tensor[1])
        assert torch.equal(destination_tensor[3], source_tensor[4])
        assert receiver.committed == ["request"]
        assert receiver.aborted == []
        assert lease.released
        assert relay.put_ops[0].acked
        assert relay.put_ops[0].waited
        assert not relay.generic_put_called
        assert [type(message) for message in control_plane.messages] == [
            KVTransferPrepareMessage,
            KVTransferReadyMessage,
            DataReadyMessage,
        ]

    asyncio.run(_run())


def test_stage_uses_common_data_ready_path_for_kv_pages() -> None:
    async def _run() -> None:
        relay = _PagedRelay()
        control_plane = RecordingStageControlPlane()
        stage = make_stage(
            name="destination",
            endpoints={"source": "unused"},
            relay=relay,
            control_plane=control_plane,
        )
        source_tensor = torch.arange(16, dtype=torch.uint8).reshape(4, 4)
        destination_tensor = torch.zeros_like(source_tensor)
        source_pool = _pool("source_pool", source_tensor)
        destination_pool = _pool("destination_pool", destination_tensor)
        relay.register_kv_pool(source_pool)
        stage._comm.register_kv_pool(destination_pool)
        receiver = _Receiver((2,))
        stage._comm.register_kv_receiver("destination_pool", receiver)
        prepare = KVTransferPrepareMessage(
            request_id="request",
            transfer_id="transfer",
            from_stage="source",
            to_stage="destination",
            source_pool_id="source_pool",
            target_pool_id="destination_pool",
            source_page_indices=(1,),
            source_layout=source_pool.layout,
        )
        ready = stage._comm.prepare_kv_receive(prepare, relay=relay)
        assert ready.success
        destination_ref = BackendRef.from_dict(ready.destination_ref or {})
        op = await relay.put_kv_pages(
            source_pool_id="source_pool",
            source_page_indices=(1,),
            destination_ref=destination_ref.info,
            destination_page_indices=ready.destination_page_indices,
            request_id="request",
            receiver_id="destination",
        )
        data_ref = DataRef(
            version=1,
            object_id="transfer",
            kind=DataKind.KV_PAGES,
            transport=destination_ref.transport,
            layout=DataLayout.PAGED,
            buffer=BackendRef.from_relay_info(
                transport=destination_ref.transport,
                relay_info=op.metadata,
            ),
        )

        await stage._on_data_ready(
            DataReadyMessage(
                request_id="request",
                from_stage="source",
                to_stage="destination",
                data_ref=data_ref.to_dict(),
            )
        )

        assert torch.equal(destination_tensor[2], source_tensor[1])
        assert receiver.committed == ["request"]
        assert stage.scheduler.inbox.empty()
        assert len(control_plane.sent_to_stage) == 1
        _, _, ack = control_plane.sent_to_stage[0]
        assert isinstance(ack, DataAckMessage)
        assert ack.success

    asyncio.run(_run())


def test_kv_transfer_rejects_layout_mismatch_and_releases_lease() -> None:
    async def _run() -> None:
        relay = _PagedRelay()
        source_engine = _engine("source", "destination", relay)
        destination_engine = _engine("destination", "source", relay)
        control_plane = _LinkedControlPlane(
            {"source": source_engine, "destination": destination_engine}
        )
        source_engine.register_kv_pool(
            _pool("source_pool", torch.zeros((4, 4), dtype=torch.uint8))
        )
        destination_engine.register_kv_pool(
            KVPool(
                pool_id="destination_pool",
                layout_id="NHD",
                page_size=1,
                buffers=(
                    KVBufferRegion(
                        name="different",
                        tensor=torch.zeros((4, 4), dtype=torch.uint8),
                        bytes_per_page=4,
                    ),
                ),
            )
        )
        receiver = _Receiver((0,))
        destination_engine.register_kv_receiver("destination_pool", receiver)
        lease = _Lease()

        with pytest.raises(RuntimeError, match="layouts do not match"):
            await source_engine.send_kv_pages(
                control_plane=control_plane,
                request_id="request",
                transfer_id="transfer",
                source_pool_id="source_pool",
                source_page_indices=(1,),
                target_pool_id="destination_pool",
                from_stage="source",
                to_stage="destination",
                target_endpoint="unused",
                lease=lease,
            )

        assert receiver.aborted == ["request"]
        assert lease.released
        assert not relay.put_ops

    asyncio.run(_run())


def test_kv_transfer_releases_lease_when_source_setup_fails() -> None:
    async def _run() -> None:
        relay = _PagedRelay()
        source_engine = _engine("source", "destination", relay)
        lease = _Lease()

        with pytest.raises(KeyError, match="unknown source KV pool"):
            await source_engine.send_kv_pages(
                control_plane=_LinkedControlPlane({"source": source_engine}),
                request_id="request",
                source_pool_id="missing",
                source_page_indices=(0,),
                target_pool_id="destination_pool",
                from_stage="source",
                to_stage="destination",
                target_endpoint="unused",
                lease=lease,
            )

        assert lease.released

    asyncio.run(_run())


def test_kv_cleanup_aborts_reserved_destination() -> None:
    relay = _PagedRelay()
    destination_engine = _engine("destination", "source", relay)
    destination_pool = _pool("destination_pool", torch.zeros((4, 4), dtype=torch.uint8))
    destination_engine.register_kv_pool(destination_pool)
    receiver = _Receiver((2,))
    destination_engine.register_kv_receiver("destination_pool", receiver)
    prepare = KVTransferPrepareMessage(
        request_id="request",
        transfer_id="transfer",
        from_stage="source",
        to_stage="destination",
        source_pool_id="source_pool",
        target_pool_id="destination_pool",
        source_page_indices=(0,),
        source_layout=destination_pool.layout,
    )

    ready = destination_engine.prepare_kv_receive(prepare, relay=relay)
    assert ready.success

    destination_engine.cleanup("request")

    assert receiver.aborted == ["request"]
