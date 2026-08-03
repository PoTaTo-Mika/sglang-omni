# SPDX-License-Identifier: Apache-2.0
"""Model-neutral contracts for page-oriented KV cache movement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

from sglang_omni.proto import KVBufferSpec, KVPoolLayout, KVTransferPrepareMessage


@dataclass(frozen=True)
class KVBufferRegion:
    """One page-addressable tensor in a registered KV pool."""

    name: str
    tensor: torch.Tensor
    bytes_per_page: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("KV buffer region name must not be empty")
        if not self.tensor.is_contiguous():
            raise ValueError(f"KV buffer region {self.name!r} must be contiguous")
        if self.bytes_per_page <= 0:
            raise ValueError("KV buffer bytes_per_page must be positive")
        if self.tensor.numel() * self.tensor.element_size() % self.bytes_per_page:
            raise ValueError(
                f"KV buffer region {self.name!r} byte length must be divisible "
                "by bytes_per_page"
            )

    @property
    def page_count(self) -> int:
        return (
            int(self.tensor.numel()) * int(self.tensor.element_size())
        ) // self.bytes_per_page

    def byte_view(self) -> torch.Tensor:
        return self.tensor.view(torch.uint8).reshape(-1)


@dataclass(frozen=True)
class KVPool:
    """Long-lived local KV storage registered once with transfer backends."""

    pool_id: str
    layout_id: str
    page_size: int
    buffers: tuple[KVBufferRegion, ...]

    def __post_init__(self) -> None:
        if not self.pool_id:
            raise ValueError("KV pool_id must not be empty")
        if not self.layout_id:
            raise ValueError("KV pool layout_id must not be empty")
        if self.page_size <= 0:
            raise ValueError("KV pool page_size must be positive")
        if not self.buffers:
            raise ValueError("KV pool requires at least one buffer")
        names = [buffer.name for buffer in self.buffers]
        if len(set(names)) != len(names):
            raise ValueError("KV pool buffer names must be unique")
        devices = {buffer.tensor.device for buffer in self.buffers}
        if len(devices) != 1:
            raise ValueError("all KV pool buffers must live on the same device")

    @property
    def device(self) -> torch.device:
        return self.buffers[0].tensor.device

    @property
    def layout(self) -> KVPoolLayout:
        return KVPoolLayout(
            layout_id=self.layout_id,
            page_size=self.page_size,
            buffers=tuple(
                KVBufferSpec(
                    name=buffer.name,
                    bytes_per_page=buffer.bytes_per_page,
                )
                for buffer in self.buffers
            ),
        )

    def validate_page_indices(self, page_indices: tuple[int, ...]) -> None:
        if not page_indices:
            raise ValueError("KV page indices must not be empty")
        if min(page_indices) < 0:
            raise ValueError("KV page indices must be non-negative")
        max_page_index = max(page_indices)
        for buffer in self.buffers:
            if max_page_index >= buffer.page_count:
                raise ValueError(
                    f"KV page index {max_page_index} exceeds buffer "
                    f"{buffer.name!r} capacity {buffer.page_count}"
                )


@dataclass(frozen=True)
class KVPageDestination:
    """Receiver-owned allocation reserved for one incoming transfer."""

    pool_id: str
    page_indices: tuple[int, ...]


class KVReceiver(Protocol):
    """Request-scoped allocation lifecycle owned by the receiving stage.

    A successful ``commit`` transfers responsibility for releasing the pages
    from the transfer engine to the receiver's normal request lifecycle.
    """

    def reserve(self, request: KVTransferPrepareMessage) -> KVPageDestination: ...

    def commit(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination,
    ) -> None: ...

    def abort(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination | None,
        error: BaseException,
    ) -> None: ...


class KVPageLease(Protocol):
    """Pins source pages until the receiver acknowledges the transfer."""

    def release(self) -> None: ...


@runtime_checkable
class KVTransferRelay(Protocol):
    """Optional paged-transfer capability implemented by a physical relay."""

    def register_kv_pool(self, pool: KVPool) -> None: ...

    def prepare_kv_destination(self, pool_id: str) -> dict[str, Any]: ...

    async def put_kv_pages(
        self,
        *,
        source_pool_id: str,
        source_page_indices: tuple[int, ...],
        destination_ref: dict[str, Any],
    ) -> Any: ...

    async def get_kv_pages(
        self,
        metadata: dict[str, Any],
        *,
        destination_pool_id: str,
        source_page_indices: tuple[int, ...],
        destination_page_indices: tuple[int, ...],
        request_id: str,
    ) -> Any: ...
