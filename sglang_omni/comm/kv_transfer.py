# SPDX-License-Identifier: Apache-2.0
"""Model-neutral contracts for page-oriented KV cache movement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from sglang_omni.proto import KVBufferSpec, KVPoolLayout, KVTransferPrepareMessage


@dataclass(frozen=True)
class KVBufferRegion:
    """One page-addressable tensor in a registered KV pool."""

    name: str
    tensor: torch.Tensor
    bytes_per_page: int

    @property
    def page_count(self) -> int:
        return (self.tensor.numel() * self.tensor.element_size()) // self.bytes_per_page

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
