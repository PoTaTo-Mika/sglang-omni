# SPDX-License-Identifier: Apache-2.0
"""Control-plane types for page-oriented KV cache transfers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import msgspec


@dataclass(frozen=True)
class KVBufferSpec:
    """Wire-visible shape of one independently registered KV buffer."""

    name: str
    bytes_per_page: int

    def __post_init__(self) -> None:
        _require_str(self.name, "name")
        _require_positive_int(self.bytes_per_page, "bytes_per_page")


@dataclass(frozen=True)
class KVPoolLayout:
    """Layout contract that source and destination pools must share."""

    layout_id: str
    page_size: int
    buffers: tuple[KVBufferSpec, ...]

    def __post_init__(self) -> None:
        _require_str(self.layout_id, "layout_id")
        _require_positive_int(self.page_size, "page_size")
        if not isinstance(self.buffers, tuple):
            raise TypeError("buffers must be tuple[KVBufferSpec, ...]")
        if not self.buffers:
            raise ValueError("KV pool layout requires at least one buffer")
        if any(not isinstance(buffer, KVBufferSpec) for buffer in self.buffers):
            raise TypeError("buffers must contain KVBufferSpec values")
        names = [buffer.name for buffer in self.buffers]
        if len(set(names)) != len(names):
            raise ValueError("KV pool layout buffer names must be unique")

    def compatible_with(self, other: "KVPoolLayout") -> bool:
        """Return whether pages can be copied without a layout transform."""

        if self.layout_id != other.layout_id or self.page_size != other.page_size:
            return False
        if len(self.buffers) != len(other.buffers):
            return False
        return all(
            source.name == destination.name
            and source.bytes_per_page == destination.bytes_per_page
            for source, destination in zip(self.buffers, other.buffers, strict=True)
        )


@dataclass(frozen=True)
class KVTransferPrepareMessage:
    """Ask a receiver to reserve destination pages for one transfer."""

    request_id: str
    transfer_id: str
    from_stage: str
    to_stage: str
    source_pool_id: str
    target_pool_id: str
    source_page_indices: tuple[int, ...]
    source_layout: KVPoolLayout
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_common_message_fields(
            request_id=self.request_id,
            transfer_id=self.transfer_id,
            from_stage=self.from_stage,
            to_stage=self.to_stage,
        )
        _require_str(self.source_pool_id, "source_pool_id")
        _require_str(self.target_pool_id, "target_pool_id")
        _validate_page_indices(self.source_page_indices, "source_page_indices")
        if not isinstance(self.source_layout, KVPoolLayout):
            raise TypeError("source_layout must be KVPoolLayout")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be dict")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "kv_transfer_prepare", **msgspec.to_builtins(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KVTransferPrepareMessage":
        return msgspec.convert(value, type=cls, strict=True)


@dataclass(frozen=True)
class KVTransferReadyMessage:
    """Return a reserved destination or a prepare failure to the sender."""

    request_id: str
    transfer_id: str
    from_stage: str
    to_stage: str
    success: bool
    destination_pool_id: str | None = None
    destination_page_indices: tuple[int, ...] = ()
    destination_ref: dict[str, Any] | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        _validate_common_message_fields(
            request_id=self.request_id,
            transfer_id=self.transfer_id,
            from_stage=self.from_stage,
            to_stage=self.to_stage,
        )
        if not isinstance(self.success, bool):
            raise TypeError("success must be bool")
        if self.success:
            _require_str(self.destination_pool_id, "destination_pool_id")
            _validate_page_indices(
                self.destination_page_indices, "destination_page_indices"
            )
            if self.destination_ref is None:
                raise TypeError("successful KV transfer ready requires destination_ref")
            _require_dict(self.destination_ref, "destination_ref")
            if self.error is not None:
                raise ValueError("successful KV transfer ready cannot carry error")
        else:
            _require_str(self.error, "error")
            if self.destination_pool_id is not None:
                raise ValueError("failed KV transfer ready cannot carry destination")
            if self.destination_page_indices:
                raise ValueError("failed KV transfer ready cannot carry page indices")
            if self.destination_ref is not None:
                raise ValueError(
                    "failed KV transfer ready cannot carry destination_ref"
                )

    def to_dict(self) -> dict[str, Any]:
        value = {"type": "kv_transfer_ready", **msgspec.to_builtins(self)}
        if self.success:
            value.pop("error")
        else:
            value.pop("destination_pool_id")
            value.pop("destination_page_indices")
            value.pop("destination_ref")
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KVTransferReadyMessage":
        return msgspec.convert(value, type=cls, strict=True)


def _validate_common_message_fields(
    *, request_id: str, transfer_id: str, from_stage: str, to_stage: str
) -> None:
    _require_str(request_id, "request_id")
    _require_str(transfer_id, "transfer_id")
    _require_str(from_stage, "from_stage")
    _require_str(to_stage, "to_stage")


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TypeError(f"{name} must be a non-empty str")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise TypeError(f"{name} must be a positive int")
    return value


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be dict")
    return value


def _validate_page_indices(indices: tuple[int, ...], name: str) -> None:
    if not isinstance(indices, tuple):
        raise TypeError(f"{name} must be tuple[int, ...]")
    if not indices:
        raise ValueError(f"{name} must not be empty")
    if any(type(index) is not int or index < 0 for index in indices):
        raise TypeError(f"{name} must contain non-negative ints")
