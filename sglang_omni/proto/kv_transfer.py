# SPDX-License-Identifier: Apache-2.0
"""Control-plane types for page-oriented KV cache transfers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KVBufferSpec:
    """Wire-visible shape of one independently registered KV buffer."""

    name: str
    bytes_per_page: int

    def to_dict(self) -> dict[str, Any]:
        _require_str(self.name, "name")
        _require_positive_int(self.bytes_per_page, "bytes_per_page")
        return {
            "name": self.name,
            "bytes_per_page": self.bytes_per_page,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KVBufferSpec":
        return cls(
            name=_require_str(value.get("name"), "name"),
            bytes_per_page=_require_positive_int(
                value.get("bytes_per_page"), "bytes_per_page"
            ),
        )


@dataclass(frozen=True)
class KVPoolLayout:
    """Layout contract that source and destination pools must share."""

    layout_id: str
    page_size: int
    buffers: tuple[KVBufferSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        _require_str(self.layout_id, "layout_id")
        _require_positive_int(self.page_size, "page_size")
        if not self.buffers:
            raise ValueError("KV pool layout requires at least one buffer")
        names = [buffer.name for buffer in self.buffers]
        if len(set(names)) != len(names):
            raise ValueError("KV pool layout buffer names must be unique")
        return {
            "layout_id": self.layout_id,
            "page_size": self.page_size,
            "buffers": [buffer.to_dict() for buffer in self.buffers],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KVPoolLayout":
        raw_buffers = value.get("buffers")
        if not isinstance(raw_buffers, list):
            raise TypeError("buffers must be list")
        layout = cls(
            layout_id=_require_str(value.get("layout_id"), "layout_id"),
            page_size=_require_positive_int(value.get("page_size"), "page_size"),
            buffers=tuple(
                KVBufferSpec.from_dict(_require_dict(item, "buffer"))
                for item in raw_buffers
            ),
        )
        layout.to_dict()
        return layout

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

    def to_dict(self) -> dict[str, Any]:
        _validate_common_message_fields(
            request_id=self.request_id,
            transfer_id=self.transfer_id,
            from_stage=self.from_stage,
            to_stage=self.to_stage,
        )
        _require_str(self.source_pool_id, "source_pool_id")
        _require_str(self.target_pool_id, "target_pool_id")
        _validate_page_indices(self.source_page_indices, "source_page_indices")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be dict")
        return {
            "type": "kv_transfer_prepare",
            "request_id": self.request_id,
            "transfer_id": self.transfer_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "source_pool_id": self.source_pool_id,
            "target_pool_id": self.target_pool_id,
            "source_page_indices": list(self.source_page_indices),
            "source_layout": self.source_layout.to_dict(),
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KVTransferPrepareMessage":
        message = cls(
            request_id=_require_str(value.get("request_id"), "request_id"),
            transfer_id=_require_str(value.get("transfer_id"), "transfer_id"),
            from_stage=_require_str(value.get("from_stage"), "from_stage"),
            to_stage=_require_str(value.get("to_stage"), "to_stage"),
            source_pool_id=_require_str(value.get("source_pool_id"), "source_pool_id"),
            target_pool_id=_require_str(value.get("target_pool_id"), "target_pool_id"),
            source_page_indices=_page_indices_from_dict(value, "source_page_indices"),
            source_layout=KVPoolLayout.from_dict(
                _require_dict(value.get("source_layout"), "source_layout")
            ),
            metadata=_require_dict(value.get("metadata", {}), "metadata"),
        )
        message.to_dict()
        return message


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

    def to_dict(self) -> dict[str, Any]:
        _validate_common_message_fields(
            request_id=self.request_id,
            transfer_id=self.transfer_id,
            from_stage=self.from_stage,
            to_stage=self.to_stage,
        )
        if not isinstance(self.success, bool):
            raise TypeError("success must be bool")
        value: dict[str, Any] = {
            "type": "kv_transfer_ready",
            "request_id": self.request_id,
            "transfer_id": self.transfer_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "success": self.success,
        }
        if self.success:
            value["destination_pool_id"] = _require_str(
                self.destination_pool_id, "destination_pool_id"
            )
            _validate_page_indices(
                self.destination_page_indices, "destination_page_indices"
            )
            if self.destination_ref is None:
                raise TypeError("successful KV transfer ready requires destination_ref")
            value["destination_page_indices"] = list(self.destination_page_indices)
            value["destination_ref"] = _require_dict(
                self.destination_ref, "destination_ref"
            ).copy()
            if self.error is not None:
                raise ValueError("successful KV transfer ready cannot carry error")
        else:
            value["error"] = _require_str(self.error, "error")
            if self.destination_pool_id is not None:
                raise ValueError("failed KV transfer ready cannot carry destination")
            if self.destination_page_indices:
                raise ValueError("failed KV transfer ready cannot carry page indices")
            if self.destination_ref is not None:
                raise ValueError(
                    "failed KV transfer ready cannot carry destination_ref"
                )
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KVTransferReadyMessage":
        success = value.get("success")
        if not isinstance(success, bool):
            raise TypeError("success must be bool")
        message = cls(
            request_id=_require_str(value.get("request_id"), "request_id"),
            transfer_id=_require_str(value.get("transfer_id"), "transfer_id"),
            from_stage=_require_str(value.get("from_stage"), "from_stage"),
            to_stage=_require_str(value.get("to_stage"), "to_stage"),
            success=success,
            destination_pool_id=(
                _require_str(value.get("destination_pool_id"), "destination_pool_id")
                if success
                else None
            ),
            destination_page_indices=(
                _page_indices_from_dict(value, "destination_page_indices")
                if success
                else ()
            ),
            destination_ref=(
                _require_dict(value.get("destination_ref"), "destination_ref")
                if success
                else None
            ),
            error=(None if success else _require_str(value.get("error"), "error")),
        )
        message.to_dict()
        return message


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


def _page_indices_from_dict(value: dict[str, Any], key: str) -> tuple[int, ...]:
    raw_indices = value.get(key)
    if not isinstance(raw_indices, list):
        raise TypeError(f"{key} must be list[int]")
    indices = tuple(raw_indices)
    _validate_page_indices(indices, key)
    return indices
