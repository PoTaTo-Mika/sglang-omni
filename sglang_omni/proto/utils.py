# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for typed control-plane messages."""

from __future__ import annotations

from typing import Any, TypeVar

import msgspec

T = TypeVar("T")


def dataclass_to_wire(
    value: object, *, message_type: str | None = None
) -> dict[str, Any]:
    """Convert a typed dataclass and its nested values to wire-safe builtins."""

    payload = msgspec.to_builtins(value)
    if not isinstance(payload, dict):
        raise TypeError("wire object must encode to dict")
    if message_type is None:
        return payload
    return {"type": message_type, **payload}


def dataclass_from_wire(
    value: dict[str, Any], cls: type[T], *, message_type: str | None = None
) -> T:
    """Strictly reconstruct a typed dataclass from control-plane builtins."""

    if not isinstance(value, dict):
        raise TypeError("wire value must be dict")
    if message_type is not None and value.get("type") != message_type:
        raise ValueError(f"expected message type {message_type!r}")
    payload = {key: item for key, item in value.items() if key != "type"}
    return msgspec.convert(payload, type=cls, strict=True)
