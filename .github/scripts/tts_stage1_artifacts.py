# SPDX-License-Identifier: Apache-2.0
"""Versioned JSON contracts for TTS Stage 1 CI evidence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SUPPORTED_TOPOLOGIES = frozenset({"multi_gpu", "mps_shared"})
MPS_DISABLED_REASON = "mps_disabled_for_topology"


def _validate_topology(topology: str) -> None:
    if topology not in SUPPORTED_TOPOLOGIES:
        raise ValueError(f"Unsupported TTS Stage 1 topology: {topology!r}")


def artifact_envelope(
    *,
    artifact_name: str,
    topology: str,
    mps_only: bool,
) -> dict[str, Any]:
    """Return the initial contract for one Stage 1 artifact."""

    _validate_topology(topology)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_name": artifact_name,
        "topology": topology,
    }
    if mps_only and topology == "multi_gpu":
        payload.update(
            status="not_applicable",
            reason_code=MPS_DISABLED_REASON,
        )
    else:
        payload["status"] = "required"
    return payload


def verdict_envelope(
    *,
    artifact_name: str,
    topology: str,
    status: str,
    clean: bool | None = None,
    reason_code: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned verdict without inventing topology-specific evidence."""

    payload = artifact_envelope(
        artifact_name=artifact_name,
        topology=topology,
        mps_only=False,
    )
    payload["status"] = status
    if clean is not None:
        payload["clean"] = clean
    if reason_code is not None:
        payload["reason_code"] = reason_code
    if details:
        payload["details"] = details
    return payload


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """Write one JSON artifact atomically with deterministic formatting."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
