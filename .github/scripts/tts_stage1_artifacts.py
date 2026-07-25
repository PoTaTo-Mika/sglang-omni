# SPDX-License-Identifier: Apache-2.0
"""Versioned JSON contracts for TTS Stage 1 CI evidence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SUPPORTED_TOPOLOGIES = frozenset({"multi_gpu", "mps_shared"})
MPS_DISABLED_REASON = "mps_disabled_for_topology"
MPS_ONLY_JSON_ARTIFACTS = (
    "mps_status.json",
    "mps_attachment.json",
    "weight_share_audit.json",
    "private_tensor_audit.json",
    "equal_kv.json",
    "mps_teardown_verdict.json",
)
MPS_ONLY_TEXT_ARTIFACTS = ("raw_mps_control.txt",)
REQUIRED_JSON_ARTIFACTS = (
    "runner_quarantine_feasibility.json",
    "preflight.json",
    "baseline_gpu.json",
    "launch_manifest.json",
    "replica_manifest.json",
    *MPS_ONLY_JSON_ARTIFACTS,
    "router_workers_before.json",
    "router_workers_after.json",
    "overlap_verdict.json",
    "normal_correctness.json",
    "high_concurrency_correctness.json",
    "performance_observation.json",
    "stage1_timing_breakdown.json",
    "process_tree.json",
    "ordinary_teardown_verdict.json",
    "pre_evaluator_cleanup_verdict.json",
    "post_state.json",
    "lane_release_verdict.json",
    "artifact_index.json",
)
REQUIRED_NON_JSON_ARTIFACTS = (*MPS_ONLY_TEXT_ARTIFACTS, "replica_activity.jsonl")


def _validate_topology(topology: str) -> None:
    if topology not in SUPPORTED_TOPOLOGIES:
        raise ValueError(f"Unsupported TTS Stage 1 topology: {topology!r}")


def _is_mps_only(artifact_name: str) -> bool:
    return artifact_name in MPS_ONLY_JSON_ARTIFACTS + MPS_ONLY_TEXT_ARTIFACTS


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
        mps_only=_is_mps_only(artifact_name),
    )
    payload["status"] = status
    if clean is not None:
        payload["clean"] = clean
    if reason_code is not None:
        payload["reason_code"] = reason_code
    if details:
        payload["details"] = details
    return payload


def artifact_manifest(topology: str) -> dict[str, Any]:
    """Describe every required artifact without fabricating runtime evidence."""

    _validate_topology(topology)
    names = REQUIRED_JSON_ARTIFACTS + REQUIRED_NON_JSON_ARTIFACTS
    return {
        "schema_version": SCHEMA_VERSION,
        "topology": topology,
        "status": "contract",
        "artifacts": [
            artifact_envelope(
                artifact_name=name,
                topology=topology,
                mps_only=_is_mps_only(name),
            )
            for name in names
        ],
    }


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


def initialize_artifact_contracts(output_dir: str | Path, *, topology: str) -> None:
    """Write the manifest and honest N/A files for the selected topology."""

    root = Path(output_dir)
    write_json_atomic(root / "artifact_manifest.json", artifact_manifest(topology))
    if topology != "multi_gpu":
        return
    for artifact_name in MPS_ONLY_JSON_ARTIFACTS:
        write_json_atomic(
            root / artifact_name,
            artifact_envelope(
                artifact_name=artifact_name,
                topology=topology,
                mps_only=True,
            ),
        )
    for artifact_name in MPS_ONLY_TEXT_ARTIFACTS:
        payload = artifact_envelope(
            artifact_name=artifact_name,
            topology=topology,
            mps_only=True,
        )
        (root / artifact_name).write_text(
            "\n".join(f"{key}={value}" for key, value in payload.items()) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--output-dir", required=True)
    initialize.add_argument("--topology", required=True)
    args = parser.parse_args()
    if args.command == "initialize":
        initialize_artifact_contracts(args.output_dir, topology=args.topology)


if __name__ == "__main__":
    main()
