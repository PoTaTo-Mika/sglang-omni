# SPDX-License-Identifier: Apache-2.0
"""Ordinary multi-GPU TTS Stage 1 lifecycle evidence."""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from tts_stage1_artifacts import SCHEMA_VERSION, write_json_atomic
from tts_stage1_lifecycle import (
    FAILURE_INJECTION_ENV,
    build_ordinary_teardown_verdict,
    lane_context,
    write_pre_evaluator_cleanup,
)

TOPOLOGY = "multi_gpu"


def read_process_groups(manifest: str | Path) -> tuple[list[int], bool]:
    path = Path(manifest)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], False
    groups: set[int] = set()
    valid = True
    for line in lines:
        try:
            pgid = int(line.strip())
        except ValueError:
            valid = False
            continue
        if pgid <= 0:
            valid = False
            continue
        groups.add(pgid)
    return sorted(groups), valid and bool(groups)


def _process_tree(
    process_groups: Sequence[int],
    *,
    parse_succeeded: bool,
    command_runner: Any = subprocess.run,
) -> dict[str, Any]:
    groups = []
    query_succeeded = parse_succeeded
    for pgid in process_groups:
        result = command_runner(
            ["ps", "-o", "pid=,ppid=,pgid=,stat=,comm=", "-g", str(pgid)],
            check=False,
            capture_output=True,
            text=True,
        )
        query_succeeded = query_succeeded and result.returncode == 0
        groups.append(
            {
                "tracked_pgid": pgid,
                "ps_exit_code": result.returncode,
                "processes": result.stdout.splitlines(),
                "stderr": result.stderr,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "topology": TOPOLOGY,
        "status": "pass" if query_succeeded else "error",
        "process_manifest_parse_succeeded": parse_succeeded,
        "groups": groups,
    }


def write_startup_artifacts(
    output_dir: str | Path,
    *,
    model: str,
    router_pid: int,
    router_port: int,
    worker_ports: Sequence[int],
    num_gpus_per_worker: int,
    router_ready_s: float | None,
    cleanup_manifest: str | Path,
    before_workers: Mapping[str, Any],
    command_runner: Any = subprocess.run,
) -> None:
    root = Path(output_dir)
    process_groups, parse_succeeded = read_process_groups(cleanup_manifest)
    replicas = [
        {
            "replica_id": index,
            "port": port,
            "gpu_slots": list(
                range(
                    index * num_gpus_per_worker,
                    (index + 1) * num_gpus_per_worker,
                )
            ),
            "scheduler_and_kv_scope": "worker_local",
        }
        for index, port in enumerate(worker_ports)
    ]
    write_json_atomic(
        root / "launch_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "topology": TOPOLOGY,
            "status": "pass",
            "model": model,
            "router": {"pid": router_pid, "port": router_port},
            "worker_count": len(replicas),
            "num_gpus_per_worker": num_gpus_per_worker,
            "router_ready_s": router_ready_s,
            "cleanup_manifest": str(cleanup_manifest),
        },
    )
    write_json_atomic(
        root / "replica_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "topology": TOPOLOGY,
            "status": "pass",
            "replicas": replicas,
        },
    )
    write_json_atomic(
        root / "router_workers_before.json",
        {
            "schema_version": SCHEMA_VERSION,
            "topology": TOPOLOGY,
            "status": "pass",
            "snapshot": dict(before_workers),
        },
    )
    write_json_atomic(
        root / "process_tree.json",
        _process_tree(
            process_groups,
            parse_succeeded=parse_succeeded,
            command_runner=command_runner,
        ),
    )


def write_router_after(
    output_dir: str | Path,
    *,
    snapshot: Mapping[str, Any] | None,
    error: str | None = None,
) -> None:
    write_json_atomic(
        Path(output_dir) / "router_workers_after.json",
        {
            "schema_version": SCHEMA_VERSION,
            "topology": TOPOLOGY,
            "status": "pass" if snapshot is not None and error is None else "error",
            "snapshot": dict(snapshot or {}),
            "error": error,
        },
    )


def _artifact_pass(path: Path) -> bool:
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("topology") == TOPOLOGY
        and payload.get("status") == "pass"
    )


def _process_group_alive(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass

    try:
        status = subprocess.run(
            ["ps", "-o", "stat=", "-g", str(process_group_id)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return True
    if status.returncode != 0:
        return True
    # A group containing only zombies owns no GPU/port resources. Match the
    # production launcher while keeping process-status query failures dirty.
    return any(
        state.strip() and not state.lstrip().startswith("Z")
        for state in status.stdout.splitlines()
    )


def _occupied_ports(ports: Sequence[int]) -> list[int]:
    occupied = []
    for port in sorted(set(ports)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                occupied.append(port)
    return occupied


def finalize_ordinary_teardown(
    output_dir: str | Path,
    *,
    cleanup_manifest: str | Path,
    router_port: int,
    worker_ports: Sequence[int],
    router_stopped: bool,
    requests_drained_or_cancelled: bool,
    gpu_memory_released: bool,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    process_groups, process_query_succeeded = read_process_groups(cleanup_manifest)
    alive = sorted(pgid for pgid in process_groups if _process_group_alive(pgid))
    occupied = _occupied_ports([router_port, *worker_ports])
    env = os.environ if environ is None else environ
    injection = (env.get(FAILURE_INJECTION_ENV) or "").strip() or None
    retained_state = (
        str(cleanup_manifest)
        if alive or occupied or not process_query_succeeded or injection
        else None
    )
    startup_evidence_clean = all(
        _artifact_pass(root / name)
        for name in (
            "launch_manifest.json",
            "replica_manifest.json",
            "router_workers_before.json",
            "process_tree.json",
        )
    )
    router_attribution_recorded = _artifact_pass(root / "router_workers_after.json")
    verdict = build_ordinary_teardown_verdict(
        router_stopped=router_stopped,
        requests_drained_or_cancelled=requests_drained_or_cancelled,
        process_query_succeeded=process_query_succeeded,
        tracked_process_groups_alive=alive,
        occupied_ports=occupied,
        gpu_memory_released=gpu_memory_released,
        retained_state=retained_state,
        startup_evidence_clean=startup_evidence_clean,
        router_attribution_recorded=router_attribution_recorded,
        failure_injection=injection,
    )
    verdict["lane_context"] = lane_context(env)
    write_json_atomic(root / "ordinary_teardown_verdict.json", verdict)
    write_pre_evaluator_cleanup(root, topology=TOPOLOGY, environ=env)
    if verdict["clean"] is not True:
        raise RuntimeError(
            "ordinary Stage 1 teardown is dirty; evaluator is blocked and "
            f"state is retained at {retained_state or root}"
        )
    return verdict
