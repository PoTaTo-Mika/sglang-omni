# SPDX-License-Identifier: Apache-2.0
"""TTS Stage 1 cleanup, evaluator barrier, and lane-release evidence.

Repository-side producer only: release requires runner-injected, verified lease
identity and a deployed host-side quarantine consumer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from tts_stage1_artifacts import SCHEMA_VERSION, write_json_atomic

SUPPORTED_TOPOLOGIES = frozenset({"mps_shared", "multi_gpu"})
FAILURE_INJECTION_ENV = "TTS_STAGE1_FAILURE_INJECTION"
LEASE_ID_ENV = "TTS_STAGE1_LANE_LEASE_ID"
LEASE_VERIFIED_ENV = "TTS_STAGE1_LANE_LEASE_VERIFIED"
CONSUMER_STATUS_ENV = "TTS_RUNNER_QUARANTINE_STATUS"
CONSUMER_VERSION_ENV = "TTS_RUNNER_QUARANTINE_CONSUMER_VERSION"
CONSUMER_OWNER_ENV = "TTS_RUNNER_QUARANTINE_OWNER"


def _now_ns() -> int:
    return time.time_ns()


def _json_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"required lifecycle artifact is unreadable: {path}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"invalid lifecycle artifact schema: {path}")
    return payload


def _read_or_missing(path: Path, *, topology: str) -> dict[str, Any]:
    try:
        return _read_json(path)
    except RuntimeError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "topology": topology,
            "status": "missing_or_invalid",
            "clean": False,
            "artifact": path.name,
            "error": str(exc),
        }


def runner_feasibility(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    status = (env.get(CONSUMER_STATUS_ENV) or "blocked").strip()
    if status not in {"blocked", "deployable", "deployed"}:
        raise ValueError(f"unsupported runner quarantine status: {status!r}")
    version = (env.get(CONSUMER_VERSION_ENV) or "").strip() or None
    owner = (env.get(CONSUMER_OWNER_ENV) or "").strip() or None
    if status == "deployed" and (version is None or owner is None):
        raise ValueError(
            "deployed runner quarantine requires consumer version and owner"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "checked_at_ns": _now_ns(),
        "evidence": {
            "repository_consumer_found": False,
            "runner_injected_consumer_version": version,
            "runner_injected_owner": owner,
        },
        "missing_capabilities": (
            []
            if status == "deployed"
            else [
                "exclusive_lane_lease_verification",
                "host_visible_verdict_consumer",
                "dirty_lane_lease_rejection",
                "independent_recovery_verification",
            ]
        ),
        "responsible_owner": {
            "role": "sglang_self_hosted_runner_infrastructure_owner",
            "confirmed_identity": owner,
        },
        "required_action": (
            "exercise quarantine failure injection on the deployed consumer"
            if status == "deployed"
            else "deploy a versioned host-side verdict consumer and identify its owner"
        ),
        "promotion_eligible": status == "deployed",
    }


def _parse_csv_rows(stdout: str, fields: Sequence[str]) -> list[dict[str, str]]:
    rows = []
    for raw in stdout.splitlines():
        values = [item.strip() for item in raw.split(",")]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values)))
    return rows


def capture_gpu_state(*, command_runner: Any = subprocess.run) -> dict[str, Any]:
    gpu = command_runner(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    clients = command_runner(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "pass" if gpu.returncode == 0 and clients.returncode == 0 else "error"
        ),
        "captured_at_ns": _now_ns(),
        "gpus": _parse_csv_rows(
            gpu.stdout, ("index", "uuid", "name", "driver_version")
        ),
        "compute_clients": _parse_csv_rows(
            clients.stdout, ("pid", "gpu_uuid", "process_name")
        ),
        "commands": {
            "gpu_exit_code": gpu.returncode,
            "client_exit_code": clients.returncode,
            "gpu_stderr": gpu.stderr,
            "client_stderr": clients.stderr,
        },
    }


def lane_context(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    lease_id = (env.get(LEASE_ID_ENV) or "").strip() or None
    verified = (env.get(LEASE_VERIFIED_ENV) or "").strip().lower() == "true"
    return {
        "runner_name": (env.get("RUNNER_NAME") or "").strip() or None,
        "run_id": (env.get("GITHUB_RUN_ID") or "").strip() or None,
        "run_attempt": (env.get("GITHUB_RUN_ATTEMPT") or "").strip() or None,
        "job": (env.get("GITHUB_JOB") or "").strip() or None,
        "lane_lease_id": lease_id,
        "lane_lease_verified": bool(lease_id and verified),
    }


def initialize_lifecycle(
    output_dir: str | Path,
    *,
    topology: str,
    model: str,
    environ: Mapping[str, str] | None = None,
    command_runner: Any = subprocess.run,
) -> None:
    if topology not in SUPPORTED_TOPOLOGIES:
        raise ValueError(f"unsupported topology: {topology!r}")
    root = Path(output_dir)
    baseline = capture_gpu_state(command_runner=command_runner)
    baseline.update(topology=topology, model=model, lane_context=lane_context(environ))
    write_json_atomic(
        root / "runner_quarantine_feasibility.json", runner_feasibility(environ)
    )
    write_json_atomic(root / "baseline_gpu.json", baseline)


def build_mps_teardown_verdict(
    *,
    run_id: str,
    router_stopped: bool,
    requests_drained_or_cancelled: bool,
    down_exit_code: int,
    process_query_succeeded: bool,
    tracked_processes_alive: Sequence[int],
    mps_control_pid_observed: bool,
    mps_control_alive: bool,
    mps_namespace_active: bool,
    occupied_ports: Sequence[int],
    gpu_client_query_succeeded: bool,
    tracked_gpu_clients_alive: Sequence[int],
    retained_state: str | None,
    failure_injection: str | None = None,
) -> dict[str, Any]:
    checks = {
        "router_stopped": router_stopped,
        "requests_drained_or_cancelled": requests_drained_or_cancelled,
        "launch_down_succeeded": down_exit_code == 0,
        "process_query_succeeded": process_query_succeeded,
        "tracked_pgid_descendants_exited": not tracked_processes_alive,
        "mps_control_pid_observed": mps_control_pid_observed,
        "mps_clients_and_daemon_exited": not mps_control_alive
        and not tracked_gpu_clients_alive,
        "mps_namespace_inactive": not mps_namespace_active,
        "tts_ports_released": not occupied_ports,
        "gpu_client_query_succeeded": gpu_client_query_succeeded,
        "gpu_client_delta_compliant": not tracked_gpu_clients_alive,
    }
    clean = all(checks.values()) and failure_injection != "dirty_mps_teardown"
    return {
        "schema_version": SCHEMA_VERSION,
        "topology": "mps_shared",
        "status": "pass" if clean else "dirty",
        "clean": clean,
        "run_id": run_id,
        "checks": checks,
        "tracked_processes_alive": list(tracked_processes_alive),
        "tracked_gpu_clients_alive": list(tracked_gpu_clients_alive),
        "occupied_ports": list(occupied_ports),
        "failure_injection": failure_injection,
        "retained_state": retained_state,
        "quarantine_required": not clean,
    }


def write_pre_evaluator_cleanup(
    output_dir: str | Path,
    *,
    topology: str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    source_name = (
        "mps_teardown_verdict.json"
        if topology == "mps_shared"
        else "ordinary_teardown_verdict.json"
    )
    source = _read_json(root / source_name)
    clean = (
        source.get("topology") == topology
        and source.get("status") == "pass"
        and source.get("clean") is True
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "topology": topology,
        "status": "pass" if clean else "dirty",
        "clean": clean,
        "source_artifact": source_name,
        "source_sha256": _json_digest(source),
        "evaluator_start_allowed": clean,
        "lane_context": lane_context(environ),
        "quarantine_required": not clean,
    }
    write_json_atomic(root / "pre_evaluator_cleanup_verdict.json", payload)
    return payload


def require_pre_evaluator_clean(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    payload = _read_json(root / "pre_evaluator_cleanup_verdict.json")
    if (
        payload.get("status") != "pass"
        or payload.get("clean") is not True
        or payload.get("evaluator_start_allowed") is not True
    ):
        raise RuntimeError("pre-evaluator cleanup verdict is not clean")
    source = _read_json(root / str(payload.get("source_artifact")))
    if _json_digest(source) != payload.get("source_sha256"):
        raise RuntimeError("pre-evaluator cleanup source verdict changed")
    return payload


def write_evaluator_lifecycle(
    output_dir: str | Path,
    *,
    status: str,
    pid: int | None,
    port: int | None,
    error: str | None = None,
) -> None:
    if status not in {"running", "completed", "error"}:
        raise ValueError(f"unsupported evaluator status: {status}")
    write_json_atomic(
        Path(output_dir) / "evaluator_lifecycle.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "pid": pid,
            "port": port,
            "error": error,
            "recorded_at_ns": _now_ns(),
        },
    )


def build_lane_release_verdict(
    *,
    topology: str,
    teardown: Mapping[str, Any],
    pre_evaluator: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    post_state: Mapping[str, Any],
    feasibility: Mapping[str, Any],
    baseline_lane_context: Mapping[str, Any],
    final_lane_context: Mapping[str, Any],
    failure_injection: str | None = None,
) -> dict[str, Any]:
    lease_id = baseline_lane_context.get("lane_lease_id")
    lease_match = bool(
        lease_id
        and baseline_lane_context.get("lane_lease_verified") is True
        and final_lane_context.get("lane_lease_verified") is True
        and lease_id == final_lane_context.get("lane_lease_id")
    )
    checks = {
        "topology_teardown_clean": teardown.get("clean") is True,
        "pre_evaluator_cleanup_clean": pre_evaluator.get("clean") is True,
        "evaluator_completed_and_clean": evaluator.get("status") == "completed",
        "final_post_state_clean": post_state.get("clean") is True,
        "runner_consumer_deployed": feasibility.get("status") == "deployed",
        "lane_lease_verified_and_matching": lease_match,
    }
    if failure_injection == "lease_mismatch":
        checks["lane_lease_verified_and_matching"] = False
    if failure_injection == "dirty_post_state":
        checks["final_post_state_clean"] = False
    clean = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "topology": topology,
        "status": "pass" if clean else "blocked",
        "clean": clean,
        "release_eligible": clean,
        "checks": checks,
        "lane_lease_id": lease_id,
        "failure_injection": failure_injection,
        "quarantine_required": not clean,
        "required_runner_action": (
            "release matching exclusive lane lease"
            if clean
            else "quarantine lane and perform independent recovery verification"
        ),
    }


def _new_compute_clients(
    baseline: Mapping[str, Any], post: Mapping[str, Any]
) -> list[dict[str, str]]:
    before = {
        (x.get("pid"), x.get("gpu_uuid")) for x in baseline.get("compute_clients", [])
    }
    return [
        x
        for x in post.get("compute_clients", [])
        if (x.get("pid"), x.get("gpu_uuid")) not in before
    ]


def finalize_lifecycle(
    output_dir: str | Path,
    *,
    topology: str,
    model: str,
    exact_sha: str,
    workflow_url: str,
    environ: Mapping[str, str] | None = None,
    command_runner: Any = subprocess.run,
) -> dict[str, Any]:
    root = Path(output_dir)
    baseline = _read_or_missing(root / "baseline_gpu.json", topology=topology)
    feasibility = _read_or_missing(
        root / "runner_quarantine_feasibility.json", topology=topology
    )
    teardown_name = (
        "mps_teardown_verdict.json"
        if topology == "mps_shared"
        else "ordinary_teardown_verdict.json"
    )
    teardown = _read_or_missing(root / teardown_name, topology=topology)
    pre = _read_or_missing(
        root / "pre_evaluator_cleanup_verdict.json", topology=topology
    )
    evaluator = _read_or_missing(root / "evaluator_lifecycle.json", topology=topology)
    post = capture_gpu_state(command_runner=command_runner)
    new_clients = _new_compute_clients(baseline, post)
    post.update(
        topology=topology,
        clean=baseline.get("status") == "pass"
        and post["status"] == "pass"
        and not new_clients,
        new_compute_clients=new_clients,
    )
    env = os.environ if environ is None else environ
    injection = (env.get(FAILURE_INJECTION_ENV) or "").strip() or None
    if injection == "dirty_post_state":
        post.update(clean=False, failure_injection=injection)
    write_json_atomic(root / "post_state.json", post)
    final_context = lane_context(env)
    write_json_atomic(
        root / "artifact_index.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "indexed",
            "evidence_complete": all(
                item.get("status") not in {"missing_or_invalid", "error"}
                for item in (baseline, feasibility, teardown, pre, evaluator, post)
            ),
            "exact_sha": exact_sha,
            "workflow_url": workflow_url,
            "model": model,
            "topology": topology,
            "run_attempt": final_context.get("run_attempt"),
            "cleanup": {
                "source": teardown_name,
                "clean": teardown.get("clean"),
                "pre_evaluator_clean": pre.get("clean"),
            },
            "evaluator": evaluator,
            "threshold_provenance": "pending_h100_calibration",
            "retained_state": teardown.get("retained_state"),
            "runner_quarantine_status": feasibility.get("status"),
        },
    )
    verdict = build_lane_release_verdict(
        topology=topology,
        teardown=teardown,
        pre_evaluator=pre,
        evaluator=evaluator,
        post_state=post,
        feasibility=feasibility,
        baseline_lane_context=baseline.get("lane_context") or {},
        final_lane_context=final_context,
        failure_injection=injection,
    )
    write_json_atomic(root / "lane_release_verdict.json", verdict)
    return verdict


def _workflow_url(env: Mapping[str, str]) -> str:
    server = (env.get("GITHUB_SERVER_URL") or "https://github.com").rstrip("/")
    repository = (env.get("GITHUB_REPOSITORY") or "").strip()
    run_id = (env.get("GITHUB_RUN_ID") or "").strip()
    return (
        f"{server}/{repository}/actions/runs/{run_id}" if repository and run_id else ""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("initialize")
    initialize.add_argument("--output-dir", required=True)
    initialize.add_argument("--topology", required=True)
    initialize.add_argument("--model", required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--output-dir", required=True)
    finalize.add_argument("--topology", required=True)
    finalize.add_argument("--model", required=True)
    finalize.add_argument("--exact-sha", required=True)
    args = parser.parse_args()
    if args.command == "initialize":
        initialize_lifecycle(args.output_dir, topology=args.topology, model=args.model)
        return
    verdict = finalize_lifecycle(
        args.output_dir,
        topology=args.topology,
        model=args.model,
        exact_sha=args.exact_sha,
        workflow_url=_workflow_url(os.environ),
    )
    if verdict["clean"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
