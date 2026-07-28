# SPDX-License-Identifier: Apache-2.0
"""Aggregated, fail-closed evidence for TTS Stage 1 CI."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SUPPORTED_TOPOLOGIES = frozenset({"multi_gpu", "mps_shared"})
REQUIRED_CORRECTNESS = ("canonical_generation", "wer", "similarity", "utmos")
# Note (Jiaxin Deng): wide model-specific floors retain enough slack to catch only
# obvious MPS regressions while the exact H100 samples remain auditable here.
MPS_NORMAL_LOAD_FLOORS: dict[str, dict[str, Any]] = {
    "higgs": {
        "concurrency": 16,
        "minimum": {
            "throughput_qps": 10.0,
            "audio_throughput_s_per_s": 43.0,
            "output_throughput": 1100.0,
            "output_tok_per_req_s": 80.0,
        },
        "maximum": {"latency_mean_s": 1.8, "rtf_mean": 0.45},
        "provenance": {
            "temporary": True,
            "source_sha": "fe7e5dd802cb575b754d705232de2ff8b4f1795e",
            "accepted_run_ids": [30196509700, 30199556014],
            "method": "wide_floor_below_worst_of_two_exact_sha_h100_mps_samples",
            "raw_samples": [
                {
                    "run_id": 30196509700,
                    "throughput_qps": 13.985,
                    "latency_mean_s": 1.137,
                    "rtf_mean": 0.2723,
                    "audio_throughput_s_per_s": 59.64,
                    "output_throughput": 1588.9,
                    "output_tok_per_req_s": 111.8,
                },
                {
                    "run_id": 30199556014,
                    "throughput_qps": 13.624,
                    "latency_mean_s": 1.168,
                    "rtf_mean": 0.2764,
                    "audio_throughput_s_per_s": 58.987,
                    "output_throughput": 1570.1,
                    "output_tok_per_req_s": 111.1,
                },
            ],
        },
    },
    "moss": {
        "concurrency": 16,
        "minimum": {
            "throughput_qps": 7.0,
            "audio_throughput_s_per_s": 30.0,
            "output_throughput": 380.0,
            "output_tok_per_req_s": 40.0,
        },
        "maximum": {"latency_mean_s": 2.4, "rtf_mean": 0.6},
        "provenance": {
            "temporary": True,
            "source_sha": "fe7e5dd802cb575b754d705232de2ff8b4f1795e",
            "accepted_run_ids": [30202304743],
            "method": "wide_floor_with_at_least_30_percent_slack_from_exact_sha_h100_mps_sample",
            "raw_samples": [
                {
                    "run_id": 30202304743,
                    "throughput_qps": 9.905,
                    "latency_mean_s": 1.609,
                    "rtf_mean": 0.3778,
                    "audio_throughput_s_per_s": 43.593,
                    "output_throughput": 544.9,
                    "output_tok_per_req_s": 58.8,
                },
            ],
        },
    },
}


def _validate_topology(topology: str) -> None:
    if topology not in SUPPORTED_TOPOLOGIES:
        raise ValueError(f"Unsupported TTS Stage 1 topology: {topology!r}")


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_evidence(path: str | Path, *, expected_kind: str) -> dict[str, Any]:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"missing or invalid Stage 1 evidence: {target}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Stage 1 evidence is not an object: {target}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"Stage 1 evidence schema_version mismatch: {target}")
    if expected_kind == "preflight":
        if payload.get("selected_model") not in MPS_NORMAL_LOAD_FLOORS:
            raise RuntimeError(f"invalid preflight selected_model: {target}")
        _validate_topology(str(payload.get("tts_stage1_topology")))
    elif payload.get("artifact_kind") != expected_kind:
        raise RuntimeError(f"Stage 1 evidence artifact_kind mismatch: {target}")
    return payload


def _mutate(
    path: Path, *, expected_kind: str, mutate: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    payload = read_evidence(path, expected_kind=expected_kind)
    mutate(payload)
    write_json_atomic(path, payload)
    return payload


def _run_metadata(environ: Mapping[str, str]) -> dict[str, Any]:
    server = (environ.get("GITHUB_SERVER_URL") or "https://github.com").rstrip("/")
    repository = (environ.get("GITHUB_REPOSITORY") or "").strip()
    run_id = (environ.get("GITHUB_RUN_ID") or "").strip()
    return {
        "run_id": run_id,
        "run_attempt": (environ.get("GITHUB_RUN_ATTEMPT") or "").strip(),
        "exact_sha": (environ.get("GITHUB_SHA") or "").strip(),
        "workflow_url": (
            f"{server}/{repository}/actions/runs/{run_id}"
            if repository and run_id
            else ""
        ),
    }


def initialize_evidence(
    output_dir: str | Path,
    *,
    topology: str,
    model: str,
    configured_timeout_minutes: int,
    environ: Mapping[str, str] | None = None,
) -> None:
    _validate_topology(topology)
    if model not in MPS_NORMAL_LOAD_FLOORS:
        raise ValueError(f"Unsupported TTS Stage 1 model: {model!r}")
    if configured_timeout_minutes <= 0:
        raise ValueError("configured timeout must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ if environ is None else environ
    common = {
        "schema_version": SCHEMA_VERSION,
        "topology": topology,
        "model": model,
    }
    mps = (
        {"status": "pending"}
        if topology == "mps_shared"
        else {"status": "not_applicable", "reason": "mps_disabled_for_topology"}
    )
    write_json_atomic(
        root / "stage1_runtime.json",
        {
            **common,
            "artifact_kind": "runtime",
            "status": "collecting",
            "run": _run_metadata(env),
            "launch": {"status": "pending"},
            "replicas": [],
            "router": {"before": None, "after": None},
            "processes": {"status": "pending"},
            "mps": mps,
            "raw_state_directory": None,
            "evidence_index": {"status": "pending"},
        },
    )
    write_json_atomic(
        root / "stage1_correctness.json",
        {
            **common,
            "artifact_kind": "correctness",
            "status": "incomplete",
            "canonical_stage3_input_only": True,
            "required_components": list(REQUIRED_CORRECTNESS),
            "components": {},
            "overlap_canary": (
                {"status": "pending", "canonical_stage3_input": False}
                if topology == "mps_shared"
                else {"status": "not_applicable", "reason": "mps_disabled_for_topology"}
            ),
        },
    )
    write_json_atomic(
        root / "stage1_timing.json",
        {
            **common,
            "artifact_kind": "timing",
            "status": "collecting",
            "started_at_epoch_ns": time.time_ns(),
            "phases": (
                {"hc_canary": {"status": "not_applicable"}}
                if topology == "multi_gpu"
                else {}
            ),
            "normal_load_performance": {
                "status": "pending",
                "gate": (
                    "mps_model_specific_safety_floor"
                    if topology == "mps_shared"
                    else "existing_multi_gpu_rollback_contract"
                ),
            },
            "total_observed_duration_s": None,
            "configured_timeout_minutes": configured_timeout_minutes,
        },
    )
    write_json_atomic(
        root / "stage1_cleanup.json",
        {
            **common,
            "artifact_kind": "cleanup",
            "verdict": "pending",
            "clean": False,
            "generation": {"status": "pending", "clean": False, "checks": {}},
            "evaluator_start_allowed": False,
            "evaluator": {
                "status": "pending",
                "pid": None,
                "port": None,
                "error": None,
            },
            "final_post_state": {"status": "pending", "clean": False},
            "retained_diagnostics": str(root),
        },
    )


def record_runtime_section(
    output_dir: str | Path, *, section: str, value: Any
) -> dict[str, Any]:
    path = Path(output_dir) / "stage1_runtime.json"
    return _mutate(
        path,
        expected_kind="runtime",
        mutate=lambda item: item.__setitem__(section, value),
    )


def record_phase(
    output_dir: str | Path,
    *,
    phase: str,
    duration_s: float,
    status: str = "pass",
    details: Mapping[str, Any] | None = None,
) -> None:
    if duration_s < 0:
        raise ValueError("phase duration must be non-negative")
    path = Path(output_dir) / "stage1_timing.json"

    def mutate(payload: dict[str, Any]) -> None:
        payload["phases"][phase] = {
            "status": status,
            "duration_s": duration_s,
            "details": dict(details or {}),
        }

    _mutate(path, expected_kind="timing", mutate=mutate)


def _refresh_correctness(payload: dict[str, Any]) -> None:
    complete = all(
        payload["components"].get(name, {}).get("status") == "pass"
        for name in REQUIRED_CORRECTNESS
    )
    if payload["topology"] == "mps_shared":
        complete = complete and payload["overlap_canary"].get("status") == "pass"
    payload["status"] = "pass" if complete else "incomplete"


def record_correctness_component(
    output_dir: str | Path, *, component: str, details: Mapping[str, Any]
) -> None:
    if component not in REQUIRED_CORRECTNESS:
        raise ValueError(f"unsupported correctness component: {component}")
    path = Path(output_dir) / "stage1_correctness.json"

    def mutate(payload: dict[str, Any]) -> None:
        payload["components"][component] = {"status": "pass", "details": dict(details)}
        _refresh_correctness(payload)

    _mutate(path, expected_kind="correctness", mutate=mutate)


def record_overlap_summary(output_dir: str | Path, verdict: Mapping[str, Any]) -> None:
    path = Path(output_dir) / "stage1_correctness.json"
    summary = dict(verdict)
    summary["canonical_stage3_input"] = False

    def mutate(payload: dict[str, Any]) -> None:
        if payload["topology"] != "mps_shared":
            raise ValueError("overlap evidence applies only to mps_shared")
        payload["overlap_canary"] = summary
        _refresh_correctness(payload)

    _mutate(path, expected_kind="correctness", mutate=mutate)


def record_normal_performance(
    output_dir: str | Path, *, concurrency: int, summary: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(output_dir) / "stage1_timing.json"
    payload = read_evidence(path, expected_kind="timing")
    if payload["topology"] == "multi_gpu":
        verdict = {"status": "recorded", "gate": "existing_multi_gpu_rollback_contract"}
        payload["normal_load_performance"] = {
            **verdict,
            "concurrency": concurrency,
            "metrics": dict(summary),
        }
        write_json_atomic(path, payload)
        return verdict
    floor = MPS_NORMAL_LOAD_FLOORS[payload["model"]]
    if concurrency != floor["concurrency"]:
        raise ValueError(
            f"MPS normal-load floor is calibrated only for concurrency {floor['concurrency']}, got {concurrency}"
        )
    failed: list[str] = []
    checks: dict[str, Any] = {}
    for metric, threshold in floor["minimum"].items():
        value = summary.get(metric)
        passed = isinstance(value, (int, float)) and value >= threshold
        checks[metric] = {
            "operator": ">=",
            "threshold": threshold,
            "observed": value,
            "pass": passed,
        }
        if not passed:
            failed.append(metric)
    for metric, threshold in floor["maximum"].items():
        value = summary.get(metric)
        passed = isinstance(value, (int, float)) and value <= threshold
        checks[metric] = {
            "operator": "<=",
            "threshold": threshold,
            "observed": value,
            "pass": passed,
        }
        if not passed:
            failed.append(metric)
    total_requests = summary.get("total_requests")
    if (
        summary.get("failed_requests") != 0
        or not isinstance(total_requests, int)
        or total_requests <= 0
        or summary.get("completed_requests") != total_requests
    ):
        failed.append("canonical_request_completion")
    verdict = {
        "status": "pass" if not failed else "fail",
        "gate": "mps_model_specific_safety_floor",
        "concurrency": concurrency,
        "checks": checks,
        "failed_checks": failed,
        "provenance": floor["provenance"],
    }
    payload["normal_load_performance"] = verdict
    write_json_atomic(path, payload)
    if failed:
        raise AssertionError(
            f"MPS normal-load safety floor failed: {', '.join(failed)}"
        )
    return verdict


def record_generation_cleanup(
    output_dir: str | Path,
    *,
    checks: Mapping[str, bool],
    details: Mapping[str, Any] | None = None,
    retained_diagnostics: str | None = None,
) -> dict[str, Any]:
    path = Path(output_dir) / "stage1_cleanup.json"
    clean = bool(checks) and all(value is True for value in checks.values())

    def mutate(payload: dict[str, Any]) -> None:
        payload["generation"] = {
            "status": "clean" if clean else "dirty",
            "clean": clean,
            "checks": dict(checks),
            "details": dict(details or {}),
        }
        payload["evaluator_start_allowed"] = clean
        if retained_diagnostics:
            payload["retained_diagnostics"] = retained_diagnostics
        if not clean:
            payload["verdict"] = "dirty"

    result = _mutate(path, expected_kind="cleanup", mutate=mutate)
    if not clean:
        raise RuntimeError("Stage 1 generation cleanup is dirty; evaluator is blocked")
    return result["generation"]


def require_evaluator_start_allowed(output_dir: str | Path) -> dict[str, Any]:
    payload = read_evidence(
        Path(output_dir) / "stage1_cleanup.json", expected_kind="cleanup"
    )
    if (
        payload["generation"].get("clean") is not True
        or payload.get("evaluator_start_allowed") is not True
    ):
        raise RuntimeError(
            "Stage 1 generation cleanup is not clean; evaluator is blocked"
        )
    return payload


def record_evaluator(
    output_dir: str | Path,
    *,
    status: str,
    pid: int | None,
    port: int | None,
    error: str | None = None,
) -> None:
    if status not in {"running", "completed", "error"}:
        raise ValueError(f"unsupported evaluator status: {status}")
    path = Path(output_dir) / "stage1_cleanup.json"

    def mutate(payload: dict[str, Any]) -> None:
        previous = payload.get("evaluator") or {}
        payload["evaluator"] = {
            "status": status,
            "pid": pid,
            "port": port,
            "observed_pid": pid if pid is not None else previous.get("observed_pid"),
            "observed_port": (
                port if port is not None else previous.get("observed_port")
            ),
            "error": error,
        }

    _mutate(path, expected_kind="cleanup", mutate=mutate)


def finalize_timing(
    output_dir: str | Path, *, finalization_started_at_ns: int
) -> dict[str, Any]:
    path = Path(output_dir) / "stage1_timing.json"
    now = time.time_ns()

    def mutate(payload: dict[str, Any]) -> None:
        payload["phases"]["finalization"] = {
            "status": "pass",
            "duration_s": max(0.0, (now - finalization_started_at_ns) / 1e9),
            "details": {},
        }
        payload["total_observed_duration_s"] = max(
            0.0, (now - int(payload["started_at_epoch_ns"])) / 1e9
        )
        payload["status"] = "complete"

    return _mutate(path, expected_kind="timing", mutate=mutate)
