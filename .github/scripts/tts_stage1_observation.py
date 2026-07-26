# SPDX-License-Identifier: Apache-2.0
"""Observation-only calibration evidence for TTS Stage 1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

from tts_stage1_artifacts import SCHEMA_VERSION, write_json_atomic
from tts_stage1_lifecycle import lane_context

REQUIRED_NORMAL_COMPONENTS = (
    "canonical_generation",
    "wer",
    "similarity",
    "utmos",
)
PERFORMANCE_METRICS = (
    "throughput_qps",
    "latency_mean_s",
    "latency_p95_s",
    "rtf_mean",
    "rtf_p95",
    "audio_throughput_s_per_s",
    "output_throughput",
    "output_tok_per_req_s",
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"invalid observation artifact: {path}")
    return payload


def initialize_observation(
    output_dir: str | Path,
    *,
    topology: str,
    model: str,
    configured_timeout_minutes: int,
    environ: Mapping[str, str] | None = None,
) -> None:
    root = Path(output_dir)
    identity = lane_context(environ)
    now = time.time_ns()
    phases: dict[str, Any] = {}
    if topology == "multi_gpu":
        phases["hc_canary"] = {
            "status": "not_applicable",
            "reason_code": "mps_disabled_for_topology",
        }
    write_json_atomic(
        root / "stage1_timing_breakdown.json",
        {
            "schema_version": SCHEMA_VERSION,
            "topology": topology,
            "model": model,
            "status": "collecting",
            "started_at_epoch_ns": now,
            "lane_context": identity,
            "phases": phases,
            "total_stage1_wall_time_s": None,
            "repeated_run_variance": {
                "status": "pending_h100_repeated_runs",
                "accepted_samples": 0,
            },
            "outer_timeout": {
                "configured_minutes": configured_timeout_minutes,
                "status": "unvalidated",
                "adequate": None,
                "promotion_eligible": False,
                "reason": "requires accepted Higgs and MOSS H100 timing samples including artifact handling",
            },
        },
    )
    write_json_atomic(
        root / "performance_observation.json",
        {
            "schema_version": SCHEMA_VERSION,
            "topology": topology,
            "model": model,
            "status": "pending",
            "gate_mode": (
                "observation_only_pending_h100_calibration"
                if topology == "mps_shared"
                else "existing_multi_gpu_rollback_contract"
            ),
            "normal_load_samples": [],
            "threshold_calibration": {
                "status": "pending_h100_repeated_runs",
                "promotion_eligible": False,
                "required_metrics": list(PERFORMANCE_METRICS),
            },
            "lane_context": identity,
        },
    )
    write_json_atomic(
        root / "normal_correctness.json",
        {
            "schema_version": SCHEMA_VERSION,
            "topology": topology,
            "model": model,
            "status": "incomplete",
            "clean": False,
            "canonical_workload_only": True,
            "required_components": list(REQUIRED_NORMAL_COMPONENTS),
            "components": {},
            "lane_context": identity,
        },
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
    path = Path(output_dir) / "stage1_timing_breakdown.json"
    payload = _read(path)
    payload["phases"][phase] = {
        "status": status,
        "duration_s": duration_s,
        "details": dict(details or {}),
    }
    write_json_atomic(path, payload)


def record_performance_observation(
    output_dir: str | Path,
    *,
    concurrency: int,
    summary: Mapping[str, Any],
) -> None:
    path = Path(output_dir) / "performance_observation.json"
    payload = _read(path)
    payload["normal_load_samples"].append(
        {
            "concurrency": concurrency,
            "metrics": {key: summary.get(key) for key in PERFORMANCE_METRICS},
            "completed_requests": summary.get("completed_requests"),
            "failed_requests": summary.get("failed_requests"),
        }
    )
    payload["status"] = "observed"
    write_json_atomic(path, payload)


def record_correctness_component(
    output_dir: str | Path,
    *,
    component: str,
    details: Mapping[str, Any],
) -> None:
    if component not in REQUIRED_NORMAL_COMPONENTS:
        raise ValueError(f"unsupported normal correctness component: {component}")
    path = Path(output_dir) / "normal_correctness.json"
    payload = _read(path)
    payload["components"][component] = {
        "status": "pass",
        "details": dict(details),
    }
    complete = all(
        payload["components"].get(name, {}).get("status") == "pass"
        for name in REQUIRED_NORMAL_COMPONENTS
    )
    payload["status"] = "pass" if complete else "incomplete"
    payload["clean"] = complete
    write_json_atomic(path, payload)


def finalize_observation(
    output_dir: str | Path,
    *,
    finalization_started_at_ns: int,
) -> dict[str, Any]:
    root = Path(output_dir)
    path = root / "stage1_timing_breakdown.json"
    payload = _read(path)
    now = time.time_ns()
    payload["phases"]["final_cleanup_artifact_preparation"] = {
        "status": "pass",
        "duration_s": max(0.0, (now - finalization_started_at_ns) / 1e9),
        "details": {},
    }
    payload["phases"]["artifact_upload"] = {
        "status": "pending_external_step_measurement",
        "duration_s": None,
        "details": {},
    }
    payload["total_until_artifact_upload_s"] = max(
        0.0, (now - int(payload["started_at_epoch_ns"])) / 1e9
    )
    payload["status"] = "sample_incomplete"
    payload["completed_at_epoch_ns"] = now
    write_json_atomic(path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("initialize")
    initialize.add_argument("--output-dir", required=True)
    initialize.add_argument("--topology", required=True)
    initialize.add_argument("--model", required=True)
    initialize.add_argument("--configured-timeout-minutes", type=int, required=True)
    args = parser.parse_args()
    if args.command == "initialize":
        initialize_observation(
            args.output_dir,
            topology=args.topology,
            model=args.model,
            configured_timeout_minutes=args.configured_timeout_minutes,
        )


if __name__ == "__main__":
    main()
