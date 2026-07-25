# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


overlap = _load("tts_stage1_overlap", ".github/scripts/tts_stage1_overlap.py")
activity = _load("replica_activity", "sglang_omni/profiler/replica_activity.py")


def test_replica_activity_uses_monotonic_clock_and_dedupes(tmp_path: Path) -> None:
    output = tmp_path / "replica-0.jsonl"
    ticks = iter((100, 250))
    recorder = activity.ReplicaActivityRecorder(
        environ={
            activity.ENV_PATH: str(output),
            activity.ENV_RUN_ID: "run-1",
            activity.ENV_REPLICA_ID: "0",
        },
        monotonic_ns=lambda: next(ticks),
        pid=42,
        boot_id="boot-abc",
    )

    recorder.start("req-a")
    recorder.start("req-a")
    recorder.end("req-a", status="success")
    recorder.end("req-a", status="success")

    events = [json.loads(line) for line in output.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "model_path_start",
        "model_path_end",
    ]
    assert [event["monotonic_ns"] for event in events] == [100, 250]
    assert all(event["clock"] == "CLOCK_MONOTONIC" for event in events)
    assert all(event["run_id"] == "run-1" for event in events)
    assert all(event["replica_id"] == 0 for event in events)
    assert all(event["pid"] == 42 for event in events)
    assert all(event["host_boot_id"] == "boot-abc" for event in events)


def _interval_events() -> list[dict]:
    common = {
        "schema_version": 1,
        "clock": "CLOCK_MONOTONIC",
        "run_id": "run-1",
        "host_boot_id": "boot-1",
    }
    intervals = [
        (0, "r0-a", 100, 500),
        (0, "r0-b", 600, 1000),
        (1, "r1-a", 200, 550),
        (1, "r1-b", 700, 1100),
    ]
    events = []
    for replica_id, request_id, start, end in intervals:
        events.extend(
            [
                {
                    **common,
                    "event": "model_path_start",
                    "monotonic_ns": start,
                    "replica_id": replica_id,
                    "request_id": request_id,
                    "pid": replica_id + 10,
                },
                {
                    **common,
                    "event": "model_path_end",
                    "monotonic_ns": end,
                    "replica_id": replica_id,
                    "request_id": request_id,
                    "pid": replica_id + 10,
                    "status": "success",
                },
            ]
        )
    return events


def test_composite_oracle_requires_repeated_one_to_one_cross_replica_overlap() -> None:
    verdict = overlap.build_overlap_verdict(
        _interval_events(),
        request_ids={"r0-a", "r0-b", "r1-a", "r1-b"},
        min_successes_per_replica=2,
        min_matched_overlap_count=2,
        measurement_uncertainty_ns=10,
    )

    assert verdict["status"] == "pass"
    assert verdict["per_replica_successes"] == {"0": 2, "1": 2}
    assert verdict["matched_overlap_count"] == 2
    assert verdict["aggregate_overlap_ns"] == 600
    assert verdict["maximum_overlap_ns"] == 300
    assert len({pair["replica_0_request_id"] for pair in verdict["matches"]}) == 2
    assert len({pair["replica_1_request_id"] for pair in verdict["matches"]}) == 2
    assert verdict["threshold_provenance"] == (
        "provisional_repository_default_pending_h100_calibration"
    )
    assert verdict["promotion_eligible"] is False


def test_composite_oracle_rejects_a_single_accidental_interval() -> None:
    events = _interval_events()[:4]
    with pytest.raises(ValueError, match="successful requests"):
        overlap.build_overlap_verdict(
            events,
            request_ids={"r0-a", "r1-a"},
            min_successes_per_replica=2,
            min_matched_overlap_count=2,
            measurement_uncertainty_ns=10,
        )


def test_composite_oracle_prioritizes_matching_cardinality() -> None:
    common = {
        "schema_version": 1,
        "clock": "CLOCK_MONOTONIC",
        "run_id": "run-cardinality",
        "host_boot_id": "boot-1",
    }
    intervals = [
        (0, "left-wide", 0, 200),
        (0, "left-late", 200, 250),
        (1, "right-wide", 0, 250),
        (1, "right-early", 0, 50),
    ]
    events = []
    for replica_id, request_id, start, end in intervals:
        events.extend(
            [
                {
                    **common,
                    "event": "model_path_start",
                    "monotonic_ns": start,
                    "replica_id": replica_id,
                    "request_id": request_id,
                    "pid": replica_id + 20,
                },
                {
                    **common,
                    "event": "model_path_end",
                    "monotonic_ns": end,
                    "replica_id": replica_id,
                    "request_id": request_id,
                    "pid": replica_id + 20,
                    "status": "success",
                },
            ]
        )

    verdict = overlap.build_overlap_verdict(
        events,
        request_ids={item[1] for item in intervals},
        min_successes_per_replica=2,
        min_matched_overlap_count=2,
        measurement_uncertainty_ns=1,
    )

    assert verdict["matched_overlap_count"] == 2
    assert verdict["aggregate_overlap_ns"] == 100
