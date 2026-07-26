# SPDX-License-Identifier: Apache-2.0
"""Composite cross-replica overlap oracle for the bounded TTS Stage 1 canary."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

SCHEMA_VERSION = 1
OVERLAP_PROVENANCE = "exact_sha_h100_runs_30196509700_30199556014_30202304743"


@dataclass(frozen=True)
class Interval:
    replica_id: int
    request_id: str
    start_ns: int
    end_ns: int


def _build_intervals(
    events: Iterable[dict[str, Any]], *, request_ids: set[str]
) -> list[Interval]:
    points: dict[tuple[int, str], dict[str, int]] = {}
    run_ids: set[str] = set()
    boot_ids: set[str] = set()
    for event in events:
        if event.get("request_id") not in request_ids:
            continue
        if event.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("replica activity schema version mismatch")
        if event.get("clock") != "CLOCK_MONOTONIC":
            raise ValueError("replica activity must use CLOCK_MONOTONIC")
        run_ids.add(str(event.get("run_id")))
        boot_ids.add(str(event.get("host_boot_id")))
        replica_id = int(event["replica_id"])
        request_id = str(event["request_id"])
        kind = event.get("event")
        if kind not in {"model_path_start", "model_path_end"}:
            continue
        if kind == "model_path_end" and event.get("status") != "success":
            continue
        point_name = "start" if kind == "model_path_start" else "end"
        key = (replica_id, request_id)
        point_set = points.setdefault(key, {})
        if point_name in point_set:
            raise ValueError(f"duplicate {point_name} event for {key}")
        point_set[point_name] = int(event["monotonic_ns"])
    if len(run_ids) != 1 or len(boot_ids) != 1:
        raise ValueError("replica activity must come from one run and host boot")

    intervals: list[Interval] = []
    for (replica_id, request_id), point_set in points.items():
        if set(point_set) != {"start", "end"}:
            continue
        if point_set["end"] <= point_set["start"]:
            raise ValueError(f"non-positive model-path interval for {request_id}")
        intervals.append(
            Interval(
                replica_id=replica_id,
                request_id=request_id,
                start_ns=point_set["start"],
                end_ns=point_set["end"],
            )
        )
    return intervals


def build_overlap_verdict(
    events: Iterable[dict[str, Any]],
    *,
    request_ids: set[str],
    min_successes_per_replica: int,
    min_matched_overlap_count: int,
    measurement_uncertainty_ns: int,
) -> dict[str, Any]:
    """Maximize one-to-one overlap count, then aggregate overlap duration."""

    if min_successes_per_replica < 2 or min_matched_overlap_count < 2:
        raise ValueError("composite overlap requires at least two repeated intervals")
    if measurement_uncertainty_ns < 0:
        raise ValueError("measurement uncertainty must be non-negative")
    intervals = _build_intervals(events, request_ids=request_ids)
    by_replica = {
        replica_id: [item for item in intervals if item.replica_id == replica_id]
        for replica_id in {item.replica_id for item in intervals}
    }
    if set(by_replica) != {0, 1}:
        raise ValueError(
            "overlap canary requires successful requests on replicas 0 and 1"
        )
    counts = {replica_id: len(items) for replica_id, items in by_replica.items()}
    if any(count < min_successes_per_replica for count in counts.values()):
        raise ValueError(
            f"insufficient successful requests per replica: {counts}; "
            f"required {min_successes_per_replica}"
        )

    left_intervals = by_replica[0]
    right_intervals = by_replica[1]

    @lru_cache(maxsize=None)
    def best_matching(
        left_index: int, used_right_mask: int
    ) -> tuple[int, int, list[dict[str, Any]]]:
        if left_index == len(left_intervals):
            return 0, 0, []
        best = best_matching(left_index + 1, used_right_mask)
        left = left_intervals[left_index]
        for right_index, right in enumerate(right_intervals):
            if used_right_mask & (1 << right_index):
                continue
            overlap_ns = min(left.end_ns, right.end_ns) - max(
                left.start_ns, right.start_ns
            )
            if overlap_ns <= measurement_uncertainty_ns:
                continue
            count, aggregate, matches = best_matching(
                left_index + 1, used_right_mask | (1 << right_index)
            )
            candidate = (
                count + 1,
                aggregate + overlap_ns,
                [
                    {
                        "replica_0_request_id": left.request_id,
                        "replica_1_request_id": right.request_id,
                        "overlap_ns": overlap_ns,
                    },
                    *matches,
                ],
            )
            if candidate[:2] > best[:2]:
                best = candidate
        return best

    _, _, matches = best_matching(0, 0)
    if len(matches) < min_matched_overlap_count:
        raise ValueError(
            f"insufficient repeated cross-replica overlap: {len(matches)}; "
            f"required {min_matched_overlap_count}"
        )
    overlaps = [match["overlap_ns"] for match in matches]
    return {
        "schema_version": SCHEMA_VERSION,
        "topology": "mps_shared",
        "status": "pass",
        "clock": "CLOCK_MONOTONIC",
        "per_replica_successes": {str(key): value for key, value in counts.items()},
        "minimum_successes_per_replica": min_successes_per_replica,
        "matched_overlap_count": len(matches),
        "minimum_matched_overlap_count": min_matched_overlap_count,
        "aggregate_overlap_ns": sum(overlaps),
        "maximum_overlap_ns": max(overlaps),
        "measurement_uncertainty_ns": measurement_uncertainty_ns,
        "matches": matches,
        "threshold_provenance": OVERLAP_PROVENANCE,
    }
