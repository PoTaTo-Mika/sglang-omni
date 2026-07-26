# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))
spec = importlib.util.spec_from_file_location(
    "tts_stage1_observation",
    REPO_ROOT / ".github/scripts/tts_stage1_observation.py",
)
observation = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = observation
spec.loader.exec_module(observation)


def test_mps_observation_is_not_a_promoted_threshold_or_validated_timeout(
    tmp_path: Path,
) -> None:
    observation.initialize_observation(
        tmp_path,
        topology="mps_shared",
        model="higgs",
        configured_timeout_minutes=25,
        environ={"GITHUB_RUN_ID": "1", "GITHUB_RUN_ATTEMPT": "2"},
    )
    observation.record_phase(
        tmp_path,
        phase="startup_attachment",
        duration_s=12.5,
    )
    observation.record_performance_observation(
        tmp_path,
        concurrency=16,
        summary={
            "throughput_qps": 10.0,
            "latency_mean_s": 1.2,
            "rtf_mean": 0.3,
            "completed_requests": 50,
            "failed_requests": 0,
        },
    )
    timing = observation.finalize_observation(
        tmp_path,
        finalization_started_at_ns=observation.time.time_ns(),
    )

    performance = json.loads((tmp_path / "performance_observation.json").read_text())
    assert performance["status"] == "observed"
    assert performance["gate_mode"] == "observation_only_pending_h100_calibration"
    assert performance["threshold_calibration"]["promotion_eligible"] is False
    assert timing["outer_timeout"] == {
        "adequate": None,
        "configured_minutes": 25,
        "promotion_eligible": False,
        "reason": "requires accepted Higgs and MOSS H100 timing samples including artifact handling",
        "status": "unvalidated",
    }
    assert timing["phases"]["artifact_upload"]["status"] == (
        "pending_external_step_measurement"
    )
    assert timing["repeated_run_variance"]["status"] == ("pending_h100_repeated_runs")


def test_normal_correctness_passes_only_after_all_canonical_components(
    tmp_path: Path,
) -> None:
    observation.initialize_observation(
        tmp_path,
        topology="multi_gpu",
        model="moss",
        configured_timeout_minutes=25,
        environ={},
    )
    for component in observation.REQUIRED_NORMAL_COMPONENTS[:-1]:
        observation.record_correctness_component(
            tmp_path,
            component=component,
            details={"value": 1},
        )
    incomplete = json.loads((tmp_path / "normal_correctness.json").read_text())
    assert incomplete["status"] == "incomplete"
    assert incomplete["clean"] is False

    observation.record_correctness_component(
        tmp_path,
        component=observation.REQUIRED_NORMAL_COMPONENTS[-1],
        details={"value": 1},
    )
    complete = json.loads((tmp_path / "normal_correctness.json").read_text())
    assert complete["status"] == "pass"
    assert complete["clean"] is True
    timing = json.loads((tmp_path / "stage1_timing_breakdown.json").read_text())
    assert timing["phases"]["hc_canary"]["status"] == "not_applicable"
