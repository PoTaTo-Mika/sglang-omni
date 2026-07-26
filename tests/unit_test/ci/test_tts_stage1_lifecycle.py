# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))
spec = importlib.util.spec_from_file_location(
    "tts_stage1_lifecycle",
    REPO_ROOT / ".github/scripts/tts_stage1_lifecycle.py",
)
lifecycle = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = lifecycle
spec.loader.exec_module(lifecycle)


def _command_runner(command, **kwargs):
    if "--query-gpu=index,uuid,name,driver_version" in command:
        return SimpleNamespace(
            returncode=0,
            stdout="0, GPU-abc, H100, 570.1\n",
            stderr="",
        )
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def _clean_teardown() -> dict:
    return lifecycle.build_mps_teardown_verdict(
        run_id="run-1",
        router_stopped=True,
        requests_drained_or_cancelled=True,
        down_exit_code=0,
        process_query_succeeded=True,
        tracked_processes_alive=[],
        mps_control_pid_observed=True,
        mps_control_alive=False,
        mps_namespace_active=False,
        occupied_ports=[],
        gpu_client_query_succeeded=True,
        tracked_gpu_clients_alive=[],
        retained_state=None,
    )


def test_default_feasibility_is_blocked_without_fabricated_consumer() -> None:
    payload = lifecycle.runner_feasibility({})

    assert payload["status"] == "blocked"
    assert payload["promotion_eligible"] is False
    assert payload["evidence"]["repository_consumer_found"] is False
    assert "host_visible_verdict_consumer" in payload["missing_capabilities"]


def test_clean_ordinary_teardown_allows_bound_evaluator_barrier(
    tmp_path: Path,
) -> None:
    teardown = lifecycle.build_ordinary_teardown_verdict(
        router_stopped=True,
        requests_drained_or_cancelled=True,
        process_query_succeeded=True,
        tracked_process_groups_alive=[],
        occupied_ports=[],
        gpu_memory_released=True,
        retained_state=None,
    )
    lifecycle.write_json_atomic(tmp_path / "ordinary_teardown_verdict.json", teardown)
    barrier = lifecycle.write_pre_evaluator_cleanup(
        tmp_path,
        topology="multi_gpu",
        environ={},
    )

    assert teardown["clean"] is True
    assert barrier["source_artifact"] == "ordinary_teardown_verdict.json"
    assert barrier["evaluator_start_allowed"] is True
    lifecycle.require_pre_evaluator_clean(tmp_path, environ={})


def test_dirty_ordinary_teardown_failure_injection_blocks_evaluator() -> None:
    teardown = lifecycle.build_ordinary_teardown_verdict(
        router_stopped=True,
        requests_drained_or_cancelled=True,
        process_query_succeeded=True,
        tracked_process_groups_alive=[],
        occupied_ports=[],
        gpu_memory_released=True,
        retained_state="/state/router_pgids.txt",
        failure_injection="dirty_ordinary_teardown",
    )

    assert teardown["clean"] is False
    assert teardown["quarantine_required"] is True
    assert teardown["retained_state"] == "/state/router_pgids.txt"


def test_pre_evaluator_barrier_binds_to_clean_source(tmp_path: Path) -> None:
    teardown = _clean_teardown()
    lifecycle.write_json_atomic(tmp_path / "mps_teardown_verdict.json", teardown)
    verdict = lifecycle.write_pre_evaluator_cleanup(
        tmp_path,
        topology="mps_shared",
        environ={},
    )

    assert verdict["clean"] is True
    lifecycle.require_pre_evaluator_clean(tmp_path, environ={})

    teardown["clean"] = False
    (tmp_path / "mps_teardown_verdict.json").write_text(json.dumps(teardown))
    with pytest.raises(RuntimeError, match="source verdict changed"):
        lifecycle.require_pre_evaluator_clean(tmp_path, environ={})


def test_pre_evaluator_barrier_rejects_a_different_run_attempt(
    tmp_path: Path,
) -> None:
    first_attempt = {
        "GITHUB_RUN_ID": "100",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_JOB": "stage-1",
        "TTS_STAGE1_LANE_LEASE_ID": "lease-100",
        "TTS_STAGE1_LANE_LEASE_VERIFIED": "true",
    }
    second_attempt = {**first_attempt, "GITHUB_RUN_ATTEMPT": "2"}
    teardown = _clean_teardown()
    teardown["lane_context"] = lifecycle.lane_context(first_attempt)
    lifecycle.write_json_atomic(tmp_path / "mps_teardown_verdict.json", teardown)
    lifecycle.write_pre_evaluator_cleanup(
        tmp_path,
        topology="mps_shared",
        environ=first_attempt,
    )

    with pytest.raises(RuntimeError, match="run or lane identity"):
        lifecycle.require_pre_evaluator_clean(tmp_path, environ=second_attempt)


def test_dirty_mps_failure_injection_blocks_evaluator() -> None:
    verdict = lifecycle.build_mps_teardown_verdict(
        run_id="run-1",
        router_stopped=True,
        requests_drained_or_cancelled=True,
        down_exit_code=0,
        process_query_succeeded=True,
        tracked_processes_alive=[],
        mps_control_pid_observed=True,
        mps_control_alive=False,
        mps_namespace_active=False,
        occupied_ports=[],
        gpu_client_query_succeeded=True,
        tracked_gpu_clients_alive=[],
        retained_state="/state/run-1",
        failure_injection="dirty_mps_teardown",
    )

    assert verdict["clean"] is False
    assert verdict["quarantine_required"] is True
    assert verdict["retained_state"] == "/state/run-1"


def test_lane_release_requires_real_consumer_and_matching_lease() -> None:
    context = {
        "lane_lease_id": "lease-1",
        "lane_lease_verified": True,
    }
    common = {
        "topology": "mps_shared",
        "teardown": {"clean": True, "lane_context": context},
        "pre_evaluator": {"clean": True, "lane_context": context},
        "evaluator": {
            "status": "completed",
            "topology": "mps_shared",
            "lane_context": context,
        },
        "post_state": {"clean": True},
        "baseline_lane_context": context,
        "final_lane_context": context,
    }
    blocked = lifecycle.build_lane_release_verdict(
        **common,
        feasibility={"status": "blocked"},
    )
    clean = lifecycle.build_lane_release_verdict(
        **common,
        feasibility={"status": "deployed"},
    )

    assert blocked["clean"] is False
    assert blocked["quarantine_required"] is True
    assert clean["clean"] is True
    assert clean["release_eligible"] is True


def test_finalize_writes_auditable_blocked_verdict_without_lease(
    tmp_path: Path,
) -> None:
    lifecycle.initialize_lifecycle(
        tmp_path,
        topology="mps_shared",
        model="higgs",
        environ={},
        command_runner=_command_runner,
    )
    lifecycle.write_json_atomic(
        tmp_path / "mps_teardown_verdict.json", _clean_teardown()
    )
    lifecycle.write_pre_evaluator_cleanup(
        tmp_path,
        topology="mps_shared",
        environ={},
    )
    lifecycle.write_evaluator_lifecycle(
        tmp_path,
        topology="mps_shared",
        status="completed",
        pid=10,
        port=8000,
    )

    verdict = lifecycle.finalize_lifecycle(
        tmp_path,
        topology="mps_shared",
        model="higgs",
        exact_sha="abc123",
        workflow_url="https://example.test/run/1",
        environ={},
        command_runner=_command_runner,
    )

    assert verdict["clean"] is False
    assert verdict["checks"]["runner_consumer_deployed"] is False
    assert verdict["checks"]["lane_lease_verified_and_matching"] is False
    index = json.loads((tmp_path / "artifact_index.json").read_text())
    assert index["exact_sha"] == "abc123"
    assert index["workflow_url"] == "https://example.test/run/1"


def test_finalize_multi_gpu_uses_ordinary_teardown_chain(tmp_path: Path) -> None:
    lifecycle.initialize_lifecycle(
        tmp_path,
        topology="multi_gpu",
        model="moss",
        environ={},
        command_runner=_command_runner,
    )
    teardown = lifecycle.build_ordinary_teardown_verdict(
        router_stopped=True,
        requests_drained_or_cancelled=True,
        process_query_succeeded=True,
        tracked_process_groups_alive=[],
        occupied_ports=[],
        gpu_memory_released=True,
        retained_state=None,
    )
    lifecycle.write_json_atomic(tmp_path / "ordinary_teardown_verdict.json", teardown)
    lifecycle.write_pre_evaluator_cleanup(
        tmp_path,
        topology="multi_gpu",
        environ={},
    )
    lifecycle.write_evaluator_lifecycle(
        tmp_path,
        topology="multi_gpu",
        status="completed",
        pid=None,
        port=None,
    )
    lifecycle.write_json_atomic(
        tmp_path / "preflight.json",
        {
            "schema_version": 1,
            "selected_model": "moss",
            "tts_stage1_topology": "multi_gpu",
            "run_id": None,
            "run_attempt": None,
            "selection_digest": "digest",
        },
    )
    lifecycle.write_json_atomic(
        tmp_path / "normal_correctness.json",
        {"schema_version": 1, "status": "pass", "clean": True},
    )
    lifecycle.write_json_atomic(
        tmp_path / "performance_observation.json",
        {
            "schema_version": 1,
            "status": "observed",
            "threshold_calibration": {
                "status": "pending_h100_repeated_runs",
                "promotion_eligible": False,
            },
        },
    )
    lifecycle.write_json_atomic(
        tmp_path / "stage1_timing_breakdown.json",
        {
            "schema_version": 1,
            "status": "collecting",
            "started_at_epoch_ns": lifecycle.time.time_ns(),
            "phases": {},
            "outer_timeout": {
                "configured_minutes": 25,
                "status": "unvalidated",
                "adequate": None,
                "promotion_eligible": False,
            },
        },
    )

    verdict = lifecycle.finalize_lifecycle(
        tmp_path,
        topology="multi_gpu",
        model="moss",
        exact_sha="rollback-sha",
        workflow_url="https://example.test/run/rollback",
        environ={},
        command_runner=_command_runner,
    )

    index = json.loads((tmp_path / "artifact_index.json").read_text())
    assert index["cleanup"]["source"] == "ordinary_teardown_verdict.json"
    assert index["cleanup"]["pre_evaluator_clean"] is True
    assert index["evidence_complete"] is True
    assert index["preflight_matches_current_run"] is True
    assert index["promotion_ready"] is False
    assert index["timeout_validation"]["status"] == "unvalidated"
    assert verdict["checks"]["topology_teardown_clean"] is True
    assert verdict["checks"]["runner_consumer_deployed"] is False


def test_finalize_missing_evidence_still_writes_quarantine_verdict(
    tmp_path: Path,
) -> None:
    lifecycle.write_json_atomic(
        tmp_path / "performance_observation.json",
        {
            "schema_version": 1,
            "status": "observed",
            "threshold_calibration": {
                "status": "calibrated",
                "promotion_eligible": True,
            },
        },
    )
    lifecycle.write_json_atomic(
        tmp_path / "stage1_timing_breakdown.json",
        {
            "schema_version": 1,
            "status": "sample_complete",
            "started_at_epoch_ns": lifecycle.time.time_ns(),
            "phases": {},
            "outer_timeout": {
                "status": "validated",
                "adequate": True,
                "promotion_eligible": True,
            },
        },
    )
    verdict = lifecycle.finalize_lifecycle(
        tmp_path,
        topology="mps_shared",
        model="moss",
        exact_sha="def456",
        workflow_url="https://example.test/run/2",
        environ={},
        command_runner=_command_runner,
    )

    assert verdict["clean"] is False
    assert verdict["quarantine_required"] is True
    assert (tmp_path / "post_state.json").is_file()
    assert (tmp_path / "artifact_index.json").is_file()
    assert (tmp_path / "lane_release_verdict.json").is_file()
    index = json.loads((tmp_path / "artifact_index.json").read_text())
    assert index["status"] == "indexed"
    assert index["evidence_complete"] is False
    assert index["promotion_ready"] is False
