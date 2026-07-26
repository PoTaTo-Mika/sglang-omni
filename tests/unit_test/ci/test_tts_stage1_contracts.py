# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))

import tts_stage1_evidence as evidence  # noqa: E402
import tts_stage1_selection as selection  # noqa: E402


def _select(**overrides):
    args = {
        "run_id": "123456789",
        "run_attempt": "1",
        "labels": [],
        "override": "",
        "repository_root": REPO_ROOT,
        "topology": "multi_gpu",
    }
    args.update(overrides)
    return selection.select_tts_stage1(**args)


def test_run_attempt_does_not_change_digest_or_selected_model() -> None:
    expected_digest = hashlib.sha256(b"123456789").hexdigest()
    expected_model = ("higgs", "moss")[int(expected_digest, 16) % 2]
    first = _select(run_attempt="1")
    rerun = _select(run_attempt="9")
    assert first.selection_digest == expected_digest
    assert first.selected_model == expected_model
    assert rerun.selection_digest == first.selection_digest
    assert rerun.selected_model == first.selected_model
    assert rerun.run_attempt == "9"


def test_conflicting_model_labels_fail_before_selection() -> None:
    with pytest.raises(ValueError, match="run-higgs.*run-moss"):
        _select(labels=["run-ci", "run-higgs", "run-moss"])


@pytest.mark.parametrize(
    ("override", "expected"), [("higgs", "higgs"), ("MOSS", "moss")]
)
def test_manual_override_selects_supported_model(override: str, expected: str) -> None:
    result = _select(override=override)
    assert result.selected_model == expected
    assert result.selection_reason == "workflow_dispatch_override"


def test_invalid_manual_override_and_topology_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported tts_ci_model"):
        _select(override="qwen3")
    with pytest.raises(ValueError, match="Unsupported TTS Stage 1 topology"):
        _select(topology="one_gpu")


def test_single_model_label_wins_without_override() -> None:
    result = _select(labels=["run-ci", "run-moss"])
    assert result.selected_model == "moss"
    assert result.selection_reason == "label:run-moss"


def test_default_topology_and_merged_1124_registry_resolution() -> None:
    higgs = _select(labels=["run-higgs"])
    moss = _select(labels=["run-moss"])
    assert higgs.tts_stage1_topology == "multi_gpu"
    assert higgs.resolved_mps_config == "examples/mps_dp/configs/higgs_h100_dp2.yaml"
    assert higgs.resolved_config_class == "HiggsTtsPipelineConfig"
    assert (
        moss.resolved_mps_config == "examples/mps_dp/configs/moss_local_h100_dp2.yaml"
    )
    assert moss.resolved_config_class == "MossTTSLocalPipelineConfig"


def test_config_path_is_derived_from_registry(tmp_path: Path) -> None:
    config_dir = tmp_path / "examples" / "mps_dp" / "configs"
    config_dir.mkdir(parents=True)
    (tmp_path / "examples" / "mps_dp" / "config.py").write_text(
        "WEIGHT_SHARE_VALIDATED_CONFIGS = {\n"
        '    "HiggsTtsPipelineConfig": "HiggsArchitecture",\n'
        "}\n"
        "TTS_STAGE1_VALIDATED_CONFIGS = {\n"
        '    "higgs": "configs/custom_higgs.yaml",\n'
        "}\n",
        encoding="utf-8",
    )
    (config_dir / "custom_higgs.yaml").write_text(
        "config_cls: HiggsTtsPipelineConfig\n", encoding="utf-8"
    )
    resolved, config_class = selection._resolve_config(
        repository_root=tmp_path, model="higgs"
    )
    assert resolved == "examples/mps_dp/configs/custom_higgs.yaml"
    assert config_class == "HiggsTtsPipelineConfig"


def test_preflight_json_contains_one_immutable_selection(tmp_path: Path) -> None:
    result = _select(
        run_id="84",
        run_attempt="7",
        override="higgs",
        topology="mps_shared",
        exact_sha="abc123",
        workflow_url="https://example.test/runs/84",
    )
    output = tmp_path / "preflight.json"
    selection.write_preflight(output, result)
    payload = evidence.read_evidence(output, expected_kind="preflight")
    assert payload == result.as_dict()
    assert payload["run_attempt"] == "7"
    assert payload["exact_sha"] == "abc123"
    assert payload["workflow_url"].endswith("/84")


@pytest.mark.parametrize("topology", ["mps_shared", "multi_gpu"])
def test_initialize_writes_only_four_aggregate_json_files(
    tmp_path: Path, topology: str
) -> None:
    evidence.initialize_evidence(
        tmp_path,
        topology=topology,
        model="higgs",
        configured_timeout_minutes=25,
        environ={"GITHUB_RUN_ID": "10", "GITHUB_RUN_ATTEMPT": "2"},
    )
    assert {path.name for path in tmp_path.glob("*.json")} == {
        "stage1_runtime.json",
        "stage1_correctness.json",
        "stage1_timing.json",
        "stage1_cleanup.json",
    }
    runtime = evidence.read_evidence(
        tmp_path / "stage1_runtime.json", expected_kind="runtime"
    )
    if topology == "multi_gpu":
        assert runtime["mps"] == {
            "status": "not_applicable",
            "reason": "mps_disabled_for_topology",
        }
    else:
        assert runtime["mps"]["status"] == "pending"


def test_atomic_reinitialize_replaces_stale_attempt_state(tmp_path: Path) -> None:
    evidence.initialize_evidence(
        tmp_path, topology="mps_shared", model="higgs", configured_timeout_minutes=25
    )
    evidence.update_evidence(
        tmp_path / "stage1_cleanup.json",
        expected_kind="cleanup",
        updates={"clean": True, "verdict": "clean"},
    )
    evidence.initialize_evidence(
        tmp_path, topology="mps_shared", model="higgs", configured_timeout_minutes=25
    )
    cleanup = evidence.read_evidence(
        tmp_path / "stage1_cleanup.json", expected_kind="cleanup"
    )
    assert cleanup["clean"] is False
    assert cleanup["verdict"] == "pending"
    assert not list(tmp_path.glob(".*.tmp"))


def test_schema_validation_is_fail_closed(tmp_path: Path) -> None:
    bad = tmp_path / "stage1_runtime.json"
    bad.write_text('{"schema_version": 99, "artifact_kind": "runtime"}')
    with pytest.raises(RuntimeError, match="schema_version"):
        evidence.read_evidence(bad, expected_kind="runtime")
    bad.write_text('{"schema_version": 1, "artifact_kind": "cleanup"}')
    with pytest.raises(RuntimeError, match="artifact_kind"):
        evidence.read_evidence(bad, expected_kind="runtime")


def test_higgs_stage1_config_is_explicit_h100_dp2() -> None:
    config = __import__("yaml").safe_load(
        (REPO_ROOT / "examples/mps_dp/configs/higgs_h100_dp2.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["config_cls"] == "HiggsTtsPipelineConfig"
    assert config["runtime_overrides"]["tts_engine"]["server_args_overrides"] == {
        "max_total_tokens": 100000,
        "max_running_requests": 64,
        "cuda_graph_max_bs": 64,
    }


@pytest.mark.parametrize(
    ("model", "summary"),
    [
        (
            "higgs",
            {
                "throughput_qps": 13.624,
                "latency_mean_s": 1.168,
                "rtf_mean": 0.2764,
                "audio_throughput_s_per_s": 58.987,
                "output_throughput": 1570.1,
                "output_tok_per_req_s": 111.1,
                "completed_requests": 1088,
                "failed_requests": 0,
            },
        ),
        (
            "moss",
            {
                "throughput_qps": 9.905,
                "latency_mean_s": 1.609,
                "rtf_mean": 0.3778,
                "audio_throughput_s_per_s": 43.593,
                "output_throughput": 544.9,
                "output_tok_per_req_s": 58.8,
                "completed_requests": 1088,
                "failed_requests": 0,
            },
        ),
    ],
)
def test_mps_normal_load_floor_accepts_raw_h100_samples(
    tmp_path: Path, model: str, summary: dict
) -> None:
    evidence.initialize_evidence(
        tmp_path, topology="mps_shared", model=model, configured_timeout_minutes=25
    )
    verdict = evidence.record_normal_performance(
        tmp_path, concurrency=16, summary=summary
    )
    assert verdict["status"] == "pass"
    assert verdict["provenance"]["temporary"] is True
    assert verdict["provenance"]["accepted_run_ids"]
    timing = json.loads((tmp_path / "stage1_timing.json").read_text())
    assert timing["normal_load_performance"]["status"] == "pass"


def test_mps_normal_load_floor_hard_fails_obvious_regression(tmp_path: Path) -> None:
    evidence.initialize_evidence(
        tmp_path, topology="mps_shared", model="higgs", configured_timeout_minutes=25
    )
    summary = {
        "throughput_qps": 1.0,
        "latency_mean_s": 10.0,
        "rtf_mean": 2.0,
        "audio_throughput_s_per_s": 2.0,
        "output_throughput": 100.0,
        "output_tok_per_req_s": 5.0,
        "completed_requests": 1088,
        "failed_requests": 0,
    }
    with pytest.raises(AssertionError, match="MPS normal-load safety floor"):
        evidence.record_normal_performance(tmp_path, concurrency=16, summary=summary)
    timing = json.loads((tmp_path / "stage1_timing.json").read_text())
    assert timing["normal_load_performance"]["status"] == "fail"
    assert timing["normal_load_performance"]["failed_checks"]


def test_multi_gpu_records_metrics_without_mps_specific_gate(tmp_path: Path) -> None:
    evidence.initialize_evidence(
        tmp_path, topology="multi_gpu", model="moss", configured_timeout_minutes=25
    )
    verdict = evidence.record_normal_performance(
        tmp_path,
        concurrency=16,
        summary={"completed_requests": 1088, "failed_requests": 0},
    )
    assert verdict == {
        "status": "recorded",
        "gate": "existing_multi_gpu_rollback_contract",
    }
