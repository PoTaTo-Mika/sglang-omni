# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))


def _load(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


selection = _load("tts_stage1_selection", ".github/scripts/tts_stage1_selection.py")
artifacts = _load("tts_stage1_artifacts", ".github/scripts/tts_stage1_artifacts.py")


def test_run_attempt_does_not_change_digest_or_selected_model() -> None:
    run_id = "123456789"
    expected_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    expected_model = ("higgs", "moss")[int(expected_digest, 16) % 2]

    first = selection.select_tts_stage1(
        run_id=run_id,
        run_attempt="1",
        labels=[],
        override="",
        repository_root=REPO_ROOT,
    )
    rerun = selection.select_tts_stage1(
        run_id=run_id,
        run_attempt="9",
        labels=[],
        override="",
        repository_root=REPO_ROOT,
    )

    assert first.selection_digest == expected_digest
    assert first.selected_model == expected_model
    assert rerun.selection_digest == first.selection_digest
    assert rerun.selected_model == first.selected_model
    assert first.run_attempt == "1"
    assert rerun.run_attempt == "9"


def test_conflicting_model_labels_fail_before_selection() -> None:
    with pytest.raises(ValueError, match="run-higgs.*run-moss"):
        selection.select_tts_stage1(
            run_id="42",
            run_attempt="1",
            labels=["run-ci", "run-higgs", "run-moss"],
            override="",
            repository_root=REPO_ROOT,
        )


@pytest.mark.parametrize(
    ("override", "expected"),
    [("higgs", "higgs"), ("MOSS", "moss")],
)
def test_manual_override_selects_supported_model(override: str, expected: str) -> None:
    result = selection.select_tts_stage1(
        run_id="42",
        run_attempt="2",
        labels=[],
        override=override,
        repository_root=REPO_ROOT,
    )
    assert result.selected_model == expected
    assert result.selection_reason == "workflow_dispatch_override"


def test_invalid_manual_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported tts_ci_model"):
        selection.select_tts_stage1(
            run_id="42",
            run_attempt="1",
            labels=[],
            override="qwen3",
            repository_root=REPO_ROOT,
        )


def test_single_model_label_wins_without_override() -> None:
    result = selection.select_tts_stage1(
        run_id="42",
        run_attempt="3",
        labels=["run-ci", "run-moss"],
        override="",
        repository_root=REPO_ROOT,
    )
    assert result.selected_model == "moss"
    assert result.selection_reason == "label:run-moss"


def test_default_topology_is_multi_gpu_and_resolves_validated_config() -> None:
    result = selection.select_tts_stage1(
        run_id="42",
        run_attempt="1",
        labels=["run-higgs"],
        override="",
        repository_root=REPO_ROOT,
    )
    resolved = REPO_ROOT / result.resolved_mps_config
    assert result.tts_stage1_topology == "multi_gpu"
    assert result.resolved_mps_config == ("examples/mps_dp/configs/higgs_h100_dp3.yaml")
    assert resolved.is_file()
    assert result.resolved_config_class == "HiggsTtsPipelineConfig"


def test_moss_resolves_the_merged_1124_validated_config() -> None:
    result = selection.select_tts_stage1(
        run_id="42",
        run_attempt="1",
        labels=["run-moss"],
        override="",
        repository_root=REPO_ROOT,
    )
    assert result.resolved_mps_config == (
        "examples/mps_dp/configs/moss_local_h100_dp2.yaml"
    )
    assert result.resolved_config_class == "MossTTSLocalPipelineConfig"


def test_preflight_json_contains_immutable_selection_and_provenance(
    tmp_path: Path,
) -> None:
    result = selection.select_tts_stage1(
        run_id="84",
        run_attempt="7",
        labels=[],
        override="higgs",
        topology="mps_shared",
        repository_root=REPO_ROOT,
    )
    output = tmp_path / "preflight.json"
    selection.write_preflight(output, result)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == result.as_dict()
    assert payload["schema_version"] == 1
    assert payload["run_attempt"] == "7"
    assert payload["tts_stage1_topology"] == "mps_shared"


def test_multi_gpu_mps_artifact_is_explicitly_not_applicable() -> None:
    payload = artifacts.artifact_envelope(
        artifact_name="mps_attachment.json",
        topology="multi_gpu",
        mps_only=True,
    )
    assert payload == {
        "schema_version": 1,
        "artifact_name": "mps_attachment.json",
        "topology": "multi_gpu",
        "status": "not_applicable",
        "reason_code": "mps_disabled_for_topology",
    }


def test_mps_artifact_is_required_for_mps_topology() -> None:
    payload = artifacts.artifact_envelope(
        artifact_name="mps_attachment.json",
        topology="mps_shared",
        mps_only=True,
    )
    assert payload["status"] == "required"
    assert "reason_code" not in payload


def test_unknown_topology_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported TTS Stage 1 topology"):
        artifacts.artifact_envelope(
            artifact_name="normal_correctness.json",
            topology="one_gpu",
            mps_only=False,
        )
