# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def _workflow(relative_path: str) -> dict:
    return yaml.load(
        (REPO_ROOT / relative_path).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )


def test_model_selection_is_a_cpu_preflight_output_before_h100_setup() -> None:
    workflow = _workflow(".github/workflows/omni-ci.yaml")
    jobs = workflow["jobs"]
    preflight = jobs["preflight"]

    assert preflight["runs-on"] == "ubuntu-latest"
    assert "selected_model" in preflight["outputs"]
    assert "selection_digest" in preflight["outputs"]
    assert "tts_stage1_topology" in preflight["outputs"]
    assert "pick-tts-model" not in jobs
    assert "preflight" in jobs["setup"]["needs"]
    upload = next(
        step
        for step in preflight["steps"]
        if step.get("name") == "Upload TTS preflight artifact"
    )
    assert upload["with"]["path"] == "${{ runner.temp }}/tts-stage1-preflight"


def test_tts_job_passes_one_preflight_selection_to_stages_one_through_three() -> None:
    parent = _workflow(".github/workflows/omni-ci.yaml")
    child = _workflow(".github/workflows/test-tts-ci.yaml")

    tts_job = parent["jobs"]["tts-ci"]
    assert "preflight" in tts_job["needs"]
    assert tts_job["with"]["tts_ci_model"] == (
        "${{ needs.preflight.outputs.selected_model }}"
    )

    child_jobs = child["jobs"]
    for job_name in (
        "stage-1-non-streaming",
        "stage-2-streaming",
        "stage-3-consistency",
    ):
        env = child_jobs[job_name]["steps"]
        run_steps = [
            step for step in env if "env" in step and "TTS_CI_MODEL" in step["env"]
        ]
        assert run_steps
        assert all(
            step["env"]["TTS_CI_MODEL"] == "${{ inputs.tts_ci_model }}"
            for step in run_steps
        )

    assert "TTS_CI_MODEL" not in str(child_jobs["stage-4-serving"])


def test_stage1_alone_consumes_explicit_topology_and_validated_config() -> None:
    parent = _workflow(".github/workflows/omni-ci.yaml")
    child = _workflow(".github/workflows/test-tts-ci.yaml")

    parent_inputs = parent["on"]["workflow_dispatch"]["inputs"]
    assert parent_inputs["tts_stage1_topology"]["default"] == "multi_gpu"
    assert parent_inputs["tts_stage1_topology"]["options"] == [
        "multi_gpu",
        "mps_shared",
    ]

    parent_call = parent["jobs"]["tts-ci"]["with"]
    assert parent_call["tts_stage1_topology"] == (
        "${{ needs.preflight.outputs.tts_stage1_topology }}"
    )
    assert parent_call["tts_stage1_mps_config"] == (
        "${{ needs.preflight.outputs.resolved_mps_config }}"
    )

    child_inputs = child["on"]["workflow_call"]["inputs"]
    assert child_inputs["tts_stage1_topology"]["default"] == "multi_gpu"
    assert child_inputs["tts_stage1_mps_config"]["required"] == "true"

    stage1_text = str(child["jobs"]["stage-1-non-streaming"])
    assert "TTS_STAGE1_TOPOLOGY" in stage1_text
    assert "TTS_STAGE1_MPS_CONFIG" in stage1_text
    assert "TTS_STAGE1_MPS_STATE_ROOT" in stage1_text
    assert "tts-stage1-mps-state" in stage1_text
    assert "examples/mps_dp/launch.sh" in stage1_text
    for job_name in (
        "stage-2-streaming",
        "stage-3-consistency",
        "stage-4-serving",
    ):
        job_text = str(child["jobs"][job_name])
        assert "TTS_STAGE1_TOPOLOGY" not in job_text
        assert "TTS_STAGE1_MPS_CONFIG" not in job_text
