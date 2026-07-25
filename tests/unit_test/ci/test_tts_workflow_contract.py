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
