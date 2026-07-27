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
    checkout = next(
        step
        for step in preflight["steps"]
        if step.get("name") == "Checkout workflow scripts"
    )
    assert "with" not in checkout
    selection_step = next(
        step
        for step in preflight["steps"]
        if "tts_stage1_selection.py" in step.get("run", "")
    )
    assert "--exact-sha" in selection_step["run"]
    assert "--workflow-url" in selection_step["run"]


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
    assert parent_inputs["tts_stage1_topology"]["default"] == "mps_shared"
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
    assert child_inputs["tts_stage1_topology"]["default"] == "mps_shared"
    assert child_inputs["tts_stage1_mps_config"]["required"] == "true"

    stage1_text = str(child["jobs"]["stage-1-non-streaming"])
    assert "TTS_STAGE1_TOPOLOGY" in stage1_text
    assert "TTS_STAGE1_MPS_CONFIG" in stage1_text
    assert "TTS_STAGE1_MPS_STATE_ROOT" in stage1_text
    state_root = next(
        step["env"]["TTS_STAGE1_MPS_STATE_ROOT"]
        for step in child["jobs"]["stage-1-non-streaming"]["steps"]
        if "TTS_STAGE1_MPS_STATE_ROOT" in step.get("env", {})
    )
    rendered_root = state_root.replace(
        "${{ env.OMNI_CI_HOME }}", "/data/omni-ci/run-30255610170"
    )
    control_socket = f"{rendered_root}/gpu-0/run-tts-30255610170-1/mps/pipe/control"
    assert len(control_socket.encode()) <= 107
    assert "examples/mps_dp/launch.sh" in stage1_text
    for job_name in (
        "stage-2-streaming",
        "stage-3-consistency",
        "stage-4-serving",
    ):
        job_text = str(child["jobs"][job_name])
        assert "TTS_STAGE1_TOPOLOGY" not in job_text
        assert "TTS_STAGE1_MPS_CONFIG" not in job_text


def test_stage1_all_topologies_finalize_cleanup_before_always_upload() -> None:
    child = _workflow(".github/workflows/test-tts-ci.yaml")
    steps = child["jobs"]["stage-1-non-streaming"]["steps"]
    names = [step.get("name") for step in steps]

    download = steps[names.index("Download current TTS preflight evidence")]
    mps_prepare = steps[names.index("Prepare explicit MPS Stage 1 runtime")]
    benchmark = steps[names.index("Run TTS non-streaming benchmark stage")]
    finalize = steps[names.index("Finalize Stage 1 evidence and cleanup")]
    upload = steps[names.index("Upload Stage 1 evidence")]
    speed_upload = steps[names.index("Upload non-streaming speed artifact")]

    audit_root = benchmark["env"]["TTS_STAGE1_AUDIT_ROOT"]
    assert "run-${{ github.run_id }}" in audit_root
    assert "attempt-${{ github.run_attempt }}" in audit_root
    assert download["with"] == {"name": "tts-stage1-preflight", "path": audit_root}
    assert "tts_stage1_runtime.py" in mps_prepare["run"]
    assert "derive-core-blocks --gpu-id 0" in mps_prepare["run"]
    assert "run_tts_stage1_attempt.sh" in benchmark["run"]
    wrapper = (REPO_ROOT / ".github/scripts/run_tts_stage1_attempt.sh").read_text()
    assert "launch.sh down" in wrapper
    assert wrapper.index("launch.sh down") < wrapper.index(
        "tts_stage1_runtime.py initialize"
    )
    assert wrapper.count("tts_stage1_runtime.py initialize") == 1
    assert "tts_stage1_runtime.py finalize" in finalize["run"]
    assert "always()" in finalize["if"]
    assert "always()" in upload["if"]
    assert "always()" in speed_upload["if"]
    assert names.index("Finalize Stage 1 evidence and cleanup") < names.index(
        "Upload Stage 1 evidence"
    )


def test_stage1_container_allows_numa_membind_for_mps_launcher() -> None:
    child = _workflow(".github/workflows/test-tts-ci.yaml")
    options = child["jobs"]["stage-1-non-streaming"]["container"]["options"]

    assert "--cap-add SYS_NICE" in options


def test_mps_normal_speed_thresholds_use_model_specific_h100_floor() -> None:
    source = (REPO_ROOT / "tests/test_model/test_tts_ci.py").read_text(encoding="utf-8")
    evidence = (REPO_ROOT / ".github/scripts/tts_stage1_evidence.py").read_text(
        encoding="utf-8"
    )

    assert "record_normal_performance" in source
    assert "MPS_NORMAL_LOAD_FLOORS" in evidence
    assert "30196509700" in evidence
    assert "30202304743" in evidence
    assert "observation_only_pending_h100_calibration" not in evidence
