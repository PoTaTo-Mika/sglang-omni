# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))

import tts_stage1_evidence as evidence  # noqa: E402
import tts_stage1_runtime as runtime  # noqa: E402


def _gpu_runner(command, **kwargs):
    if "--query-gpu=index,uuid,name,driver_version" in command:
        return SimpleNamespace(
            returncode=0, stdout="0, GPU-abc, H100, 570.1\n", stderr=""
        )
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def _init(root: Path, *, topology: str = "mps_shared", model: str = "higgs") -> None:
    runtime.initialize_stage1(
        root,
        topology=topology,
        model=model,
        configured_timeout_minutes=25,
        command_runner=_gpu_runner,
    )


def _spec(tmp_path: Path, *, run_id: str = "run-123") -> runtime.MpsLaunchSpec:
    return runtime.MpsLaunchSpec(
        repository_root=REPO_ROOT,
        output_dir=tmp_path / "audit",
        state_root=tmp_path / "state-root",
        run_id=run_id,
        config_path=REPO_ROOT / "examples/mps_dp/configs/higgs_h100_dp2.yaml",
        gpu_id=0,
        base_port=9100,
        core_blocks=("0-3", "4-7"),
        python_bin="/opt/omni/bin/python",
    )


def _write_state(tmp_path: Path, *, moss: bool = False) -> Path:
    state = tmp_path / "state"
    logs = state / "logs"
    logs.mkdir(parents=True)
    (state / "manifest").write_text(
        "run_id=run-123\ngpu_id=0\ngpu_uuid=GPU-abc\nnuma_node=0\n"
        "config=/repo/config.yaml\nn=2\nbase_port=9100\n"
        "core_blocks=0-3 4-7\nmax_total_tokens=30000\nweight_share=1\n",
        encoding="utf-8",
    )
    (state / "replicas.tsv").write_text(
        "0\t101\t101\t9100\tlogs/replica_0.log\tstart0\n"
        "1\t102\t102\t9101\tlogs/replica_1.log\tstart1\n",
        encoding="utf-8",
    )
    for index in range(2):
        (logs / f"replica_{index}.log").write_text(
            "KV pool is allocated. #tokens: 30000\n", encoding="utf-8"
        )
    (state / "mps_attach.txt").write_text(
        "replica 0 (port 9100): attached clients: 201\n"
        "replica 1 (port 9101): attached clients: 202\nRESULT: PASS\n",
        encoding="utf-8",
    )
    (state / "mps_control_raw.txt").write_text(
        "$ get_server_list\n99\n", encoding="utf-8"
    )
    audit_dir = state / "weight_audit"
    audit_dir.mkdir()
    architecture = (
        "MossTTSLocalSGLangModel"
        if moss
        else "HiggsMultimodalQwen3ForConditionalGeneration"
    )
    private = ["_decode_input_embedding.weight"] if moss else []
    local = (
        {
            "status": "pass",
            "scope": "process_local_unregistered_tensors",
            "tensor_names": [
                "audio_token_presence",
                "feedback_embeds",
                "generation_steps",
                "sampling_steps",
            ],
            "history_tensor_names": ["audio_token_presence"],
            "shared_record_intersection": [],
        }
        if moss
        else {
            "status": "not_applicable",
            "reason_code": "no_architecture_specific_mutable_state",
            "scope": "none",
            "tensor_names": [],
            "history_tensor_names": [],
            "shared_record_intersection": [],
        }
    )
    for role, pid in (("leader", 101), ("follower", 102)):
        payload = {
            "schema_version": 1,
            "status": "pass",
            "verified_attachment": True,
            "role": role,
            "run_id": "run-123",
            "pid": pid,
            "gpu_uuid": "GPU-abc",
            "architecture": architecture,
            "model_class": "TtsModel",
            "manifest_hash": "manifest-1",
            "shared_tensor_count": 2,
            "shared_tensor_names": ["embed.weight", "head.weight"],
            "private_tensor_names": private,
            "private_shared_storage_intersection": [],
            "private_storage_preserved_after_attachment": True,
            "replica_local_state": local,
        }
        (audit_dir / f"weight_share_{role}_{pid}.json").write_text(
            json.dumps(payload) + "\n", encoding="utf-8"
        )
    return state


def test_launch_spec_uses_production_launcher_and_fixed_dp2(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    assert spec.command == ("bash", str(REPO_ROOT / "examples/mps_dp/launch.sh"), "up")
    assert spec.worker_urls == ("http://127.0.0.1:9100", "http://127.0.0.1:9101")
    assert spec.environment["N"] == "2"
    assert spec.environment["WEIGHT_SHARE"] == "1"
    assert spec.environment["WEIGHT_SHARE_AUDIT"] == "1"
    assert spec.environment["REPLICA_ACTIVITY"] == "1"
    with pytest.raises(ValueError, match="run-"):
        _spec(tmp_path, run_id="unsafe/path")


def test_launcher_manifest_aggregates_attachment_kv_and_moss_private_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "audit"
    _init(root, model="moss")
    snapshot = runtime.read_launcher_state(_write_state(tmp_path, moss=True))
    runtime.write_mps_runtime(
        snapshot,
        output_dir=root,
        command_runner=lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )

    payload = evidence.read_evidence(
        root / "stage1_runtime.json", expected_kind="runtime"
    )
    assert payload["launch"]["replica_count"] == 2
    assert [item["max_total_tokens"] for item in payload["replicas"]] == [30000, 30000]
    assert payload["mps"]["attachment"]["replicas"] == [0, 1]
    assert payload["mps"]["equal_kv"] == {"status": "pass", "max_total_tokens": 30000}
    private = payload["model_audits"]["private_state"]
    assert private["private_tensor_names"] == ["_decode_input_embedding.weight"]
    assert private["follower_private_storage_preserved"] is True
    assert private["replica_local_state"]["shared_record_intersection"] == []
    assert private["history_provenance"]["tensor_names"] == ["audio_token_presence"]
    assert (root / "raw/mps/mps_control_raw.txt").is_file()
    assert not (root / "mps_attachment.json").exists()


def test_launcher_rejects_private_tensor_alias_to_shared_storage(
    tmp_path: Path,
) -> None:
    state = _write_state(tmp_path, moss=True)
    audit = next((state / "weight_audit").glob("weight_share_follower_*.json"))
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["private_shared_storage_intersection"] = ["_decode_input_embedding.weight"]
    audit.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="private tensors alias shared storage"):
        runtime.read_launcher_state(state)


def test_failed_launcher_start_runs_production_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path)
    spec.state_dir.mkdir(parents=True)
    calls: list[tuple[str, ...]] = []

    def run(command, **kwargs):
        calls.append(tuple(command))
        if tuple(command) == spec.command:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        runtime.launch_replicas(spec)

    assert calls == [spec.command, spec.teardown_command]


def test_cpu_list_round_trip_and_zombie_is_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cores = runtime.parse_cpu_list("0-3,8,10-11")
    assert runtime.format_cpu_list(cores) == "0-3,8,10-11"
    monkeypatch.setattr(runtime.os, "kill", lambda *_: None)
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="Z\n"),
    )
    assert runtime._pid_alive(123) is False


def test_post_launch_validation_failure_still_runs_production_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, run_id="run-cleanup")
    spec.state_dir.mkdir(parents=True)
    commands = []
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda command, **kwargs: commands.append(tuple(command)),
    )
    monkeypatch.setattr(
        runtime,
        "read_launcher_state",
        lambda _: (_ for _ in ()).throw(ValueError("invalid attachment")),
    )
    with pytest.raises(ValueError, match="invalid attachment"):
        runtime.launch_replicas(spec)
    assert commands == [spec.command, spec.teardown_command]


def test_collect_replica_activity_requires_server_monotonic_identity(
    tmp_path: Path,
) -> None:
    snapshot = runtime.read_launcher_state(_write_state(tmp_path))
    for replica_id in (0, 1):
        events = [
            {
                "schema_version": 1,
                "clock": "CLOCK_MONOTONIC",
                "monotonic_ns": 100 + replica_id,
                "run_id": "run-123",
                "replica_id": replica_id,
                "request_id": f"req-{replica_id}",
                "pid": 101 + replica_id,
                "host_boot_id": "boot-1",
                "event": "model_path_start",
            },
            {
                "schema_version": 1,
                "clock": "CLOCK_MONOTONIC",
                "monotonic_ns": 200 + replica_id,
                "run_id": "run-123",
                "replica_id": replica_id,
                "request_id": f"req-{replica_id}",
                "pid": 101 + replica_id,
                "host_boot_id": "boot-1",
                "event": "model_path_end",
                "status": "success",
            },
        ]
        (snapshot.state_dir / f"replica_activity_{replica_id}.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in events), encoding="utf-8"
        )
    collected = runtime.collect_replica_activity(
        snapshot, output_dir=tmp_path / "audit"
    )
    assert len(collected) == 4
    assert {item["replica_id"] for item in collected} == {0, 1}


def test_clean_mps_teardown_permits_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "audit"
    _init(root)
    state = _write_state(tmp_path)
    control = state / "mps/pipe/nvidia-cuda-mps-control.pid"
    control.parent.mkdir(parents=True)
    control.write_text("999\n")
    snapshot = runtime.read_launcher_state(state)
    spec = _spec(tmp_path)
    monkeypatch.setattr(runtime, "read_launcher_state", lambda _: snapshot)
    monkeypatch.setattr(runtime, "_tracked_pids", lambda _: ({101, 102}, True))
    monkeypatch.setattr(runtime, "_pid_alive", lambda _: False)
    monkeypatch.setattr(
        runtime,
        "capture_gpu_state",
        lambda: {"status": "pass", "gpus": [], "compute_clients": [], "errors": []},
    )
    monkeypatch.setattr(runtime, "_occupied_ports", lambda _: [])

    def clean_down(*args, **kwargs):
        shutil.rmtree(state)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime.subprocess, "run", clean_down)
    verdict = runtime.teardown_mps(
        spec, router_stopped=True, requests_drained_or_cancelled=True
    )
    assert verdict["clean"] is True
    assert (
        evidence.require_evaluator_start_allowed(root)["evaluator_start_allowed"]
        is True
    )


def test_corrupt_mps_evidence_runs_down_and_blocks_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "audit"
    _init(root)
    spec = _spec(tmp_path, run_id="run-corrupt")
    spec.state_dir.mkdir(parents=True)
    (spec.state_dir / "manifest").write_text("corrupt\n")
    commands = []
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda command, **kwargs: commands.append(tuple(command))
        or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        runtime,
        "capture_gpu_state",
        lambda: {"status": "pass", "gpus": [], "compute_clients": [], "errors": []},
    )
    monkeypatch.setattr(runtime, "_occupied_ports", lambda _: [])
    with pytest.raises(RuntimeError, match="evaluator is blocked"):
        runtime.teardown_mps(
            spec, router_stopped=True, requests_drained_or_cancelled=True
        )
    assert spec.teardown_command in commands
    with pytest.raises(RuntimeError, match="evaluator is blocked"):
        evidence.require_evaluator_start_allowed(root)


def test_multi_gpu_runtime_and_cleanup_stay_simple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "audit"
    _init(root, topology="multi_gpu")
    manifest = tmp_path / "router_pgids.txt"
    manifest.write_text("100\n101\n")
    runtime.write_multi_gpu_runtime(
        root,
        model="higgs",
        router_pid=10,
        router_port=3000,
        worker_ports=[3001, 3002],
        num_gpus_per_worker=1,
        router_ready_s=4.5,
        cleanup_manifest=manifest,
        before_workers={"workers": ["a", "b"]},
        command_runner=lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )
    runtime.write_router_snapshot(root, when="after", snapshot={"workers": ["a", "b"]})
    monkeypatch.setattr(runtime, "_process_group_alive", lambda _: False)
    monkeypatch.setattr(runtime, "_occupied_ports", lambda _: [])
    verdict = runtime.teardown_multi_gpu(
        root,
        cleanup_manifest=manifest,
        router_port=3000,
        worker_ports=[3001, 3002],
        router_stopped=True,
        requests_drained_or_cancelled=True,
        gpu_memory_released=True,
    )
    assert verdict["clean"] is True
    payload = evidence.read_evidence(
        root / "stage1_runtime.json", expected_kind="runtime"
    )
    assert payload["mps"]["status"] == "not_applicable"
    assert not (root / "ordinary_teardown_verdict.json").exists()


def test_dirty_multi_gpu_cleanup_blocks_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "audit"
    _init(root, topology="multi_gpu")
    manifest = tmp_path / "router_pgids.txt"
    manifest.write_text("corrupt\n")
    monkeypatch.setattr(runtime, "_occupied_ports", lambda _: [])
    with pytest.raises(RuntimeError, match="evaluator is blocked"):
        runtime.teardown_multi_gpu(
            root,
            cleanup_manifest=manifest,
            router_port=3000,
            worker_ports=[3001, 3002],
            router_stopped=True,
            requests_drained_or_cancelled=True,
            gpu_memory_released=True,
        )
    cleanup = evidence.read_evidence(
        root / "stage1_cleanup.json", expected_kind="cleanup"
    )
    assert cleanup["verdict"] == "dirty"
    assert cleanup["retained_diagnostics"] == str(manifest)


def test_record_router_process_requires_cleanup_manifest_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "audit"
    _init(root)
    manifest = tmp_path / "router_pgids.txt"
    manifest.write_text("300\n", encoding="utf-8")
    monkeypatch.setattr(runtime.os, "getpgid", lambda pid: pid, raising=False)

    runtime.record_router_process(root, pid=300, port=3000, cleanup_manifest=manifest)

    payload = evidence.read_evidence(
        root / "stage1_runtime.json", expected_kind="runtime"
    )
    assert payload["router"]["process"] == {
        "pid": 300,
        "port": 3000,
        "process_groups": [300],
        "cleanup_manifest": str(manifest),
    }


def test_final_cleanup_detects_mps_router_leftover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "audit"
    _init(root, topology="mps_shared")
    snapshot = runtime.read_launcher_state(_write_state(tmp_path))
    runtime.write_mps_runtime(
        snapshot,
        output_dir=root,
        command_runner=lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )
    runtime.write_router_snapshot(root, when="before", snapshot={"workers": ["a", "b"]})
    runtime.write_router_snapshot(root, when="after", snapshot={"workers": ["a", "b"]})
    runtime_payload = evidence.read_evidence(
        root / "stage1_runtime.json", expected_kind="runtime"
    )
    runtime_payload["router"]["process"] = {
        "pid": 300,
        "port": 3000,
        "process_groups": [300],
    }
    evidence.write_json_atomic(root / "stage1_runtime.json", runtime_payload)
    for component in evidence.REQUIRED_CORRECTNESS:
        evidence.record_correctness_component(
            root, component=component, details={"ok": True}
        )
    evidence.record_overlap_summary(root, {"status": "pass"})
    evidence.record_model_audits(root, {"status": "pass"})
    evidence.record_normal_performance(
        root,
        concurrency=16,
        summary={
            "total_requests": 1088,
            "completed_requests": 1088,
            "failed_requests": 0,
            "throughput_qps": 14.0,
            "latency_mean_s": 1.1,
            "rtf_mean": 0.27,
            "audio_throughput_s_per_s": 59.0,
            "output_throughput": 1500.0,
            "output_tok_per_req_s": 110.0,
        },
    )
    evidence.record_generation_cleanup(root, checks={"clean": True})
    evidence.record_evaluator(root, status="running", pid=400, port=4000)
    evidence.record_evaluator(root, status="completed", pid=None, port=None)
    evidence.write_json_atomic(
        root / "preflight.json",
        {
            "schema_version": 1,
            "selected_model": "higgs",
            "selection_digest": "digest",
            "selection_reason": "test",
            "tts_stage1_topology": "mps_shared",
            "resolved_mps_config": "config.yaml",
            "resolved_config_class": "HiggsTtsPipelineConfig",
            "run_id": "",
            "run_attempt": "",
            "exact_sha": "abc123",
            "workflow_url": "https://github.test/run/1",
        },
    )

    monkeypatch.setattr(runtime, "_pid_alive", lambda pid: pid == 300)
    monkeypatch.setattr(runtime, "_process_group_alive", lambda pgid: pgid == 300)
    monkeypatch.setattr(
        runtime, "_occupied_ports", lambda ports: [3000] if 3000 in ports else []
    )
    with pytest.raises(RuntimeError, match="final cleanup/evidence is dirty"):
        runtime.finalize_stage1(
            root,
            topology="mps_shared",
            model="higgs",
            exact_sha="abc123",
            command_runner=_gpu_runner,
        )

    cleanup = evidence.read_evidence(
        root / "stage1_cleanup.json", expected_kind="cleanup"
    )
    assert cleanup["final_post_state"]["live_tracked_processes"] == [300]
    assert cleanup["final_post_state"]["occupied_ports"] == [3000]

    runtime_payload = evidence.read_evidence(
        root / "stage1_runtime.json", expected_kind="runtime"
    )
    runtime_payload["router"].pop("process")
    evidence.write_json_atomic(root / "stage1_runtime.json", runtime_payload)
    monkeypatch.setattr(runtime, "_pid_alive", lambda _: False)
    monkeypatch.setattr(runtime, "_process_group_alive", lambda _: False)
    monkeypatch.setattr(runtime, "_occupied_ports", lambda _: [])
    with pytest.raises(RuntimeError, match="final cleanup/evidence is dirty"):
        runtime.finalize_stage1(
            root,
            topology="mps_shared",
            model="higgs",
            exact_sha="abc123",
            command_runner=_gpu_runner,
        )
    cleanup = evidence.read_evidence(
        root / "stage1_cleanup.json", expected_kind="cleanup"
    )
    assert cleanup["final_checks"]["router_process_recorded"] is False


def test_final_cleanup_detects_evaluator_port_and_process_leftovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "audit"
    _init(root, topology="multi_gpu")
    manifest = tmp_path / "router_pgids.txt"
    manifest.write_text("100\n", encoding="utf-8")
    runtime.write_multi_gpu_runtime(
        root,
        model="higgs",
        router_pid=10,
        router_port=3000,
        worker_ports=[3001, 3002],
        num_gpus_per_worker=1,
        router_ready_s=1.0,
        cleanup_manifest=manifest,
        before_workers={"workers": ["a", "b"]},
        command_runner=lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )
    runtime.write_router_snapshot(root, when="after", snapshot={"workers": ["a", "b"]})
    for component in evidence.REQUIRED_CORRECTNESS:
        evidence.record_correctness_component(
            root, component=component, details={"ok": True}
        )
    evidence.record_normal_performance(
        root,
        concurrency=16,
        summary={"completed_requests": 1088, "failed_requests": 0},
    )
    monkeypatch.setattr(runtime, "_process_group_alive", lambda _: False)
    monkeypatch.setattr(runtime, "_occupied_ports", lambda _: [])
    runtime.teardown_multi_gpu(
        root,
        cleanup_manifest=manifest,
        router_port=3000,
        worker_ports=[3001, 3002],
        router_stopped=True,
        requests_drained_or_cancelled=True,
        gpu_memory_released=True,
    )
    evidence.write_json_atomic(
        root / "preflight.json",
        {
            "schema_version": 1,
            "selected_model": "higgs",
            "selection_digest": "digest",
            "selection_reason": "test",
            "tts_stage1_topology": "multi_gpu",
            "resolved_mps_config": "examples/mps_dp/configs/higgs_h100_dp2.yaml",
            "resolved_config_class": "HiggsTtsPipelineConfig",
            "run_id": "",
            "run_attempt": "",
            "exact_sha": "abc123",
            "workflow_url": "https://github.test/run/1",
        },
    )
    evidence.record_evaluator(root, status="running", pid=400, port=4000)
    evidence.record_evaluator(root, status="completed", pid=None, port=None)

    monkeypatch.setattr(runtime, "_pid_alive", lambda pid: pid == 400)
    monkeypatch.setattr(
        runtime, "_occupied_ports", lambda ports: [4000] if 4000 in ports else []
    )
    with pytest.raises(RuntimeError, match="final cleanup/evidence is dirty"):
        runtime.finalize_stage1(
            root,
            topology="multi_gpu",
            model="higgs",
            exact_sha="abc123",
            command_runner=_gpu_runner,
        )

    cleanup = evidence.read_evidence(
        root / "stage1_cleanup.json", expected_kind="cleanup"
    )
    assert cleanup["clean"] is False
    assert cleanup["final_post_state"]["live_tracked_processes"] == [400]
    assert cleanup["final_post_state"]["occupied_ports"] == [4000]
    assert cleanup["evaluator"]["observed_port"] == 4000
