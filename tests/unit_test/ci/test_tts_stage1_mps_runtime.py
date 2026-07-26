# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))


def _load(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


runtime = _load(
    "tts_stage1_mps_runtime",
    ".github/scripts/tts_stage1_mps_runtime.py",
)


def test_launch_spec_uses_production_launcher_and_fixed_two_replica_contract(
    tmp_path: Path,
) -> None:
    spec = runtime.MpsLaunchSpec(
        repository_root=REPO_ROOT,
        output_dir=tmp_path / "audit",
        state_root=tmp_path / "state",
        run_id="run-123-1",
        config_path=REPO_ROOT / "examples/mps_dp/configs/moss_local_h100_dp2.yaml",
        gpu_id=0,
        base_port=9100,
        core_blocks=("0-3", "4-7"),
        python_bin="/opt/omni/bin/python",
        serve_extra_args="--allowed-local-media-path /tmp",
    )

    assert spec.command == (
        "bash",
        str(REPO_ROOT / "examples/mps_dp/launch.sh"),
        "up",
    )
    assert spec.worker_urls == (
        "http://127.0.0.1:9100",
        "http://127.0.0.1:9101",
    )
    assert spec.environment == {
        "BASE_PORT": "9100",
        "CONFIG": str(spec.config_path),
        "CORE_BLOCKS": "0-3 4-7",
        "GPU_ID": "0",
        "N": "2",
        "PYTHON_BIN": "/opt/omni/bin/python",
        "REPLICA_ACTIVITY": "1",
        "RUN_ID": "run-123-1",
        "SERVE_EXTRA_ARGS": "--allowed-local-media-path /tmp",
        "STATE_ROOT": str(tmp_path / "state"),
        "WEIGHT_SHARE": "1",
    }
    assert "MODEL" not in spec.environment


@pytest.mark.parametrize(
    ("run_id", "core_blocks", "match"),
    [
        ("123", ("0-3", "4-7"), "run-"),
        ("run-bad/path", ("0-3", "4-7"), "run-"),
        ("run-ok", ("0-3",), "exactly two"),
    ],
)
def test_launch_spec_rejects_unsafe_or_non_dp2_inputs(
    tmp_path: Path,
    run_id: str,
    core_blocks: tuple[str, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        runtime.MpsLaunchSpec(
            repository_root=REPO_ROOT,
            output_dir=tmp_path / "audit",
            state_root=tmp_path / "state",
            run_id=run_id,
            config_path=REPO_ROOT / "examples/mps_dp/configs/moss_local_h100_dp2.yaml",
            gpu_id=0,
            base_port=9100,
            core_blocks=core_blocks,
            python_bin="python",
        )


def _write_state(tmp_path: Path) -> Path:
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
        "mps_server 99\n"
        "  client 201 -> replica 0 (pgid 101, port 9100)\n"
        "  client 202 -> replica 1 (pgid 102, port 9101)\n"
        "replica 0 (port 9100): attached clients: 201\n"
        "replica 1 (port 9101): attached clients: 202\n"
        "RESULT: PASS\n",
        encoding="utf-8",
    )
    (state / "mps_control_raw.txt").write_text(
        "$ get_server_list\n99\n$ get_client_list 99\n201\n202\n",
        encoding="utf-8",
    )
    audit_dir = state / "weight_audit"
    audit_dir.mkdir()
    for role, pid in (("leader", 101), ("follower", 102)):
        (audit_dir / f"weight_share_{role}_{pid}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "pass",
                    "verified_attachment": True,
                    "role": role,
                    "run_id": "run-123",
                    "pid": pid,
                    "gpu_uuid": "GPU-abc",
                    "architecture": "HiggsMultimodalQwen3ForConditionalGeneration",
                    "model_class": "HiggsModel",
                    "manifest_hash": "manifest-1",
                    "shared_tensor_count": 2,
                    "shared_tensor_names": ["embed.weight", "head.weight"],
                    "private_tensor_names": [],
                    "private_storage_preserved_after_attachment": True,
                    "replica_local_state": {
                        "status": "not_applicable",
                        "reason_code": "no_architecture_specific_mutable_state",
                        "scope": "none",
                        "tensor_names": [],
                        "history_tensor_names": [],
                        "shared_record_intersection": [],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return state


def test_read_launcher_state_requires_two_replicas_and_equal_kv(
    tmp_path: Path,
) -> None:
    snapshot = runtime.read_launcher_state(_write_state(tmp_path))

    assert snapshot.manifest["weight_share"] == "1"
    assert [replica.port for replica in snapshot.replicas] == [9100, 9101]
    assert [replica.kv_tokens for replica in snapshot.replicas] == [30000, 30000]
    assert snapshot.equal_kv
    assert snapshot.attachment_passed


def test_write_startup_artifacts_records_runtime_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "audit"
    runtime.write_startup_artifacts(
        runtime.read_launcher_state(_write_state(tmp_path)),
        output_dir=output,
    )

    launch = json.loads((output / "launch_manifest.json").read_text())
    replicas = json.loads((output / "replica_manifest.json").read_text())
    status = json.loads((output / "mps_status.json").read_text())
    attachment = json.loads((output / "mps_attachment.json").read_text())
    equal_kv = json.loads((output / "equal_kv.json").read_text())
    weight_audit = json.loads((output / "weight_share_audit.json").read_text())
    private_audit = json.loads((output / "private_tensor_audit.json").read_text())

    assert launch["topology"] == "mps_shared"
    assert launch["run_id"] == "run-123"
    assert replicas["replicas"][0]["role"] == "leader"
    assert replicas["replicas"][1]["role"] == "follower"
    assert status["status"] == "pass"
    assert attachment["status"] == "pass"
    assert equal_kv["status"] == "pass"
    assert equal_kv["expected_max_total_tokens"] == 30000
    assert weight_audit["status"] == "pass"
    assert {entry["role"] for entry in weight_audit["replicas"]} == {
        "leader",
        "follower",
    }
    assert private_audit["status"] == "pass"
    assert private_audit["private_tensor_names"] == []
    assert (output / "raw_mps_control.txt").read_text() == (
        "$ get_server_list\n99\n$ get_client_list 99\n201\n202\n"
    )


def test_cpu_list_round_trip_for_numa_affinity() -> None:
    cores = runtime.parse_cpu_list("0-3,8,10-11")
    assert cores == [0, 1, 2, 3, 8, 10, 11]
    assert runtime.format_cpu_list(cores) == "0-3,8,10-11"


def test_post_launch_validation_failure_attempts_run_specific_teardown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = runtime.MpsLaunchSpec(
        repository_root=REPO_ROOT,
        output_dir=tmp_path / "audit",
        state_root=tmp_path / "state",
        run_id="run-cleanup-test",
        config_path=REPO_ROOT / "examples/mps_dp/configs/moss_local_h100_dp2.yaml",
        gpu_id=0,
        base_port=9100,
        core_blocks=("0-3", "4-7"),
        python_bin="python",
    )
    commands: list[tuple[str, ...]] = []

    def record_run(command, **kwargs):
        commands.append(tuple(command))

    def fail_validation(state_dir):
        raise ValueError("invalid attachment evidence")

    monkeypatch.setattr(runtime.subprocess, "run", record_run)
    monkeypatch.setattr(runtime, "read_launcher_state", fail_validation)

    with pytest.raises(ValueError, match="invalid attachment evidence"):
        runtime.launch_replicas(spec)

    assert commands == [spec.command, spec.teardown_command]


def test_collect_replica_activity_requires_same_run_monotonic_host_boot(
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
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )

    output = tmp_path / "audit"
    collected = runtime.collect_replica_activity(snapshot, output_dir=output)

    assert len(collected) == 4
    persisted = [
        json.loads(line)
        for line in (output / "replica_activity.jsonl").read_text().splitlines()
    ]
    assert persisted == collected
    assert {event["replica_id"] for event in collected} == {0, 1}


def test_moss_private_and_history_provenance_is_preserved(tmp_path: Path) -> None:
    state = _write_state(tmp_path)
    local_state = {
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
    for path in (state / "weight_audit").glob("weight_share_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["architecture"] = "MossTTSLocalSGLangModel"
        payload["private_tensor_names"] = ["_decode_input_embedding.weight"]
        payload["private_storage_preserved_after_attachment"] = True
        payload["replica_local_state"] = local_state
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    output = tmp_path / "audit"
    runtime.write_startup_artifacts(
        runtime.read_launcher_state(state),
        output_dir=output,
    )

    private = json.loads((output / "private_tensor_audit.json").read_text())
    assert private["private_tensor_names"] == ["_decode_input_embedding.weight"]
    assert private["follower_private_storage_preserved"] is True
    assert private["replica_local_state"] == local_state
    assert private["history_provenance"] == {
        "scope": "replica_process",
        "tensor_names": ["audio_token_presence"],
        "exported_via_weight_share": False,
    }


def test_teardown_writes_clean_mps_and_pre_evaluator_verdicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _write_state(tmp_path)
    control = state / "mps" / "pipe" / "nvidia-cuda-mps-control.pid"
    control.parent.mkdir(parents=True)
    control.write_text("999\n", encoding="utf-8")
    snapshot = runtime.read_launcher_state(state)
    spec = runtime.MpsLaunchSpec(
        repository_root=REPO_ROOT,
        output_dir=tmp_path / "audit",
        state_root=tmp_path / "state-root",
        run_id="run-clean-teardown",
        config_path=REPO_ROOT / "examples/mps_dp/configs/higgs_h100_dp2.yaml",
        gpu_id=0,
        base_port=9100,
        core_blocks=("0-3", "4-7"),
        python_bin="python",
    )

    monkeypatch.setattr(runtime, "read_launcher_state", lambda _: snapshot)
    monkeypatch.setattr(runtime, "_tracked_pids", lambda _: ({101, 102}, True))
    monkeypatch.setattr(runtime, "_pid_alive", lambda _: False)
    monkeypatch.setattr(runtime, "_gpu_client_pids", lambda: (set(), True))
    monkeypatch.setattr(runtime, "_occupied_ports", lambda _: [])

    def clean_down(*args, **kwargs):
        shutil.rmtree(state)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime.subprocess, "run", clean_down)
    runtime.teardown_replicas(
        spec,
        router_stopped=True,
        requests_drained_or_cancelled=True,
    )

    teardown = json.loads((spec.output_dir / "mps_teardown_verdict.json").read_text())
    barrier = json.loads(
        (spec.output_dir / "pre_evaluator_cleanup_verdict.json").read_text()
    )
    assert teardown["clean"] is True
    assert teardown["checks"]["mps_control_pid_observed"] is True
    assert barrier["clean"] is True
    assert barrier["evaluator_start_allowed"] is True


def test_teardown_runs_production_down_when_launcher_evidence_is_corrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = runtime.MpsLaunchSpec(
        repository_root=REPO_ROOT,
        output_dir=tmp_path / "audit",
        state_root=tmp_path / "state-root",
        run_id="run-corrupt-evidence",
        config_path=REPO_ROOT / "examples/mps_dp/configs/higgs_h100_dp2.yaml",
        gpu_id=0,
        base_port=9200,
        core_blocks=("0-3", "4-7"),
        python_bin="python",
    )
    spec.state_dir.mkdir(parents=True)
    (spec.state_dir / "manifest").write_text("corrupt\n", encoding="utf-8")
    commands = []

    def record_down(command, **kwargs):
        commands.append(tuple(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime.subprocess, "run", record_down)
    monkeypatch.setattr(runtime, "_gpu_client_pids", lambda: (set(), True))
    monkeypatch.setattr(runtime, "_occupied_ports", lambda _: [])

    with pytest.raises(RuntimeError, match="MPS teardown is dirty"):
        runtime.teardown_replicas(
            spec,
            router_stopped=True,
            requests_drained_or_cancelled=True,
        )

    assert spec.teardown_command in commands
    teardown = json.loads((spec.output_dir / "mps_teardown_verdict.json").read_text())
    barrier = json.loads(
        (spec.output_dir / "pre_evaluator_cleanup_verdict.json").read_text()
    )
    assert teardown["clean"] is False
    assert teardown["checks"]["launch_down_succeeded"] is True
    assert teardown["checks"]["process_query_succeeded"] is False
    assert teardown["evidence_errors"]
    assert barrier["evaluator_start_allowed"] is False
