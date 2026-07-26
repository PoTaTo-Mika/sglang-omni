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
    "tts_stage1_ordinary_runtime",
    REPO_ROOT / ".github/scripts/tts_stage1_ordinary_runtime.py",
)
runtime = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)


def test_startup_artifacts_record_router_ports_gpu_mapping_and_process_tree(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "router_pgids.txt"
    manifest.write_text("100\n101\n", encoding="utf-8")

    def ps(command, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=f"1 0 {command[-1]} S python\n",
            stderr="",
        )

    runtime.write_startup_artifacts(
        tmp_path / "audit",
        model="moss",
        router_pid=10,
        router_port=3000,
        worker_ports=[3001, 3002],
        num_gpus_per_worker=1,
        router_ready_s=4.5,
        cleanup_manifest=manifest,
        before_workers={"workers": ["a", "b"]},
        command_runner=ps,
    )

    launch = json.loads((tmp_path / "audit/launch_manifest.json").read_text())
    replicas = json.loads((tmp_path / "audit/replica_manifest.json").read_text())
    tree = json.loads((tmp_path / "audit/process_tree.json").read_text())
    assert launch["topology"] == "multi_gpu"
    assert launch["router"] == {"pid": 10, "port": 3000}
    assert [item["gpu_slots"] for item in replicas["replicas"]] == [[0], [1]]
    assert tree["status"] == "pass"
    assert [item["tracked_pgid"] for item in tree["groups"]] == [100, 101]


def test_clean_ordinary_teardown_writes_evaluator_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "router_pgids.txt"
    manifest.write_text("100\n101\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "_process_group_alive", lambda _: False)
    monkeypatch.setattr(runtime, "_occupied_ports", lambda _: [])
    runtime.write_startup_artifacts(
        tmp_path / "audit",
        model="higgs",
        router_pid=10,
        router_port=3000,
        worker_ports=[3001, 3002],
        num_gpus_per_worker=1,
        router_ready_s=4.5,
        cleanup_manifest=manifest,
        before_workers={"workers": ["a", "b"]},
        command_runner=lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )
    runtime.write_router_after(tmp_path / "audit", snapshot={"workers": ["a", "b"]})

    verdict = runtime.finalize_ordinary_teardown(
        tmp_path / "audit",
        cleanup_manifest=manifest,
        router_port=3000,
        worker_ports=[3001, 3002],
        router_stopped=True,
        requests_drained_or_cancelled=True,
        gpu_memory_released=True,
        environ={},
    )

    barrier = json.loads(
        (tmp_path / "audit/pre_evaluator_cleanup_verdict.json").read_text()
    )
    assert verdict["clean"] is True
    assert barrier["clean"] is True
    assert barrier["source_artifact"] == "ordinary_teardown_verdict.json"


def test_invalid_process_manifest_fails_closed_and_retains_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "router_pgids.txt"
    manifest.write_text("corrupt\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "_occupied_ports", lambda _: [])

    with pytest.raises(RuntimeError, match="ordinary Stage 1 teardown is dirty"):
        runtime.finalize_ordinary_teardown(
            tmp_path / "audit",
            cleanup_manifest=manifest,
            router_port=3000,
            worker_ports=[3001, 3002],
            router_stopped=True,
            requests_drained_or_cancelled=True,
            gpu_memory_released=True,
            environ={},
        )

    verdict = json.loads(
        (tmp_path / "audit/ordinary_teardown_verdict.json").read_text()
    )
    assert verdict["clean"] is False
    assert verdict["retained_state"] == str(manifest)
