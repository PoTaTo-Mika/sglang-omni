# SPDX-License-Identifier: Apache-2.0
"""Topology runtime and cleanup control for TTS Stage 1 CI."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tts_stage1_evidence import (
    finalize_timing,
    initialize_evidence,
    read_evidence,
    record_generation_cleanup,
    record_model_audits,
    record_runtime_section,
    write_json_atomic,
)

REPLICA_COUNT = 2
RUN_ID_PATTERN = re.compile(r"^run-[A-Za-z0-9_-]+$")
KV_PATTERN = re.compile(r"#tokens:\s*([0-9]+)")

EXPECTED_PRIVATE_TENSORS = {
    "HiggsMultimodalQwen3ForConditionalGeneration": [],
    "MossTTSLocalSGLangModel": ["_decode_input_embedding.weight"],
}


def parse_cpu_list(value: str) -> list[int]:
    cores: set[int] = set()
    for item in value.strip().split(","):
        if not item:
            continue
        start_text, separator, end_text = item.partition("-")
        start = int(start_text)
        end = int(end_text) if separator else start
        if start < 0 or end < start:
            raise ValueError(f"invalid CPU list element: {item!r}")
        cores.update(range(start, end + 1))
    if not cores:
        raise ValueError("CPU list is empty")
    return sorted(cores)


def format_cpu_list(cores: list[int]) -> str:
    if not cores:
        raise ValueError("cannot format an empty CPU list")
    ordered = sorted(set(cores))
    ranges: list[str] = []
    start = previous = ordered[0]
    for core in ordered[1:]:
        if core == previous + 1:
            previous = core
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = core
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def derive_core_blocks(gpu_id: int) -> tuple[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=pci.bus_id",
            "--format=csv,noheader",
            "-i",
            str(gpu_id),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    bus_suffix = result.stdout.strip().lower().split(":", 1)[-1]
    matches = list(Path("/sys/bus/pci/devices").glob(f"*:{bus_suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve one PCI sysfs device for GPU {gpu_id}")
    numa_node = int((matches[0] / "numa_node").read_text().strip())
    if numa_node < 0:
        raise RuntimeError(f"GPU {gpu_id} has no usable NUMA node")
    node_cores = set(
        parse_cpu_list(
            Path(f"/sys/devices/system/node/node{numa_node}/cpulist")
            .read_text()
            .strip()
        )
    )
    allowed = sorted(node_cores & set(os.sched_getaffinity(0)))
    server_cores = allowed[: max(REPLICA_COUNT, len(allowed) * 3 // 4)]
    if len(server_cores) < 4:
        raise RuntimeError("need at least four allowed CPU cores on the GPU NUMA node")
    split = len(server_cores) // 2
    return format_cpu_list(server_cores[:split]), format_cpu_list(server_cores[split:])


@dataclass(frozen=True)
class MpsLaunchSpec:
    repository_root: Path
    output_dir: Path
    state_root: Path
    run_id: str
    config_path: Path
    gpu_id: int
    base_port: int
    core_blocks: tuple[str, ...]
    python_bin: str
    serve_extra_args: str = ""

    def __post_init__(self) -> None:
        if not RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError("run_id must be a safe run-<suffix> path component")
        if len(self.core_blocks) != REPLICA_COUNT or any(
            not item.strip() for item in self.core_blocks
        ):
            raise ValueError("mps_shared requires exactly two non-empty core blocks")
        if self.gpu_id < 0 or not 1 <= self.base_port < 65535:
            raise ValueError("invalid MPS GPU or base port")
        if not self.config_path.is_file():
            raise ValueError(f"validated MPS config does not exist: {self.config_path}")
        if not (self.repository_root / "examples/mps_dp/launch.sh").is_file():
            raise ValueError("production MPS launcher does not exist")

    @property
    def command(self) -> tuple[str, ...]:
        return ("bash", str(self.repository_root / "examples/mps_dp/launch.sh"), "up")

    @property
    def teardown_command(self) -> tuple[str, ...]:
        return (
            "bash",
            str(self.repository_root / "examples/mps_dp/launch.sh"),
            "down",
            self.run_id,
        )

    @property
    def state_dir(self) -> Path:
        return self.state_root / f"gpu-{self.gpu_id}" / self.run_id

    @property
    def worker_urls(self) -> tuple[str, ...]:
        return tuple(
            f"http://127.0.0.1:{self.base_port + index}"
            for index in range(REPLICA_COUNT)
        )

    @property
    def environment(self) -> dict[str, str]:
        return {
            "BASE_PORT": str(self.base_port),
            "CONFIG": str(self.config_path),
            "CORE_BLOCKS": " ".join(self.core_blocks),
            "GPU_ID": str(self.gpu_id),
            "N": "2",
            "PYTHON_BIN": self.python_bin,
            "REPLICA_ACTIVITY": "1",
            "RUN_ID": self.run_id,
            "SERVE_EXTRA_ARGS": self.serve_extra_args,
            "STATE_ROOT": str(self.state_root),
            "WEIGHT_SHARE": "1",
            "WEIGHT_SHARE_AUDIT": "1",
        }


@dataclass(frozen=True)
class ReplicaState:
    index: int
    pid: int
    pgid: int
    port: int
    log_path: Path
    leader_start: str
    kv_tokens: int


@dataclass(frozen=True)
class LauncherState:
    state_dir: Path
    manifest: dict[str, str]
    replicas: tuple[ReplicaState, ...]
    attachment_text: str
    weight_audits: tuple[dict[str, Any], ...]

    @property
    def attachment_passed(self) -> bool:
        return "RESULT: PASS" in self.attachment_text and all(
            f"replica {item.index} (port {item.port}): attached clients:"
            in self.attachment_text
            for item in self.replicas
        )


def _read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key:
            raise ValueError(f"invalid launcher manifest line: {line!r}")
        values[key] = value
    return values


def _read_kv_tokens(path: Path) -> int:
    matches = KV_PATTERN.findall(path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"resolved KV capacity is missing from {path}")
    return int(matches[-1])


def _read_weight_audits(
    state: Path, manifest: Mapping[str, str]
) -> tuple[dict[str, Any], ...]:
    paths = sorted((state / "weight_audit").glob("weight_share_*.json"))
    if len(paths) != 2:
        raise ValueError("expected exactly two verified weight-share audit files")
    audits = tuple(json.loads(path.read_text(encoding="utf-8")) for path in paths)
    if {item.get("role") for item in audits} != {"leader", "follower"}:
        raise ValueError("weight-share audits require leader and follower roles")
    equality_fields = (
        "architecture",
        "model_class",
        "manifest_hash",
        "shared_tensor_count",
        "shared_tensor_names",
        "private_tensor_names",
        "private_storage_preserved_after_attachment",
        "replica_local_state",
    )
    for audit in audits:
        if (
            audit.get("schema_version") != 1
            or audit.get("status") != "pass"
            or audit.get("verified_attachment") is not True
        ):
            raise ValueError("weight-share audit was not emitted after verification")
        if audit.get("private_shared_storage_intersection") != []:
            raise ValueError("private tensors alias shared storage")
        if (
            audit.get("run_id") != manifest["run_id"]
            or audit.get("gpu_uuid") != manifest["gpu_uuid"]
        ):
            raise ValueError(
                "weight-share audit identity does not match launcher state"
            )
    for field in equality_fields:
        if len({json.dumps(item.get(field), sort_keys=True) for item in audits}) != 1:
            raise ValueError(f"weight-share leader/follower disagree on {field}")
    first = audits[0]
    architecture = first["architecture"]
    if first["private_tensor_names"] != EXPECTED_PRIVATE_TENSORS.get(architecture):
        raise ValueError(f"private tensor policy mismatch for {architecture}")
    if set(first["shared_tensor_names"]) & set(first["private_tensor_names"]):
        raise ValueError("private tensors leaked into shared payload")
    if first.get("private_storage_preserved_after_attachment") is not True:
        raise ValueError("replica-private storage preservation was not verified")
    local = first.get("replica_local_state") or {}
    if architecture == "MossTTSLocalSGLangModel" and (
        local.get("status") != "pass"
        or local.get("scope") != "process_local_unregistered_tensors"
        or local.get("shared_record_intersection") != []
        or "audio_token_presence" not in local.get("history_tensor_names", [])
    ):
        raise ValueError("MOSS replica-local history isolation audit failed")
    return audits


def read_launcher_state(state_dir: str | Path) -> LauncherState:
    state = Path(state_dir)
    manifest = _read_key_values(state / "manifest")
    required = {
        "run_id",
        "gpu_id",
        "gpu_uuid",
        "numa_node",
        "config",
        "n",
        "base_port",
        "core_blocks",
        "max_total_tokens",
        "weight_share",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"launcher manifest is missing fields: {missing}")
    if (
        manifest["n"] != "2"
        or manifest["weight_share"] != "1"
        or not manifest["max_total_tokens"].isdigit()
    ):
        raise ValueError("launcher did not establish DP2 weight sharing and fixed KV")
    replicas: list[ReplicaState] = []
    for line in (state / "replicas.tsv").read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 6:
            raise ValueError(f"invalid replicas.tsv line: {line!r}")
        index, pid, pgid, port, raw_log, leader_start = fields
        log_path = Path(raw_log)
        if not log_path.is_absolute():
            log_path = state / log_path
        replicas.append(
            ReplicaState(
                int(index),
                int(pid),
                int(pgid),
                int(port),
                log_path,
                leader_start,
                _read_kv_tokens(log_path),
            )
        )
    if [item.index for item in replicas] != [0, 1]:
        raise ValueError("replicas.tsv must contain ordered replica indices 0 and 1")
    if [item.port for item in replicas] != [
        int(manifest["base_port"]),
        int(manifest["base_port"]) + 1,
    ]:
        raise ValueError("replica ports do not match launcher manifest")
    expected_kv = int(manifest["max_total_tokens"])
    if any(item.kv_tokens != expected_kv for item in replicas):
        raise ValueError("replicas do not have equal resolved KV capacity")
    snapshot = LauncherState(
        state,
        manifest,
        tuple(replicas),
        (state / "mps_attach.txt").read_text(encoding="utf-8"),
        _read_weight_audits(state, manifest),
    )
    if not snapshot.attachment_passed:
        raise ValueError("both replicas are not proven attached to private MPS")
    return snapshot


def _process_tree(
    pgids: Sequence[int], *, command_runner: Any = subprocess.run
) -> dict[str, Any]:
    groups = []
    for pgid in pgids:
        result = command_runner(
            ["ps", "-o", "pid=,ppid=,pgid=,stat=,args=", "-g", str(pgid)],
            check=False,
            capture_output=True,
            text=True,
        )
        groups.append(
            {
                "tracked_pgid": pgid,
                "query_exit_code": result.returncode,
                "processes": result.stdout.splitlines(),
                "stderr": result.stderr,
            }
        )
    return {
        "status": (
            "pass"
            if groups and all(item["query_exit_code"] == 0 for item in groups)
            else "error"
        ),
        "groups": groups,
    }


def _copy_raw_mps(snapshot: LauncherState, output_dir: Path) -> Path:
    raw = output_dir / "raw" / "mps"
    raw.mkdir(parents=True, exist_ok=True)
    for name in ("manifest", "replicas.tsv", "mps_attach.txt", "mps_control_raw.txt"):
        shutil.copy2(snapshot.state_dir / name, raw / name)
    for directory in ("logs", "weight_audit"):
        shutil.copytree(
            snapshot.state_dir / directory, raw / directory, dirs_exist_ok=True
        )
    return raw


def _copy_failed_launch_diagnostics(state_dir: Path, output_dir: Path) -> Path:
    raw = output_dir / "raw" / "mps" / "failed_launch"
    raw.mkdir(parents=True, exist_ok=True)
    for name in (
        "manifest",
        "replicas.tsv",
        "mps_ctl.err",
        "mps_attach.txt",
        "mps_control_raw.txt",
    ):
        source = state_dir / name
        if not source.is_symlink() and source.is_file():
            shutil.copy2(source, raw / name)
    for source in state_dir.glob("replica_activity_*.jsonl"):
        if not source.is_symlink() and source.is_file():
            shutil.copy2(source, raw / source.name)
    for relative in (Path("logs"), Path("mps/log"), Path("weight_audit")):
        source_dir = state_dir / relative
        if source_dir.is_symlink() or not source_dir.is_dir():
            continue
        target_dir = raw / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in source_dir.iterdir():
            if not source.is_symlink() and source.is_file():
                shutil.copy2(source, target_dir / source.name)
    return raw


def write_mps_runtime(
    snapshot: LauncherState,
    *,
    output_dir: str | Path,
    command_runner: Any = subprocess.run,
) -> None:
    root = Path(output_dir)
    raw = _copy_raw_mps(snapshot, root)
    audits = snapshot.weight_audits
    first = audits[0]
    follower = next(item for item in audits if item["role"] == "follower")
    local = first["replica_local_state"]
    model_audits = {
        "status": "pass",
        "architecture": first["architecture"],
        "weight_share": {
            "status": "pass",
            "manifest_hash": first["manifest_hash"],
            "shared_tensor_count": first["shared_tensor_count"],
            "roles": sorted(item["role"] for item in audits),
        },
        "private_state": {
            "status": "pass",
            "private_tensor_names": first["private_tensor_names"],
            "shared_tensor_intersection": [],
            "follower_private_storage_preserved": follower[
                "private_storage_preserved_after_attachment"
            ],
            "replica_local_state": local,
            "history_provenance": {
                "scope": "replica_process" if local["status"] == "pass" else "none",
                "tensor_names": local["history_tensor_names"],
                "exported_via_weight_share": False,
            },
        },
    }
    replicas = [
        {
            "replica_id": item.index,
            "role": "leader" if item.index == 0 else "follower",
            "pid": item.pid,
            "pgid": item.pgid,
            "port": item.port,
            "log_path": str(raw / "logs" / item.log_path.name),
            "leader_start": item.leader_start,
            "max_total_tokens": item.kv_tokens,
        }
        for item in snapshot.replicas
    ]
    record_runtime_section(
        root,
        section="launch",
        value={
            "status": "pass",
            "run_id": snapshot.manifest["run_id"],
            "gpu_id": int(snapshot.manifest["gpu_id"]),
            "gpu_uuid": snapshot.manifest["gpu_uuid"],
            "numa_node": int(snapshot.manifest["numa_node"]),
            "config": snapshot.manifest["config"],
            "replica_count": 2,
            "base_port": int(snapshot.manifest["base_port"]),
            "core_blocks": snapshot.manifest["core_blocks"].split(),
            "weight_share": True,
        },
    )
    record_runtime_section(root, section="replicas", value=replicas)
    record_runtime_section(
        root,
        section="mps",
        value={
            "status": "pass",
            "attachment": {"status": "pass", "replicas": [0, 1]},
            "equal_kv": {
                "status": "pass",
                "max_total_tokens": int(snapshot.manifest["max_total_tokens"]),
            },
            "private_pipe_dir": str(snapshot.state_dir / "mps/pipe"),
            "private_log_dir": str(snapshot.state_dir / "mps/log"),
            "raw_control_path": str(raw / "mps_control_raw.txt"),
        },
    )
    record_runtime_section(root, section="model_audits", value=model_audits)
    record_runtime_section(
        root,
        section="processes",
        value=_process_tree(
            [item.pgid for item in snapshot.replicas], command_runner=command_runner
        ),
    )
    record_runtime_section(root, section="raw_state_directory", value=str(raw))
    record_model_audits(root, model_audits)


def read_replica_activity(snapshot: LauncherState) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for replica in snapshot.replicas:
        path = snapshot.state_dir / f"replica_activity_{replica.index}.jsonl"
        if not path.is_file():
            raise ValueError(f"replica activity file is missing: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if (
                event.get("schema_version") != 1
                or event.get("clock") != "CLOCK_MONOTONIC"
            ):
                raise ValueError("replica activity schema/clock mismatch")
            if (
                event.get("run_id") != snapshot.manifest["run_id"]
                or event.get("replica_id") != replica.index
            ):
                raise ValueError("replica activity identity mismatch")
            if not event.get("host_boot_id") or event["host_boot_id"] == "unavailable":
                raise ValueError("replica activity is missing host boot ID")
            events.append(event)
    if len({item["host_boot_id"] for item in events}) != 1:
        raise ValueError("replica activity spans more than one host boot")
    return sorted(
        events,
        key=lambda item: (
            int(item["monotonic_ns"]),
            int(item["replica_id"]),
            str(item["request_id"]),
        ),
    )


def collect_replica_activity(
    snapshot: LauncherState, *, output_dir: str | Path
) -> list[dict[str, Any]]:
    events = read_replica_activity(snapshot)
    target = Path(output_dir) / "replica_activity.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return events


def write_router_snapshot(
    output_dir: str | Path,
    *,
    when: str,
    snapshot: Mapping[str, Any] | None,
    error: str | None = None,
) -> None:
    if when not in {"before", "after"}:
        raise ValueError("router snapshot must be before or after")
    path = Path(output_dir) / "stage1_runtime.json"
    payload = read_evidence(path, expected_kind="runtime")
    payload["router"][when] = {
        "status": "pass" if snapshot is not None and error is None else "error",
        "snapshot": dict(snapshot or {}),
        "error": error,
    }
    if when == "after" and payload["launch"].get("status") == "pass":
        before = payload["router"].get("before") or {}
        payload["status"] = (
            "pass"
            if before.get("status") == "pass"
            and payload["router"]["after"]["status"] == "pass"
            else "error"
        )
    write_json_atomic(path, payload)


def record_router_process(
    output_dir: str | Path,
    *,
    pid: int,
    port: int,
    cleanup_manifest: str | Path,
) -> None:
    groups, parsed = read_process_groups(cleanup_manifest)
    if not parsed or pid <= 0 or not 1 <= port < 65536:
        raise ValueError("router cleanup identity is missing or invalid")
    if os.getpgid(pid) not in groups:
        raise ValueError("router process group is missing from its cleanup manifest")
    path = Path(output_dir) / "stage1_runtime.json"
    payload = read_evidence(path, expected_kind="runtime")
    payload["router"]["process"] = {
        "pid": pid,
        "port": port,
        "process_groups": groups,
        "cleanup_manifest": str(cleanup_manifest),
    }
    write_json_atomic(path, payload)


def launch_replicas(spec: MpsLaunchSpec) -> LauncherState:
    environment = os.environ.copy()
    environment.update(spec.environment)
    try:
        subprocess.run(
            spec.command, cwd=spec.repository_root, env=environment, check=True
        )
        snapshot = read_launcher_state(spec.state_dir)
        write_mps_runtime(snapshot, output_dir=spec.output_dir)
        return snapshot
    except BaseException as original:
        if not spec.state_dir.exists():
            raise
        retention_error: OSError | None = None
        try:
            _copy_failed_launch_diagnostics(spec.state_dir, spec.output_dir)
        except OSError as error:
            retention_error = error
        try:
            subprocess.run(
                spec.teardown_command,
                cwd=spec.repository_root,
                env=environment,
                check=True,
            )
        except BaseException as cleanup:
            raise RuntimeError(
                f"MPS launch/validation and run-scoped teardown failed; state retained at {spec.state_dir}; original={original!r}"
            ) from cleanup
        if retention_error is not None:
            raise RuntimeError(
                f"MPS launch failed and diagnostics could not be retained before run-scoped teardown; original={original!r}"
            ) from retention_error
        raise


def _tracked_pids(snapshot: LauncherState) -> tuple[set[int], bool]:
    tracked = {item.pid for item in snapshot.replicas}
    succeeded = True
    for replica in snapshot.replicas:
        result = subprocess.run(
            ["ps", "-o", "pid=", "-g", str(replica.pgid)],
            check=False,
            capture_output=True,
            text=True,
        )
        succeeded = succeeded and result.returncode == 0
        tracked.update(int(raw) for raw in result.stdout.split() if raw.isdigit())
    return tracked, succeeded


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    try:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return True
    state = result.stdout.strip()
    return result.returncode == 0 and bool(state) and not state.startswith("Z")


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    try:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-g", str(pgid)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return True
    if result.returncode != 0:
        return True
    return any(
        state.strip() and not state.lstrip().startswith("Z")
        for state in result.stdout.splitlines()
    )


def _occupied_ports(ports: Sequence[int]) -> list[int]:
    occupied = []
    for port in sorted(set(ports)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                occupied.append(port)
    return occupied


def teardown_mps(
    spec: MpsLaunchSpec,
    *,
    router_stopped: bool = False,
    requests_drained_or_cancelled: bool = False,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(spec.environment)
    errors: list[str] = []
    tracked: set[int] = set()
    process_query_succeeded = False
    ports = [spec.base_port, spec.base_port + 1]
    try:
        snapshot = read_launcher_state(spec.state_dir)
        ports = [item.port for item in snapshot.replicas]
        tracked, process_query_succeeded = _tracked_pids(snapshot)
        _copy_raw_mps(snapshot, spec.output_dir)
    except BaseException as exc:
        snapshot = None
        errors.append(f"launcher_state_or_process_query: {exc!r}")
    state = snapshot.state_dir if snapshot else spec.state_dir
    control_pid = None
    control_path = state / "mps/pipe/nvidia-cuda-mps-control.pid"
    if control_path.is_file():
        try:
            control_pid = int(control_path.read_text().strip())
        except (OSError, ValueError) as exc:
            errors.append(f"mps_control_pid: {exc!r}")
    try:
        result = subprocess.run(
            spec.teardown_command,
            cwd=spec.repository_root,
            env=environment,
            check=False,
        )
        down_exit_code = result.returncode
    except BaseException as exc:
        down_exit_code = -1
        errors.append(f"launch_down: {exc!r}")
    alive = sorted(pid for pid in tracked if _pid_alive(pid))
    try:
        runtime = read_evidence(
            Path(spec.output_dir) / "stage1_runtime.json", expected_kind="runtime"
        )
        baseline = runtime.get("baseline_gpu") or {
            "status": "error",
            "compute_clients": [],
        }
        post_gpu = capture_gpu_state()
        new_clients = _new_clients(baseline, post_gpu)
        gpu_query = (
            baseline.get("status") == "pass" and post_gpu.get("status") == "pass"
        )
    except BaseException as exc:
        post_gpu = {"status": "error", "compute_clients": []}
        new_clients = []
        gpu_query = False
        errors.append(f"gpu_client_query: {exc!r}")
    try:
        occupied = _occupied_ports(ports)
    except BaseException as exc:
        occupied = sorted(ports)
        errors.append(f"port_query: {exc!r}")
    checks = {
        "router_stopped": router_stopped,
        "requests_drained_or_cancelled": requests_drained_or_cancelled,
        "launch_down_succeeded": down_exit_code == 0,
        "process_query_succeeded": process_query_succeeded,
        "tracked_processes_exited": not alive,
        "mps_control_pid_observed": control_pid is not None,
        "mps_control_exited": not bool(control_pid and _pid_alive(control_pid)),
        "mps_namespace_inactive": not (state / "mps/pipe").exists(),
        "ports_released": not occupied,
        "gpu_client_query_succeeded": gpu_query,
        "no_new_gpu_clients": not new_clients,
        "launcher_evidence_valid": snapshot is not None and not errors,
    }
    details = {
        "run_id": spec.run_id,
        "down_exit_code": down_exit_code,
        "tracked_processes_alive": alive,
        "occupied_ports": occupied,
        "new_compute_clients": new_clients,
        "post_teardown_gpu_state": post_gpu,
        "evidence_errors": errors,
    }
    return record_generation_cleanup(
        spec.output_dir,
        checks=checks,
        details=details,
        retained_diagnostics=str(spec.output_dir / "raw"),
    )


def read_process_groups(manifest: str | Path) -> tuple[list[int], bool]:
    try:
        lines = Path(manifest).read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], False
    groups: set[int] = set()
    valid = True
    for line in lines:
        try:
            pgid = int(line.strip())
        except ValueError:
            valid = False
            continue
        if pgid <= 0:
            valid = False
        else:
            groups.add(pgid)
    return sorted(groups), valid and bool(groups)


def write_multi_gpu_runtime(
    output_dir: str | Path,
    *,
    model: str,
    router_pid: int,
    router_port: int,
    worker_ports: Sequence[int],
    num_gpus_per_worker: int,
    router_ready_s: float | None,
    cleanup_manifest: str | Path,
    before_workers: Mapping[str, Any],
    command_runner: Any = subprocess.run,
) -> None:
    groups, parsed = read_process_groups(cleanup_manifest)
    replicas = [
        {
            "replica_id": index,
            "port": port,
            "gpu_slots": list(
                range(index * num_gpus_per_worker, (index + 1) * num_gpus_per_worker)
            ),
            "scheduler_and_kv_scope": "worker_local",
        }
        for index, port in enumerate(worker_ports)
    ]
    record_runtime_section(
        output_dir,
        section="launch",
        value={
            "status": "pass",
            "model": model,
            "router": {"pid": router_pid, "port": router_port},
            "worker_count": len(replicas),
            "num_gpus_per_worker": num_gpus_per_worker,
            "router_ready_s": router_ready_s,
            "cleanup_manifest": str(cleanup_manifest),
            "process_manifest_parse_succeeded": parsed,
        },
    )
    record_runtime_section(output_dir, section="replicas", value=replicas)
    record_runtime_section(
        output_dir,
        section="processes",
        value=_process_tree(groups, command_runner=command_runner),
    )
    write_router_snapshot(output_dir, when="before", snapshot=before_workers)


def teardown_multi_gpu(
    output_dir: str | Path,
    *,
    cleanup_manifest: str | Path,
    router_port: int,
    worker_ports: Sequence[int],
    router_stopped: bool,
    requests_drained_or_cancelled: bool,
    gpu_memory_released: bool,
) -> dict[str, Any]:
    root = Path(output_dir)
    groups, parsed = read_process_groups(cleanup_manifest)
    alive = sorted(pgid for pgid in groups if _process_group_alive(pgid))
    occupied = _occupied_ports([router_port, *worker_ports])
    runtime = read_evidence(root / "stage1_runtime.json", expected_kind="runtime")
    checks = {
        "router_stopped": router_stopped,
        "requests_drained_or_cancelled": requests_drained_or_cancelled,
        "process_manifest_valid": parsed,
        "tracked_process_groups_exited": not alive,
        "ports_released": not occupied,
        "gpu_memory_released": gpu_memory_released,
        "router_attribution_recorded": (runtime["router"].get("after") or {}).get(
            "status"
        )
        == "pass",
    }
    details = {
        "tracked_process_groups_alive": alive,
        "occupied_ports": occupied,
        "cleanup_manifest": str(cleanup_manifest),
    }
    return record_generation_cleanup(
        root, checks=checks, details=details, retained_diagnostics=str(cleanup_manifest)
    )


def _parse_csv(stdout: str, fields: Sequence[str]) -> list[dict[str, str]]:
    rows = []
    for line in stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values)))
    return rows


def capture_gpu_state(command_runner: Any = subprocess.run) -> dict[str, Any]:
    gpu = command_runner(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    clients = command_runner(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    ok = gpu.returncode == 0 and clients.returncode == 0
    return {
        "status": "pass" if ok else "error",
        "gpus": (
            _parse_csv(gpu.stdout, ("index", "uuid", "name", "driver_version"))
            if gpu.returncode == 0
            else []
        ),
        "compute_clients": (
            _parse_csv(clients.stdout, ("pid", "gpu_uuid", "process_name"))
            if clients.returncode == 0
            else []
        ),
        "errors": [
            text
            for text in (
                gpu.stderr if gpu.returncode else "",
                clients.stderr if clients.returncode else "",
            )
            if text
        ],
    }


def initialize_stage1(
    output_dir: str | Path,
    *,
    topology: str,
    model: str,
    configured_timeout_minutes: int,
    command_runner: Any = subprocess.run,
) -> None:
    initialize_evidence(
        output_dir,
        topology=topology,
        model=model,
        configured_timeout_minutes=configured_timeout_minutes,
    )
    record_runtime_section(
        output_dir, section="baseline_gpu", value=capture_gpu_state(command_runner)
    )


def _new_clients(
    baseline: Mapping[str, Any], post: Mapping[str, Any]
) -> list[dict[str, str]]:
    before = {
        (item.get("pid"), item.get("gpu_uuid"))
        for item in baseline.get("compute_clients", [])
    }
    return [
        item
        for item in post.get("compute_clients", [])
        if (item.get("pid"), item.get("gpu_uuid")) not in before
    ]


def finalize_stage1(
    output_dir: str | Path,
    *,
    topology: str,
    model: str,
    exact_sha: str,
    command_runner: Any = subprocess.run,
) -> dict[str, Any]:
    root = Path(output_dir)
    started = time.time_ns()
    runtime = read_evidence(root / "stage1_runtime.json", expected_kind="runtime")
    correctness = read_evidence(
        root / "stage1_correctness.json", expected_kind="correctness"
    )
    timing = finalize_timing(root, finalization_started_at_ns=started)
    cleanup = read_evidence(root / "stage1_cleanup.json", expected_kind="cleanup")
    try:
        preflight = read_evidence(root / "preflight.json", expected_kind="preflight")
    except (RuntimeError, ValueError):
        preflight = {}
    post = capture_gpu_state(command_runner)
    baseline = runtime.get("baseline_gpu") or {"status": "error", "compute_clients": []}
    new_clients = _new_clients(baseline, post)
    replicas = runtime.get("replicas") or []
    ports = [item["port"] for item in replicas if isinstance(item.get("port"), int)]
    router_launch = runtime.get("launch", {}).get("router") or {}
    router_process = runtime.get("router", {}).get("process") or {}
    if isinstance(router_launch.get("port"), int):
        ports.append(router_launch["port"])
    if isinstance(router_process.get("port"), int):
        ports.append(router_process["port"])
    evaluator_pid = cleanup.get("evaluator", {}).get("observed_pid")
    evaluator_port = cleanup.get("evaluator", {}).get("observed_port")
    if isinstance(evaluator_port, int):
        ports.append(evaluator_port)
    process_errors: list[str] = []
    live_processes: set[int] = set()
    try:
        for group in (runtime.get("processes") or {}).get("groups", []):
            pgid = group.get("tracked_pgid")
            if isinstance(pgid, int) and _process_group_alive(pgid):
                live_processes.add(pgid)
        router_pid = router_process.get("pid")
        if isinstance(router_pid, int) and _pid_alive(router_pid):
            live_processes.add(router_pid)
        for pgid in router_process.get("process_groups", []):
            if isinstance(pgid, int) and _process_group_alive(pgid):
                live_processes.add(pgid)
        if isinstance(evaluator_pid, int) and _pid_alive(evaluator_pid):
            live_processes.add(evaluator_pid)
    except BaseException as exc:
        process_errors.append(repr(exc))
    try:
        occupied = _occupied_ports(ports)
    except BaseException as exc:
        occupied = ports
        process_errors.append(repr(exc))
    post_checks = {
        "gpu_query_succeeded": baseline.get("status") == "pass"
        and post.get("status") == "pass",
        "no_new_gpu_clients": not new_clients,
        "tracked_processes_exited": not live_processes,
        "tracked_ports_released": not occupied,
        "post_state_query_succeeded": not process_errors,
    }
    post_clean = all(post_checks.values())
    final_post = {
        "status": "clean" if post_clean else "dirty",
        "clean": post_clean,
        "checks": post_checks,
        "gpu_state": post,
        "new_compute_clients": new_clients,
        "live_tracked_processes": sorted(live_processes),
        "occupied_ports": occupied,
        "errors": process_errors,
    }
    performance_status = timing["normal_load_performance"].get("status")
    expected_performance = "pass" if topology == "mps_shared" else "recorded"
    preflight_matches = (
        preflight.get("selected_model") == model
        and preflight.get("tts_stage1_topology") == topology
        and preflight.get("run_id") == runtime.get("run", {}).get("run_id")
        and preflight.get("exact_sha") == exact_sha
    )
    config_matches = True
    if topology == "mps_shared":
        resolved = str(preflight.get("resolved_mps_config") or "").replace("\\", "/")
        launched = str(runtime.get("launch", {}).get("config") or "").replace("\\", "/")
        config_matches = bool(resolved and launched.endswith(resolved))
    router_process_recorded = (
        isinstance(router_process.get("pid"), int)
        and isinstance(router_process.get("port"), int)
        and bool(router_process.get("process_groups"))
        if topology == "mps_shared"
        else isinstance(router_launch.get("pid"), int)
        and isinstance(router_launch.get("port"), int)
    )
    evidence_checks = {
        "preflight_selection_matches": preflight_matches,
        "router_process_recorded": router_process_recorded,
        "validated_config_matches": config_matches,
        "runtime_identity_matches": runtime.get("topology") == topology
        and runtime.get("model") == model,
        "runtime_complete": runtime.get("status") == "pass",
        "correctness_complete": correctness.get("status") == "pass",
        "normal_load_performance_gate": performance_status == expected_performance,
        "generation_cleanup_clean": cleanup["generation"].get("clean") is True,
        "evaluator_completed": cleanup["evaluator"].get("status") == "completed",
        "final_post_state_clean": post_clean,
    }
    clean = all(evidence_checks.values())
    cleanup.update(
        final_post_state=final_post,
        verdict="clean" if clean else "dirty",
        clean=clean,
        final_checks=evidence_checks,
    )
    write_json_atomic(root / "stage1_cleanup.json", cleanup)
    runtime["run"].update(exact_sha=exact_sha)
    files = [
        "preflight.json",
        "stage1_runtime.json",
        "stage1_correctness.json",
        "stage1_timing.json",
        "stage1_cleanup.json",
    ]
    if topology == "mps_shared":
        files.append("replica_activity.jsonl")
    runtime["evidence_index"] = {
        "status": "complete" if clean else "incomplete",
        "exact_sha": exact_sha,
        "workflow_url": runtime["run"].get("workflow_url", ""),
        "model": model,
        "topology": topology,
        "run_attempt": runtime["run"].get("run_attempt", ""),
        "selection_digest": preflight.get("selection_digest"),
        "resolved_support_config": preflight.get("resolved_mps_config"),
        "files": files,
        "cleanup_clean": clean,
        "evaluator_status": cleanup["evaluator"].get("status"),
        "performance_provenance": timing["normal_load_performance"].get("provenance"),
        "retained_diagnostics": cleanup.get("retained_diagnostics"),
    }
    write_json_atomic(root / "stage1_runtime.json", runtime)
    if not clean:
        raise RuntimeError(
            f"Stage 1 final cleanup/evidence is dirty: {evidence_checks}"
        )
    return cleanup


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    derive = commands.add_parser("derive-core-blocks")
    derive.add_argument("--gpu-id", type=int, default=0)
    initialize = commands.add_parser("initialize")
    initialize.add_argument("--output-dir", required=True)
    initialize.add_argument("--topology", required=True)
    initialize.add_argument("--model", required=True)
    initialize.add_argument("--configured-timeout-minutes", type=int, required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--output-dir", required=True)
    finalize.add_argument("--topology", required=True)
    finalize.add_argument("--model", required=True)
    finalize.add_argument("--exact-sha", required=True)
    args = parser.parse_args()
    if args.command == "derive-core-blocks":
        print(" ".join(derive_core_blocks(args.gpu_id)))
    elif args.command == "initialize":
        initialize_stage1(
            args.output_dir,
            topology=args.topology,
            model=args.model,
            configured_timeout_minutes=args.configured_timeout_minutes,
        )
    else:
        finalize_stage1(
            args.output_dir,
            topology=args.topology,
            model=args.model,
            exact_sha=args.exact_sha,
        )


if __name__ == "__main__":
    main()
