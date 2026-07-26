# SPDX-License-Identifier: Apache-2.0
"""Production-launcher adapter and structured startup evidence for TTS Stage 1."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tts_stage1_lifecycle import (
    FAILURE_INJECTION_ENV,
    build_mps_teardown_verdict,
    lane_context,
    write_pre_evaluator_cleanup,
)

SCHEMA_VERSION = 1
TOPOLOGY = "mps_shared"
REPLICA_COUNT = 2
RUN_ID_PATTERN = re.compile(r"^run-[A-Za-z0-9_-]+$")
KV_PATTERN = re.compile(r"#tokens:\s*([0-9]+)")
EXPECTED_PRIVATE_TENSORS = {
    "HiggsMultimodalQwen3ForConditionalGeneration": [],
    "MossTTSLocalSGLangModel": ["_decode_input_embedding.weight"],
}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


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
    """Split the assigned GPU NUMA node's allowed CPU set between two replicas."""

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
        raise RuntimeError(
            f"cannot resolve one PCI sysfs device for GPU {gpu_id}: {result.stdout.strip()}"
        )
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
        raise RuntimeError(
            f"need at least four allowed CPU cores on GPU {gpu_id}'s NUMA node"
        )
    split = len(server_cores) // 2
    return format_cpu_list(server_cores[:split]), format_cpu_list(server_cores[split:])


@dataclass(frozen=True)
class MpsLaunchSpec:
    """Resolved, auditable input to the production same-GPU launcher."""

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
            not block.strip() for block in self.core_blocks
        ):
            raise ValueError("mps_shared requires exactly two non-empty core blocks")
        if self.gpu_id < 0:
            raise ValueError("gpu_id must be non-negative")
        if not 1 <= self.base_port < 65535:
            raise ValueError("base_port must leave room for exactly two replicas")
        if not self.config_path.is_file():
            raise ValueError(f"validated MPS config does not exist: {self.config_path}")
        launcher = self.repository_root / "examples/mps_dp/launch.sh"
        if not launcher.is_file():
            raise ValueError(f"production MPS launcher does not exist: {launcher}")

    @property
    def command(self) -> tuple[str, ...]:
        return (
            "bash",
            str(self.repository_root / "examples/mps_dp/launch.sh"),
            "up",
        )

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
            f"http://127.0.0.1:{self.base_port + offset}"
            for offset in range(REPLICA_COUNT)
        )

    @property
    def environment(self) -> dict[str, str]:
        return {
            "BASE_PORT": str(self.base_port),
            "CONFIG": str(self.config_path),
            "CORE_BLOCKS": " ".join(self.core_blocks),
            "GPU_ID": str(self.gpu_id),
            "N": str(REPLICA_COUNT),
            "PYTHON_BIN": self.python_bin,
            "REPLICA_ACTIVITY": "1",
            "RUN_ID": self.run_id,
            "SERVE_EXTRA_ARGS": self.serve_extra_args,
            "STATE_ROOT": str(self.state_root),
            "WEIGHT_SHARE": "1",
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
    raw_control_text: str
    weight_audits: tuple[dict[str, Any], ...]

    @property
    def equal_kv(self) -> bool:
        expected = int(self.manifest["max_total_tokens"])
        return all(replica.kv_tokens == expected for replica in self.replicas)

    @property
    def attachment_passed(self) -> bool:
        return "RESULT: PASS" in self.attachment_text and all(
            f"replica {replica.index} (port {replica.port}): attached clients:"
            in self.attachment_text
            for replica in self.replicas
        )


def _read_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key:
            raise ValueError(f"invalid launcher manifest line: {line!r}")
        values[key] = value
    return values


def _read_kv_tokens(log_path: Path) -> int:
    matches = KV_PATTERN.findall(log_path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"resolved KV capacity is missing from {log_path}")
    return int(matches[-1])


def _read_weight_audits(
    state: Path, manifest: dict[str, str]
) -> tuple[dict[str, Any], ...]:
    paths = sorted((state / "weight_audit").glob("weight_share_*.json"))
    if len(paths) != REPLICA_COUNT:
        raise ValueError("expected exactly two verified weight-share audit files")
    audits = tuple(json.loads(path.read_text(encoding="utf-8")) for path in paths)
    roles = {audit.get("role") for audit in audits}
    if roles != {"leader", "follower"}:
        raise ValueError(f"weight-share audits have invalid roles: {sorted(roles)}")
    for audit in audits:
        if audit.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("weight-share audit schema version mismatch")
        if (
            audit.get("status") != "pass"
            or audit.get("verified_attachment") is not True
        ):
            raise ValueError("weight-share audit was not emitted after verification")
        if audit.get("run_id") != manifest["run_id"]:
            raise ValueError("weight-share audit run_id does not match launcher state")
        if audit.get("gpu_uuid") != manifest["gpu_uuid"]:
            raise ValueError("weight-share audit GPU does not match launcher state")
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
    for field in equality_fields:
        if len({json.dumps(audit.get(field), sort_keys=True) for audit in audits}) != 1:
            raise ValueError(f"weight-share leader/follower disagree on {field}")
    architecture = audits[0]["architecture"]
    if architecture not in EXPECTED_PRIVATE_TENSORS:
        raise ValueError(
            f"unsupported TTS weight-share audit architecture: {architecture}"
        )
    if audits[0]["private_tensor_names"] != EXPECTED_PRIVATE_TENSORS[architecture]:
        raise ValueError(
            f"private tensor policy mismatch for {architecture}: "
            f"{audits[0]['private_tensor_names']!r}"
        )
    shared = set(audits[0]["shared_tensor_names"])
    private = set(audits[0]["private_tensor_names"])
    if shared & private:
        raise ValueError("private tensors leaked into the shared tensor audit")
    if audits[0].get("private_storage_preserved_after_attachment") is not True:
        raise ValueError("replica-private storage preservation was not verified")
    local_state = audits[0].get("replica_local_state") or {}
    if architecture == "MossTTSLocalSGLangModel":
        if (
            local_state.get("status") != "pass"
            or local_state.get("scope") != "process_local_unregistered_tensors"
            or local_state.get("shared_record_intersection") != []
            or "audio_token_presence" not in local_state.get("history_tensor_names", [])
        ):
            raise ValueError("MOSS replica-local history isolation audit failed")
    return audits


def read_launcher_state(state_dir: str | Path) -> LauncherState:
    """Validate and parse the state emitted by a successful launch.sh up."""

    state = Path(state_dir)
    manifest = _read_key_value_file(state / "manifest")
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
    if manifest["n"] != str(REPLICA_COUNT):
        raise ValueError("launcher did not start exactly two replicas")
    if manifest["weight_share"] != "1":
        raise ValueError("launcher did not enable leader/follower weight sharing")
    if not manifest["max_total_tokens"].isdigit():
        raise ValueError("launcher did not resolve a common positive KV capacity")

    replicas: list[ReplicaState] = []
    for line in (state / "replicas.tsv").read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 6:
            raise ValueError(f"invalid replicas.tsv line: {line!r}")
        index, pid, pgid, port, raw_log_path, leader_start = fields
        log_path = Path(raw_log_path)
        if not log_path.is_absolute():
            log_path = state / log_path
        replicas.append(
            ReplicaState(
                index=int(index),
                pid=int(pid),
                pgid=int(pgid),
                port=int(port),
                log_path=log_path,
                leader_start=leader_start,
                kv_tokens=_read_kv_tokens(log_path),
            )
        )
    if [replica.index for replica in replicas] != list(range(REPLICA_COUNT)):
        raise ValueError("replicas.tsv must contain ordered replica indices 0 and 1")
    expected_ports = [int(manifest["base_port"]) + index for index in range(2)]
    if [replica.port for replica in replicas] != expected_ports:
        raise ValueError("replica ports do not match the launcher manifest")

    snapshot = LauncherState(
        state_dir=state,
        manifest=manifest,
        replicas=tuple(replicas),
        attachment_text=(state / "mps_attach.txt").read_text(encoding="utf-8"),
        raw_control_text=(state / "mps_control_raw.txt").read_text(encoding="utf-8"),
        weight_audits=_read_weight_audits(state, manifest),
    )
    if not snapshot.equal_kv:
        raise ValueError("replicas do not have the same resolved KV capacity")
    if not snapshot.attachment_passed:
        raise ValueError("both replicas are not proven attached to private MPS")
    return snapshot


def write_startup_artifacts(
    snapshot: LauncherState,
    *,
    output_dir: str | Path,
) -> None:
    """Materialize structured startup evidence without claiming later lifecycle gates."""

    output = Path(output_dir)
    common = {"schema_version": SCHEMA_VERSION, "topology": TOPOLOGY}
    _write_json_atomic(
        output / "launch_manifest.json",
        {
            **common,
            "status": "pass",
            "run_id": snapshot.manifest["run_id"],
            "gpu_id": int(snapshot.manifest["gpu_id"]),
            "gpu_uuid": snapshot.manifest["gpu_uuid"],
            "numa_node": int(snapshot.manifest["numa_node"]),
            "config": snapshot.manifest["config"],
            "replica_count": REPLICA_COUNT,
            "base_port": int(snapshot.manifest["base_port"]),
            "core_blocks": snapshot.manifest["core_blocks"].split(),
            "weight_share": True,
        },
    )
    replica_payload = []
    for replica in snapshot.replicas:
        replica_payload.append(
            {
                "replica_id": replica.index,
                "role": "leader" if replica.index == 0 else "follower",
                "pid": replica.pid,
                "pgid": replica.pgid,
                "port": replica.port,
                "log_path": str(replica.log_path),
                "leader_start": replica.leader_start,
                "kv_tokens": replica.kv_tokens,
            }
        )
    _write_json_atomic(
        output / "replica_manifest.json",
        {**common, "status": "pass", "replicas": replica_payload},
    )
    _write_json_atomic(
        output / "mps_status.json",
        {
            **common,
            "status": "pass",
            "run_id": snapshot.manifest["run_id"],
            "private_pipe_dir": str(snapshot.state_dir / "mps/pipe"),
            "private_log_dir": str(snapshot.state_dir / "mps/log"),
        },
    )
    _write_json_atomic(
        output / "mps_attachment.json",
        {
            **common,
            "status": "pass",
            "replicas_attached": [replica.index for replica in snapshot.replicas],
            "source": str(snapshot.state_dir / "mps_attach.txt"),
        },
    )
    representative_audit = snapshot.weight_audits[0]
    follower_audit = next(
        audit for audit in snapshot.weight_audits if audit["role"] == "follower"
    )
    local_state = representative_audit["replica_local_state"]
    history_provenance = {
        "scope": "replica_process" if local_state["status"] == "pass" else "none",
        "tensor_names": local_state["history_tensor_names"],
        "exported_via_weight_share": False,
    }
    _write_json_atomic(
        output / "weight_share_audit.json",
        {
            **common,
            "status": "pass",
            "architecture": representative_audit["architecture"],
            "model_class": representative_audit["model_class"],
            "manifest_hash": representative_audit["manifest_hash"],
            "shared_tensor_count": representative_audit["shared_tensor_count"],
            "shared_tensor_names": representative_audit["shared_tensor_names"],
            "replicas": list(snapshot.weight_audits),
        },
    )
    _write_json_atomic(
        output / "private_tensor_audit.json",
        {
            **common,
            "status": "pass",
            "architecture": representative_audit["architecture"],
            "private_tensor_names": representative_audit["private_tensor_names"],
            "replica_local_storage_required": bool(
                representative_audit["private_tensor_names"]
            ),
            "shared_tensor_intersection": [],
            "follower_private_storage_preserved": follower_audit[
                "private_storage_preserved_after_attachment"
            ],
            "replica_local_state": local_state,
            "history_provenance": history_provenance,
        },
    )
    _write_json_atomic(
        output / "equal_kv.json",
        {
            **common,
            "status": "pass",
            "expected_max_total_tokens": int(snapshot.manifest["max_total_tokens"]),
            "replicas": [
                {"replica_id": replica.index, "kv_tokens": replica.kv_tokens}
                for replica in snapshot.replicas
            ],
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw_mps_control.txt").write_text(
        snapshot.raw_control_text, encoding="utf-8"
    )


def read_replica_activity(snapshot: LauncherState) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for replica in snapshot.replicas:
        path = snapshot.state_dir / f"replica_activity_{replica.index}.jsonl"
        if not path.is_file():
            raise ValueError(f"replica activity file is missing: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("replica activity schema version mismatch")
            if event.get("clock") != "CLOCK_MONOTONIC":
                raise ValueError("replica activity did not use CLOCK_MONOTONIC")
            if event.get("run_id") != snapshot.manifest["run_id"]:
                raise ValueError("replica activity run_id mismatch")
            if event.get("replica_id") != replica.index:
                raise ValueError("replica activity file contains the wrong replica_id")
            if not event.get("host_boot_id") or event["host_boot_id"] == "unavailable":
                raise ValueError("replica activity is missing the host boot ID")
            events.append(event)
    if len({event["host_boot_id"] for event in events}) != 1:
        raise ValueError("replica activity spans more than one host boot")
    return sorted(
        events,
        key=lambda event: (
            int(event["monotonic_ns"]),
            int(event["replica_id"]),
            str(event["request_id"]),
        ),
    )


def collect_replica_activity(
    snapshot: LauncherState, *, output_dir: str | Path
) -> list[dict[str, Any]]:
    events = read_replica_activity(snapshot)
    output = Path(output_dir) / "replica_activity.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True))
                handle.write("\n")
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return events


def write_router_snapshot(
    *,
    output_dir: str | Path,
    artifact_name: str,
    snapshot: dict[str, Any],
) -> None:
    if artifact_name not in {"router_workers_before.json", "router_workers_after.json"}:
        raise ValueError(f"unsupported router snapshot artifact: {artifact_name}")
    _write_json_atomic(
        Path(output_dir) / artifact_name,
        {
            "schema_version": SCHEMA_VERSION,
            "topology": TOPOLOGY,
            "status": "pass",
            "router": snapshot,
        },
    )


def write_process_tree(snapshot: LauncherState, *, output_dir: str | Path) -> None:
    """Record the tracked process groups without scanning or signalling foreign jobs."""

    groups = []
    for replica in snapshot.replicas:
        result = subprocess.run(
            [
                "ps",
                "-o",
                "pid=,ppid=,pgid=,lstart=,args=",
                "-g",
                str(replica.pgid),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        groups.append(
            {
                "replica_id": replica.index,
                "tracked_pgid": replica.pgid,
                "ps_exit_code": result.returncode,
                "processes": result.stdout.splitlines(),
                "stderr": result.stderr,
            }
        )
    _write_json_atomic(
        Path(output_dir) / "process_tree.json",
        {
            "schema_version": SCHEMA_VERSION,
            "topology": TOPOLOGY,
            "status": (
                "pass"
                if all(not group["ps_exit_code"] for group in groups)
                else "error"
            ),
            "groups": groups,
        },
    )


def launch_replicas(spec: MpsLaunchSpec) -> LauncherState:
    """Invoke launch.sh up, then require and emit its structured startup evidence."""

    environment = os.environ.copy()
    environment.update(spec.environment)
    subprocess.run(
        spec.command,
        cwd=spec.repository_root,
        env=environment,
        check=True,
    )
    try:
        snapshot = read_launcher_state(spec.state_dir)
        write_startup_artifacts(snapshot, output_dir=spec.output_dir)
        write_process_tree(snapshot, output_dir=spec.output_dir)
        return snapshot
    except BaseException as post_launch_error:
        try:
            teardown_replicas(spec, emit_lifecycle=False)
        except BaseException as cleanup_error:
            raise RuntimeError(
                "post-launch validation failed and run-specific teardown also failed; "
                f"state retained at {spec.state_dir}; original error: "
                f"{post_launch_error!r}"
            ) from cleanup_error
        raise


def _tracked_pids(snapshot: LauncherState) -> tuple[set[int], bool]:
    tracked = {replica.pid for replica in snapshot.replicas}
    query_succeeded = True
    for replica in snapshot.replicas:
        result = subprocess.run(
            ["ps", "-o", "pid=", "-g", str(replica.pgid)],
            check=False,
            capture_output=True,
            text=True,
        )
        query_succeeded = query_succeeded and result.returncode == 0
        for raw in result.stdout.split():
            if raw.isdigit():
                tracked.add(int(raw))
    return tracked, query_succeeded


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass

    try:
        status = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return True
    process_state = status.stdout.strip()
    if status.returncode != 0 or not process_state:
        return False
    # Match production launch.sh: zombies hold no GPU/port resources and cannot
    # be reaped by this verifier when its parent is an init-less container.
    return not process_state.startswith("Z")


def _occupied_ports(ports: set[int]) -> list[int]:
    occupied = []
    for port in sorted(ports):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                occupied.append(port)
    return occupied


def _gpu_client_pids() -> tuple[set[int], bool]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return set(), False
    return (
        {
            int(raw.strip())
            for raw in result.stdout.splitlines()
            if raw.strip().isdigit()
        },
        True,
    )


def teardown_replicas(
    spec: MpsLaunchSpec,
    *,
    router_stopped: bool = False,
    requests_drained_or_cancelled: bool = False,
    emit_lifecycle: bool = True,
) -> None:
    """Stop this run and emit a fail-closed MPS-sub-lifecycle verdict.

    The production ``launch.sh down RUN_ID`` command is attempted even when
    startup evidence is missing or corrupt. Evidence failures make the verdict
    dirty; they never bypass run-scoped teardown.
    """

    environment = os.environ.copy()
    environment.update(spec.environment)
    if not emit_lifecycle:
        subprocess.run(
            spec.teardown_command,
            cwd=spec.repository_root,
            env=environment,
            check=True,
        )
        return

    evidence_errors: list[str] = []
    tracked_before: set[int] = set()
    process_query_succeeded = False
    ports = {spec.base_port + offset for offset in range(REPLICA_COUNT)}
    try:
        snapshot = read_launcher_state(spec.state_dir)
    except BaseException as exc:
        snapshot = None
        evidence_errors.append(f"launcher_state: {exc!r}")
    else:
        ports = {replica.port for replica in snapshot.replicas}
        try:
            tracked_before, process_query_succeeded = _tracked_pids(snapshot)
        except BaseException as exc:
            tracked_before = {replica.pid for replica in snapshot.replicas}
            evidence_errors.append(f"process_query: {exc!r}")

    lifecycle_state_dir = snapshot.state_dir if snapshot is not None else spec.state_dir
    control_pid_path = (
        lifecycle_state_dir / "mps" / "pipe" / "nvidia-cuda-mps-control.pid"
    )
    control_pid = None
    if control_pid_path.is_file():
        try:
            control_pid = int(control_pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            evidence_errors.append(f"mps_control_pid: {exc!r}")

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
        evidence_errors.append(f"launch_down: {exc!r}")

    alive = sorted(pid for pid in tracked_before if _pid_alive(pid))
    try:
        gpu_client_pids, gpu_client_query_succeeded = _gpu_client_pids()
    except BaseException as exc:
        gpu_client_pids, gpu_client_query_succeeded = set(), False
        evidence_errors.append(f"gpu_client_query: {exc!r}")
    tracked_gpu_clients = sorted(gpu_client_pids & tracked_before)
    try:
        occupied_ports = _occupied_ports(ports)
    except BaseException as exc:
        occupied_ports = sorted(ports)
        evidence_errors.append(f"port_query: {exc!r}")
    retained_state = str(lifecycle_state_dir) if lifecycle_state_dir.exists() else None
    verdict = build_mps_teardown_verdict(
        run_id=spec.run_id,
        router_stopped=router_stopped,
        requests_drained_or_cancelled=requests_drained_or_cancelled,
        down_exit_code=down_exit_code,
        process_query_succeeded=process_query_succeeded,
        tracked_processes_alive=alive,
        mps_control_pid_observed=control_pid is not None,
        mps_control_alive=bool(control_pid and _pid_alive(control_pid)),
        mps_namespace_active=(lifecycle_state_dir / "mps" / "pipe").exists(),
        occupied_ports=occupied_ports,
        gpu_client_query_succeeded=gpu_client_query_succeeded,
        tracked_gpu_clients_alive=tracked_gpu_clients,
        retained_state=retained_state,
        failure_injection=(os.environ.get(FAILURE_INJECTION_ENV) or "").strip() or None,
    )
    verdict["evidence_errors"] = evidence_errors
    verdict["lane_context"] = lane_context(os.environ)
    _write_json_atomic(spec.output_dir / "mps_teardown_verdict.json", verdict)
    write_pre_evaluator_cleanup(spec.output_dir, topology=TOPOLOGY)
    if verdict["clean"] is not True:
        raise RuntimeError(
            "MPS teardown is dirty; evaluator is blocked and state is retained at "
            f"{retained_state or spec.output_dir}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    derive = subparsers.add_parser("derive-core-blocks")
    derive.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    if args.command == "derive-core-blocks":
        print(" ".join(derive_core_blocks(args.gpu_id)))


if __name__ == "__main__":
    main()
