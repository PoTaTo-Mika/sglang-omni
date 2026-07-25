# SPDX-License-Identifier: Apache-2.0
"""Production-launcher adapter and structured startup evidence for TTS Stage 1."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
TOPOLOGY = "mps_shared"
REPLICA_COUNT = 2
RUN_ID_PATTERN = re.compile(r"^run-[A-Za-z0-9_-]+$")
KV_PATTERN = re.compile(r"#tokens:\s*([0-9]+)")


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
    snapshot = read_launcher_state(spec.state_dir)
    write_startup_artifacts(snapshot, output_dir=spec.output_dir)
    write_process_tree(snapshot, output_dir=spec.output_dir)
    return snapshot


def teardown_replicas(spec: MpsLaunchSpec) -> None:
    """Ask launch.sh to stop only the run identified by this immutable spec."""

    environment = os.environ.copy()
    environment.update(spec.environment)
    subprocess.run(
        spec.teardown_command,
        cwd=spec.repository_root,
        env=environment,
        check=True,
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
