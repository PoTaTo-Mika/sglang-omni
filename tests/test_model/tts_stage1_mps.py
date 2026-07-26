# SPDX-License-Identifier: Apache-2.0
"""TTS Stage 1 composition for production MPS replicas and external router."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / ".github" / "scripts"))

from tts_stage1_artifacts import write_json_atomic  # noqa: E402
from tts_stage1_mps_runtime import (  # noqa: E402
    MpsLaunchSpec,
    collect_replica_activity,
    launch_replicas,
    read_replica_activity,
    teardown_replicas,
    write_router_snapshot,
)
from tts_stage1_overlap import build_overlap_verdict  # noqa: E402

from tests.test_model.omni_router_utils import (  # noqa: E402
    ManagedRouterHandle,
    launch_router_for_workers,
    router_get_json,
)

TOPOLOGY_ENV = "TTS_STAGE1_TOPOLOGY"
CONFIG_ENV = "TTS_STAGE1_MPS_CONFIG"
OUTPUT_ENV = "TTS_STAGE1_AUDIT_ROOT"
CORE_BLOCKS_ENV = "TTS_STAGE1_MPS_CORE_BLOCKS"
BASE_PORT_ENV = "TTS_STAGE1_MPS_BASE_PORT"
GPU_ID_ENV = "TTS_STAGE1_MPS_GPU_ID"
STATE_ROOT_ENV = "TTS_STAGE1_MPS_STATE_ROOT"
RUN_ID_ENV = "TTS_STAGE1_MPS_RUN_ID"


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required for mps_shared Stage 1")
    return value


def build_launch_spec(*, serve_extra_args: str) -> MpsLaunchSpec:
    output_dir = Path(_required_environment(OUTPUT_ENV)).resolve()
    raw_config = Path(_required_environment(CONFIG_ENV))
    config_path = raw_config if raw_config.is_absolute() else PROJECT_ROOT / raw_config
    state_root = Path(_required_environment(STATE_ROOT_ENV)).resolve()
    return MpsLaunchSpec(
        repository_root=PROJECT_ROOT,
        output_dir=output_dir,
        state_root=state_root,
        run_id=_required_environment(RUN_ID_ENV),
        config_path=config_path.resolve(),
        gpu_id=int(os.environ.get(GPU_ID_ENV, "0")),
        base_port=int(os.environ.get(BASE_PORT_ENV, "8801")),
        core_blocks=tuple(_required_environment(CORE_BLOCKS_ENV).split()),
        python_bin=sys.executable,
        serve_extra_args=serve_extra_args,
    )


@contextmanager
def launch_stage1_mps_router(
    *,
    tmp_path_factory: pytest.TempPathFactory,
    model_name: str,
    worker_extra_args: str,
    wait_timeout: int,
) -> Iterator[ManagedRouterHandle]:
    """Launch exactly two shared-weight replicas and the production router."""

    if os.environ.get(TOPOLOGY_ENV) != "mps_shared":
        raise ValueError("launch_stage1_mps_router requires mps_shared topology")
    spec = build_launch_spec(serve_extra_args=worker_extra_args)
    launcher_snapshot = launch_replicas(spec)
    cleaned = False
    router: ManagedRouterHandle | None = None

    def record_after_workers() -> None:
        assert router is not None
        collect_replica_activity(launcher_snapshot, output_dir=spec.output_dir)
        write_router_snapshot(
            output_dir=spec.output_dir,
            artifact_name="router_workers_after.json",
            snapshot=router_get_json(router.port, "/workers"),
        )

    def cleanup_replicas(*, router_was_stopped: bool) -> None:
        nonlocal cleaned
        if not cleaned:
            cleaned = True
            teardown_replicas(
                spec,
                router_stopped=router_was_stopped,
                requests_drained_or_cancelled=router_was_stopped,
            )

    try:
        with launch_router_for_workers(
            tmp_path_factory=tmp_path_factory,
            worker_urls=list(spec.worker_urls),
            model_name=model_name,
            wait_timeout=wait_timeout,
            log_prefix="tts_stage1_mps_router_logs",
            before_stop_callback=record_after_workers,
            cleanup_callback=lambda: cleanup_replicas(router_was_stopped=True),
        ) as running_router:
            router = running_router
            router.mps_state_dir = launcher_snapshot.state_dir
            router.audit_root = spec.output_dir
            write_router_snapshot(
                output_dir=spec.output_dir,
                artifact_name="router_workers_before.json",
                snapshot=router_get_json(router.port, "/workers"),
            )
            yield router
    finally:
        cleanup_replicas(router_was_stopped=False)


def write_bounded_canary_artifacts(
    *,
    router: ManagedRouterHandle,
    results: dict,
    concurrency: int,
    wall_time_s: float,
) -> dict:
    if router.mps_state_dir is None or router.audit_root is None:
        raise ValueError("bounded canary requires an mps_shared router handle")
    successful = [item for item in results["per_request"] if item["is_success"]]
    request_ids = {str(item.get("server_request_id") or "") for item in successful}
    request_ids.discard("")
    if len(successful) != len(results["per_request"]):
        raise AssertionError("bounded canary contains failed requests")
    if len(request_ids) != len(successful):
        raise AssertionError("bounded canary is missing unique server request IDs")
    if any(
        not item.get("wav_path")
        or float(item.get("audio_duration_s") or 0) <= 0
        or item.get("error")
        for item in successful
    ):
        raise AssertionError("bounded canary audio integrity failed")

    snapshot = read_launcher_state(router.mps_state_dir)
    events = read_replica_activity(snapshot)
    verdict = build_overlap_verdict(
        events,
        request_ids=request_ids,
        min_successes_per_replica=2,
        min_matched_overlap_count=2,
        measurement_uncertainty_ns=1_000_000,
    )
    correctness = {
        "schema_version": 1,
        "topology": "mps_shared",
        "status": "pass",
        "request_ids": sorted(request_ids),
        "total_requests": len(results["per_request"]),
        "completed_requests": len(successful),
        "failed_requests": 0,
        "concurrency": concurrency,
        "canary_wall_time_s": wall_time_s,
        "canonical_stage3_input": False,
        "audio_integrity": "pass",
        "threshold_provenance": verdict["threshold_provenance"],
        "promotion_eligible": False,
    }
    write_json_atomic(router.audit_root / "overlap_verdict.json", verdict)
    write_json_atomic(
        router.audit_root / "high_concurrency_correctness.json", correctness
    )
    return verdict
