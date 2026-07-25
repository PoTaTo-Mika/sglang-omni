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

from tts_stage1_mps_runtime import (  # noqa: E402
    MpsLaunchSpec,
    launch_replicas,
    teardown_replicas,
    write_router_snapshot,
)

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
    launch_replicas(spec)
    cleaned = False
    router: ManagedRouterHandle | None = None

    def record_after_workers() -> None:
        assert router is not None
        write_router_snapshot(
            output_dir=spec.output_dir,
            artifact_name="router_workers_after.json",
            snapshot=router_get_json(router.port, "/workers"),
        )

    def cleanup_replicas() -> None:
        nonlocal cleaned
        if not cleaned:
            teardown_replicas(spec)
            cleaned = True

    try:
        with launch_router_for_workers(
            tmp_path_factory=tmp_path_factory,
            worker_urls=list(spec.worker_urls),
            model_name=model_name,
            wait_timeout=wait_timeout,
            log_prefix="tts_stage1_mps_router_logs",
            before_stop_callback=record_after_workers,
            cleanup_callback=cleanup_replicas,
        ) as running_router:
            router = running_router
            write_router_snapshot(
                output_dir=spec.output_dir,
                artifact_name="router_workers_before.json",
                snapshot=router_get_json(router.port, "/workers"),
            )
            yield router
    finally:
        cleanup_replicas()
