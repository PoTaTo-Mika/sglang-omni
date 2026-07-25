# SPDX-License-Identifier: Apache-2.0
"""CI-only replica-internal model-path activity recorder."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
ENV_PATH = "SGLANG_OMNI_REPLICA_ACTIVITY_PATH"
ENV_RUN_ID = "SGLANG_OMNI_REPLICA_ACTIVITY_RUN_ID"
ENV_REPLICA_ID = "SGLANG_OMNI_REPLICA_ACTIVITY_REPLICA_ID"


def _host_boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return "unavailable"


class ReplicaActivityRecorder:
    """Append start/end points measured on CLOCK_MONOTONIC inside one replica."""

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        pid: int | None = None,
        boot_id: str | None = None,
    ) -> None:
        env = os.environ if environ is None else environ
        raw_path = (env.get(ENV_PATH) or "").strip()
        self._path = Path(raw_path) if raw_path else None
        self._run_id = (env.get(ENV_RUN_ID) or "").strip()
        raw_replica_id = (env.get(ENV_REPLICA_ID) or "").strip()
        self._replica_id = int(raw_replica_id) if raw_replica_id else None
        if self._path is not None and (not self._run_id or self._replica_id is None):
            raise ValueError(f"{ENV_PATH} requires {ENV_RUN_ID} and {ENV_REPLICA_ID}")
        self._clock = monotonic_ns
        self._pid = os.getpid() if pid is None else pid
        self._boot_id = _host_boot_id() if boot_id is None else boot_id
        self._lock = threading.Lock()
        self._started: set[str] = set()
        self._ended: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def start(self, request_id: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            if request_id in self._started:
                return
            self._started.add(request_id)
            self._append_unlocked(
                request_id=request_id,
                event="model_path_start",
                status=None,
            )

    def end(self, request_id: str, *, status: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            if request_id not in self._started or request_id in self._ended:
                return
            self._ended.add(request_id)
            self._append_unlocked(
                request_id=request_id,
                event="model_path_end",
                status=status,
            )

    def _append_unlocked(
        self,
        *,
        request_id: str,
        event: str,
        status: str | None,
    ) -> None:
        assert self._path is not None
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "clock": "CLOCK_MONOTONIC",
            "monotonic_ns": self._clock(),
            "run_id": self._run_id,
            "replica_id": self._replica_id,
            "request_id": request_id,
            "pid": self._pid,
            "host_boot_id": self._boot_id,
            "event": event,
        }
        if status is not None:
            payload["status"] = status
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")


_RECORDER = ReplicaActivityRecorder()


def model_path_start(request_id: str) -> None:
    _RECORDER.start(request_id)


def model_path_end(request_id: str, *, status: str = "success") -> None:
    _RECORDER.end(request_id, status=status)
