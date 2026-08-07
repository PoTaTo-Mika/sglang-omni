# SPDX-License-Identifier: Apache-2.0
"""Exact-owner routing state for uploaded TTS voices."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Set
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from sglang_omni_router.config import can_own_uploaded_voices
from sglang_omni_router.worker import Worker

logger = logging.getLogger(__name__)

DEFAULT_VOICE_NAME = "default"


@dataclass(frozen=True)
class VoiceMutation:
    operation: Literal["upload", "delete"]
    name: str

    @classmethod
    def create(
        cls,
        operation: Literal["upload", "delete"],
        name: str,
    ) -> VoiceMutation | None:
        normalized = _normalize_voice_name(name)
        if normalized is None:
            return None
        return cls(operation=operation, name=normalized)


class VoiceRoutingState:
    """Track which voice names require the configured state owner."""

    def __init__(
        self,
        *,
        workers: list[Worker],
        owner_url: str | None,
        client: httpx.AsyncClient,
        timeout_secs: int,
        retry_interval_secs: int,
    ) -> None:
        self._workers = workers
        self._owner_url = owner_url
        self._client = client
        self._timeout_secs = timeout_secs
        self._retry_interval_secs = retry_interval_secs
        self._uploaded_names: set[str] = set()
        self._active = False
        self._hydrated = False
        self._registry_available: bool | None = None
        self._is_reconciling = False
        self._mutations_inflight = 0
        self._last_refresh_error: str | None = None
        self._uncertainty_generation = 0
        self._pending_mutations: dict[str, bool] = {}
        self._refresh_requested = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def resolve_owner(self) -> Worker | None:
        """Return the fixed voice owner, selecting a routable owner once."""
        if self._owner_url is None:
            owner = next(
                (
                    worker
                    for worker in self._workers
                    if worker.is_routable
                    and can_own_uploaded_voices(worker.capabilities)
                ),
                None,
            )
            if owner is not None:
                self._owner_url = owner.url
            return owner
        return next(
            (worker for worker in self._workers if worker.url == self._owner_url),
            None,
        )

    def is_owner(self, worker: Worker) -> bool:
        owner = self.resolve_owner()
        return owner is not None and owner.url == worker.url

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_reconciliation())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def request_refresh(self) -> None:
        if self._active and not self._is_reconciling:
            self._refresh_requested.set()

    def activate(self) -> None:
        self._active = True
        self.request_refresh()

    def mark_uncertain(self) -> None:
        self._active = True
        self._uncertainty_generation += 1
        self._hydrated = False
        self._last_refresh_error = "mutation outcome is uncertain"
        self._refresh_requested.set()

    def begin_mutation(self) -> None:
        self._active = True
        self._mutations_inflight += 1
        self._uncertainty_generation += 1
        self._hydrated = False

    def end_mutation(self) -> None:
        assert self._mutations_inflight > 0
        self._mutations_inflight -= 1
        if self._mutations_inflight == 0:
            self._pending_mutations.clear()
            self._refresh_requested.set()

    def to_dict(self) -> dict[str, Any]:
        owner = self.resolve_owner()
        if owner is None:
            registry_state = "unassigned"
        elif not self._active:
            registry_state = "inactive"
        elif self._mutations_inflight:
            registry_state = "mutating"
        elif self._is_reconciling:
            registry_state = "refreshing" if self._hydrated else "hydrating"
        elif not self._hydrated:
            registry_state = (
                "degraded" if self._last_refresh_error is not None else "pending"
            )
        elif self._registry_available is False:
            registry_state = "unsupported"
        elif self._last_refresh_error is not None:
            registry_state = "stale"
        else:
            registry_state = "ready"
        return {
            "owner_worker_id": owner.worker_id if owner is not None else None,
            "owner_url": owner.url if owner is not None else None,
            "owner_routable": owner.is_routable if owner is not None else False,
            "registry_state": registry_state,
            "uploaded_voice_count": len(self._uploaded_names),
            "last_refresh_error": self._last_refresh_error,
        }

    def requires_owner(
        self,
        voice_names: Set[str],
        *,
        body_exceeds_metadata_limit: bool = False,
    ) -> bool:
        if self.resolve_owner() is None:
            return False
        if body_exceeds_metadata_limit:
            return True
        names = {
            normalized
            for name in voice_names
            if (normalized := _normalize_voice_name(name)) is not None
        }
        names.discard(DEFAULT_VOICE_NAME)
        if not names:
            return False
        self.activate()
        if not self._hydrated:
            self.request_refresh()
            # The owner can serve both presets and uploaded voices. Falling back
            # to it preserves correctness when registry discovery is unavailable.
            return True
        return bool(names & self._uploaded_names)

    async def _run_reconciliation(self) -> None:
        await self._refresh_requested.wait()
        loop = asyncio.get_running_loop()
        while True:
            self._refresh_requested.clear()
            owner = self.resolve_owner()
            if (
                self._mutations_inflight == 0
                and owner is not None
                and owner.is_routable
            ):
                await self._hydrate_from(owner)
            periodic_refresh = loop.call_later(
                self._retry_interval_secs,
                self._refresh_requested.set,
            )
            try:
                await self._refresh_requested.wait()
            finally:
                periodic_refresh.cancel()

    async def _hydrate_from(self, owner: Worker) -> None:
        uncertainty_generation = self._uncertainty_generation
        self._is_reconciling = True
        try:
            response = await self._client.get(
                f"{owner.url}/v1/audio/voices",
                timeout=self._timeout_secs,
            )
            if response.status_code == 404:
                uploaded_names: set[str] = set()
                registry_available = False
            else:
                response.raise_for_status()
                uploaded_names = _uploaded_voice_names(response.json())
                registry_available = True
        except (httpx.HTTPError, ValueError) as exc:
            self._last_refresh_error = type(exc).__name__
            logger.warning(
                "voice_registry_hydration_failed worker=%s error=%s",
                owner.display_id,
                type(exc).__name__,
            )
            return
        else:
            for name, uploaded in self._pending_mutations.items():
                if uploaded:
                    uploaded_names.add(name)
                else:
                    uploaded_names.discard(name)
            self._uploaded_names = uploaded_names
            self._registry_available = registry_available
            self._pending_mutations.clear()
            if self._uncertainty_generation == uncertainty_generation:
                self._hydrated = True
                self._last_refresh_error = None
                logger.info(
                    "voice_registry_hydrated worker=%s uploaded_voices=%d",
                    owner.display_id,
                    len(uploaded_names),
                )
        finally:
            self._is_reconciling = False

    def record_upload(self, name: str) -> None:
        normalized = _normalize_voice_name(name)
        if normalized is None:
            return
        self._uploaded_names.add(normalized)
        self._pending_mutations[normalized] = True
        if not self._hydrated:
            self.activate()

    def record_delete(self, name: str) -> None:
        normalized = _normalize_voice_name(name)
        if normalized is None:
            return
        self._uploaded_names.discard(normalized)
        self._pending_mutations[normalized] = False
        if not self._hydrated:
            self.activate()

    def apply(self, mutation: VoiceMutation) -> None:
        if mutation.operation == "upload":
            self.record_upload(mutation.name)
        else:
            self.record_delete(mutation.name)


def _uploaded_voice_names(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        raise ValueError("voice list response must be an object")
    uploaded = payload.get("uploaded_voices")
    if not isinstance(uploaded, list):
        raise ValueError("voice list response must include uploaded_voices")
    names: set[str] = set()
    for item in uploaded:
        if not isinstance(item, dict):
            raise ValueError("uploaded voice metadata must be an object")
        normalized = _normalize_voice_name(item.get("name"))
        if normalized is None:
            raise ValueError("uploaded voice metadata must include a name")
        names.add(normalized)
    return names


def _normalize_voice_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None
