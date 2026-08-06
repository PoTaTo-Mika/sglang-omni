# SPDX-License-Identifier: Apache-2.0
"""Exact-owner routing state for uploaded TTS voices."""

from __future__ import annotations

import asyncio
import logging
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
        self._hydrated = False
        self._pending_mutations: dict[str, bool] = {}
        self._task: asyncio.Task[None] | None = None

    def resolve_owner(self) -> Worker | None:
        """Return the fixed voice owner, selecting it once when configured as auto."""
        if self._owner_url is None:
            owner = next(
                (
                    worker
                    for worker in self._workers
                    if can_own_uploaded_voices(worker.capabilities)
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
        if self._hydrated:
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_hydration())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def requires_owner(
        self,
        voice_names: set[str],
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
        if not self._hydrated:
            # The owner can serve both presets and uploaded voices. Falling back
            # to it preserves correctness when registry discovery is unavailable.
            return True
        return bool(names & self._uploaded_names)

    async def _run_hydration(self) -> None:
        while not self._hydrated:
            owner = self.resolve_owner()
            if owner is not None and owner.is_routable:
                await self._hydrate_from(owner)
            if not self._hydrated:
                await asyncio.sleep(self._retry_interval_secs)

    async def _hydrate_from(self, owner: Worker) -> None:
        try:
            response = await self._client.get(
                f"{owner.url}/v1/audio/voices",
                timeout=self._timeout_secs,
            )
            response.raise_for_status()
            uploaded_names = _uploaded_voice_names(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "voice_registry_hydration_failed worker=%s error=%s",
                owner.display_id,
                type(exc).__name__,
            )
            return
        for name, uploaded in self._pending_mutations.items():
            if uploaded:
                uploaded_names.add(name)
            else:
                uploaded_names.discard(name)
        self._uploaded_names = uploaded_names
        self._pending_mutations.clear()
        self._hydrated = True
        logger.info(
            "voice_registry_hydrated worker=%s uploaded_voices=%d",
            owner.display_id,
            len(uploaded_names),
        )

    def record_upload(self, name: str) -> None:
        normalized = _normalize_voice_name(name)
        if normalized is None:
            return
        self._uploaded_names.add(normalized)
        if not self._hydrated:
            self._pending_mutations[normalized] = True

    def record_delete(self, name: str) -> None:
        normalized = _normalize_voice_name(name)
        if normalized is None:
            return
        self._uploaded_names.discard(normalized)
        if not self._hydrated:
            self._pending_mutations[normalized] = False

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
