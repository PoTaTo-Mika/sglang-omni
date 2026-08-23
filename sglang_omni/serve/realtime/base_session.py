from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from sglang_omni.client import Client
from sglang_omni.serve.realtime.events import make_event, parse_client_event


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class BaseRealtimeSession:
    handlers: dict[type, str] = {}

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        model_name: str,
        session_id: str | None = None,
    ) -> None:
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
        self.session_id = session_id or new_id("sess")
        self.closed = False
        self.event_index = 0
        self._send_lock = asyncio.Lock()

    def initial_event(self) -> dict[str, Any]:
        raise NotImplementedError

    async def run(self) -> None:
        await self.send(self.initial_event())
        while not self.closed:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message["type"] != "websocket.receive":
                continue
            try:
                payload = json.loads(message["text"])
            except (KeyError, TypeError, json.JSONDecodeError):
                await self.send_error(
                    "invalid_request_error", "invalid_json", "Invalid JSON event."
                )
                continue
            if not isinstance(payload, dict):
                await self.send_error(
                    "invalid_request_error",
                    "invalid_event",
                    "Top-level payload must be a JSON object.",
                )
                continue
            await self.dispatch(payload)

    async def dispatch(self, payload: dict[str, Any]) -> None:
        try:
            event = parse_client_event(payload)
        except ValueError as exc:
            await self.send_error("invalid_request_error", "invalid_event", str(exc))
            return
        if event is None or type(event) not in self.handlers:
            await self.send_error(
                "invalid_request_error",
                "unsupported_event",
                f"Unsupported event type: {payload.get('type')!r}",
            )
            return
        await getattr(self, self.handlers[type(event)])(event)

    async def send(self, event: dict[str, Any]) -> None:
        if self.closed:
            return
        if self.websocket.application_state != WebSocketState.CONNECTED:
            return
        async with self._send_lock:
            self.event_index += 1
            event.setdefault("event_id", new_id("evt"))
            event.setdefault("event_index", self.event_index)
            await self.websocket.send_text(json.dumps(event))

    async def send_error(self, type_: str, code: str, message: str) -> None:
        await self.send(
            make_event(
                "error",
                error={"type": type_, "code": code, "message": message},
            )
        )

    async def cancel_and_abort(
        self, task: asyncio.Task[Any] | None, request_id: str | None
    ) -> None:
        if task is None or task.done():
            return
        task.cancel()
        try:
            if request_id is not None:
                await self.client.abort(request_id)
        finally:
            await asyncio.gather(task, return_exceptions=True)

    async def teardown(self) -> None:
        self.closed = True
        if self.websocket.client_state == WebSocketState.CONNECTED:
            await self.websocket.close()
