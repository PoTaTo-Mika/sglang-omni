# SPDX-License-Identifier: Apache-2.0
"""Pinned WebSocket relay for TTS streaming sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState
from websockets.exceptions import ConnectionClosed, WebSocketException

try:
    from websockets.asyncio.client import connect as websocket_connect

    _WEBSOCKET_HEADERS_ARGUMENT = "additional_headers"
except ImportError:  # websockets 12 compatibility
    from websockets.client import connect as websocket_connect

    _WEBSOCKET_HEADERS_ARGUMENT = "extra_headers"

from sglang_omni.serve.speech_errors import (
    SpeechAPIError,
    bad_request,
    speech_websocket_error_payload,
)
from sglang_omni.serve.speech_limits import (
    MAX_SPEECH_WS_CONFIG_MESSAGE_BYTES,
    MAX_SPEECH_WS_TEXT_MESSAGE_BYTES,
    SPEECH_WS_CONFIG_TIMEOUT_S,
)
from sglang_omni_router.config import Capability, RouterConfig
from sglang_omni_router.proxy import WORKER_EVICTION_STATUS_CODES, AdmissionController
from sglang_omni_router.selector import (
    NoEligibleWorkerError,
    WorkerSelector,
    require_eligible_worker,
)
from sglang_omni_router.voice_routing import VoiceRoutingState
from sglang_omni_router.worker import Worker

logger = logging.getLogger(__name__)

RelayOutcome = Literal[
    "application_failure",
    "client_disconnected",
    "client_message_too_large",
    "completed",
    "upstream_failure",
]
HandshakeOutcome = Literal[
    "application_failure",
    "upstream_failure",
    "upstream_unavailable",
]
ClientRelayOutcome = Literal[
    "client_disconnected",
    "message_too_large",
    "upstream_closed_while_sending",
]


@dataclass(frozen=True)
class _HandshakeFailure:
    error: SpeechAPIError
    close_code: int
    outcome: HandshakeOutcome


_APPLICATION_CLOSE_CODES = {1002, 1003, 1007, 1008, 1009, 1010, 1013}
_MAX_UPSTREAM_SERVER_MESSAGE_BYTES = 512 * 1024 * 1024
_REQUEST_HEADERS_TO_STRIP = {
    "connection",
    "host",
    "sec-websocket-accept",
    "sec-websocket-extensions",
    "sec-websocket-key",
    "sec-websocket-protocol",
    "sec-websocket-version",
    "upgrade",
}


class TTSWebSocketProxy:
    def __init__(
        self,
        *,
        config: RouterConfig,
        workers: list[Worker],
        selector: WorkerSelector,
        admission: AdmissionController,
        voice_routing: VoiceRoutingState,
    ) -> None:
        self._config = config
        self._workers = workers
        self._selector = selector
        self._admission = admission
        self._voice_routing = voice_routing

    async def forward(self, websocket: WebSocket) -> None:
        await websocket.accept()
        request_id = _request_id(websocket)
        if not self._admission.try_acquire():
            await _send_router_error(
                websocket,
                SpeechAPIError(
                    message="router overloaded: max in-flight requests reached",
                    status_code=503,
                    error_type="overloaded_error",
                    code=503,
                ),
                close_code=1013,
            )
            return

        worker: Worker | None = None
        worker_active = False
        start_time = time.perf_counter()
        try:
            first_message = await asyncio.wait_for(
                websocket.receive(),
                timeout=SPEECH_WS_CONFIG_TIMEOUT_S,
            )
            if first_message.get("type") == "websocket.disconnect":
                return
            first_message_limit = min(
                self._config.max_payload_size,
                MAX_SPEECH_WS_CONFIG_MESSAGE_BYTES,
            )
            if _message_size(first_message) > first_message_limit:
                await _send_router_error(
                    websocket,
                    SpeechAPIError(
                        message="session.config WebSocket message exceeds payload limit",
                        status_code=413,
                        error_type="BadRequestError",
                        code=413,
                    ),
                    close_code=1009,
                )
                return

            facts = _session_route_facts(first_message)
            try:
                worker = self._select_worker(facts)
            except NoEligibleWorkerError:
                await _send_router_error(
                    websocket,
                    SpeechAPIError(
                        message="no eligible upstream",
                        status_code=503,
                        error_type="server_error",
                        code=503,
                    ),
                    close_code=1013,
                )
                return

            worker.increment_active()
            worker_active = True
            try:
                connect_options = {
                    _WEBSOCKET_HEADERS_ARGUMENT: _forward_headers(websocket),
                    "open_timeout": self._config.health_check_timeout_secs,
                    "close_timeout": self._config.health_check_timeout_secs,
                    "compression": None,
                    "max_queue": 1,
                    "max_size": _MAX_UPSTREAM_SERVER_MESSAGE_BYTES,
                }
                async with websocket_connect(
                    _upstream_url(worker, websocket),
                    **connect_options,
                ) as upstream:
                    await _send_upstream(upstream, first_message)
                    outcome = await _relay(
                        websocket,
                        upstream,
                        max_client_message_bytes=min(
                            self._config.max_payload_size,
                            MAX_SPEECH_WS_TEXT_MESSAGE_BYTES,
                        ),
                    )
            except asyncio.CancelledError:
                raise
            except (WebSocketException, OSError, asyncio.TimeoutError) as exc:
                handshake_status = _handshake_status_code(exc)
                if handshake_status is not None:
                    failure = _handshake_failure(handshake_status)
                    if handshake_status in WORKER_EVICTION_STATUS_CODES:
                        worker.record_request_failure(
                            failure_threshold=self._config.health_failure_threshold,
                            status_code=handshake_status,
                            error=f"status={handshake_status}",
                        )
                    worker.record_routed_request(service_class="tts_websocket")
                    log = (
                        logger.info
                        if failure.outcome == "application_failure"
                        else logger.warning
                    )
                    log(
                        "tts_websocket_completed request_id=%s worker=%s "
                        "outcome=%s status_code=%d duration_ms=%.2f",
                        request_id,
                        worker.display_id,
                        failure.outcome,
                        handshake_status,
                        (time.perf_counter() - start_time) * 1000,
                    )
                    await _send_router_error(
                        websocket,
                        failure.error,
                        close_code=failure.close_code,
                    )
                    return
                if isinstance(exc, ConnectionClosed) and _is_application_close(exc):
                    worker.record_routed_request(service_class="tts_websocket")
                    logger.info(
                        "tts_websocket_completed request_id=%s worker=%s "
                        "outcome=application_failure duration_ms=%.2f",
                        request_id,
                        worker.display_id,
                        (time.perf_counter() - start_time) * 1000,
                    )
                    await _send_router_error(
                        websocket,
                        SpeechAPIError(
                            message="upstream WebSocket rejected the request",
                            status_code=400,
                            error_type="upstream_error",
                            code=400,
                        ),
                        close_code=1008,
                    )
                    return
                worker.record_request_failure(
                    failure_threshold=self._config.health_failure_threshold,
                    error=type(exc).__name__,
                )
                worker.record_routed_request(service_class="tts_websocket")
                logger.warning(
                    "tts_websocket_upstream_failure request_id=%s worker=%s "
                    "error=%s",
                    request_id,
                    worker.display_id,
                    type(exc).__name__,
                )
                await _send_router_error(
                    websocket,
                    SpeechAPIError(
                        message="upstream WebSocket failed",
                        status_code=502,
                        error_type="upstream_error",
                        code=502,
                    ),
                    close_code=1011,
                )
                return

            if outcome == "client_message_too_large":
                worker.record_routed_request(service_class="tts_websocket")
                await _send_router_error(
                    websocket,
                    SpeechAPIError(
                        message="WebSocket message exceeds payload limit",
                        status_code=413,
                        error_type="BadRequestError",
                        code=413,
                    ),
                    close_code=1009,
                )
            elif outcome == "upstream_failure":
                worker.record_request_failure(
                    failure_threshold=self._config.health_failure_threshold,
                    error="WebSocketRelayError",
                )
                worker.record_routed_request(service_class="tts_websocket")
            elif outcome == "completed":
                worker.record_routed_request(
                    status_code=200,
                    service_class="tts_websocket",
                )
            else:
                worker.record_routed_request(service_class="tts_websocket")
            logger.info(
                "tts_websocket_completed request_id=%s worker=%s outcome=%s "
                "duration_ms=%.2f",
                request_id,
                worker.display_id,
                outcome,
                (time.perf_counter() - start_time) * 1000,
            )
        except asyncio.TimeoutError:
            await _send_router_error(
                websocket,
                bad_request("session.config was not received before timeout"),
                close_code=1008,
            )
        except WebSocketDisconnect:
            pass
        finally:
            if worker is not None and worker_active:
                worker.decrement_active()
            self._admission.release()

    def _select_worker(self, facts: "_SessionRouteFacts") -> Worker:
        required_capabilities: set[Capability] = {"speech", "streaming"}
        if facts.uses_reference_audio:
            required_capabilities.add("audio_input")
        uploaded_voice_request = self._voice_routing.requires_owner(facts.voice_names)
        if uploaded_voice_request:
            required_capabilities.add("audio_input")
            return require_eligible_worker(
                self._voice_routing.resolve_owner(),
                required_capabilities=required_capabilities,
                requested_model=facts.model,
            )
        return self._selector.select(
            self._workers,
            required_capabilities=required_capabilities,
            requested_model=facts.model,
        )


class _SessionRouteFacts:
    def __init__(
        self,
        *,
        model: str | None = None,
        voice_names: set[str] | None = None,
        uses_reference_audio: bool = False,
    ) -> None:
        self.model = model
        self.voice_names = voice_names or set()
        self.uses_reference_audio = uses_reference_audio


def _session_route_facts(message: dict[str, Any]) -> _SessionRouteFacts:
    text = message.get("text")
    if not isinstance(text, str):
        return _SessionRouteFacts()
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return _SessionRouteFacts()
    if not isinstance(payload, dict) or payload.get("type") != "session.config":
        return _SessionRouteFacts()
    session = payload.get("session")
    if session is None:
        session = {key: value for key, value in payload.items() if key != "type"}
    if not isinstance(session, dict):
        return _SessionRouteFacts()
    model = session.get("model")
    model = model.strip() if isinstance(model, str) and model.strip() else None
    voice = session.get("voice", session.get("speaker"))
    voice_names = (
        {voice.strip().lower()} if isinstance(voice, str) and voice.strip() else set()
    )
    uses_reference_audio = _uses_reference_audio(session)
    return _SessionRouteFacts(
        model=model,
        voice_names=voice_names,
        uses_reference_audio=uses_reference_audio,
    )


def _uses_reference_audio(session: dict[str, Any]) -> bool:
    reference_fields = ("audio_path", "ref_audio", "audio", "data")
    if session.get("ref_audio"):
        return True
    references = session.get("references")
    if not isinstance(references, list):
        return False
    return any(
        isinstance(reference, dict)
        and any(reference.get(field) for field in reference_fields)
        for reference in references
    )


async def _relay(
    websocket: WebSocket,
    upstream: Any,
    *,
    max_client_message_bytes: int,
) -> RelayOutcome:
    client_task = asyncio.create_task(
        _client_to_upstream(
            websocket,
            upstream,
            max_message_bytes=max_client_message_bytes,
        )
    )
    upstream_task = asyncio.create_task(_upstream_to_client(upstream, websocket))
    try:
        done, _ = await asyncio.wait(
            {client_task, upstream_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if client_task in done:
            client_outcome = await client_task
            if client_outcome == "client_disconnected":
                outcome = "client_disconnected"
            elif client_outcome == "message_too_large":
                if not upstream_task.done():
                    upstream_task.cancel()
                with suppress(Exception):
                    await upstream.close(code=1009)
                return "client_message_too_large"
            elif upstream_task in done:
                outcome = await upstream_task
            else:
                # The receive loop owns the upstream protocol outcome. A send
                # may observe closure before buffered terminal frames are read.
                outcome = await upstream_task
        else:
            outcome = await upstream_task

        if outcome == "client_disconnected":
            with suppress(Exception):
                await upstream.close(code=1000)
        else:
            if not client_task.done():
                client_task.cancel()
            if websocket.application_state != WebSocketState.CONNECTED:
                return outcome
            code = (
                1011
                if outcome == "upstream_failure"
                else getattr(upstream, "close_code", None) or 1000
            )
            reason = getattr(upstream, "close_reason", None) or ""
            with suppress(RuntimeError, WebSocketDisconnect):
                await websocket.close(code=_safe_close_code(code), reason=reason)
        return outcome
    finally:
        for task in (client_task, upstream_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(client_task, upstream_task, return_exceptions=True)


async def _client_to_upstream(
    websocket: WebSocket,
    upstream: Any,
    *,
    max_message_bytes: int,
) -> ClientRelayOutcome:
    while True:
        try:
            message = await websocket.receive()
        except (OSError, RuntimeError, WebSocketDisconnect):
            return "client_disconnected"
        message_type = message.get("type")
        if message_type == "websocket.disconnect":
            return "client_disconnected"
        if _message_size(message) > max_message_bytes:
            return "message_too_large"
        try:
            await _send_upstream(upstream, message)
        except ConnectionClosed:
            return "upstream_closed_while_sending"


async def _upstream_to_client(
    upstream: Any,
    websocket: WebSocket,
) -> RelayOutcome:
    protocol = _TTSProtocolState()
    try:
        async for message in upstream:
            if isinstance(message, str):
                protocol.observe(message)
                if not await _send_downstream(websocket.send_text(message)):
                    return "client_disconnected"
            else:
                if not await _send_downstream(websocket.send_bytes(bytes(message))):
                    return "client_disconnected"
    except ConnectionClosed as exc:
        if protocol.has_application_failure or _is_application_close(exc):
            return "application_failure"
        return "upstream_failure"
    except (WebSocketException, OSError, asyncio.TimeoutError):
        return "upstream_failure"
    return protocol.clean_close_outcome()


class _TTSProtocolState:
    def __init__(self) -> None:
        self._received_error = False
        self._failed_audio = False
        self._session_done = False

    def observe(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        message_type = payload.get("type")
        if message_type == "error":
            self._received_error = True
        elif message_type == "audio.done" and payload.get("error") is True:
            self._failed_audio = True
        elif message_type == "session.done":
            self._session_done = True
            self._received_error = False

    @property
    def has_application_failure(self) -> bool:
        return self._received_error or self._failed_audio

    def clean_close_outcome(self) -> RelayOutcome:
        if self._failed_audio:
            return "application_failure"
        if self._session_done:
            return "completed"
        if self._received_error:
            return "application_failure"
        return "upstream_failure"


async def _send_downstream(send: Awaitable[None]) -> bool:
    try:
        await send
    except (OSError, RuntimeError, WebSocketDisconnect):
        return False
    return True


def _handshake_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    legacy_status_code = getattr(exc, "status_code", None)
    return legacy_status_code if isinstance(legacy_status_code, int) else None


def _handshake_failure(status_code: int) -> _HandshakeFailure:
    if status_code in {429, 503}:
        if status_code == 429:
            message = "upstream WebSocket is overloaded"
            error_type = "overloaded_error"
        else:
            message = "upstream WebSocket is unavailable"
            error_type = "server_error"
        return _HandshakeFailure(
            error=SpeechAPIError(
                message=message,
                status_code=status_code,
                error_type=error_type,
                code=status_code,
            ),
            close_code=1013,
            outcome="upstream_unavailable",
        )
    if 400 <= status_code < 500:
        close_code = 1008
        message = "upstream WebSocket rejected the request"
        outcome = "application_failure"
    else:
        close_code = 1011
        message = "upstream WebSocket failed"
        outcome = "upstream_failure"
    return _HandshakeFailure(
        error=SpeechAPIError(
            message=message,
            status_code=status_code,
            error_type="upstream_error",
            code=status_code,
        ),
        close_code=close_code,
        outcome=outcome,
    )


def _is_application_close(exc: ConnectionClosed) -> bool:
    received = getattr(exc, "rcvd", None)
    close_code = getattr(received, "code", None)
    if not isinstance(close_code, int):
        legacy_close_code = getattr(exc, "code", None)
        close_code = legacy_close_code if isinstance(legacy_close_code, int) else None
    return close_code in _APPLICATION_CLOSE_CODES or (
        close_code is not None and 3000 <= close_code < 5000
    )


async def _send_upstream(upstream: Any, message: dict[str, Any]) -> None:
    if message.get("type") != "websocket.receive":
        return
    text = message.get("text")
    if text is not None:
        await upstream.send(text)
        return
    data = message.get("bytes")
    if data is not None:
        await upstream.send(data)


async def _send_router_error(
    websocket: WebSocket,
    error: SpeechAPIError,
    *,
    close_code: int,
) -> None:
    if websocket.application_state != WebSocketState.CONNECTED:
        return
    if websocket.client_state != WebSocketState.CONNECTED:
        return
    with suppress(OSError, RuntimeError, WebSocketDisconnect):
        await websocket.send_json(speech_websocket_error_payload(error))
        await websocket.close(code=close_code)


def _upstream_url(worker: Worker, websocket: WebSocket) -> str:
    parsed = urlsplit(worker.url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = websocket.url.query
    return urlunsplit(
        (
            scheme,
            parsed.netloc,
            "/v1/audio/speech/stream",
            query,
            "",
        )
    )


def _forward_headers(websocket: WebSocket) -> dict[str, str]:
    return {
        key: value
        for key, value in websocket.headers.items()
        if key.lower() not in _REQUEST_HEADERS_TO_STRIP
    }


def _request_id(websocket: WebSocket) -> str:
    return (
        websocket.headers.get("x-sglang-omni-request-id")
        or websocket.headers.get("x-request-id")
        or websocket.headers.get("x-correlation-id")
        or str(uuid.uuid4())
    )


def _safe_close_code(code: int) -> int:
    return code if 1000 <= code < 5000 else 1011


def _message_size(message: dict[str, Any]) -> int:
    text = message.get("text")
    if isinstance(text, str):
        return len(text.encode("utf-8"))
    data = message.get("bytes")
    return len(data) if isinstance(data, bytes) else 0
