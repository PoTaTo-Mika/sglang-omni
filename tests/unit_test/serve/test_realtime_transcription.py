# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.client import CompletionResult, GenerateRequest
from sglang_omni.config import RealtimeTranscriptionConfig
from sglang_omni.serve.realtime import transcription_session as session_module
from sglang_omni.serve.realtime.strategy import StreamingASRConfig, StreamingHypothesis
from sglang_omni.serve.realtime.transcription_session import (
    RealtimeTranscriptionSession,
)


class RecordingWebSocket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def send_text(self, payload: str) -> None:
        self.events.append(json.loads(payload))

    async def close(self) -> None:
        self.application_state = WebSocketState.DISCONNECTED
        self.client_state = WebSocketState.DISCONNECTED


class FakeVAD:
    def process(self, _pcm: bytes) -> list[Any]:
        return []

    def reset(self) -> None: ...


class FakeStrategy:
    def create_state(self, config: StreamingASRConfig) -> object:
        return {"config": config}

    def build_decode_request(self, **_: Any) -> GenerateRequest:
        return GenerateRequest(prompt="audio", stream=False)

    def update_hypothesis(
        self,
        *,
        generated_text: str,
        metadata: dict[str, Any],
        state: object,
        is_final: bool,
    ) -> StreamingHypothesis:
        del state, is_final
        return StreamingHypothesis(
            text=generated_text,
            stable_text=str(metadata.get("streaming_prefix_text", "")),
            language=metadata.get("language"),
        )

    def reset_segment(self, state: object) -> None:
        del state


class FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[str] = []
        self.aborted: list[str] = []

    async def completion(
        self, _request: GenerateRequest, *, request_id: str
    ) -> CompletionResult:
        self.calls.append(request_id)
        text = self.outputs.pop(0) if self.outputs else f"text-{len(self.calls)}"
        return CompletionResult(
            request_id=request_id,
            text=text,
            metadata={"language": "English"},
        )

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class BlockingClient(FakeClient):
    def __init__(self) -> None:
        super().__init__([])
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def completion(
        self, _request: GenerateRequest, *, request_id: str
    ) -> CompletionResult:
        self.calls.append(request_id)
        if len(self.calls) == 1:
            self.started.set()
            await self.release.wait()
        return CompletionResult(
            request_id=request_id,
            text=f"text-{len(self.calls)}",
            metadata={"language": "English"},
        )


def _pcm(seconds: float, amplitude: int = 1000) -> bytes:
    samples = int(16000 * seconds)
    return amplitude.to_bytes(2, "little", signed=True) * samples


def _audio_event(pcm: bytes, *, sequence: int | None = None) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(pcm).decode(),
    }
    if sequence is not None:
        event["sequence"] = sequence
    return event


async def _session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    outputs: list[str] | None = None,
    max_audio_clip_s: float = 60.0,
) -> tuple[RealtimeTranscriptionSession, RecordingWebSocket, FakeClient]:
    monkeypatch.setattr(session_module, "StreamingVAD", lambda _config: FakeVAD())
    websocket = RecordingWebSocket()
    client = FakeClient(outputs or [])
    session = RealtimeTranscriptionSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-asr",
        capability=RealtimeTranscriptionConfig(
            strategy_factory="unused",
            decode_interval_ms=2000,
            max_audio_clip_s=max_audio_clip_s,
        ),
        strategy=FakeStrategy(),
        session_id="sess-test",
    )
    await session.dispatch(
        {
            "type": "session.update",
            "session": {"turn_detection": None},
        }
    )
    return session, websocket, client


@pytest.mark.asyncio
async def test_partial_is_replaced_by_one_final_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _client = await _session(
        monkeypatch, outputs=["hello wor", "hello world"]
    )
    await session.dispatch(_audio_event(_pcm(2.0)))
    for _ in range(10):
        await asyncio.sleep(0)
        if any(event["type"] == "transcription.segment" for event in websocket.events):
            break

    await session.dispatch({"type": "input_audio_buffer.commit"})
    await session.dispatch({"type": "transcription.done"})

    hypotheses = [
        event for event in websocket.events if event["type"] == "transcription.segment"
    ]
    assert [
        (event["revision"], event["text"], event["is_final"]) for event in hypotheses
    ] == [
        (1, "hello wor", False),
        (2, "hello world", True),
    ]
    indexes = [event["event_index"] for event in websocket.events]
    assert indexes == sorted(indexes) and len(indexes) == len(set(indexes))
    completed = websocket.events[-1]
    assert completed["type"] == "transcription.completed"
    assert completed["text"] == "hello world"


@pytest.mark.asyncio
async def test_duplicate_sequence_is_not_appended_twice_and_conflict_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _client = await _session(monkeypatch)
    pcm = _pcm(0.1)

    await session.dispatch(_audio_event(pcm, sequence=10))
    size = session.audio_buffer.num_bytes
    await session.dispatch(_audio_event(pcm, sequence=10))
    await session.dispatch(_audio_event(_pcm(0.1, 2000), sequence=10))

    assert session.audio_buffer.num_bytes == size
    appended = [
        event
        for event in websocket.events
        if event["type"] == "input_audio_buffer.appended"
    ]
    assert [event["duplicate"] for event in appended] == [False, True]
    assert websocket.events[-1]["error"]["code"] == "idempotency_conflict"

    await session.dispatch(_audio_event(pcm, sequence=12))
    await session.dispatch(_audio_event(pcm, sequence=11))
    assert websocket.events[-1]["error"]["code"] == "out_of_order_sequence"


@pytest.mark.asyncio
async def test_clear_aborts_inflight_partial_and_resets_uncommitted_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _client = await _session(monkeypatch)
    client = BlockingClient()
    session.client = client  # type: ignore[assignment]

    await session.dispatch(_audio_event(_pcm(2.0)))
    await client.started.wait()
    await session.dispatch({"type": "input_audio_buffer.clear"})

    assert len(client.aborted) == 1
    assert session.active_segment is None
    assert session.audio_buffer.is_empty()
    assert websocket.events[-1]["type"] == "input_audio_buffer.cleared"


@pytest.mark.asyncio
async def test_audio_during_decode_coalesces_to_one_followup_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _websocket, _client = await _session(monkeypatch)
    client = BlockingClient()
    session.client = client  # type: ignore[assignment]

    await session.dispatch(_audio_event(_pcm(2.0)))
    await client.started.wait()
    await session.dispatch(_audio_event(_pcm(2.0)))
    client.release.set()
    for _ in range(20):
        await asyncio.sleep(0)
        if len(client.calls) == 2:
            break

    assert len(client.calls) == 2
    await session.dispatch({"type": "input_audio_buffer.clear"})


@pytest.mark.asyncio
async def test_hard_limit_finalizes_in_audio_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _client = await _session(monkeypatch, max_audio_clip_s=1.0)
    await session.dispatch(_audio_event(_pcm(2.25)))
    await session.dispatch({"type": "transcription.done"})

    finals = [
        event
        for event in websocket.events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    assert [event["segment_id"] for event in finals] == [0, 1, 2]
    assert websocket.events[-1]["type"] == "transcription.completed"
    assert [segment["id"] for segment in websocket.events[-1]["segments"]] == [0, 1, 2]
