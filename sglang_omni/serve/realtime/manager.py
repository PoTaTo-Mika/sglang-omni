from __future__ import annotations

import logging

from fastapi import WebSocket

from sglang_omni.client import Client
from sglang_omni.config import RealtimeTranscriptionConfig
from sglang_omni.serve.realtime.session import RealtimeSession
from sglang_omni.serve.realtime.transcription_session import (
    RealtimeTranscriptionSession,
)
from sglang_omni.utils.imports import import_string

logger = logging.getLogger(__name__)


class RealtimeSessionManager:
    def __init__(
        self,
        *,
        client: Client,
        model_name: str,
        supports_audio_output: bool = False,
        transcription_config: RealtimeTranscriptionConfig | None = None,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.supports_audio_output = supports_audio_output
        self.transcription_config = transcription_config
        self.sessions: dict[str, RealtimeSession | RealtimeTranscriptionSession] = {}

    def open(
        self, websocket: WebSocket, *, intent: str = "conversation"
    ) -> RealtimeSession | RealtimeTranscriptionSession:
        normalized_intent = intent.strip().casefold()
        if normalized_intent == "conversation":
            session: RealtimeSession | RealtimeTranscriptionSession = RealtimeSession(
                websocket,
                client=self.client,
                model_name=self.model_name,
                supports_audio_output=self.supports_audio_output,
            )
        elif normalized_intent == "transcription":
            if self.transcription_config is None:
                raise ValueError(
                    "This pipeline does not support realtime transcription."
                )
            strategy_cls = import_string(self.transcription_config.strategy_factory)
            session = RealtimeTranscriptionSession(
                websocket,
                client=self.client,
                model_name=self.model_name,
                capability=self.transcription_config,
                strategy=strategy_cls(),
            )
        else:
            raise ValueError(
                "Realtime intent must be 'conversation' or 'transcription'."
            )
        self.sessions[session.session_id] = session
        logger.info(
            "Realtime session opened: %s intent=%s",
            session.session_id,
            normalized_intent,
        )
        return session

    async def close(self, session_id: str) -> None:
        session = self.sessions[session_id]
        await session.teardown()
        del self.sessions[session_id]
        logger.info(f"Realtime session closed: {session_id}")

    def active_sessions(self) -> list[str]:
        return list(self.sessions.keys())
