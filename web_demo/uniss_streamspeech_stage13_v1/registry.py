"""Small bounded browser-session registry for microphone streaming."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from experiments.uniss_streamspeech_ctc_v1.stage11_streaming_audio.engine import (
    Stage11Session,
)


@dataclass
class BrowserState:
    session_id: str
    direction: str
    session: Stage11Session
    samples: int = 0


class Registry:
    def __init__(self, limit: int = 16) -> None:
        self.limit = limit
        self.values: dict[str, BrowserState] = {}
        self.lock = threading.Lock()

    def create(self, direction: str, session: Stage11Session) -> BrowserState:
        with self.lock:
            if len(self.values) >= self.limit:
                raise RuntimeError("too many retained browser sessions")
            value = BrowserState(uuid.uuid4().hex, direction, session)
            self.values[value.session_id] = value
            return value

    def get(self, session_id: str) -> BrowserState:
        with self.lock:
            if session_id not in self.values:
                raise KeyError(f"unknown browser session: {session_id}")
            return self.values[session_id]

    def discard(self, session_id: str | None) -> None:
        if not session_id:
            return
        with self.lock:
            self.values.pop(session_id, None)
