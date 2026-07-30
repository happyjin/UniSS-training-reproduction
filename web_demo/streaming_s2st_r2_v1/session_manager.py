"""Per-browser append-only state and artifact bookkeeping."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio_io import SAMPLE_RATE, create_request_directory, resample_mono


@dataclass
class AudioIngressState:
    max_seconds: float
    sample_rate: int = SAMPLE_RATE
    chunks: list[np.ndarray] = field(default_factory=list)
    sample_count: int = 0

    def append(self, chunk: tuple[int, np.ndarray] | None) -> np.ndarray:
        if chunk is None:
            return np.zeros(0, dtype=np.float32)
        sample_rate, audio = chunk
        values = resample_mono(audio, int(sample_rate))
        maximum = int(round(self.max_seconds * self.sample_rate))
        if self.sample_count + len(values) > maximum:
            raise ValueError(f"Microphone session exceeds {self.max_seconds:.1f} seconds")
        self.chunks.append(values)
        self.sample_count += len(values)
        return values

    @property
    def waveform(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.float32)

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate


@dataclass
class BrowserSession:
    output_root: Path
    max_microphone_seconds: float
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    request_dir: Path | None = None
    ingress: AudioIngressState = field(init=False)
    engine_state: object | None = None
    cancelled: bool = False

    def __post_init__(self) -> None:
        self.ingress = AudioIngressState(self.max_microphone_seconds)

    def ensure_request_dir(self) -> Path:
        if self.request_dir is None:
            self.request_dir = create_request_directory(self.output_root)
        return self.request_dir

    def cancel(self) -> None:
        self.cancelled = True


class SessionRegistry:
    """Small bounded in-memory registry; GPU execution remains serialized separately."""

    def __init__(self, output_root: Path, max_microphone_seconds: float, limit: int = 16):
        self.output_root = output_root
        self.max_microphone_seconds = max_microphone_seconds
        self.limit = limit
        self._sessions: dict[str, BrowserSession] = {}
        self._lock = threading.Lock()

    def create(self) -> BrowserSession:
        with self._lock:
            if len(self._sessions) >= self.limit:
                raise RuntimeError("Too many retained browser sessions")
            session = BrowserSession(self.output_root, self.max_microphone_seconds)
            self._sessions[session.session_id] = session
            return session

    def get(self, session_id: str) -> BrowserSession:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"Unknown browser session: {session_id}") from exc

    def discard(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
