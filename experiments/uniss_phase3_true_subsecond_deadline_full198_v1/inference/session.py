"""Persistent bounded state shared by file replay and WebRTC sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamingSessionState:
    speaker_global: tuple[int, ...]
    semantic_history_limit: int = 200
    committed_text_ids: list[int] = field(default_factory=list)
    semantic_history: list[int] = field(default_factory=list)
    frontend_state: Any = None
    qwen_past_key_values: Any = None
    codec_state: Any = None
    source_audio_ms: int = 0
    first_write_ms: int | None = None
    first_audio_ms: int | None = None
    forced_write_count: int = 0
    natural_write_count: int = 0

    def __post_init__(self) -> None:
        self.speaker_global = tuple(int(value) for value in self.speaker_global)
        if len(self.speaker_global) != 32:
            raise ValueError("speaker_global must contain exactly 32 tokens")
        if self.semantic_history_limit <= 0:
            raise ValueError("semantic_history_limit must be positive")

    def append_source_time(self, milliseconds: int) -> None:
        if milliseconds < 0:
            raise ValueError("source increment must be non-negative")
        self.source_audio_ms += milliseconds

    def commit_text(self, values: list[int], *, forced: bool) -> None:
        self.committed_text_ids.extend(int(value) for value in values)
        if self.first_write_ms is None:
            self.first_write_ms = self.source_audio_ms
        if forced:
            self.forced_write_count += 1
        else:
            self.natural_write_count += 1

    def append_semantic(self, values: list[int]) -> None:
        self.semantic_history.extend(int(value) for value in values)
        del self.semantic_history[: max(0, len(self.semantic_history) - self.semantic_history_limit)]

    def mark_audio(self) -> None:
        if self.first_audio_ms is None:
            self.first_audio_ms = self.source_audio_ms

    def reset_utterance(self) -> None:
        """Clear linguistic state while retaining the session speaker anchor."""

        self.committed_text_ids.clear()
        self.semantic_history.clear()
        self.frontend_state = None
        self.qwen_past_key_values = None
        self.codec_state = None
        self.source_audio_ms = 0
        self.first_write_ms = None
        self.first_audio_ms = None
        self.forced_write_count = 0
        self.natural_write_count = 0
