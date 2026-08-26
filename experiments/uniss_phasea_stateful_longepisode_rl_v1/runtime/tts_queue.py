"""Acknowledged append-only text-to-speech queue."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class QueueStatus(str, Enum):
    PENDING = "pending"
    SYNTHESIZING = "synthesizing"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class TTSQueueItem:
    item_id: int
    token_ids: tuple[int, ...]
    text: str
    source_available_ms: int
    status: QueueStatus = QueueStatus.PENDING
    attempts: int = 0
    semantic_tokens: int = 0
    continuation_count: int = 0
    audio_samples: int = 0


@dataclass
class TTSQueue:
    """Retain text until healthy decoded audio explicitly acknowledges it."""

    items: list[TTSQueueItem] = field(default_factory=list)
    _next_id: int = 0

    def append(self, token_ids: Sequence[int], text: str, source_available_ms: int) -> TTSQueueItem:
        values = tuple(int(value) for value in token_ids)
        normalized = " ".join(str(text).split())
        if not values or not normalized:
            raise ValueError("TTS queue items require non-empty token IDs and text")
        item = TTSQueueItem(
            item_id=self._next_id,
            token_ids=values,
            text=normalized,
            source_available_ms=int(source_available_ms),
        )
        self._next_id += 1
        self.items.append(item)
        return item

    @property
    def pending(self) -> list[TTSQueueItem]:
        return [item for item in self.items if item.status != QueueStatus.ACKNOWLEDGED]

    @property
    def acknowledged(self) -> list[TTSQueueItem]:
        return [item for item in self.items if item.status == QueueStatus.ACKNOWLEDGED]

    def begin(self, item_id: int) -> TTSQueueItem:
        item = self._find(item_id)
        if item.status == QueueStatus.ACKNOWLEDGED:
            raise ValueError("cannot synthesize an acknowledged TTS item")
        item.status = QueueStatus.SYNTHESIZING
        item.attempts += 1
        return item

    def fail(self, item_id: int) -> TTSQueueItem:
        item = self._find(item_id)
        if item.status == QueueStatus.ACKNOWLEDGED:
            raise ValueError("cannot fail an acknowledged TTS item")
        item.status = QueueStatus.PENDING
        return item

    def acknowledge(
        self,
        item_id: int,
        *,
        semantic_tokens: int,
        continuation_count: int,
        audio_samples: int,
        healthy: bool,
    ) -> TTSQueueItem:
        item = self._find(item_id)
        if not healthy or semantic_tokens <= 0 or audio_samples <= 0:
            item.status = QueueStatus.PENDING
            return item
        item.semantic_tokens = int(semantic_tokens)
        item.continuation_count = int(continuation_count)
        item.audio_samples = int(audio_samples)
        item.status = QueueStatus.ACKNOWLEDGED
        return item

    def _find(self, item_id: int) -> TTSQueueItem:
        for item in self.items:
            if item.item_id == int(item_id):
                return item
        raise KeyError(item_id)


__all__ = ["QueueStatus", "TTSQueue", "TTSQueueItem"]

