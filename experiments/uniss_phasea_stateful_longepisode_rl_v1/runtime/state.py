"""Session state shared across artificial long-form memory rollovers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .commit import AppendOnlyDeltaCommitter, StablePrefixCommitter
from .tts_queue import TTSQueue


@dataclass
class StreamingSessionState:
    """All mutable state for one source file or live microphone session."""

    frontend_state: Any = None
    speech_embedding_ring: list[Any] = field(default_factory=list)
    speech_ring_start_ms: int = 0
    asr_segment_committer: StablePrefixCommitter = field(
        default_factory=lambda: StablePrefixCommitter(holdback=1)
    )
    asr_committed_ids: list[int] = field(default_factory=list)
    mt_committer: AppendOnlyDeltaCommitter = field(
        default_factory=lambda: AppendOnlyDeltaCommitter(holdback=2)
    )
    tts_queue: TTSQueue = field(default_factory=TTSQueue)
    playback_cursor_samples: int = 0
    source_finished: bool = False
    memory_rollovers: int = 0
    artificial_boundary_finalizations: int = 0

    def finalize_asr_segment(self) -> list[int]:
        """Close only the decoder segment, never the acoustic source."""

        candidate = self.asr_segment_committer.previous or []
        new = self.asr_segment_committer.update(candidate, final=True)
        self.asr_committed_ids.extend(self.asr_segment_committer.committed)
        self.asr_segment_committer = StablePrefixCommitter(holdback=1)
        self.speech_embedding_ring.clear()
        self.memory_rollovers += 1
        return new

    def mark_true_final(self) -> None:
        self.source_finished = True

    def mark_artificial_boundary(self) -> None:
        """Record a memory boundary without pretending the source ended."""

        self.artificial_boundary_finalizations += 1


__all__ = ["StreamingSessionState"]
