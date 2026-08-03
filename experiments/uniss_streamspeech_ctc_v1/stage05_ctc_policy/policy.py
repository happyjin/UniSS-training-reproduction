"""Monotonic, language-aware CTC count policy for streaming UniSS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence


def collapse_ctc(path: Sequence[int], blank_id: int) -> list[int]:
    output: list[int] = []
    previous: int | None = None
    for token in path:
        value = int(token)
        if value != blank_id and value != previous:
            output.append(value)
        previous = value
    return output


def longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        count += 1
    return count


@dataclass
class StableCTCTracker:
    confirmations: int = 2
    previous: list[int] = field(default_factory=list)
    agreement: list[int] = field(default_factory=list)
    stable_count: int = 0
    conflict_events: int = 0

    def update(self, collapsed: Sequence[int]) -> int:
        current = list(map(int, collapsed))
        common = longest_common_prefix(self.previous, current)
        next_agreement = [1] * len(current)
        for position in range(common):
            prior = self.agreement[position] if position < len(self.agreement) else 1
            next_agreement[position] = prior + 1
        if common < min(self.stable_count, len(current)):
            self.conflict_events += 1
        candidate = 0
        for count in next_agreement:
            if count < self.confirmations:
                break
            candidate += 1
        # Stable count is monotonic.  If a later prefix conflicts, the caller
        # receives a diagnostic while the already established commitment holds.
        self.stable_count = max(self.stable_count, candidate)
        self.previous = current
        self.agreement = next_agreement
        return min(self.stable_count, len(current))

    def stable_tokens(self) -> list[int]:
        return self.previous[: min(self.stable_count, len(self.previous))]


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    new_target_tokens: tuple[int, ...]
    stable_source_count: int
    stable_target_count: int
    committed_target_count: int
    source_conflicts: int
    target_conflicts: int


class LanguageBoundary:
    def __init__(self, target_language: str, id_to_piece: Callable[[int], str]) -> None:
        self.language = target_language.lower()
        self.id_to_piece = id_to_piece

    def safe_count(self, tokens: Sequence[int], requested: int, final: bool) -> int:
        requested = min(requested, len(tokens))
        if final or self.language in {"cmn", "zh", "zho"}:
            return requested
        if self.language not in {"eng", "en"}:
            return requested
        # SentencePiece marks a new English word with U+2581.  Without seeing
        # the next word boundary, the last visible word remains provisional.
        last_complete = 0
        for position in range(1, requested):
            piece = self.id_to_piece(int(tokens[position]))
            if piece.startswith("▁"):
                last_complete = position
        return last_complete


class CTCReadWritePolicy:
    def __init__(
        self,
        *,
        source_blank_id: int,
        target_blank_id: int,
        target_language: str,
        target_id_to_piece: Callable[[int], str],
        confirmations: int = 2,
        lagging_k: int = 0,
    ) -> None:
        if lagging_k < 0:
            raise ValueError("lagging_k must be non-negative")
        self.source_blank_id = source_blank_id
        self.target_blank_id = target_blank_id
        self.source = StableCTCTracker(confirmations)
        self.target = StableCTCTracker(confirmations)
        self.boundary = LanguageBoundary(target_language, target_id_to_piece)
        self.lagging_k = lagging_k
        self.last_source_event_count = 0
        self.committed_target: list[int] = []

    def update(
        self,
        source_frame_path: Sequence[int],
        target_frame_path: Sequence[int],
        *,
        final: bool = False,
    ) -> PolicyDecision:
        source_path = collapse_ctc(source_frame_path, self.source_blank_id)
        target_path = collapse_ctc(target_frame_path, self.target_blank_id)
        stable_source = self.source.update(source_path)
        stable_target = self.target.update(target_path)
        source_event = stable_source > self.last_source_event_count
        if source_event:
            self.last_source_event_count = stable_source
        available = max(0, stable_target - self.lagging_k)
        safe = self.boundary.safe_count(target_path, available, final)
        safe = max(len(self.committed_target), safe)
        new_tokens: tuple[int, ...] = ()
        if (source_event or final) and safe > len(self.committed_target):
            candidate_prefix = target_path[:safe]
            common = longest_common_prefix(self.committed_target, candidate_prefix)
            if common == len(self.committed_target):
                new_tokens = tuple(candidate_prefix[len(self.committed_target) :])
                self.committed_target.extend(new_tokens)
        action = "WRITE" if new_tokens else "WAIT"
        return PolicyDecision(
            action=action,
            new_target_tokens=new_tokens,
            stable_source_count=stable_source,
            stable_target_count=stable_target,
            committed_target_count=len(self.committed_target),
            source_conflicts=self.source.conflict_events,
            target_conflicts=self.target.conflict_events,
        )

