"""Append-only commit policies for a genuinely stateful streaming session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


def longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for first, second in zip(left, right):
        if int(first) != int(second):
            break
        length += 1
    return length


@dataclass
class StablePrefixCommitter:
    """Commit only the stable prefix of repeated full-prefix hypotheses."""

    holdback: int
    committed: list[int] = field(default_factory=list)
    previous: list[int] | None = None
    revision_conflicts: int = 0

    def update(self, candidate: Sequence[int], *, final: bool = False) -> list[int]:
        current = [int(value) for value in candidate]
        if current[: len(self.committed)] != self.committed:
            self.revision_conflicts += 1
            self.previous = current
            return []
        if final:
            stable = len(current)
        elif self.previous is None:
            stable = len(self.committed)
        else:
            stable = max(
                len(self.committed),
                longest_common_prefix(self.previous, current) - max(0, self.holdback),
            )
        stable = min(stable, len(current))
        new = current[len(self.committed) : stable]
        self.committed.extend(new)
        self.previous = current
        return new


@dataclass
class AppendOnlyDeltaCommitter:
    """Commit stable continuations generated after an immutable prefix.

    The model is prompted with all already committed target tokens.  Therefore
    every candidate passed here is only the *new delta*.  Once a stable delta
    prefix is committed, it is removed from the comparison buffer and can
    never be revised by a later source prefix.
    """

    holdback: int
    committed: list[int] = field(default_factory=list)
    previous_delta: list[int] | None = None
    revision_conflicts: int = 0

    def update(self, candidate_delta: Sequence[int], *, final: bool = False) -> list[int]:
        current = [int(value) for value in candidate_delta]
        if final:
            stable = len(current)
        elif self.previous_delta is None:
            stable = 0
        else:
            stable = max(
                0,
                longest_common_prefix(self.previous_delta, current)
                - max(0, self.holdback),
            )
            if stable == 0 and self.previous_delta and current:
                self.revision_conflicts += 1
        new = current[:stable]
        self.committed.extend(new)
        self.previous_delta = current[stable:]
        return new

    def force_pending(self) -> list[int]:
        """Commit the last observed delta at true source EOS only."""

        if not self.previous_delta:
            return []
        new = list(self.previous_delta)
        self.committed.extend(new)
        self.previous_delta = []
        return new


__all__ = [
    "AppendOnlyDeltaCommitter",
    "StablePrefixCommitter",
    "longest_common_prefix",
]

