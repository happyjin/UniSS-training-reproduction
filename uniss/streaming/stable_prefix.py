"""Irreversible stable-prefix commit logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


def longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


@dataclass
class StablePrefixCommitter:
    holdback_tokens: int = 2
    committed: list[int] = field(default_factory=list)
    previous_candidate: list[int] | None = None
    revision_events: int = 0

    def update(self, candidate: Sequence[int], *, is_final: bool = False) -> list[int]:
        current = [int(token) for token in candidate]
        if current[: len(self.committed)] != self.committed:
            self.revision_events += 1
            current = [*self.committed, *current[len(self.committed) :]]
        if self.previous_candidate is None:
            stable_length = len(current) if is_final else len(self.committed)
        else:
            stable_length = longest_common_prefix(self.previous_candidate, current)
            if not is_final:
                stable_length = max(len(self.committed), stable_length - self.holdback_tokens)
            else:
                stable_length = len(current)
        stable_length = max(len(self.committed), min(stable_length, len(current)))
        new_tokens = current[len(self.committed) : stable_length]
        self.committed.extend(new_tokens)
        self.previous_candidate = current
        return new_tokens
