#!/usr/bin/env python3
"""Remove the guaranteed empty first commit from the stable-prefix rule.

``StablePrefixCommitter.update`` needs two hypotheses to compare:

    elif self.previous is None:
        stable = len(self.committed)      # zero on the first call

So the first invocation of every stage commits nothing no matter what the
model produced, and the cascade has two stages in series -- ASR must commit
before the switch rule reaches MT, and MT must commit before it reaches TTS.
That is two blocks, 320 ms, spent before any token can be released, on top of
the blocks spent waiting for the longest common prefix to grow past the
holdback.

Seeding ``previous`` with the first hypothesis makes the first call behave like
an agreement with itself: ``stable = len(current) - holdback``.  The holdback
still trims the unstable tail, so this is not the same as holdback=0, which
was swept and found to cost content -- short-audio MT chrF fell from 72.88 to
50.85 and English ASR error rose from 0.120 to 0.200 because the raw longest
common prefix pollutes both prefixes.  Here the trim is kept and only the
"wait for a second opinion" requirement is dropped on the very first block.

Subclassed rather than changed in place: ``StablePrefixCommitter`` belongs to
another experiment and is used by its gates.
"""
from __future__ import annotations

from typing import Sequence

from experiments.uniss_phasea_stateful_longepisode_rl_v1.runtime.commit import (
    StablePrefixCommitter,
)


class SeededPrefixCommitter(StablePrefixCommitter):
    """``StablePrefixCommitter`` whose first call is not forced to commit zero."""

    def update(self, candidate: Sequence[int], *, final: bool = False) -> list[int]:
        if self.previous is None and not final:
            current = [int(value) for value in candidate]
            if current[: len(self.committed)] == self.committed:
                self.previous = current
        return super().update(candidate, final=final)


__all__ = ["SeededPrefixCommitter"]
