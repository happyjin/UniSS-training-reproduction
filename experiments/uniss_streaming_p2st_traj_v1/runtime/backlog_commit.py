"""Commit the target prefix on a deadline, not only when it stops changing.

The defect
----------
``StablePrefixCommitter`` releases only the longest common prefix of two
consecutive hypotheses.  Near the end of an utterance the MT stage keeps
revising its ending as more source arrives, so that common prefix stops
growing, nothing becomes speakable, and the cascade stays silent.  Then the
last read step arrives with ``final=True``, which commits the whole remaining
hypothesis at once.

Measured over eight long-form utterances, the fragment emitted at that moment
carries **21.4% of the entire translation in 3.5 s**, and on
emilia_zh_0003980703 it is 108 characters after 2.5 s of silence -- the single
most audible stall in the set, and the one a playout buffer cannot reach,
because the source boundary that makes it legal is the end of the audio.

What this changes
-----------------
When nothing is stable and the uncommitted backlog has grown past
``max_backlog`` tokens, release enough of it to leave only ``keep_tail``
uncommitted.  The released prefix has *not* been confirmed by a second
hypothesis, so this trades revision risk for timeliness: audio cannot be
retracted, and a token spoken here may turn out to be one MT would have
revised.  That is the whole cost, and it is why the default is off.

``max_backlog = 0`` disables the deadline entirely and reproduces
``StablePrefixCommitter`` token for token.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from experiments.uniss_phasea_stateful_longepisode_rl_v1.runtime.commit import (
    StablePrefixCommitter,
    longest_common_prefix,
)


@dataclass
class BacklogCappedCommitter(StablePrefixCommitter):
    """A stable-prefix committer with a deadline on how far it may fall behind."""

    max_backlog: int = 0
    keep_tail: int = 0
    forced_tokens: int = 0
    forced_events: int = 0

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
        # A deadline needs something to be late against.  On the first
        # hypothesis there is no previous one to compare with, so nothing has
        # stalled yet and firing here would release text on no evidence.
        if not final and int(self.max_backlog) > 0 and self.previous is not None:
            backlog = len(current) - len(self.committed)
            if backlog >= int(self.max_backlog):
                deadline = len(current) - max(0, int(self.keep_tail))
                if deadline > stable:
                    self.forced_tokens += deadline - stable
                    self.forced_events += 1
                    stable = deadline
        stable = min(max(stable, len(self.committed)), len(current))
        new = current[len(self.committed) : stable]
        self.committed.extend(new)
        self.previous = current
        return new
