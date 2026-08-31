#!/usr/bin/env python3
"""Oracle ceiling: what if the model always took its chance to speak?

Action distribution over the 95 S2S events of the paced holdback-2 run:

| continuation      | count | per event |
|-------------------|------:|----------:|
| ``WAIT``          |    82 |     0.863 |
| ``WRITE_ASR``     |    78 |     0.821 |
| ``WRITE_MT``      |    16 |     0.168 |
| ``WRITE_SEMANTIC``|    15 |     0.158 |
| ``READ_NEXT``     |     9 |     0.095 |
| ``EOS``           |     4 |     0.042 |

The model recognises on 82% of events but translates on only 17% and speaks on
only 16%.  ``emilia_zh_0006795452`` speaks once in eleven events.  Low target
coverage, long internal silences and over-long individual fragments are all the
same behaviour seen from different angles: the policy speaks rarely, so when it
does speak it has to say a lot at once.

This session forces the opposite extreme -- always continue to the next family
when the grammar allows it -- to measure the ceiling that the *content* heads can
reach if the action policy were not the bottleneck.  It is a **diagnostic, not a
shippable policy**: forcing MT and TTS on an event with no new stable content is
exactly the empty-write behaviour a real policy must avoid.

Only ``_choice`` is overridden, so the event grammar, family ordering, EOS
legality and the pacing budget all stay as the established implementation
defines them.
"""

from __future__ import annotations

from typing import Sequence

from experiments.uniss_phase3_e2e_commit_policy_v1.runtime.semantic_pacing import (
    PacedInterleavedSession,
)
from training import constants_uniss as c


class EagerSpeakSession(PacedInterleavedSession):
    """Prefer ``WRITE_GENERATE``, and optionally the earliest allowed family."""

    def __init__(self, *args, force_family_order: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.force_family_order = bool(force_family_order)
        self.forced_continuations = 0
        self.forced_families = 0

    def _choice(self, candidates: Sequence[int]) -> int:
        values = [int(value) for value in candidates]
        if c.TOKEN_WRITE_GENERATE in values:
            self.forced_continuations += 1
            return c.TOKEN_WRITE_GENERATE
        if self.force_family_order and values and all(
            value in (c.TOKEN_TASK_ASR, c.TOKEN_TASK_S2T_TRANSLATION, c.TOKEN_TASK_TTS)
            for value in values
        ):
            self.forced_families += 1
            return values[0]
        return super()._choice(values)


__all__ = ["EagerSpeakSession"]
