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

from experiments.uniss_phase3_e2e_commit_policy_v1.runtime.local_agreement import (
    display_units,
)
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


FAMILY_TOKENS = (
    c.TOKEN_TASK_ASR,
    c.TOKEN_TASK_S2T_TRANSLATION,
    c.TOKEN_TASK_TTS,
)


def is_family_decision(values: Sequence[int]) -> bool:
    """``_choice`` serves two decisions; the family one offers only families."""

    return bool(values) and all(int(value) in FAMILY_TOKENS for value in values)


class ContentGatedSpeakSession(PacedInterleavedSession):
    """Speak when this event's ASR produced new content, otherwise wait.

    The oracle bracketed the two extremes and neither is acceptable:

    | policy       | speak rate | natural_eos | first speech | cmn coverage |
    |--------------|-----------:|------------:|-------------:|-------------:|
    | conservative |      0.168 |        0.50 |       680 ms |        0.514 |
    | eager        |      1.000 |        1.00 |       280 ms |        0.478 |

    Forcing every event brings the timing but degenerates into repetition
    ("of of", "new new"); leaving it to the sampled continuation logit speaks on
    17% of events and starves coverage.  This is the middle ground the design
    actually asked for: the plan's section 24 calls WAIT_READ/WRITE_GENERATE
    "a deterministic event delimiter, not a policy classification target", so the
    delimiter is derived from content rather than sampled.

    Within an event the grammar runs ASR before MT before TTS, so by the time
    the second continuation decision is made ``source_text`` has already grown
    if ASR committed anything.  The rule is therefore:

    * first decision of the event -- always continue, so ASR runs at all;
    * later decisions -- continue only if ASR added source units this event;
    * otherwise fall through to the model's own choice, which is normally WAIT.

    Only ``_choice`` and the per-event bookkeeping are overridden.  The event
    grammar, family ordering, EOS legality and the inherited pacing budget are
    untouched.
    """

    def __init__(self, *args, force_family_order: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.force_family_order = bool(force_family_order)
        self._event_asr_done = False
        self._event_start_source_units = 0
        self.gate_opened = 0
        self.gate_withheld = 0

    def _source_units(self) -> int:
        return len(display_units(self.source_text, self.trajectory.src_lang))

    def run_event(self, event, **kwargs):
        self._event_asr_done = False
        self._event_start_source_units = self._source_units()
        return super().run_event(event, **kwargs)

    def _choice(self, candidates: Sequence[int]) -> int:
        values = [int(value) for value in candidates]
        if is_family_decision(values):
            choice = values[0] if self.force_family_order else super()._choice(values)
            if choice == c.TOKEN_TASK_ASR:
                self._event_asr_done = True
            return choice
        if c.TOKEN_WRITE_GENERATE in values:
            if not self._event_asr_done:
                # Let ASR run; it is the input side and already fires on 82% of
                # events, so this does not change the speaking decision.
                return c.TOKEN_WRITE_GENERATE
            if self._source_units() > self._event_start_source_units:
                self.gate_opened += 1
                return c.TOKEN_WRITE_GENERATE
            self.gate_withheld += 1
        return super()._choice(values)


__all__ = ["ContentGatedSpeakSession", "EagerSpeakSession", "FAMILY_TOKENS", "is_family_decision"]
