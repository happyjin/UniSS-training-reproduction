#!/usr/bin/env python3
"""Graded logit bias on the two decisions that actually gate translation.

The decision-logit probe located them: within an event the first
WRITE_GENERATE/WAIT_READ choice leads by a median 28.58 logits and is never the
constraint, while ``logit[TASK_MT] - logit[TASK_ASR]`` has median -6.75 and the
continue-after-fragment WRITE/WAIT choice has median -2.88.

S0.1 falsified a *hard* content gate: forcing MT on every event drove the
session text length ratio 1.70 -> 15.40 with a repetition loop.  A graded bias
is a different instrument.  Sweeping delta answers the question a hard gate
cannot: is the model able to translate and merely mis-ranked, or is the ranking
correct because the incremental MT is not there yet?

* If some delta reaches the gold 1.95:1 family ratio without repetition
  degradation, the ranking is wrong and a position-split margin loss will fix
  it.
* If every delta that raises the MT share also produces repetition or garbage,
  the ranking is *right* and the wall is incremental MT capability, not the
  decision.

The bias is applied only inside ``_choice``; ``self.logits`` is untouched, so
generation elsewhere is bit-for-bit unchanged.  The recording probe stays
installed so each run yields both behaviour and logits.
"""
from __future__ import annotations

import os

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation import runtime
from experiments.uniss_phase3_e2e_commit_policy_v1.evaluation import (
    run_worker_local_agreement,
)
from experiments.uniss_phase3_e2e_commit_policy_v1.runtime.semantic_pacing import (
    PacedInterleavedSession,
)
from experiments.uniss_phase3_e2e_speak_decision_v1.diagnostics import (
    family_logit_probe as probe,
)
from training import constants_uniss as c

ENV_FAMILY_BIAS = "UNISS_E2E_FAMILY_MT_BIAS"
ENV_CONTINUE_BIAS = "UNISS_E2E_CONTINUE_WRITE_BIAS"

FAMILY_TOKENS = frozenset(
    {c.TOKEN_TASK_ASR, c.TOKEN_TASK_S2T_TRANSLATION, c.TOKEN_TASK_TTS}
)


def install_bias(family_bias: float, continue_bias: float) -> None:
    """Bias TASK_MT, and WRITE_GENERATE only after a fragment already ran."""

    established_choice = runtime.PersistentInterleavedSession._choice
    established_run_event = PacedInterleavedSession.run_event

    def counting_run_event(self, *args, **kwargs):
        self._fragments_this_event = 0
        return established_run_event(self, *args, **kwargs)

    def biased_choice(self, candidates) -> int:
        tokens = [int(token) for token in candidates]
        if not tokens:
            return established_choice(self, candidates)
        values = self.logits.reshape(-1)
        offsets = {token: 0.0 for token in tokens}
        if set(tokens) <= FAMILY_TOKENS:
            if c.TOKEN_TASK_S2T_TRANSLATION in offsets:
                offsets[c.TOKEN_TASK_S2T_TRANSLATION] = family_bias
            chosen = max(tokens, key=lambda t: float(values[t]) + offsets[t])
            self._fragments_this_event = getattr(self, "_fragments_this_event", 0) + 1
            return chosen
        if (
            getattr(self, "_fragments_this_event", 0) >= 1
            and c.TOKEN_WRITE_GENERATE in offsets
        ):
            offsets[c.TOKEN_WRITE_GENERATE] = continue_bias
        return max(tokens, key=lambda t: float(values[t]) + offsets[t])

    # Both classes are wrapped: the worker selects the paced session when
    # pacing is on and the base session otherwise, and a missed reset would
    # leak the fragment counter across events.
    established_base_run_event = runtime.PersistentInterleavedSession.run_event

    def counting_base_run_event(self, *args, **kwargs):
        self._fragments_this_event = 0
        return established_base_run_event(self, *args, **kwargs)

    runtime.PersistentInterleavedSession.run_event = counting_base_run_event
    PacedInterleavedSession.run_event = counting_run_event
    runtime.PersistentInterleavedSession._choice = biased_choice


def main() -> None:
    family_bias = float(os.environ.get(ENV_FAMILY_BIAS, "0") or 0.0)
    continue_bias = float(os.environ.get(ENV_CONTINUE_BIAS, "0") or 0.0)
    if family_bias < 0 or continue_bias < 0:
        raise SystemExit("biases must be non-negative")
    install_bias(family_bias, continue_bias)
    probe.install_probe()  # wraps the biased choice; records what it returns
    import atexit

    atexit.register(probe._flush)
    run_worker_local_agreement.main()


if __name__ == "__main__":
    main()
