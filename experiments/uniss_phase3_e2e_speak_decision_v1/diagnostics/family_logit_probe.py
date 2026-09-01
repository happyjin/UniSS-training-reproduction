#!/usr/bin/env python3
"""Record every argmax the interleaved runtime makes, with the losing logits.

The speak-decision run raised no ``WRITE_MT``: 0.168 -> 0.147 over 95 events.
The recorded decision sequences say why the target may have been wrong.  No
event ever runs ASR -> MT -> TTS; events are either ``WRITE_ASR -> WAIT`` (78 of
95) or ``WRITE_MT -> WRITE_SEMANTIC -> ...`` (16 of 95).  An event that starts
at MT skipped ASR at ``_choice(allowed_families)`` -- the three-way family
argmax in ``runtime.run_event`` -- not at the WRITE_GENERATE / WAIT_READ
decision this experiment trained.

This probe separates the two.  It patches ``PersistentInterleavedSession._choice``
to record the candidate logits before returning the established argmax, so the
session behaves bit-for-bit as the gate ran it, and delegates to the
local-agreement worker so the commit policy and pacing are also unchanged.

What the numbers decide: if the family gap ``logit[TASK_MT] - logit[TASK_ASR]``
is small, a margin term on the family row can flip it and the corrected S2 is
cheap.  If it is large, the family choice is entrenched and the wall is
elsewhere.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
from typing import Sequence

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation import runtime
from experiments.uniss_phase3_e2e_commit_policy_v1.evaluation import (
    run_worker_local_agreement,
)
from training import constants_uniss as c

ENV_OUTPUT = "UNISS_E2E_FAMILY_PROBE_OUTPUT"

TOKEN_NAMES = {
    c.TOKEN_WRITE_GENERATE: "WRITE_GENERATE",
    c.TOKEN_WAIT_READ: "WAIT_READ",
    c.TOKEN_START_GLM: "READ_NEXT",
    c.TOKEN_EOS: "EOS",
    c.TOKEN_TASK_ASR: "TASK_ASR",
    c.TOKEN_TASK_S2T_TRANSLATION: "TASK_MT",
    c.TOKEN_TASK_TTS: "TASK_TTS",
}
FAMILY_TOKENS = (
    c.TOKEN_TASK_ASR,
    c.TOKEN_TASK_S2T_TRANSLATION,
    c.TOKEN_TASK_TTS,
)

_RECORDS: list[dict[str, object]] = []


def _name(token: int) -> str:
    return TOKEN_NAMES.get(int(token), f"token_{int(token)}")


def install_probe() -> None:
    """Wrap the established argmax; the returned choice is never altered."""

    established = runtime.PersistentInterleavedSession._choice

    def probing_choice(self, candidates: Sequence[int]) -> int:
        chosen = established(self, candidates)
        values = self.logits.reshape(-1)
        scores = {_name(token): float(values[int(token)]) for token in candidates}
        kind = (
            "family"
            if set(int(t) for t in candidates) <= set(FAMILY_TOKENS)
            else "continuation"
        )
        ordered = sorted(scores.values(), reverse=True)
        _RECORDS.append(
            {
                "kind": kind,
                "chosen": _name(chosen),
                "logits": scores,
                "top2_margin": (ordered[0] - ordered[1]) if len(ordered) > 1 else None,
            }
        )
        return chosen

    runtime.PersistentInterleavedSession._choice = probing_choice


def _flush() -> None:
    destination = os.environ.get(ENV_OUTPUT, "").strip()
    if not destination or not _RECORDS:
        return
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        for record in _RECORDS:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    if not os.environ.get(ENV_OUTPUT, "").strip():
        raise SystemExit(f"{ENV_OUTPUT} is required")
    install_probe()
    atexit.register(_flush)
    run_worker_local_agreement.main()


if __name__ == "__main__":
    main()
