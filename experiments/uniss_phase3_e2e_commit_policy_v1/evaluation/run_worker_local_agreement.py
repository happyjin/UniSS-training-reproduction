#!/usr/bin/env python3
"""Run the established E2E gate worker with a local-agreement MT committer.

Only one symbol is rebound: ``run_worker.incremental_mt_rollout``.  Everything
else -- model loading, the ASR rollout, the interleaved S2S session, audio
decoding, metric computation and the report schema -- is the established
implementation, so the resulting worker reports stay directly comparable with
every historical gate run.

The holdback is read from ``UNISS_E2E_MT_HOLDBACK`` rather than a new command
line flag, because ``run_worker.main`` owns its own parser and must keep
receiving exactly the arguments the established gate script passes.
"""

from __future__ import annotations

import functools
import os

import experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.run_worker as worker
from experiments.uniss_phase3_e2e_commit_policy_v1.runtime.local_agreement import (
    DEFAULT_HOLDBACK,
    local_agreement_mt_rollout,
)


ENV_HOLDBACK = "UNISS_E2E_MT_HOLDBACK"


def resolve_holdback() -> int:
    raw = os.environ.get(ENV_HOLDBACK, "").strip()
    if not raw:
        return DEFAULT_HOLDBACK
    value = int(raw)
    if value < 0:
        raise ValueError(f"{ENV_HOLDBACK} must be non-negative, got {value}")
    return value


def main() -> None:
    holdback = resolve_holdback()
    worker.incremental_mt_rollout = functools.partial(
        local_agreement_mt_rollout, holdback=holdback
    )
    print(
        f"{{\"mt_commit_policy\": \"local_agreement\", \"holdback\": {holdback}}}",
        flush=True,
    )
    worker.main()


if __name__ == "__main__":
    main()
