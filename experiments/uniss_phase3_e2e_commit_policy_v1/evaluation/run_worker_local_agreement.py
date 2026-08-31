#!/usr/bin/env python3
"""Run the established E2E gate worker with the two isolated policy fixes.

At most two symbols are rebound on the established worker:

* ``incremental_mt_rollout`` -> local-agreement commits, always on, holdback
  from ``UNISS_E2E_MT_HOLDBACK`` (default 1).  This affects only the ``e_mt_*``
  measurement.
* ``PersistentInterleavedSession`` -> the source-paced session, enabled by
  ``UNISS_E2E_SEMANTIC_PACE=1``.  This is the one that changes the demo audio.

Everything else -- model loading, the ASR rollout, the event grammar, audio
decoding, metric computation and the report schema -- is the established
implementation, so worker reports stay comparable with every historical run.

Configuration comes from the environment rather than new command line flags,
because ``run_worker.main`` owns its own parser and must keep receiving exactly
the arguments the established gate script passes.
"""

from __future__ import annotations

import functools
import json
import os

import experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.run_worker as worker
from experiments.uniss_phase3_e2e_commit_policy_v1.runtime.local_agreement import (
    DEFAULT_HOLDBACK,
    local_agreement_mt_rollout,
)
from experiments.uniss_phase3_e2e_commit_policy_v1.runtime.semantic_pacing import (
    DEFAULT_MARGIN_MS,
    DEFAULT_TAIL_MS,
    MINIMUM_FRAGMENT_TOKENS,
    PacedInterleavedSession,
)


ENV_HOLDBACK = "UNISS_E2E_MT_HOLDBACK"
ENV_PACE = "UNISS_E2E_SEMANTIC_PACE"
ENV_PACE_MARGIN_MS = "UNISS_E2E_SEMANTIC_PACE_MARGIN_MS"
ENV_PACE_TAIL_MS = "UNISS_E2E_SEMANTIC_TAIL_MS"
ENV_PACE_MINIMUM = "UNISS_E2E_SEMANTIC_MIN_FRAGMENT"


def resolve_holdback() -> int:
    raw = os.environ.get(ENV_HOLDBACK, "").strip()
    if not raw:
        return DEFAULT_HOLDBACK
    value = int(raw)
    if value < 0:
        raise ValueError(f"{ENV_HOLDBACK} must be non-negative, got {value}")
    return value


def _number(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    value = float(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def resolve_pacing() -> dict[str, float] | None:
    """``None`` means keep the established flat semantic cap."""

    if os.environ.get(ENV_PACE, "").strip() not in {"1", "true", "yes"}:
        return None
    minimum = _number(ENV_PACE_MINIMUM, MINIMUM_FRAGMENT_TOKENS)
    if minimum < 2:
        # One token cannot be followed by END, so the fragment is always
        # reported malformed.  Refuse rather than silently degrade structure.
        raise ValueError(f"{ENV_PACE_MINIMUM} must be at least 2")
    return {
        "pace_margin_ms": _number(ENV_PACE_MARGIN_MS, DEFAULT_MARGIN_MS),
        "pace_tail_ms": _number(ENV_PACE_TAIL_MS, DEFAULT_TAIL_MS),
        "minimum_fragment_tokens": int(minimum),
    }


def main() -> None:
    holdback = resolve_holdback()
    worker.incremental_mt_rollout = functools.partial(
        local_agreement_mt_rollout, holdback=holdback
    )
    manifest: dict[str, object] = {
        "mt_commit_policy": "local_agreement",
        "holdback": holdback,
        "semantic_pacing": None,
    }
    pacing = resolve_pacing()
    if pacing is not None:
        worker.PersistentInterleavedSession = functools.partial(
            PacedInterleavedSession, **pacing
        )
        manifest["semantic_pacing"] = pacing
    print(json.dumps(manifest, sort_keys=True), flush=True)
    worker.main()


if __name__ == "__main__":
    main()
