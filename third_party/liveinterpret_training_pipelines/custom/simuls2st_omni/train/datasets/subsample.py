"""Uniform subsample of an in-memory record list.

`max_samples` used to take a prefix of sorted jsonl, which dropped later
corpora (CoVoST behind GigaST). This draws a seed-stable permutation and
stops at a sample count and/or an hour budget.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


def subsample_uniform(
    records: Sequence[T],
    *,
    max_samples: int = 0,
    max_hours: float = 0.0,
    seed: int = 0,
    duration_sec_fn: Optional[Callable[[T], float]] = None,
) -> list[T]:
    """Keep a uniform subset totaling ``max_hours`` and/or ``max_samples``.

    Each record is equally likely. Hours use ``duration_sec_fn``; if the pool
    is already under the hour cap the list is returned unchanged (unless
    ``max_samples`` still binds).
    """
    if not records:
        return []
    max_samples = int(max_samples or 0)
    max_hours = float(max_hours or 0.0)
    if max_samples <= 0 and max_hours <= 0:
        return list(records)

    target_sec = max_hours * 3600.0 if max_hours > 0 else 0.0
    if target_sec and duration_sec_fn is not None:
        total_sec = sum(float(duration_sec_fn(r)) for r in records)
        if total_sec <= target_sec and (max_samples <= 0 or len(records) <= max_samples):
            return list(records)

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(records))
    picked: list[T] = []
    acc = 0.0
    for pos in order:
        rec = records[int(pos)]
        picked.append(rec)
        if target_sec and duration_sec_fn is not None:
            acc += float(duration_sec_fn(rec))
            if acc >= target_sec:
                break
        if max_samples > 0 and len(picked) >= max_samples:
            break
    return picked
