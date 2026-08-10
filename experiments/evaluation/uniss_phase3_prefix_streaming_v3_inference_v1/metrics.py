"""Latency metrics for token commits on a source-time axis."""

from __future__ import annotations

from statistics import fmean
from typing import Sequence


def latency_metrics(
    emission_ms: Sequence[float], source_duration_ms: float
) -> dict[str, float | None]:
    times = [float(value) for value in emission_ms]
    if not times or source_duration_ms <= 0:
        return {"al_ms": None, "laal_ms": None, "ap": None}
    target_length = len(times)
    ideal_step = source_duration_ms / target_length
    lag = [time - index * ideal_step for index, time in enumerate(times)]
    # Length-adaptive AL uses the observed target length.  Here it equals AL
    # because no reference target length is available during an upload demo;
    # both fields are retained so dataset evaluation can later substitute a
    # reference-aware denominator without changing the artifact schema.
    return {
        "al_ms": fmean(lag),
        "laal_ms": fmean(lag),
        "ap": fmean(min(1.0, max(0.0, time / source_duration_ms)) for time in times),
    }

