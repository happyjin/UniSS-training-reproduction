"""Deterministic dense timing schedule with a real 800 ms observation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TickPlan:
    chunk_end_ms: int
    future_1_end_ms: int
    future_2_end_ms: int
    kind: str


def _stable_uniform(*values: object) -> float:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    integer = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return integer / float(1 << 64)


def _middle_late_tick(sample_id: str, duration_ms: int) -> int | None:
    if duration_ms < 960:
        return None
    lower = min(duration_ms, max(960, math.ceil(duration_ms * 0.45 / 160) * 160))
    upper = min(duration_ms, max(lower, math.ceil(duration_ms * 0.85 / 160) * 160))
    candidates = list(range(lower, upper + 1, 160)) or [duration_ms]
    index = min(
        int(_stable_uniform(sample_id, "middle_late_v2") * len(candidates)),
        len(candidates) - 1,
    )
    return min(duration_ms, candidates[index])


def tick_times(sample_id: str, duration_ms: int) -> tuple[int, ...]:
    """Return dense early ticks plus one deterministic middle/late tick.

    Every utterance that is at least 800 ms long contains an exact 800 ms
    observation. Short utterances end at their physical duration and are not
    eligible for the grouped hard-deadline loss.
    """

    if not sample_id:
        raise ValueError("sample_id must not be empty")
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    values = {value for value in (320, 480, 640, 800) if value <= duration_ms}
    if not values:
        values.add(duration_ms)
    elif duration_ms < 800:
        values.add(duration_ms)
    middle = _middle_late_tick(sample_id, duration_ms)
    if middle is not None:
        values.add(middle)
    result = tuple(sorted(values))
    if duration_ms >= 800 and 800 not in result:
        raise AssertionError("hard-deadline-capable session lost its exact 800 ms tick")
    return result


def plans_for_row(sample_id: str, duration_ms: int) -> tuple[TickPlan, ...]:
    result = []
    for tick in tick_times(sample_id, duration_ms):
        kind = f"fixed_{tick}" if tick in {320, 480, 640, 800} else "middle_late"
        result.append(
            TickPlan(
                chunk_end_ms=tick,
                future_1_end_ms=min(duration_ms, tick + 160),
                future_2_end_ms=min(duration_ms, tick + 320),
                kind=kind,
            )
        )
    return tuple(result)


__all__ = ["TickPlan", "plans_for_row", "tick_times"]
