"""Pure CPU window planning and target-timeline helpers.

The released WhisperVQ tokenizer internally caps one acoustic encoder call at
30 seconds.  Long-form inference therefore uses sentence-friendly, bounded
source windows instead of letting cumulative prefix re-encoding grow without
limit.  These helpers deliberately contain no model imports so their coverage
does not require a GPU or the UniSS runtime environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class WindowSpan:
    """One non-overlapping source window on the global sample timeline."""

    index: int
    start_sample: int
    end_sample: int
    boundary_rms: float

    @property
    def samples(self) -> int:
        return self.end_sample - self.start_sample

    def to_dict(self, sample_rate: int) -> dict[str, int | float]:
        payload = asdict(self)
        payload.update(
            {
                "start_seconds": self.start_sample / sample_rate,
                "end_seconds": self.end_sample / sample_rate,
                "duration_seconds": self.samples / sample_rate,
            }
        )
        return payload


def _boundary_rms(values: np.ndarray, center: int, radius: int) -> float:
    start = max(0, int(center) - int(radius))
    end = min(len(values), int(center) + int(radius))
    if end <= start:
        return 0.0
    chunk = np.asarray(values[start:end], dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))


def plan_bounded_windows(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    target_seconds: float = 25.0,
    minimum_seconds: float = 18.0,
    maximum_seconds: float = 30.0,
    search_seconds: float = 5.0,
    analysis_seconds: float = 0.20,
    search_step_seconds: float = 0.02,
) -> list[WindowSpan]:
    """Cover ``waveform`` with contiguous, silence-seeking bounded windows.

    A boundary is selected near ``target_seconds`` by minimizing local RMS.
    Windows never overlap and never leave gaps, which prevents duplicate target
    speech.  Except for a recording shorter than ``minimum_seconds``, every
    span is kept within ``[minimum_seconds, maximum_seconds]``.
    """

    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if values.size == 0:
        raise ValueError("waveform must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("waveform contains non-finite samples")
    if not 0 < minimum_seconds <= target_seconds <= maximum_seconds:
        raise ValueError("expected 0 < minimum <= target <= maximum")
    if search_seconds < 0 or analysis_seconds <= 0 or search_step_seconds <= 0:
        raise ValueError("invalid boundary-search geometry")

    minimum = max(1, int(round(minimum_seconds * sample_rate)))
    target = max(minimum, int(round(target_seconds * sample_rate)))
    maximum = max(target, int(round(maximum_seconds * sample_rate)))
    search = int(round(search_seconds * sample_rate))
    radius = max(1, int(round(analysis_seconds * sample_rate / 2.0)))
    step = max(1, int(round(search_step_seconds * sample_rate)))

    spans: list[WindowSpan] = []
    start = 0
    total = len(values)
    while total - start > maximum:
        desired = start + target
        # Leave at least one minimum-size tail.  This prevents a final tiny
        # fragment that would otherwise have to be merged above 30 seconds.
        latest = min(start + maximum, total - minimum)
        earliest = start + minimum
        low = max(earliest, desired - search)
        high = min(latest, desired + search)
        if high < low:
            low, high = earliest, latest
        candidates = list(range(low, high + 1, step))
        if not candidates or candidates[-1] != high:
            candidates.append(high)

        rms_values = [_boundary_rms(values, point, radius) for point in candidates]
        # RMS is primary.  Distance only breaks near-identical silence ties so
        # the number of windows remains predictable.
        scale = max(max(rms_values, default=0.0), 1e-8)
        best = min(
            range(len(candidates)),
            key=lambda i: (
                rms_values[i] / scale
                + 1e-4 * abs(candidates[i] - desired) / max(search, 1)
            ),
        )
        end = int(candidates[best])
        spans.append(
            WindowSpan(
                index=len(spans),
                start_sample=start,
                end_sample=end,
                boundary_rms=float(rms_values[best]),
            )
        )
        start = end

    spans.append(
        WindowSpan(
            index=len(spans),
            start_sample=start,
            end_sample=total,
            boundary_rms=_boundary_rms(values, total, radius),
        )
    )
    validate_window_plan(spans, total, minimum, maximum)
    return spans


def validate_window_plan(
    spans: Sequence[WindowSpan], total_samples: int, minimum: int, maximum: int
) -> None:
    if not spans:
        raise ValueError("window plan is empty")
    cursor = 0
    for index, span in enumerate(spans):
        if span.index != index:
            raise ValueError("window indices are not contiguous")
        if span.start_sample != cursor or span.end_sample <= span.start_sample:
            raise ValueError("window plan contains a gap, overlap, or empty span")
        if span.samples > maximum:
            raise ValueError("window exceeds maximum size")
        if total_samples >= minimum and span.samples < minimum:
            raise ValueError("window is shorter than minimum size")
        cursor = span.end_sample
    if cursor != total_samples:
        raise ValueError("window plan does not cover the complete waveform")


def place_target_without_overlap(
    placements: list[tuple[int, np.ndarray]],
    waveform: np.ndarray,
    *,
    available_sample: int,
    cursor: int,
) -> tuple[int, int]:
    """Place target audio after both source availability and prior output."""

    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("target waveform contains non-finite samples")
    start = max(0, int(available_sample), int(cursor))
    end = start + len(values)
    if values.size:
        placements.append((start, values))
    return start, end


def render_target_timeline(
    placements: Sequence[tuple[int, np.ndarray]], minimum_samples: int
) -> np.ndarray:
    total = max(
        [int(minimum_samples), *[int(start) + len(values) for start, values in placements]]
    )
    timeline = np.zeros(total, dtype=np.float32)
    for start, raw in placements:
        values = np.asarray(raw, dtype=np.float32).reshape(-1)
        if np.any(timeline[start : start + len(values)]):
            raise ValueError("target placements overlap")
        timeline[start : start + len(values)] = values
    return timeline


def stereo_waveform(source: np.ndarray, target_timeline: np.ndarray) -> np.ndarray:
    left = np.asarray(source, dtype=np.float32).reshape(-1)
    right = np.asarray(target_timeline, dtype=np.float32).reshape(-1)
    total = max(len(left), len(right))
    stereo = np.zeros((total, 2), dtype=np.float32)
    stereo[: len(left), 0] = left
    stereo[: len(right), 1] = right
    return stereo
