"""Trade onset latency for continuity, without ever playing audio early.

The problem
-----------
Silence ratio is silence over span, and our silence is not decoder noise or a
wrong translation -- both were measured and ruled out.  It is the wait between
fragments: ``p2st_cascade`` starts a fragment at
``max(source_end_ms, previous_end)``, so whenever the previous fragment
finishes speaking before the read boundary that justifies the next one, the
timeline gets a hole.

Measured over 777 RealSI segments, silence ratio tracks speech volume
(r = -0.35 / -0.40) and not text length (-0.10 / +0.03), and the only lever on
speech volume -- the length prior -- is exhausted: pushing it from 4.0 to 8.0
buys 4% of silence and costs 0.8 ASR-BLEU on zh->en.

What this does
--------------
A playout buffer, exactly as a network jitter buffer works.  The listener
starts B milliseconds later than the first fragment could legally be heard,
and from then on fragments play back-to-back for as long as data is available.
A fragment is still never scheduled before ``delays[k]``, the read boundary
that produced it, so the invariant this project has held throughout --
**no audio before the source that justifies it** -- is untouched.  What changes
is only that audio may be *later*.

``B = 0`` reproduces the streaming behaviour; a large ``B`` converges on
consecutive interpretation, where all the speech is contiguous.  The point is
not to pick a B here but to measure the curve, which is the trade-off
NaturalFlow describes as "the sweet spot between the low-latency benefits of
simultaneous translation and the natural flow of consecutive translation".

Because this is arithmetic on timings plus a rewrite of the placed waveform,
it needs no GPU and no re-inference: the fragments themselves are unchanged,
so ASR-BLEU is unchanged by construction.  Only when they are heard moves.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
FRAME_MS = 15
FRAME = SAMPLE_RATE * FRAME_MS // 1000
FLOOR_FRACTION = 0.02
MIN_SILENCE_MS = 100


def schedule(delays: list[float], durations: list[float], buffer_ms: float) -> list[float]:
    """Start times under a ``buffer_ms`` playout buffer.

    ``delays[k]`` is the source boundary that justified fragment ``k`` and is
    a hard lower bound -- the buffer can only ever push a fragment later.
    Between fragments the scheduler plays continuously, so a fragment starts at
    whichever is later: the end of the previous one, or its own legality bound.
    """
    if not delays:
        return []
    starts: list[float] = []
    cursor = float(delays[0]) + float(buffer_ms)
    for delay, duration in zip(delays, durations):
        start = max(cursor, float(delay))
        starts.append(start)
        cursor = start + float(duration)
    return starts


def silence_ratio_from_schedule(
    starts: list[float], durations: list[float]
) -> float | None:
    """Silence over span, from timings alone, with the 100 ms floor applied."""
    if not starts:
        return None
    span_start = starts[0]
    span_end = starts[-1] + durations[-1]
    span = span_end - span_start
    if span <= 0:
        return None
    silence = 0.0
    previous_end = span_start
    for start, duration in zip(starts, durations):
        gap = start - previous_end
        if gap >= MIN_SILENCE_MS:
            silence += gap
        previous_end = start + duration
    return silence / span


def render(
    fragments: list[np.ndarray], starts: list[float], total_ms: float
) -> np.ndarray:
    """Place the unchanged fragment audio at the scheduled times."""
    length = int(round(total_ms * SAMPLE_RATE / 1000)) + 1
    out = np.zeros(max(length, 1), dtype=np.float32)
    for audio, start in zip(fragments, starts):
        offset = int(round(start * SAMPLE_RATE / 1000))
        end = min(offset + len(audio), len(out))
        if end > offset:
            out[offset:end] += audio[: end - offset]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--buffer-ms",
        type=float,
        action="append",
        required=True,
        help="repeat for a sweep; 0 reproduces the streaming placement",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    samples = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["samples"]
    curve: list[dict[str, object]] = []
    for buffer_ms in sorted(set(args.buffer_ms)):
        rows: list[dict[str, float]] = []
        for sample in samples:
            delays = [float(v) for v in sample.get("delays", [])]
            durations = [float(v) for v in sample.get("durations", [])]
            if not delays or len(delays) != len(durations):
                continue
            starts = schedule(delays, durations, buffer_ms)
            ratio = silence_ratio_from_schedule(starts, durations)
            if ratio is None:
                continue
            rows.append(
                {
                    "direction": sample.get("direction")
                    or ("zh2en" if sample.get("src_lang") == "cmn" else "en2zh"),
                    "silence_ratio": ratio,
                    "onset_ms": starts[0],
                    "end_offset_ms": starts[-1]
                    + durations[-1]
                    - float(sample.get("source_duration_ms") or 0.0),
                    "span_ms": starts[-1] + durations[-1] - starts[0],
                    # Mean lag of each fragment behind the boundary that
                    # justified it: what the buffer actually costs.
                    "added_lag_ms": statistics.mean(
                        s - d for s, d in zip(starts, delays)
                    ),
                }
            )
        entry: dict[str, object] = {"buffer_ms": buffer_ms, "samples": len(rows)}
        for name in ("en2zh", "zh2en"):
            group = [r for r in rows if r["direction"] == name]
            if not group:
                continue
            entry[name] = {
                key: statistics.mean(float(r[key]) for r in group)
                for key in ("silence_ratio", "onset_ms", "end_offset_ms", "added_lag_ms")
            }
        curve.append(entry)

    report = {
        "schema_version": "uniss_streaming_p2st_playout_buffer_v1",
        "manifest": str(Path(args.manifest).resolve()),
        "invariant": "a fragment is never scheduled before delays[k]",
        "curve": curve,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{'buffer':>8s}{'dir':>7s}{'SR':>8s}{'onset ms':>10s}"
        f"{'added lag':>11s}{'end off':>10s}"
    )
    for entry in curve:
        for name in ("en2zh", "zh2en"):
            stats = entry.get(name)
            if not stats:
                continue
            print(
                f"{entry['buffer_ms']:>8.0f}{name:>7s}{stats['silence_ratio']:>8.3f}"
                f"{stats['onset_ms']:>10.0f}{stats['added_lag_ms']:>11.0f}"
                f"{stats['end_offset_ms']:>10.0f}"
            )
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
