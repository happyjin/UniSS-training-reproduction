"""Put the gap where the speech is quiet, not where the fragment happens to end.

The observation this rests on
----------------------------
``StreamingBiCodecDecoder`` carries 50 tokens of left context across fragment
pushes, so ``translation_concat`` -- the fragments end to end -- is one
continuous stream with no cut in it at all.  The seam step there measures 0.09x
the interior p99.  The cut the ear hears is made by the *placement*, which
breaks that stream at each fragment boundary, and 40.2% of those boundaries sit
inside a phone with full voice on both sides.

The gap has to go somewhere.  It does not have to go there.

What this does
--------------
For each boundary, search backwards within ``window_ms`` for the quietest
frame and split there instead.  The audio after the new split point is carried
into the next fragment, so not a sample is lost -- it is simply heard a little
later.

Why only backwards
------------------
Moving a split *later* would play audio from the next fragment before the
source that justified it had been read, which is the one thing the streaming
contract forbids.  Moving it earlier only ever delays audio, which is always
legal, and is what the playout buffer does at a coarser grain.

By construction the content is untouched -- the same samples in the same order
-- so ASR-BLEU can only move through where the silence lands.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
FRAME_MS = 5
FRAME = SAMPLE_RATE * FRAME_MS // 1000
DEFAULT_WINDOW_MS = 120.0
DEFAULT_MIN_KEEP_MS = 60.0


def frame_energy(audio: np.ndarray) -> np.ndarray:
    n = len(audio) // FRAME
    if n == 0:
        return np.zeros(0)
    block = audio[: n * FRAME].reshape(n, FRAME).astype(np.float64)
    return np.sqrt((block**2).mean(axis=1))


def snap(
    durations: list[float],
    audio: np.ndarray,
    *,
    window_ms: float = DEFAULT_WINDOW_MS,
    min_keep_ms: float = DEFAULT_MIN_KEEP_MS,
) -> tuple[list[float], list[float]]:
    """New durations whose boundaries sit at local energy minima.

    Returns the adjusted durations and, per boundary, how far back it moved.
    """
    energy = frame_energy(audio)
    if len(energy) == 0 or len(durations) < 2:
        return list(durations), [0.0] * max(0, len(durations) - 1)
    window = max(1, int(round(window_ms / FRAME_MS)))
    out = [float(d) for d in durations]
    shifts: list[float] = []
    cumulative = 0.0
    for k in range(len(out) - 1):
        cumulative += out[k]
        edge = int(round(cumulative / FRAME_MS))
        # never shorten a fragment below min_keep_ms, and never move past its start
        floor = int(round((cumulative - out[k] + float(min_keep_ms)) / FRAME_MS))
        lo = max(floor, edge - window, 0)
        hi = min(edge, len(energy) - 1)
        if hi <= lo:
            shifts.append(0.0)
            continue
        best = lo + int(np.argmin(energy[lo : hi + 1]))
        moved = (edge - best) * FRAME_MS
        if moved <= 0:
            shifts.append(0.0)
            continue
        out[k] -= moved
        out[k + 1] += moved
        cumulative -= moved
        shifts.append(float(moved))
    return out, shifts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="manifest with snapped durations")
    parser.add_argument("--window-ms", type=float, default=DEFAULT_WINDOW_MS)
    parser.add_argument("--min-keep-ms", type=float, default=DEFAULT_MIN_KEEP_MS)
    args = parser.parse_args()

    data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    moved: list[float] = []
    touched = 0
    for sample in data["samples"]:
        durations = [float(v) for v in sample.get("durations", [])]
        if len(durations) < 2:
            continue
        audio, rate = sf.read(str(sample["translation_concat"]), dtype="float32")
        if audio.ndim == 2:
            audio = audio[:, -1]
        if int(rate) != SAMPLE_RATE:
            raise ValueError(f"{sample['translation_concat']} is {rate} Hz")
        new, shifts = snap(
            durations, audio, window_ms=args.window_ms, min_keep_ms=args.min_keep_ms
        )
        sample["durations"] = new
        moved += [s for s in shifts if s > 0]
        touched += sum(1 for s in shifts if s > 0)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"snapped {touched} boundaries, median move {statistics.median(moved) if moved else 0:.0f} ms, "
          f"max {max(moved) if moved else 0:.0f} ms")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
