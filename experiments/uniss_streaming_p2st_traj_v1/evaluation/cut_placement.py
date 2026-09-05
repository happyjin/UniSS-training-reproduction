"""Where do the fragment boundaries land -- in a pause, or mid-word?

The chopped-audio diagnosis found that the cause is not the codec and not
clicks.  The seam cosine across a fragment boundary measured 0.861 against
0.851 for a random pair, and the sample-to-sample step at a seam was 0.23-0.44x
the p99 of the fragment interior, so the decoder is joining cleanly.  What the
ear hears is *inserted timeline silence*: 24-53% of the placed timeline is
silence the model put between fragments, and **69-79% of the boundaries sit
where the signal has the same energy on both sides** -- that is, in the middle
of a word rather than at a pause.

That last number is the one this module makes reproducible.  It was computed
once, by hand, during the diagnosis; a target of "<= 40%" is meaningless if
nobody can recompute the 74% it is measured against.

Definition
----------
Fragment ``k`` ends at cumulative duration ``sum(durations[:k+1])`` in the
*concatenated* audio -- concatenated rather than placed, so the measurement is
about where the model chose to stop speaking and not about how the timeline
placed it.  Around each interior boundary the RMS of a short window on each
side is compared:

* **mid_voice** -- both sides voiced and their ratio within ``[1/1.3, 1.3]``.
  The model stopped in the middle of continuous speech.
* **onset** or **offset** -- one side voiced and the other not.  The boundary
  is at a natural edge, which is what a word boundary looks like.
* **silent** -- neither side voiced.  Nothing was being said.

``mid_voice_fraction`` counts mid_voice over all interior boundaries.  The
window is 30 ms, two frames of the 15 ms grid ``audio_quality`` uses, which is
short enough to sit inside one phone and long enough to be a stable RMS.

Why not measure the placed audio
--------------------------------
Because the placed timeline confounds two different faults.  A gap there can
mean "the model stopped mid-word" or "the model spoke a complete phrase and
the pacing pushed the next one late".  Only the first is a truncation, and
only the concatenated audio isolates it.  ``audio_quality`` already reports
the placed timeline's gaps and internal silence; this is the complement.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
WINDOW_MS = 30
WINDOW = SAMPLE_RATE * WINDOW_MS // 1000
# Same shape as audio_quality's voiced test: a floor relative to the loud end
# of the file, so a quiet recording is not called silent throughout.
FLOOR_FRACTION = 0.02
RATIO = 1.3
# A fragment whose last window is at least this fraction of its own mean
# voiced level was still speaking when it stopped.  0.90 is the lower edge
# of the 0.90-1.17x band the chopped-audio diagnosis reported.
TAIL_RATIO = 0.90


def _rms(window: np.ndarray) -> float:
    if not len(window):
        return 0.0
    return float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))


def classify_boundaries(
    audio: np.ndarray, durations_ms: list[float]
) -> list[dict[str, object]]:
    """One record per interior fragment boundary."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(durations_ms) < 2 or not len(audio):
        return []
    # A floor from the whole file, so the two sides are judged on one scale.
    frames = len(audio) // WINDOW
    if frames < 2:
        return []
    block = audio[: frames * WINDOW].reshape(frames, WINDOW)
    block_rms = np.sqrt(np.mean(block.astype(np.float64) ** 2, axis=1))
    floor = max(1e-4, FLOOR_FRACTION * float(np.percentile(block_rms, 99)))

    out: list[dict[str, object]] = []
    cursor = 0.0
    for index, duration in enumerate(durations_ms[:-1]):
        cursor += float(duration)
        cut = int(round(cursor * SAMPLE_RATE / 1000))
        if cut - WINDOW < 0 or cut + WINDOW > len(audio):
            continue
        before = _rms(audio[cut - WINDOW : cut])
        after = _rms(audio[cut : cut + WINDOW])
        loud_before, loud_after = before > floor, after > floor
        if loud_before and loud_after:
            ratio = max(before, after) / max(min(before, after), 1e-12)
            kind = "mid_voice" if ratio <= RATIO else "step"
        elif loud_before:
            kind = "offset"
        elif loud_after:
            kind = "onset"
        else:
            kind = "silent"
        out.append(
            {
                "boundary": index,
                "cut_ms": round(cursor, 1),
                "rms_before": before,
                "rms_after": after,
                "ratio": (
                    max(before, after) / max(min(before, after), 1e-12)
                    if loud_before and loud_after
                    else None
                ),
                "kind": kind,
            }
        )
    return out


def fragment_tails(
    audio: np.ndarray, durations_ms: list[float]
) -> list[dict[str, object]]:
    """Was each fragment still at full voice when it stopped?

    This is the direct reading of "truncated": the RMS of the fragment's last
    ``WINDOW_MS`` against the fragment's own mean voiced RMS.  A fragment that
    trailed off into a pause ends well below its own average; a fragment that
    was cut off mid-word ends at roughly its average, which is where the
    diagnosis's "0.90-1.17x RMS" band came from.

    Unlike ``classify_boundaries`` this looks only at the fragment that ends,
    so it does not depend on what the next fragment happens to start with.
    The last fragment is included: an utterance that ends mid-word is the same
    fault as one that breaks mid-word.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if not len(durations_ms) or not len(audio):
        return []
    out: list[dict[str, object]] = []
    start_ms = 0.0
    for index, duration in enumerate(durations_ms):
        end_ms = start_ms + float(duration)
        lo = int(round(start_ms * SAMPLE_RATE / 1000))
        hi = min(int(round(end_ms * SAMPLE_RATE / 1000)), len(audio))
        start_ms = end_ms
        if hi - lo < 2 * WINDOW:
            continue
        fragment = audio[lo:hi]
        frames = len(fragment) // WINDOW
        block = fragment[: frames * WINDOW].reshape(frames, WINDOW)
        block_rms = np.sqrt(np.mean(block.astype(np.float64) ** 2, axis=1))
        floor = max(1e-4, FLOOR_FRACTION * float(np.percentile(block_rms, 99)))
        voiced = block_rms[block_rms > floor]
        if not len(voiced):
            continue
        mean_voiced = float(np.mean(voiced))
        tail = _rms(fragment[-WINDOW:])
        ratio = tail / max(mean_voiced, 1e-12)
        out.append(
            {
                "fragment": index,
                "tail_rms": tail,
                "mean_voiced_rms": mean_voiced,
                "tail_ratio": ratio,
                "truncated": bool(ratio >= TAIL_RATIO),
            }
        )
    return out


def summarise_tails(records: list[dict[str, object]]) -> dict[str, object]:
    total = len(records)
    cut = sum(1 for r in records if r["truncated"])
    ratios = [float(r["tail_ratio"]) for r in records]
    return {
        "fragments": total,
        "truncated": cut,
        "truncated_fraction": cut / total if total else None,
        "tail_ratio_median": statistics.median(ratios) if ratios else None,
    }


def summarise(records: list[dict[str, object]]) -> dict[str, object]:
    total = len(records)
    counts: dict[str, int] = {}
    for record in records:
        kind = str(record["kind"])
        counts[kind] = counts.get(kind, 0) + 1
    return {
        "boundaries": total,
        "counts": counts,
        "mid_voice_fraction": counts.get("mid_voice", 0) / total if total else None,
        "edge_fraction": (
            (counts.get("onset", 0) + counts.get("offset", 0)) / total
            if total
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="a realsi_rollout MANIFEST.json; durations come from it",
    )
    parser.add_argument("--label", default="run")
    parser.add_argument("--output")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    every: list[dict[str, object]] = []
    every_tail: list[dict[str, object]] = []
    for sample in manifest["samples"]:
        path = Path(str(sample["translation_concat"]))
        if not path.exists():
            continue
        audio, rate = sf.read(str(path), dtype="float32")
        if audio.ndim == 2:
            audio = audio[:, -1]
        if int(rate) != SAMPLE_RATE:
            raise ValueError(f"{path} is {rate} Hz")
        durations = list(sample.get("durations", []))
        records = classify_boundaries(audio, durations)
        tails = fragment_tails(audio, durations)
        every.extend(records)
        every_tail.extend(tails)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "direction": sample.get("direction"),
                "fragments": sample.get("fragments"),
                **summarise(records),
                "tails": summarise_tails(tails),
            }
        )
    report = {
        "schema_version": "uniss_streaming_p2st_cut_placement_v1",
        "label": args.label,
        "manifest": str(Path(args.manifest).resolve()),
        "samples": len(rows),
        "overall": summarise(every),
        "overall_tails": summarise_tails(every_tail),
        "median_mid_voice_fraction": (
            statistics.median(
                float(r["mid_voice_fraction"])
                for r in rows
                if r["mid_voice_fraction"] is not None
            )
            if any(r["mid_voice_fraction"] is not None for r in rows)
            else None
        ),
        "rows": rows,
    }
    text = json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    overall = report["overall"]
    tails = report["overall_tails"]
    print(
        f"{args.label}: boundaries={overall['boundaries']} "
        f"mid_voice={overall['mid_voice_fraction']} "
        f"edge={overall['edge_fraction']}"
    )
    print(
        f"{' ' * len(args.label)}  fragments={tails['fragments']} "
        f"truncated={tails['truncated_fraction']} "
        f"tail_ratio_median={tails['tail_ratio_median']}"
    )
    if args.output:
        print(f"-> {args.output}")


if __name__ == "__main__":
    main()
