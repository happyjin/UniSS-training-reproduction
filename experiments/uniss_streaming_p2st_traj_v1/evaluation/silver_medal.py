"""Pick among sampled rollouts the way NaturalFlow picks among candidates.

The idea, from NaturalFlow (arXiv 2606.13121)
---------------------------------------------
Their fix for disruptive pauses is *not* to make the model speak slower.  It
is to prefer a translation that naturally takes longer to say -- a paraphrase
with more syllables -- so the timeline fills with speech rather than with
silence.  They build a preference set by sampling 32 candidates per utterance
at temperature 1.0, measuring each one's silence ratio, and then applying what
they call a Silver-Medal rule.

The rule is the part that matters, and it is counter-intuitive:
**they do not take the best candidate.**  Candidates are stratified into five
quintiles by silence ratio and the *second* quintile is chosen, with the first
(most extreme) explicitly rejected alongside quintiles 3-5.  Their ablations
say why -- taking the lowest-silence candidate collapses the model to BLEU
1.50, and removing the penalty on the extreme band gives "non-stop, extremely
fast speech that becomes unintelligible" at BLEU 0.92.

That is the guard our own beam-search attempt lacked.  Searching for the best
sequence made the translation loop; bounding the search away from the extreme
is what keeps it honest.

What this module does
---------------------
Applies the rule at the utterance level over complete rollouts of the same
audio decoded with different text-stage sampling seeds.  This is a pure
inference-time test of whether the *direction* transfers to our cascade before
any training is spent on it; NaturalFlow itself applies the preference through
DPO, which this project has no loop for.

Quality is guarded by chrF against the greedy rollout's own text rather than
by BLEU against a reference, so the rule can be applied to audio with no
reference at all, and so a candidate is rejected for drifting away from what
the model would otherwise have said.
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
# NaturalFlow's VAD ignores silences under 100 ms.
MIN_SILENCE_MS = 100
QUINTILES = 5
# Their band: the second quintile, counting from lowest silence ratio.
CHOSEN_QUINTILE = 1


def silence_ratio(audio: np.ndarray) -> float | None:
    """``1 - voiced/span`` over the output span, silences under 100 ms ignored."""
    audio = np.asarray(audio, dtype=np.float32)
    frames = len(audio) // FRAME
    if frames < 2:
        return None
    block = audio[: frames * FRAME].reshape(frames, FRAME)
    rms = np.sqrt(np.mean(block.astype(np.float64) ** 2, axis=1))
    floor = max(1e-3, FLOOR_FRACTION * float(np.percentile(rms, 99)))
    voiced = rms > floor
    if not voiced.any():
        return None
    first = int(np.argmax(voiced))
    last = int(len(voiced) - 1 - np.argmax(voiced[::-1]))
    span = voiced[first : last + 1]
    runs: list[int] = []
    current = 0
    for value in span:
        if not value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    silent = sum(r for r in runs if r * FRAME_MS >= MIN_SILENCE_MS)
    return silent / len(span)


def choose(
    candidates: list[dict], *, min_chrf: float
) -> tuple[dict | None, str]:
    """The Silver-Medal pick, or a reason it could not be made.

    Candidates carry ``silence_ratio`` and ``chrf``.  Those below ``min_chrf``
    are dropped first: a candidate that has drifted from what the model would
    otherwise say is not a paraphrase, it is a different translation.
    """
    usable = [c for c in candidates if c.get("silence_ratio") is not None]
    if not usable:
        return None, "no candidate produced measurable audio"
    passing = [c for c in usable if float(c.get("chrf", 0.0)) >= min_chrf]
    if not passing:
        return None, f"no candidate cleared chrF {min_chrf}"
    ordered = sorted(passing, key=lambda c: float(c["silence_ratio"]))
    if len(ordered) < QUINTILES:
        # Too few to stratify.  Take the second best rather than the best, so
        # the rule's shape -- never the extreme -- is preserved.
        return (ordered[1] if len(ordered) > 1 else ordered[0]), "too few to stratify"
    size = len(ordered) / QUINTILES
    lo = int(round(CHOSEN_QUINTILE * size))
    hi = max(lo + 1, int(round((CHOSEN_QUINTILE + 1) * size)))
    band = ordered[lo:hi]
    return band[len(band) // 2], "silver medal"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--greedy-arm", required=True, help="the reference rollout")
    parser.add_argument("--candidate-arm", action="append", required=True)
    parser.add_argument("--min-chrf", type=float, default=60.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import sacrebleu

    root = Path(args.rollout_root)

    def load(arm: str) -> dict[str, dict]:
        path = root / arm / "MANIFEST.json"
        if not path.exists():
            return {}
        return {
            s["sample_id"]: s
            for s in json.loads(path.read_text(encoding="utf-8"))["samples"]
        }

    greedy = load(args.greedy_arm)
    pools = {arm: load(arm) for arm in args.candidate_arm}

    rows: list[dict[str, object]] = []
    for sample_id, reference in greedy.items():
        base_text = reference.get("target_hypothesis") or ""
        candidates: list[dict] = []
        for arm, pool in [(args.greedy_arm, greedy)] + list(pools.items()):
            sample = pool.get(sample_id)
            if not sample:
                continue
            wav = Path(str(sample["translation_placed"]))
            if not wav.exists():
                continue
            audio, rate = sf.read(str(wav), dtype="float32")
            if audio.ndim == 2:
                audio = audio[:, -1]
            if int(rate) != SAMPLE_RATE:
                continue
            text = sample.get("target_hypothesis") or ""
            chrf = (
                100.0
                if arm == args.greedy_arm
                else sacrebleu.sentence_chrf(text, [base_text]).score
            )
            candidates.append(
                {
                    "arm": arm,
                    "silence_ratio": silence_ratio(audio),
                    "chrf": chrf,
                    "chars": len(text),
                    "semantic_tokens": sample.get("semantic_tokens"),
                    "fragments": sample.get("fragments"),
                }
            )
        pick, reason = choose(candidates, min_chrf=args.min_chrf)
        rows.append(
            {
                "sample_id": sample_id,
                "direction": reference.get("direction"),
                "candidates": candidates,
                "chosen_arm": pick["arm"] if pick else None,
                "chosen_silence_ratio": pick["silence_ratio"] if pick else None,
                "greedy_silence_ratio": next(
                    (c["silence_ratio"] for c in candidates if c["arm"] == args.greedy_arm),
                    None,
                ),
                "reason": reason,
            }
        )

    def mean(key: str) -> float | None:
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        return statistics.mean(values) if values else None

    best = [
        min(
            (c["silence_ratio"] for c in r["candidates"] if c["silence_ratio"] is not None),
            default=None,
        )
        for r in rows
    ]
    report = {
        "schema_version": "uniss_streaming_p2st_silver_medal_v1",
        "greedy_arm": args.greedy_arm,
        "candidate_arms": args.candidate_arm,
        "min_chrf": args.min_chrf,
        "samples": len(rows),
        "greedy_silence_ratio": mean("greedy_silence_ratio"),
        "chosen_silence_ratio": mean("chosen_silence_ratio"),
        "oracle_lowest_silence_ratio": (
            statistics.mean([b for b in best if b is not None]) if any(best) else None
        ),
        "rows": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"samples={report['samples']}  greedy SR={report['greedy_silence_ratio']:.3f}"
        f"  silver-medal SR={report['chosen_silence_ratio']:.3f}"
        f"  (oracle lowest {report['oracle_lowest_silence_ratio']:.3f})"
    )
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
