"""Score sampled candidates with HPO's reward shape, and pick one.

What this is, and is not
------------------------
HPO (arXiv 2604.21045, ACL 2026) samples 16 trajectories per segment at
top-p 0.999 / top-k 10000 and uses them as *RL rollouts*; at inference it runs
beam search with width 4 and does not sample at all.  So this is not HPO.  It
is HPO's reward shape applied as inference-time selection, on candidates
generated the way NaturalFlow generates its preference pairs -- which is the
cheapest way to find out what that reward shape is worth here before spending
sixty GPU-hours on the RL run it belongs to.

The reward, following the paper
-------------------------------
Quality and latency are scored per candidate, **separately standardised across
the group**, and combined as ``q - lambda * l`` with lambda 0.5.  A candidate
whose quality falls below a threshold has its latency score penalised to the
group maximum, so latency can never be bought with a bad translation -- that
gate is the paper's defence against reward hacking and the reason it is called
hierarchical.

Two additions this task needs
-----------------------------
*   A **speech term**.  HPO's latency is StreamLAAL, which reads when text was
    emitted and says nothing about whether the speech is continuous.  Our
    defect is silence, so silence enters the reward directly.
*   Quality **two ways**.  ``--quality reference`` scores against the reference
    and gives the ceiling this reward shape could reach; ``--quality selfscore``
    uses only quantities available at inference and is what a deployment could
    actually use.  Reporting the ceiling first is the point: if it is small,
    the RL run cannot be worth it.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
FRAME = SAMPLE_RATE * 15 // 1000


def silence_ratio(path: Path) -> float:
    """Share of the placed span that is a hole of 300 ms or more."""
    audio, rate = sf.read(str(path), dtype="float32")
    if audio.ndim == 2:
        audio = audio[:, -1]
    count = len(audio) // FRAME
    if count < 8:
        return 0.0
    block = audio[: count * FRAME].reshape(count, FRAME).astype(np.float64)
    rms = np.sqrt((block**2).mean(axis=1))
    floor = max(1e-4, 0.02 * float(np.percentile(rms, 99)))
    voiced = rms > floor
    index = np.flatnonzero(voiced)
    if len(index) == 0:
        return 0.0
    voiced = voiced[index[0] : index[-1] + 1]
    total = 0
    run = 0
    for flag in voiced:
        if not flag:
            run += 1
        elif run:
            if run * 15 >= 300:
                total += run * 15
            run = 0
    if run and run * 15 >= 300:
        total += run * 15
    return total / max(1.0, len(voiced) * 15.0)


def standardise(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [0.0] * len(values)
    mean = statistics.mean(values)
    spread = statistics.pstdev(values)
    if spread <= 0:
        return [0.0] * len(values)
    return [(v - mean) / spread for v in values]


def choose(
    rows: list[dict],
    *,
    lam: float,
    quality_quantile: float,
) -> tuple[int, list[float]]:
    """HPO's hierarchical combination over one group of candidates.

    ``quality_quantile`` sets the gate: a candidate below that quantile of the
    group's own quality has its latency term penalised to the group maximum,
    which is the paper's q_thres expressed without MetricX's absolute scale.
    """
    quality = [float(r["quality"]) for r in rows]
    latency = [float(r["latency"]) for r in rows]
    gate = float(np.quantile(quality, quality_quantile)) if len(quality) > 1 else -1e30
    worst = max(latency) if latency else 0.0
    gated = [worst if q < gate else l for q, l in zip(quality, latency)]
    zq = standardise(quality)
    zl = standardise(gated)
    scores = [q - float(lam) * l for q, l in zip(zq, zl)]
    return int(max(range(len(scores)), key=lambda i: scores[i])), scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-root", required=True, help="directory holding cand_* arms")
    parser.add_argument("--pattern", default="cand_*")
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--quality-quantile", type=float, default=0.25)
    parser.add_argument("--quality", default="reference",
                        choices=("reference", "selfscore"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    arms = sorted(Path(args.arm_root).glob(args.pattern))
    groups: dict[str, list[dict]] = {}
    for arm in arms:
        man = arm / "MANIFEST.json"
        if not man.exists():
            continue
        for sample in json.loads(man.read_text(encoding="utf-8"))["samples"]:
            groups.setdefault(sample["sample_id"], []).append(
                {"arm": arm.name, "sample": sample}
            )
    if not groups:
        raise SystemExit(f"no candidate arms matched {args.pattern} under {args.arm_root}")

    import sacrebleu

    picks: list[dict] = []
    for sample_id, members in sorted(groups.items()):
        rows = []
        for m in members:
            s = m["sample"]
            direction = s.get("direction") or (
                "zh2en" if s.get("src_lang") == "cmn" else "en2zh"
            )
            hypothesis = s["target_hypothesis"]
            if args.quality == "reference":
                quality = sacrebleu.sentence_bleu(
                    hypothesis, [s["translation_reference"]],
                    tokenize="zh" if direction == "en2zh" else "13a",
                ).score
            else:
                # reference-free stand-in: how much the model chose to say per
                # read step, which correlates with a complete translation and
                # needs nothing a deployment would not have
                quality = len(hypothesis) / max(1, int(s["read_steps"]))
            rows.append({
                "arm": m["arm"], "quality": quality,
                "latency": silence_ratio(Path(str(s["translation_placed"]))),
                "sample": s,
            })
        best, scores = choose(rows, lam=args.lam, quality_quantile=args.quality_quantile)
        picks.append({
            "sample_id": sample_id,
            "chosen_arm": rows[best]["arm"],
            "chosen_quality": rows[best]["quality"],
            "chosen_latency": rows[best]["latency"],
            "group_quality_mean": statistics.mean(r["quality"] for r in rows),
            "group_latency_mean": statistics.mean(r["latency"] for r in rows),
            "group_latency_min": min(r["latency"] for r in rows),
            "group_size": len(rows),
            "scores": scores,
            "chosen_audio": str(rows[best]["sample"]["translation_placed"]),
        })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"picks": picks}, indent=1), encoding="utf-8")
    mean = lambda k: statistics.mean(float(p[k]) for p in picks)
    print(f"{len(picks)} utterances, groups of {picks[0]['group_size']}, "
          f"quality={args.quality}, lambda={args.lam}")
    print(f"  silence  group mean {mean('group_latency_mean'):.3f}  "
          f"chosen {mean('chosen_latency'):.3f}  "
          f"group best possible {mean('group_latency_min'):.3f}")
    print(f"  quality  group mean {mean('group_quality_mean'):.2f}  "
          f"chosen {mean('chosen_quality'):.2f}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
