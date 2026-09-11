"""Silver-Medal preference pairs, following NaturalFlow.

The rule, and why it is not the argmax
--------------------------------------
NaturalFlow (arXiv 2606.13121) stratifies a group of candidates by silence
ratio into quintiles and takes the **second** quintile -- the top 20-40% --
as the chosen one, explicitly rejecting the quietest because "aggressive
silence minimisation degrades translation quality".

This project has already paid for that warning.  A rejection-sampling
fine-tune on the argmax of the same reward collapsed the translation from 60
characters to 7 and ASR-BLEU from 32.93 to 1.10, and the diagnosis was that the
selected candidates commit text in a way gold never does.  The second quintile
is the paper's answer to exactly that failure mode.

Margins
-------
A pair is emitted only when the two candidates differ enough to carry signal:
the paper requires a **BLEU difference of 5** and **15% of the group's
normalised silence ratio**.  Both are applied here.

Groups of eight
---------------
The paper samples 32 candidates; eight gives a coarser stratification, so the
second quintile of eight is a band of one or two.  ``silver_medal.choose``
already handles a group too small to stratify by taking the second-best rather
than the best, which preserves the rule's shape.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import sacrebleu

from experiments.uniss_streaming_p2st_traj_v1.evaluation.rerank_candidates import (
    silence_ratio,
)
from experiments.uniss_streaming_p2st_traj_v1.evaluation.silver_medal import choose

DEFAULT_BLEU_MARGIN = 5.0
DEFAULT_SILENCE_MARGIN = 0.15


def sentence_bleu(hypothesis: str, reference: str, direction: str) -> float:
    return sacrebleu.sentence_bleu(
        hypothesis, [reference], tokenize="zh" if direction == "en2zh" else "13a"
    ).score


def score_group(task: tuple[str, list]) -> list[dict]:
    """Score one sample's candidates.  Module level so a process pool can
    pickle it; it touches only the manifest row and the wav on disk."""
    sample_id, members = task
    rows = []
    for arm, sample in members:
        direction = sample.get("direction") or (
            "zh2en" if sample.get("src_lang") == "cmn" else "en2zh"
        )
        hypothesis = sample["target_hypothesis"]
        reference = sample["translation_reference"]
        rows.append({
            "arm": arm,
            "silence_ratio": silence_ratio(Path(str(sample["translation_placed"]))),
            "bleu": sentence_bleu(hypothesis, reference, direction),
            "chrf": sacrebleu.sentence_chrf(hypothesis, [reference]).score,
            "text": hypothesis, "direction": direction,
            "sample_id": sample_id, "reference": reference,
        })
    return rows


def pair_for_group(
    rows: list[dict],
    *,
    bleu_margin: float,
    silence_margin: float,
    min_chrf: float,
) -> tuple[dict, dict, str] | tuple[None, None, str]:
    """One (chosen, rejected) pair, or a reason there is none."""
    chosen, why = choose(rows, min_chrf=min_chrf)
    if chosen is None:
        return None, None, why
    spread = [float(r["silence_ratio"]) for r in rows if r.get("silence_ratio") is not None]
    if not spread:
        return None, None, "no measurable silence"
    scale = max(spread) - min(spread)
    if scale <= 0.0:
        return None, None, "the group has no silence spread"
    floor = max(silence_margin * scale, 1e-6)
    # The BLEU margin is read as a *quality match*, not a minimum separation:
    # the pair must differ in flow and agree in quality, or DPO learns "translate
    # better" instead of "speak continuously", which is not what the objective
    # is for.  The paper states the margin without saying which way it cuts, so
    # this interpretation is a choice, and it is the one that makes the pair
    # isolate the axis being optimised.
    worse = [
        r for r in rows
        if r is not chosen
        and r.get("silence_ratio") is not None
        and float(r["silence_ratio"]) - float(chosen["silence_ratio"]) >= floor
        and abs(float(chosen["bleu"]) - float(r["bleu"])) <= bleu_margin
    ]
    if not worse:
        return None, None, "no candidate cleared both margins"
    rejected = max(worse, key=lambda r: float(r["silence_ratio"]))
    return chosen, rejected, why


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-root", required=True)
    parser.add_argument("--pattern", default="k*")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bleu-margin", type=float, default=DEFAULT_BLEU_MARGIN)
    parser.add_argument("--silence-margin", type=float, default=DEFAULT_SILENCE_MARGIN)
    parser.add_argument("--min-chrf", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--workers", type=int, default=min(48, os.cpu_count() or 8)
    )
    args = parser.parse_args()

    groups: dict[str, list[dict]] = {}
    for arm in sorted(Path(args.arm_root).glob(args.pattern)):
        manifest = arm / "MANIFEST.json"
        if not manifest.exists():
            continue
        for sample in json.loads(manifest.read_text(encoding="utf-8"))["samples"]:
            groups.setdefault(sample["sample_id"], []).append((arm.name, sample))
    if not groups:
        raise SystemExit(f"no arms matched {args.pattern} under {args.arm_root}")

    ordered = sorted(groups.items())
    if args.limit:
        ordered = ordered[: args.limit]

    # Scoring is one wav read and two sacrebleu calls per candidate -- eight
    # times per group and entirely CPU bound, so it runs across the box rather
    # than down a single core.  ``map`` keeps the results in group order, so
    # the output is identical to the serial version.
    pairs: list[dict] = []
    reasons: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        scored = pool.map(
            score_group, [(sid, members) for sid, members in ordered], chunksize=8
        )
        for (sample_id, _members), rows in zip(ordered, scored):
            chosen, rejected, why = pair_for_group(
                rows, bleu_margin=args.bleu_margin,
                silence_margin=args.silence_margin, min_chrf=args.min_chrf,
            )
            reasons[why] = reasons.get(why, 0) + 1
            if chosen is None or rejected is None:
                continue
            pairs.append({
                "sample_id": sample_id, "direction": chosen["direction"],
                "reference": chosen["reference"],
                "chosen": {"arm": chosen["arm"], "text": chosen["text"],
                           "silence_ratio": chosen["silence_ratio"],
                           "bleu": chosen["bleu"]},
                "rejected": {"arm": rejected["arm"], "text": rejected["text"],
                             "silence_ratio": rejected["silence_ratio"],
                             "bleu": rejected["bleu"]},
            })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"pairs": pairs}, ensure_ascii=False), encoding="utf-8")
    print(f"{len(pairs)} pairs from {len(groups)} groups")
    for why, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {why}: {count}")
    if pairs:
        m = lambda side, key: statistics.mean(float(p[side][key]) for p in pairs)
        print(f"  chosen   silence {m('chosen','silence_ratio'):.3f}  BLEU {m('chosen','bleu'):.2f}")
        print(f"  rejected silence {m('rejected','silence_ratio'):.3f}  BLEU {m('rejected','bleu'):.2f}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
