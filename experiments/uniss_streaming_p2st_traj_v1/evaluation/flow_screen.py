"""Paired flow comparison of several checkpoints against one baseline.

Why this exists rather than held-out accuracy
---------------------------------------------
The DPO run's held-out pairwise accuracy sat at 0.489 -- chance -- while the
very same checkpoint cut RealSI silence by 1.52 points with a 95% confidence
interval that excludes zero.  The two measure different things: the pairwise
score ranks two *sampled* streams, while what training changed is the policy's
own decoding.  So configurations have to be screened by decoding, and the
screen reports the axes the pair statistics predicted would move -- fragments
and silence -- alongside the ones that must not move, onset and character
count.

Every arm decodes the same selection, so each comparison is paired: the
difference is taken per sample and only then averaged, which is what makes a
confidence interval this narrow meaningful on a few hundred utterances.
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from experiments.uniss_streaming_p2st_traj_v1.evaluation.rerank_candidates import (
    silence_ratio,
)


def onset_ms(record: dict) -> float:
    delays = record.get("delays") or record.get("starts_ms") or [0.0]
    return float(delays[0]) if delays else 0.0


def paired(values_a: dict, values_b: dict, ids: list[str]) -> tuple[float, float]:
    """Mean paired difference and its 95% half-width."""
    diffs = [values_a[i] - values_b[i] for i in ids]
    mean = statistics.mean(diffs)
    if len(diffs) < 2:
        return mean, 0.0
    half = 1.96 * statistics.stdev(diffs) / len(diffs) ** 0.5
    return mean, half


def _silence(task):
    arm, sample_id, path = task
    return arm, sample_id, silence_ratio(Path(path))


def load(root: Path) -> dict[str, dict[str, dict]]:
    arms = {}
    for manifest in sorted(root.glob("*/MANIFEST.json")):
        rows = json.loads(manifest.read_text(encoding="utf-8"))["samples"]
        arms[manifest.parent.name] = {r["sample_id"]: r for r in rows}
    return arms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--baseline", default="base")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output")
    args = parser.parse_args()

    arms = load(Path(args.rollout_root))
    if args.baseline not in arms:
        raise SystemExit(f"no baseline arm {args.baseline!r} under {args.rollout_root}")
    ids = sorted(set.intersection(*(set(a) for a in arms.values())))
    if not ids:
        raise SystemExit("the arms share no samples")

    tasks = [
        (arm, i, str(arms[arm][i]["translation_placed"])) for arm in arms for i in ids
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        scored = list(pool.map(_silence, tasks, chunksize=16))
    silence = {arm: {} for arm in arms}
    for arm, sample_id, value in scored:
        silence[arm][sample_id] = value

    base = args.baseline
    report = {"samples": len(ids), "baseline": base, "arms": {}}
    print(f"{len(ids)} paired samples, against {base}\n")
    print(f"{'arm':>18} {'silence':>18} {'onset ms':>12} {'frags':>14} {'chars':>8}")
    for arm in sorted(arms):
        frags = {i: float(arms[arm][i]["fragments"]) for i in ids}
        chars = {i: float(len(arms[arm][i]["target_hypothesis"])) for i in ids}
        onsets = {i: onset_ms(arms[arm][i]) for i in ids}
        if arm == base:
            print(
                f"{arm:>18} {statistics.mean(silence[arm].values()):>18.4f} "
                f"{statistics.mean(onsets.values()):>12.0f} "
                f"{statistics.mean(frags.values()):>14.2f} "
                f"{statistics.mean(chars.values()):>8.1f}"
            )
            continue
        ds, hs = paired(silence[arm], silence[base], ids)
        do, _ = paired(onsets, {i: onset_ms(arms[base][i]) for i in ids}, ids)
        df, hf = paired(frags, {i: float(arms[base][i]["fragments"]) for i in ids}, ids)
        dc, _ = paired(
            chars, {i: float(len(arms[base][i]["target_hypothesis"])) for i in ids}, ids
        )
        report["arms"][arm] = {
            "silence_delta": ds, "silence_ci": hs, "onset_delta": do,
            "fragments_delta": df, "fragments_ci": hf, "chars_delta": dc,
        }
        print(
            f"{arm:>18} {ds:>+11.4f} +-{hs:<5.4f} {do:>+12.0f} "
            f"{df:>+8.2f} +-{hf:<4.2f} {dc:>+8.1f}"
        )
    print("\nsilence intervals that exclude zero are the ones that moved.")
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"-> {args.output}")


if __name__ == "__main__":
    main()
