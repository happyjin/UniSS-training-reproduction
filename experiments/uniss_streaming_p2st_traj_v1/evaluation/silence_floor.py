"""How much of the remaining silence is actually compressible.

The deployed system now sits at 10.85% silence, down from 19.81%, and the
obvious next question is how much further there is to go.  Reporting a
reduction without a floor invites the assumption that zero is the target, and
zero is wrong: continuous speech contains pauses of its own, and a vocoder
synthesising a whole utterance at once still produces them.

So this measures the same statistic on three things that bracket the answer:

* **source** -- the human recordings the benchmark is built from, which is what
  natural speech scores on this metric;
* **offline** -- our own vocoder given the whole text at once, with no
  streaming constraint at all, which is the floor this synthesiser can reach;
* **the streaming arms**, on concatenated audio (silence inside the speech)
  and on placed audio (that plus the gaps the schedule leaves).

The gap between an arm's placed and concatenated figures is the part the
policy and the renderer can still attack.  The concatenated figure itself is
bounded below by the offline one.
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


def _measure(task):
    key, sample_id, path = task
    try:
        return key, sample_id, silence_ratio(Path(path))
    except Exception:
        return key, sample_id, None


def collect(rollout_root: Path, arms: list[str], selection: Path | None) -> list[tuple]:
    """Every (series, sample_id, wav) this report needs."""
    tasks: list[tuple] = []
    if selection is not None:
        rows = json.loads(selection.read_text(encoding="utf-8"))
        rows = rows["samples"] if isinstance(rows, dict) else rows
        for row in rows:
            tasks.append(("source", row["sample_id"], row["audio_path"]))
    for arm in arms:
        manifest = rollout_root / arm / "MANIFEST.json"
        if not manifest.exists():
            continue
        for row in json.loads(manifest.read_text(encoding="utf-8"))["samples"]:
            if row.get("translation_concat"):
                tasks.append((f"{arm} concat", row["sample_id"], row["translation_concat"]))
            if row.get("translation_placed"):
                tasks.append((f"{arm} placed", row["sample_id"], row["translation_placed"]))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--selection", default="")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--output")
    args = parser.parse_args()

    tasks = collect(
        Path(args.rollout_root), args.arm,
        Path(args.selection) if args.selection else None,
    )
    if not tasks:
        raise SystemExit("nothing to measure")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        scored = list(pool.map(_measure, tasks, chunksize=16))

    series: dict[str, dict[str, float]] = {}
    for key, sample_id, value in scored:
        if value is not None:
            series.setdefault(key, {})[sample_id] = value

    print(f"{'series':>22} {'n':>5} {'mean':>8} {'median':>8} {'p90':>8}")
    report = {}
    for key in sorted(series, key=lambda k: (k != "source", k)):
        values = sorted(series[key].values())
        report[key] = {
            "n": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p90": values[int(len(values) * 0.9)],
        }
        r = report[key]
        print(f"{key:>22} {r['n']:>5} {r['mean']:>8.4f} {r['median']:>8.4f} {r['p90']:>8.4f}")

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
