"""Sample a fixed dev subset for training-time loss curves.

The full CVSS-T dev split is 4,843 pairs, i.e. 9,686 utterances across both
directions. Scoring all of it every few hundred steps would cost more than
the signal is worth, so the curve runs on a fixed sample.

Fixed matters more than large: a curve is only readable if every point is
measured on the same material, so the subset is drawn once with a recorded
seed and reused, never resampled per evaluation.

The draw is stratified by source duration. A uniform sample over-weights the
short utterances that dominate CVSS-T, and dev loss on mostly-short audio
would miss exactly the long-form behaviour that streaming work is about.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_rows(manifest: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def duration_of(row: dict) -> float:
    for key in ("source_zh_duration_seconds", "duration_seconds"):
        if key in row:
            return float(row[key])
    return 0.0


def stratified_sample(rows: list[dict], size: int, seed: int, bins: int) -> list[dict]:
    if size >= len(rows):
        return list(rows)
    ordered = sorted(rows, key=duration_of)
    rng = random.Random(seed)
    edges = [round(i * len(ordered) / bins) for i in range(bins + 1)]
    picked: list[dict] = []
    for i in range(bins):
        stratum = ordered[edges[i] : edges[i + 1]]
        if not stratum:
            continue
        # Spread the remainder over the first few strata rather than letting
        # integer division silently drop samples.
        take = size // bins + (1 if i < size % bins else 0)
        picked.extend(rng.sample(stratum, min(take, len(stratum))))
    picked.sort(key=lambda r: str(r.get("id")))
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = load_rows(Path(args.manifest))
    picked = stratified_sample(rows, args.size, args.seed, args.bins)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in picked:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    durations = [duration_of(r) for r in picked]
    full = [duration_of(r) for r in rows]
    print(
        f"{len(picked)} of {len(rows)} pairs, seed {args.seed}\n"
        f"  subset duration  mean {sum(durations)/len(durations):.2f}s"
        f"  min {min(durations):.2f}s  max {max(durations):.2f}s\n"
        f"  full   duration  mean {sum(full)/len(full):.2f}s"
        f"  min {min(full):.2f}s  max {max(full):.2f}s"
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
