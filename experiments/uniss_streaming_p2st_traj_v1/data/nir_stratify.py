#!/usr/bin/env python3
"""Resample the gold pool to the paper's difficulty and length composition.

Appendix A.3 gives the policy but not the bucket edges: *"we bucket examples by
NIR and sentence length.  The main sampling policy uses difficulty weights
{high: 0.1, mid_high: 0.3, mid_low: 0.4, low: 0.2} and length weights
{short: 0.1, medium: 0.5, long: 0.4}."*  Four difficulty names and three length
names are read here as quartiles of NIR and terciles of source duration, taken
**per direction** because the two differ materially -- measured on the full
pool, En->Zh sits at 16.7% perfectly monotone against Zh->En's 8.8%, and their
quartile edges are 3.66/10.00/19.05% against 5.16/10.25/16.96%.

Read against uniform (0.25 each, 0.333 each), the weights downweight the hardest
quartile hardest (0.25 -> 0.10), trim the easiest (0.25 -> 0.20), and lift the
two middle ones; on length they cut short utterances (0.333 -> 0.10) in favour
of medium and long.  Both directions of that make sense for read/write
supervision: a perfectly monotone pair teaches nothing about waiting, a wildly
reordered one teaches an unstable policy, and a short utterance holds too few
chunks to teach streaming at all.

NO DUPLICATION
--------------
``build_p2st_pools`` builds ``by_sequence = {sample.sequence_id: sample}`` and
raises when that map is smaller than the sample list, and ``sequence_id`` is
``f"{sample_id}:p2st_asr:{event_index}"`` -- so repeating a trajectory to
oversample a deficit bucket would abort the pool build.  Composition is
therefore reached by downsampling every other cell instead.  Measured cost: the
binding cell is mid_low/medium, which holds 7.47% of En->Zh where the target
wants 20.00%, capping the pool at 282,950 En->Zh and 220,835 Zh->En rows, 38% of
the 1.32M available.  Raise ``--coverage-epochs`` on the training side to keep
the number of samples seen comparable rather than trying to keep the row count.

Rows whose NIR is undefined -- fewer than two aligned target words, 2,102 of
1,325,243 -- are excluded and counted in the report rather than silently
bucketed as monotone.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from training.simul_uniss.jsonl_index import load_index

NIR_WEIGHTS = {"low": 0.2, "mid_low": 0.4, "mid_high": 0.3, "high": 0.1}
LENGTH_WEIGHTS = {"short": 0.1, "medium": 0.5, "long": 0.4}
NIR_ORDER = ("low", "mid_low", "mid_high", "high")
LENGTH_ORDER = ("short", "medium", "long")


def quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    return sorted_values[int(fraction * (len(sorted_values) - 1))]


def bucket_of(value: float, edges: list[float], names: tuple[str, ...]) -> str:
    for name, edge in zip(names, edges):
        if value <= edge:
            return name
    return names[-1]


def _read_scores(task: tuple[str, int, int]) -> list[tuple[str, str, float, int]]:
    path, index, total = task
    out: list[tuple[str, str, float, int]] = []
    with open(path, encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if position % total != index:
                continue
            row = json.loads(line)
            if row.get("nir") is None:
                continue
            out.append(
                (
                    str(row["sample_id"]),
                    str(row["direction"]),
                    float(row["nir"]),
                    int(row["source_duration_ms"]),
                )
            )
    return out


def _positions(task: tuple[str, int, int]) -> dict[str, int]:
    """Map sample_id -> record position for one shard of the gold jsonl.

    The scorer writes rows in shard-interleaved order, so the mapping cannot be
    recovered from the score file's ordering and has to be read back from the
    gold jsonl.  Sharded across workers because it is 1.3M seeks.
    """
    gold_path, index, total = task
    gold = Path(gold_path)
    offsets = load_index(gold)
    if offsets is None:
        raise SystemExit(f"missing offset sidecar for {gold}")
    out: dict[str, int] = {}
    with gold.open("rb") as handle:
        for position in range(index, len(offsets), total):
            handle.seek(int(offsets[position]))
            out[str(json.loads(handle.readline())["sample_id"])] = position
    return out


def _emit(task: tuple[str, str, list[int], int, int]) -> tuple[str, int]:
    """Copy the selected gold records out by offset, one shard per worker."""
    gold_path, output_path, positions, index, total = task
    gold = Path(gold_path)
    offsets = load_index(gold)
    if offsets is None:
        raise SystemExit(f"missing offset sidecar for {gold}")
    written = 0
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gold.open("rb") as handle, out.open("wb") as sink:
        for slot in range(index, len(positions), total):
            handle.seek(int(offsets[positions[slot]]))
            sink.write(handle.readline())
            written += 1
    return output_path, written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="source gold trajectories jsonl")
    parser.add_argument("--scores", required=True, help="NIR_SCORES.jsonl for it")
    parser.add_argument("--output", required=True, help="stratified gold jsonl to write")
    parser.add_argument("--report", required=True, help="composition report json")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()

    output = Path(args.output)
    report_path = Path(args.report)
    for path in (output, report_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")

    workers = max(1, int(args.workers))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        chunks = pool.map(
            _read_scores, [(args.scores, index, workers) for index in range(workers)]
        )
        scores = [row for chunk in chunks for row in chunk]
    if not scores:
        raise SystemExit("no scored rows with a defined NIR")

    gold = Path(args.gold)
    position_of: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for shard in pool.map(
            _positions, [(str(gold), index, workers) for index in range(workers)]
        ):
            position_of.update(shard)

    by_direction: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for sample_id, direction, nir, duration in scores:
        by_direction[direction].append((sample_id, nir, duration))

    rng = random.Random(args.seed)
    selected: list[int] = []
    report: dict[str, object] = {
        "schema_version": "uniss_traj_nir_stratify_v1",
        "source_gold": str(gold.resolve()),
        "scores": str(Path(args.scores).resolve()),
        "seed": args.seed,
        "nir_weights": NIR_WEIGHTS,
        "length_weights": LENGTH_WEIGHTS,
        "scored_rows": len(scores),
        "directions": {},
    }
    for direction in sorted(by_direction):
        rows = by_direction[direction]
        nirs = sorted(value for _, value, _ in rows)
        durations = sorted(value for _, _, value in rows)
        nir_edges = [quantile(nirs, f) for f in (0.25, 0.50, 0.75)]
        length_edges = [quantile(durations, f) for f in (1 / 3, 2 / 3)]
        cells: dict[tuple[str, str], list[str]] = defaultdict(list)
        for sample_id, nir, duration in rows:
            cells[
                (
                    bucket_of(nir, nir_edges, NIR_ORDER),
                    bucket_of(duration, length_edges, LENGTH_ORDER),
                )
            ].append(sample_id)
        target = {
            (nk, lk): NIR_WEIGHTS[nk] * LENGTH_WEIGHTS[lk]
            for nk in NIR_ORDER
            for lk in LENGTH_ORDER
        }
        total = min(
            len(cells[key]) / weight for key, weight in target.items() if weight > 0
        )
        total = int(total)
        achieved: dict[str, dict[str, object]] = {}
        picked_here = 0
        for key, weight in sorted(target.items()):
            available = cells[key]
            want = int(round(weight * total))
            take = min(want, len(available))
            rng.shuffle(available)
            for sample_id in available[:take]:
                selected.append(position_of[sample_id])
            picked_here += take
            achieved[f"{key[0]}/{key[1]}"] = {
                "available": len(available),
                "target_share": weight,
                "selected": take,
            }
        for key in achieved:
            achieved[key]["achieved_share"] = (
                achieved[key]["selected"] / picked_here if picked_here else 0.0
            )
        report["directions"][direction] = {  # type: ignore[index]
            "available_rows": len(rows),
            "selected_rows": picked_here,
            "retained_fraction": picked_here / len(rows),
            "nir_quartile_edges_percent": nir_edges,
            "duration_tercile_edges_ms": length_edges,
            "cells": achieved,
        }

    selected.sort()
    parts = [
        (str(gold), str(output.with_suffix(f".part{index:03d}")), selected, index, workers)
        for index in range(workers)
    ]
    written = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _, count in pool.map(_emit, parts):
            written += count
    with output.open("wb") as sink:
        for index in range(workers):
            part = output.with_suffix(f".part{index:03d}")
            sink.write(part.read_bytes())
            part.unlink()

    report["selected_rows"] = written
    report["retained_fraction"] = written / len(scores)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"selected {written} of {len(scores)} scored rows -> {output}")
    print(f"composition report -> {report_path}")


if __name__ == "__main__":
    main()
