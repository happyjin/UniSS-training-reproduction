#!/usr/bin/env python3
"""Select the best *available* v3 checkpoint from formal validation records.

The training job validates every 250 iterations but saves every 500 iterations.
Selection therefore ranks only iterations that physically exist below the
checkpoint root.  A rank-sum is used because the six losses have different
units and scales; lower is better for every component.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "uniss_phase3_prefix_streaming_v3_checkpoint_selection_v1"
SELECTION_METRICS = (
    "loss/prefix_ce",
    "loss/semantic_ce",
    "loss/commit_suffix_ce",
    "loss/teacher_kl",
    "loss/adjacent_consistency",
    "loss/action_ce",
)
VALIDATION_RE = re.compile(r"validation loss at iteration\s+(\d+)(?! on).*?\|\s*$", re.M)


def parse_validation_rows(log_path: Path) -> list[dict[str, float | int]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, float | int]] = []
    for match in VALIDATION_RE.finditer(text):
        line = match.group(0)
        row: dict[str, float | int] = {"iteration": int(match.group(1))}
        for metric in SELECTION_METRICS:
            value = re.search(re.escape(metric) + r" value: ([0-9.Ee+-]+)", line)
            if value is None:
                raise ValueError(f"validation line is missing {metric}: {line[:200]}")
            row[metric] = float(value.group(1))
        rows.append(row)
    if not rows:
        raise ValueError(f"no validation rows found in {log_path}")
    return rows


def saved_iterations(checkpoint_root: Path) -> set[int]:
    values: set[int] = set()
    for path in checkpoint_root.glob("iter_*"):
        if path.is_dir() and (path / ".metadata").is_file():
            try:
                values.add(int(path.name.removeprefix("iter_")))
            except ValueError:
                continue
    if not values:
        raise ValueError(f"no distributed checkpoints found below {checkpoint_root}")
    return values


def rank_rows(
    rows: Iterable[dict[str, float | int]], available: set[int]
) -> list[dict[str, object]]:
    candidates = [dict(row) for row in rows if int(row["iteration"]) in available]
    by_iteration = {int(row["iteration"]): row for row in candidates}
    if len(by_iteration) != len(candidates):
        raise ValueError("duplicate formal validation iterations found")
    candidates = list(by_iteration.values())
    if not candidates:
        raise ValueError("no validation row matches a saved checkpoint")
    scores = {int(row["iteration"]): 0 for row in candidates}
    component_ranks: dict[int, dict[str, int]] = {
        int(row["iteration"]): {} for row in candidates
    }
    for metric in SELECTION_METRICS:
        ordered = sorted(candidates, key=lambda row: (float(row[metric]), int(row["iteration"])))
        for rank, row in enumerate(ordered, start=1):
            iteration = int(row["iteration"])
            scores[iteration] += rank
            component_ranks[iteration][metric] = rank
    ranked: list[dict[str, object]] = []
    for row in candidates:
        iteration = int(row["iteration"])
        ranked.append(
            {
                "iteration": iteration,
                "rank_sum": scores[iteration],
                "component_ranks": component_ranks[iteration],
                "metrics": {metric: float(row[metric]) for metric in SELECTION_METRICS},
            }
        )
    ranked.sort(key=lambda row: (int(row["rank_sum"]), int(row["iteration"])))
    return ranked


def build_manifest(log_path: Path, checkpoint_root: Path) -> dict[str, object]:
    ranked = rank_rows(parse_validation_rows(log_path), saved_iterations(checkpoint_root))
    best = ranked[0]
    iteration = int(best["iteration"])
    checkpoint = checkpoint_root / f"iter_{iteration:07d}"
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_rule": "sum of ascending ranks across six inference-relevant validation losses",
        "lower_is_better": list(SELECTION_METRICS),
        "validation_log": str(log_path.resolve()),
        "checkpoint_root": str(checkpoint_root.resolve()),
        "selected_iteration": iteration,
        "selected_checkpoint": str(checkpoint.resolve()),
        "selected": best,
        "ranked_candidates": ranked,
        "scope_note": (
            "Source-side prefix/pseudo-streaming checkpoint selection; this does not claim "
            "a causal acoustic encoder or measured end-to-end latency."
        ),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=Path,
        default=root / "logs/uniss_phase3_prefix_streaming_full198_joint_v3.log",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=root / "checkpoints/uniss_phase3_prefix_streaming_full198_joint_v3",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.log, args.checkpoint_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["selected"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

