#!/usr/bin/env python3
"""Select an RL epoch checkpoint without turning quality gates into blockers."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


VALIDATION_RE = re.compile(r"validation loss at iteration\s+(\d+)\s+\|")
METRIC_RE = re.compile(
    r"([a-zA-Z0-9_/]+) value:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)[Ee][-+]?\d+)"
)


def parse_validation_metrics(text: str) -> list[dict[str, object]]:
    """Return the last regular validation row for every saved iteration."""
    by_iteration: dict[int, dict[str, object]] = {}
    for line in text.splitlines():
        if " on validation set " in line:
            continue
        match = VALIDATION_RE.search(line)
        if match is None:
            continue
        metrics = {name: float(value) for name, value in METRIC_RE.findall(line)}
        if "loss/total" not in metrics:
            continue
        iteration = int(match.group(1))
        by_iteration[iteration] = {"iteration": iteration, "metrics": metrics}
    return [by_iteration[index] for index in sorted(by_iteration)]


def select_checkpoint(log: Path, checkpoint_root: Path) -> dict[str, object]:
    rows = parse_validation_metrics(log.read_text(encoding="utf-8", errors="replace"))
    if not rows:
        raise ValueError(f"no regular validation rows found in {log}")
    candidates: list[dict[str, object]] = []
    for row in rows:
        iteration = int(row["iteration"])
        metrics = dict(row["metrics"])
        checkpoint = checkpoint_root / f"iter_{iteration:07d}"
        if not (checkpoint / ".metadata").is_file():
            continue
        total = float(metrics["loss/total"])
        ratio = float(metrics.get("diagnostic/ratio_mean", float("nan")))
        clipped = float(
            metrics.get("diagnostic/ratio_clipped_fraction", float("nan"))
        )
        kl = float(metrics.get("loss/reference_kl", float("nan")))
        finite = all(math.isfinite(value) for value in metrics.values())
        annotations: list[str] = []
        if not finite:
            annotations.append("non_finite_metric")
        if math.isfinite(ratio) and not 0.80 <= ratio <= 1.20:
            annotations.append("ratio_outside_recording_band")
        if math.isfinite(clipped) and clipped > 0.25:
            annotations.append("high_clipped_fraction")
        if math.isfinite(kl) and kl > 0.20:
            annotations.append("high_reference_kl")
        candidates.append(
            {
                "iteration": iteration,
                "checkpoint": str(checkpoint.resolve()),
                "metrics": metrics,
                "selection_score": total if finite else float("inf"),
                "quality_annotations": annotations,
            }
        )
    if not candidates:
        raise ValueError(f"no saved checkpoint matches validation rows under {checkpoint_root}")
    selected = min(candidates, key=lambda value: float(value["selection_score"]))
    return {
        "schema_version": "uniss_phasea_longepisode_rl_checkpoint_selection_v1",
        "status": "complete",
        "selection_rule": (
            "minimum finite regular validation loss/total; ratio, clipping and KL "
            "bands are annotations only and never block downstream evaluation"
        ),
        "training_log": str(log.resolve()),
        "checkpoint_root": str(checkpoint_root.resolve()),
        "selected_iteration": selected["iteration"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_quality_annotations": selected["quality_annotations"],
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = select_checkpoint(args.log, args.checkpoint_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SELECTED={payload['selected_checkpoint']}")
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
