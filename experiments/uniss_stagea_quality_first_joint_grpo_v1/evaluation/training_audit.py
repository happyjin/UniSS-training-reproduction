#!/usr/bin/env python3
"""Parse immutable Megatron logs and GPU telemetry for the four-arm report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Sequence


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:E[+-]?\d+)?"
ITERATION_RE = re.compile(r"\[(?P<time>[^]]+)\] iteration\s+(?P<step>\d+)/\s*(?P<total>\d+)")
TRACKED = (
    "loss/asr_ce",
    "loss/mt_ce",
    "loss/semantic_ce",
    "loss/phase3_kl",
    "loss/boundary_eos",
    "grpo/policy_loss",
    "grpo/reference_kl",
    "grpo/reward_mean",
    "grpo/reward_std",
    "grpo/quality",
    "grpo/prefix_support",
    "grpo/completeness",
    "grpo/semantic_validity",
    "grpo/boundary",
    "grpo/premature_write",
    "grpo/unnecessary_wait",
    "grpo/write_coverage",
    "grpo/final_flush",
    "grpo/samples",
    "grpo/positions",
    "grpo/active",
    "grpo/reference_ready",
    "grpo/policy_update_rms",
)


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _metric(line: str, key: str) -> float | None:
    match = re.search(rf"(?:^|\| )\s*{re.escape(key)}:\s*({NUMBER})", line)
    return float(match.group(1)) if match else None


def parse_training(path: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    skipped = nan = 0
    failures: list[str] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            if any(
                marker in line
                for marker in (
                    "CUDA out of memory",
                    "received 0 items of ancdata",
                    "Traceback (most recent call last)",
                    "Pin memory thread exited unexpectedly",
                )
            ):
                failures.append(line.strip())
            match = ITERATION_RE.search(line)
            if match is None:
                continue
            row: dict[str, object] = {
                "step": int(match.group("step")),
                "total": int(match.group("total")),
                "timestamp": match.group("time"),
            }
            for key in TRACKED:
                value = _metric(line, key)
                if value is not None:
                    row[key] = value
            skipped_match = re.search(r"number of skipped iterations:\s*(\d+)", line)
            nan_match = re.search(r"number of nan iterations:\s*(\d+)", line)
            skipped = max(skipped, int(skipped_match.group(1)) if skipped_match else 0)
            nan = max(nan, int(nan_match.group(1)) if nan_match else 0)
            rows.append(row)
    if not rows:
        raise ValueError(f"training log has no iteration records: {path}")
    unique = {int(row["step"]): row for row in rows}
    rows = [unique[key] for key in sorted(unique)]
    first = datetime.fromisoformat(str(rows[0]["timestamp"]))
    last = datetime.fromisoformat(str(rows[-1]["timestamp"]))
    tail = rows[-min(20, len(rows)) :]
    tail_means = {
        key: statistics.fmean(float(row[key]) for row in tail if key in row)
        for key in TRACKED
        if any(key in row for row in tail)
    }
    return {
        "path": str(path.resolve()),
        "first_step": int(rows[0]["step"]),
        "last_step": int(rows[-1]["step"]),
        "target_steps": int(rows[-1]["total"]),
        "complete": int(rows[-1]["step"]) == int(rows[-1]["total"]),
        "logged_points": len(rows),
        "wall_seconds_between_logged_iterations": (last - first).total_seconds(),
        "skipped_iterations": skipped,
        "nan_iterations": nan,
        "failure_markers": failures,
        "first_metrics": {key: rows[0].get(key) for key in TRACKED if key in rows[0]},
        "last_metrics": {key: rows[-1].get(key) for key in TRACKED if key in rows[-1]},
        "tail20_means": tail_means,
        "curve": rows,
    }


def parse_gpu(path: Path, gpu_indices: set[int]) -> dict[str, object]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["index"].strip())
            if index not in gpu_indices:
                continue
            value = {
                "index": float(index),
                "memory": float(row["memory_used_mib"].strip()),
                "utility": float(row["utilization_gpu_percent"].strip()),
                "power": float(row["power_draw_w"].strip()),
                "power_limit": float(row["power_limit_w"].strip()),
            }
            if value["memory"] >= 10_000:
                rows.append(value)
    if not rows:
        raise ValueError(f"GPU log has no active records for {sorted(gpu_indices)}")
    utility = [row["utility"] for row in rows]
    power = [row["power"] for row in rows]
    memory = [row["memory"] for row in rows]
    return {
        "path": str(path.resolve()),
        "gpu_indices": sorted(gpu_indices),
        "active_observations": len(rows),
        "utility_mean_percent": statistics.fmean(utility),
        "utility_p50_percent": _percentile(utility, 0.50),
        "utility_p95_percent": _percentile(utility, 0.95),
        "utility_ge95_fraction": sum(value >= 95 for value in utility) / len(utility),
        "power_mean_w": statistics.fmean(power),
        "power_p50_w": _percentile(power, 0.50),
        "power_p95_w": _percentile(power, 0.95),
        "power_max_w": max(power),
        "memory_mean_mib": statistics.fmean(memory),
        "memory_max_mib": max(memory),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="ARM_ID=TRAIN.log=GPU.csv=GPU0,GPU1",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms: dict[str, object] = {}
    for raw in args.arm:
        parts = raw.split("=")
        if len(parts) != 4:
            raise ValueError("--arm must be ARM_ID=TRAIN.log=GPU.csv=GPU0,GPU1")
        arm, log, gpu, indices = parts
        arms[arm] = {
            "training": parse_training(Path(log)),
            "gpu": parse_gpu(Path(gpu), {int(value) for value in indices.split(",")}),
        }
    output = {
        "schema_version": "uniss_stagea_joint_grpo_training_audit_v1",
        "status": "complete",
        "arms": arms,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
