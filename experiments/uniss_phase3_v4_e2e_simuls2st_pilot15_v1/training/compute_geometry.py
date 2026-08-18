#!/usr/bin/env python3
"""Compute auditable E2E update geometry from an immutable task-pool report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.build_task_pools import (
    BUILD_SCHEMA,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.schedule import (
    family_blocks,
    required_total_blocks,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    FAMILY_INTERLEAVED,
    TASK_FAMILIES,
)


def compute_geometry(
    report_path: str | Path,
    *,
    global_batch_size: int,
    coverage_epochs: int,
    seed: int,
) -> dict[str, object]:
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != BUILD_SCHEMA or report.get("status") != "passed":
        raise ValueError("E2E task-pool report is not a passed build")
    families = report.get("families")
    if not isinstance(families, dict) or set(families) != set(TASK_FAMILIES):
        raise ValueError("E2E task-pool report does not contain exactly five families")
    records = {
        family: int(families[family]["records"]) for family in TASK_FAMILIES
    }
    total_blocks = required_total_blocks(
        records[FAMILY_INTERLEAVED],
        global_batch_size=int(global_batch_size),
        coverage_epochs=int(coverage_epochs),
        seed=int(seed),
    )
    blocks = family_blocks(total_blocks, seed=int(seed))
    family_block_counts = {
        family: blocks.count(family) for family in TASK_FAMILIES
    }
    consumed = {
        family: family_block_counts[family] * int(global_batch_size)
        for family in TASK_FAMILIES
    }
    cycles = {
        family: consumed[family] / records[family] for family in TASK_FAMILIES
    }
    warmup_updates = max(20, math.ceil(0.03 * total_blocks))
    return {
        "schema_version": "uniss_phase3_v4_e2e_training_geometry_v1",
        "status": "passed",
        "task_pool_report": str(report_path),
        "seq_length": int(report["seq_length"]),
        "global_batch_size": int(global_batch_size),
        "coverage_epochs": int(coverage_epochs),
        "seed": int(seed),
        "train_iters": total_blocks,
        "train_samples": total_blocks * int(global_batch_size),
        "warmup_updates": warmup_updates,
        "family_records": records,
        "family_blocks": family_block_counts,
        "family_consumed_samples": consumed,
        "family_effective_cycles": cycles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-pool-report", type=Path, required=True)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--coverage-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = compute_geometry(
        args.task_pool_report,
        global_batch_size=args.global_batch_size,
        coverage_epochs=args.coverage_epochs,
        seed=args.seed,
    )
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite E2E geometry: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
