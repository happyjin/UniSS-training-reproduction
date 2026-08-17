#!/usr/bin/env python3
"""Authorize Stage B only after the complete V7 formal Stage A run passes."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


FINAL_ITERATION = 381
OPTIMIZER_HORIZON = 127
EXPECTED_FLOOR_UPDATES = FINAL_ITERATION - OPTIMIZER_HORIZON
EXPECTED_SOURCE_PACKS = 16195
EXPECTED_EPOCH_SAMPLES = 16256
EXPECTED_TOTAL_SAMPLES = 48768
EXPECTED_SHUFFLE_SEED = 20260816
VALIDATION = re.compile(
    r"validation loss at iteration\s+(\d+)(?:\s+on validation set)?\s+\|\s+(.*)"
)
VALUE = re.compile(r"([a-zA-Z0-9_]+) value:\s+([+-]?[0-9.]+(?:E[+-]?\d+)?)")
ITERATION = re.compile(
    rf"iteration\s+(\d+)/\s*{FINAL_ITERATION}.*"
    r"consumed samples:\s+(\d+).*"
    r"number of skipped iterations:\s+(\d+).*"
    r"number of nan iterations:\s+(\d+)"
)
DATASET = re.compile(
    r"> Stage A datasets: source_packs=(\d+) coverage_epochs=(\d+) "
    r"epoch_samples=(\d+) total_samples=(\d+) global_shuffle_seed=(\d+)"
)
PREFIX_DISABLED = re.compile(r"stage_a_prefix_schedule\s+\.+\s+False")
FINAL_SAVE = re.compile(
    rf"successfully saved checkpoint from iteration\s+{FINAL_ITERATION}\b"
)


def parse_log(
    text: str,
) -> tuple[int, dict[str, float], int, int, int, int, bool, tuple[int, ...] | None, bool]:
    validation_iteration = -1
    metrics: dict[str, float] = {}
    skipped = 0
    nan = 0
    max_iteration = -1
    consumed_samples = -1
    for line in text.splitlines():
        match = VALIDATION.search(line)
        if match:
            validation_iteration = int(match.group(1))
            metrics = {
                name: float(value)
                for name, value in VALUE.findall(match.group(2))
            }
        iteration = ITERATION.search(line)
        if iteration:
            max_iteration = max(max_iteration, int(iteration.group(1)))
            consumed_samples = max(consumed_samples, int(iteration.group(2)))
            skipped += int(iteration.group(3))
            nan += int(iteration.group(4))
    dataset = DATASET.search(text)
    return (
        validation_iteration,
        metrics,
        skipped,
        nan,
        max_iteration,
        consumed_samples,
        bool(FINAL_SAVE.search(text)),
        tuple(map(int, dataset.groups())) if dataset else None,
        bool(PREFIX_DISABLED.search(text)),
    )


def evaluate(text: str) -> dict[str, object]:
    (
        iteration,
        metrics,
        skipped,
        nan,
        max_iteration,
        consumed_samples,
        final_saved,
        dataset_geometry,
        prefix_disabled,
    ) = parse_log(text)
    required = (
        "ar_asr",
        "source_ctc",
        "offline_teacher_kl",
        "ctc_blank_ratio",
        "causal_glm_agreement",
        "ctc_blank_posterior",
        "ctc_blank_budget_target",
        "codebook_commitment",
        "codebook_identity_ce",
        "teacher_code_cosine",
        "teacher_code_margin",
        "code_adapter_residual",
        "code_adapter_rms",
        "curriculum_progress",
        "curriculum_chunk_ms",
    )
    missing = [name for name in required if name not in metrics]
    checks = {
        "training_reached_iteration_381": max_iteration >= FINAL_ITERATION,
        "formal_prefix_mode_disabled": prefix_disabled,
        "exact_three_epoch_global_shuffle_geometry": dataset_geometry
        == (
            EXPECTED_SOURCE_PACKS,
            3,
            EXPECTED_EPOCH_SAMPLES,
            EXPECTED_TOTAL_SAMPLES,
            EXPECTED_SHUFFLE_SEED,
        ),
        "full_three_epoch_schedule_completed": (
            max_iteration >= FINAL_ITERATION
            and consumed_samples == EXPECTED_TOTAL_SAMPLES
        ),
        "lr_floor_hold_completed": (
            max_iteration - OPTIMIZER_HORIZON >= EXPECTED_FLOOR_UPDATES
        ),
        "final_checkpoint_saved": final_saved,
        "final_validation_reached_iteration_381": iteration >= FINAL_ITERATION,
        "final_validation_is_160ms": metrics.get("curriculum_chunk_ms") == 160.0,
        "effective_curriculum_saturated": metrics.get("curriculum_progress") == 1.0,
        "metrics_complete": not missing,
        "finite_metrics": not missing
        and all(math.isfinite(metrics[name]) for name in required),
        "zero_skipped_iterations": skipped == 0,
        "zero_nan_iterations": nan == 0,
        "strict_sustained_ctc_not_blank": metrics.get("ctc_blank_ratio", 1.0)
        <= 0.25,
        "blank_posterior_within_budget": metrics.get("ctc_blank_posterior", 1.0)
        <= metrics.get("ctc_blank_budget_target", 0.0) + 0.05,
        "causal_code_identity_retained": metrics.get("causal_glm_agreement", 0.0)
        >= 0.02,
        "teacher_geometry_retained": metrics.get("teacher_code_cosine", 0.0)
        >= 0.85,
        "adapter_is_bounded": metrics.get("code_adapter_rms", math.inf) <= 0.50,
        "ar_asr_is_learning": metrics.get("ar_asr", math.inf) < 3.0,
        "source_ctc_is_learning": metrics.get("source_ctc", math.inf) < 15.0,
    }
    passed = all(checks.values())
    return {
        "schema_version": "uniss_stage_a_v7_formal_gate_v1",
        "passed": passed,
        "validation_iteration": iteration,
        "max_training_iteration": max_iteration,
        "consumed_samples": consumed_samples,
        "dataset_geometry": {
            "source_packs": dataset_geometry[0],
            "coverage_epochs": dataset_geometry[1],
            "epoch_samples": dataset_geometry[2],
            "total_samples": dataset_geometry[3],
            "global_shuffle_seed": dataset_geometry[4],
        }
        if dataset_geometry
        else None,
        "optimizer_horizon": OPTIMIZER_HORIZON,
        "lr_floor_hold_updates": max(0, max_iteration - OPTIMIZER_HORIZON),
        "expected_lr_floor_hold_updates": EXPECTED_FLOOR_UPDATES,
        "metrics": metrics,
        "skipped_iterations": skipped,
        "nan_iterations": nan,
        "missing_metrics": missing,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "stage_b_authorized": passed,
        "blocked_next_stage": None if passed else "stage_b_incremental_mt",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite formal gate: {args.output}")
    result = evaluate(args.log.read_text(encoding="utf-8", errors="replace"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
