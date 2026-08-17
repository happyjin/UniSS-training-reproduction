#!/usr/bin/env python3
"""Gate the V8 255-update long-hold canary."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FINAL_ITERATION = 255
OPTIMIZER_HORIZON = 127
EXPECTED_PREFIX_SAMPLES = FINAL_ITERATION * 128
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
PREFIX = re.compile(
    r"> Stage A v7 prefix datasets: source_packs=(\d+) coverage_epochs=(\d+) "
    r"complete_samples=(\d+) prefix_samples=(\d+) global_shuffle_seed=(\d+)"
)
PREFIX_ENABLED = re.compile(r"stage_a_prefix_schedule\s+\.+\s+True")
FINAL_SAVE = re.compile(
    rf"successfully saved checkpoint from iteration\s+{FINAL_ITERATION}\b"
)


def evaluate(text: str) -> dict[str, object]:
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
            metrics = {name: float(value) for name, value in VALUE.findall(match.group(2))}
        iteration = ITERATION.search(line)
        if iteration:
            max_iteration = max(max_iteration, int(iteration.group(1)))
            consumed_samples = max(consumed_samples, int(iteration.group(2)))
            skipped += int(iteration.group(3))
            nan += int(iteration.group(4))
    prefix = PREFIX.search(text)
    prefix_geometry = tuple(map(int, prefix.groups())) if prefix else None
    required = (
        "ar_asr",
        "source_ctc",
        "ctc_blank_ratio",
        "causal_glm_agreement",
        "ctc_blank_posterior",
        "ctc_blank_budget_target",
        "teacher_code_cosine",
        "code_adapter_rms",
        "curriculum_progress",
        "curriculum_chunk_ms",
    )
    missing = [name for name in required if name not in metrics]
    checks = {
        "training_reached_iteration_255": max_iteration >= FINAL_ITERATION,
        "prefix_mode_enabled": bool(PREFIX_ENABLED.search(text)),
        "exact_shuffled_prefix_geometry": prefix_geometry
        == (16195, 3, 48768, EXPECTED_PREFIX_SAMPLES, 20260816),
        "prefix_samples_consumed": consumed_samples == EXPECTED_PREFIX_SAMPLES,
        "long_lr_floor_hold_completed": max_iteration - OPTIMIZER_HORIZON >= 128,
        "final_checkpoint_saved": bool(FINAL_SAVE.search(text)),
        "final_validation_reached_iteration_255": validation_iteration >= FINAL_ITERATION,
        "final_validation_is_160ms": metrics.get("curriculum_chunk_ms") == 160.0,
        "effective_curriculum_saturated": metrics.get("curriculum_progress") == 1.0,
        "metrics_complete": not missing,
        "finite_metrics": not missing and all(math.isfinite(metrics[name]) for name in required),
        "zero_skipped_iterations": skipped == 0,
        "zero_nan_iterations": nan == 0,
        "strict_sustained_ctc_not_blank": metrics.get("ctc_blank_ratio", 1.0) <= 0.25,
        "blank_posterior_is_controlled": metrics.get("ctc_blank_posterior", 1.0) <= 0.25,
        "causal_code_identity_retained": metrics.get("causal_glm_agreement", 0.0) >= 0.02,
        "teacher_geometry_retained": metrics.get("teacher_code_cosine", 0.0) >= 0.85,
        "adapter_is_bounded": metrics.get("code_adapter_rms", math.inf) <= 0.50,
        "ar_asr_is_learning": metrics.get("ar_asr", math.inf) < 3.0,
        "source_ctc_is_learning": metrics.get("source_ctc", math.inf) < 15.0,
    }
    passed = all(checks.values())
    return {
        "schema_version": "uniss_stage_a_v8_long_hold_canary_gate_v1",
        "passed": passed,
        "validation_iteration": validation_iteration,
        "max_training_iteration": max_iteration,
        "consumed_samples": consumed_samples,
        "prefix_geometry": prefix_geometry,
        "optimizer_horizon": OPTIMIZER_HORIZON,
        "lr_floor_hold_updates": max(0, max_iteration - OPTIMIZER_HORIZON),
        "metrics": metrics,
        "skipped_iterations": skipped,
        "nan_iterations": nan,
        "missing_metrics": missing,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "stage_b_authorized": False,
        "formal_v8_authorized": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite V8 canary gate: {args.output}")
    result = evaluate(args.log.read_text(encoding="utf-8", errors="replace"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
