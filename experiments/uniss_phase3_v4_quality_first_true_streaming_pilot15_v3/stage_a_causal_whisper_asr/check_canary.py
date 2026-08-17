#!/usr/bin/env python3
"""Authorize a full v3 run only when the formal-geometry canary is healthy."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


VALIDATION = re.compile(
    r"validation loss at iteration\s+(\d+)(?:\s+on validation set)?\s+\|\s+(.*)"
)
VALUE = re.compile(r"([a-zA-Z0-9_]+) value:\s+([+-]?[0-9.]+(?:E[+-]?\d+)?)")
ITERATION = re.compile(
    r"iteration\s+(\d+)/\s*127.*number of skipped iterations:\s+(\d+).*"
    r"number of nan iterations:\s+(\d+)"
)


def parse_log(text: str) -> tuple[int, dict[str, float], int, int]:
    validation_iteration = -1
    metrics: dict[str, float] = {}
    skipped = 0
    nan = 0
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
            skipped += int(iteration.group(2))
            nan += int(iteration.group(3))
    return validation_iteration, metrics, skipped, nan


def evaluate(text: str) -> dict[str, object]:
    iteration, metrics, skipped, nan = parse_log(text)
    required = (
        "ar_asr",
        "source_ctc",
        "offline_teacher_kl",
        "ctc_blank_ratio",
        "causal_glm_agreement",
        "ctc_blank_posterior",
        "ctc_blank_budget_target",
        "codebook_commitment",
        "teacher_code_cosine",
    )
    missing = [name for name in required if name not in metrics]
    checks = {
        "validation_reached_iteration_96": iteration >= 96,
        "metrics_complete": not missing,
        "finite_metrics": not missing
        and all(math.isfinite(metrics[name]) for name in required),
        "zero_skipped_iterations": skipped == 0,
        "zero_nan_iterations": nan == 0,
        "ctc_not_all_blank": metrics.get("ctc_blank_ratio", 1.0) <= 0.95,
        "blank_posterior_within_budget": metrics.get("ctc_blank_posterior", 1.0)
        <= metrics.get("ctc_blank_budget_target", 0.0) + 0.05,
        "causal_code_identity_retained": metrics.get("causal_glm_agreement", 0.0)
        >= 0.02,
        "ar_asr_is_learning": metrics.get("ar_asr", math.inf) < 3.0,
        "source_ctc_is_learning": metrics.get("source_ctc", math.inf) < 15.0,
    }
    return {
        "schema_version": "uniss_stage_a_v3_canary_gate_v1",
        "passed": all(checks.values()),
        "validation_iteration": iteration,
        "metrics": metrics,
        "skipped_iterations": skipped,
        "nan_iterations": nan,
        "missing_metrics": missing,
        "checks": checks,
        "stage_b_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.log.read_text(encoding="utf-8", errors="replace"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
