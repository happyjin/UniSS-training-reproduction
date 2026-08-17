#!/usr/bin/env python3
"""Authorize formal v5 only after the final adapter canary passes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v4.stage_a_causal_whisper_asr.check_canary import (
    parse_log,
)


def evaluate(text: str) -> dict[str, object]:
    iteration, metrics, skipped, nan, max_iteration, final_saved = parse_log(text)
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
        "curriculum_chunk_ms",
    )
    missing = [name for name in required if name not in metrics]
    checks = {
        "training_reached_iteration_127": max_iteration >= 127,
        "final_checkpoint_saved": final_saved,
        "final_validation_reached_iteration_127": iteration >= 127,
        "final_validation_is_160ms": metrics.get("curriculum_chunk_ms") == 160.0,
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
        "teacher_geometry_retained": metrics.get("teacher_code_cosine", 0.0)
        >= 0.85,
        "adapter_is_bounded": metrics.get("code_adapter_rms", math.inf) <= 0.50,
        "ar_asr_is_learning": metrics.get("ar_asr", math.inf) < 3.0,
        "source_ctc_is_learning": metrics.get("source_ctc", math.inf) < 15.0,
    }
    return {
        "schema_version": "uniss_stage_a_v5_canary_gate_v1",
        "passed": all(checks.values()),
        "validation_iteration": iteration,
        "max_training_iteration": max_iteration,
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

