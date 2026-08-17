#!/usr/bin/env python3
"""Authorize v6 formal training only after a sustained short-chunk canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v5.stage_a_causal_whisper_asr.check_canary import (
    evaluate as evaluate_v5,
)


def evaluate(text: str) -> dict[str, object]:
    result = evaluate_v5(text)
    metrics = result["metrics"]
    checks = dict(result["checks"])
    checks["strict_sustained_ctc_not_blank"] = (
        float(metrics.get("ctc_blank_ratio", 1.0)) <= 0.25
    )
    checks["effective_curriculum_saturated"] = (
        float(metrics.get("curriculum_progress", -1.0)) == 1.0
    )
    result.update(
        schema_version="uniss_stage_a_v6_hold_canary_gate_v1",
        passed=all(checks.values()),
        checks=checks,
        stage_b_authorized=False,
    )
    return result


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
