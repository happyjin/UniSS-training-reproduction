#!/usr/bin/env python3
"""Gate the complete V9 bridge-freeze formal Stage A run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v7.stage_a_causal_whisper_asr.check_formal import (
    evaluate as evaluate_v7,
)


def evaluate(text: str) -> dict[str, object]:
    result = evaluate_v7(text)
    passed = bool(result["passed"])
    result.update(
        schema_version="uniss_stage_a_v9_bridge_freeze_formal_gate_v1",
        passed=passed,
        formal_v9_completed=passed,
        stage_b_authorized=passed,
        blocked_next_stage=None if passed else "stage_b_incremental_mt",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite V9 formal gate: {args.output}")
    result = evaluate(args.log.read_text(encoding="utf-8", errors="replace"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()


__all__ = ["evaluate", "main"]
