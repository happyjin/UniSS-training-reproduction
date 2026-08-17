#!/usr/bin/env python3
"""Gate the V9 bridge-freeze 255-update canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v8.stage_a_causal_whisper_asr.check_canary import (
    evaluate as evaluate_v8,
)


def evaluate(text: str) -> dict[str, object]:
    result = evaluate_v8(text)
    passed = bool(result["passed"])
    result.update(
        schema_version="uniss_stage_a_v9_bridge_freeze_canary_gate_v1",
        passed=passed,
        formal_v9_authorized=passed,
        stage_b_authorized=False,
    )
    result.pop("formal_v8_authorized", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite V9 canary gate: {args.output}")
    result = evaluate(args.log.read_text(encoding="utf-8", errors="replace"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
