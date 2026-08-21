#!/usr/bin/env python3
"""Aggregate all GPU worker reports into the formal-training authorization gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.gate import (
    build_gate,
    write_new_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-report", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--worker-report", type=Path, action="append", required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-hf", type=Path, required=True)
    parser.add_argument("--v1-initialization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_gate(
        canary_report=args.canary_report,
        selection=args.selection,
        worker_reports=args.worker_report,
        candidate_checkpoint=args.candidate_checkpoint,
        candidate_hf=args.candidate_hf,
        v1_initialization=args.v1_initialization,
    )
    write_new_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

