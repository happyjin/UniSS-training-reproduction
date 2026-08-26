#!/usr/bin/env python3
"""Merge reference attribution workers and attach free-running runtime results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-root", type=Path, required=True)
    parser.add_argument("--expected-workers", type=int, default=8)
    parser.add_argument("--runtime-rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = sorted(args.workers_root.glob("worker_*/ATTRIBUTION.json"))
    if len(reports) != args.expected_workers:
        raise ValueError(f"expected {args.expected_workers} workers, found {len(reports)}")
    references = [
        row
        for path in reports
        for row in json.loads(path.read_text(encoding="utf-8"))["results"]
    ]
    by_id = {str(row["episode_id"]): row for row in references}
    runtime = json.loads(args.runtime_rollout.read_text(encoding="utf-8"))
    for summary in runtime["summaries"]:
        episode_id = str(summary["episode_id"])
        if episode_id not in by_id:
            raise ValueError(f"runtime episode has no reference route: {episode_id}")
        candidate = next(
            value for value in summary["candidates"] if int(value["group_index"]) == 0
        )
        by_id[episode_id]["runtime_v2_group0"] = candidate
    if len(by_id) != len(runtime["summaries"]):
        raise ValueError("reference/runtime episode sets differ")
    rows = list(by_id.values())
    payload = {
        "schema_version": "uniss_phasea_abcd_attribution_v1",
        "status": "complete",
        "protocol": {
            "A": "full-context Phase-A ASR against teacher transcription",
            "B": "gold-source Phase-A MT against teacher translation",
            "C": "gold-target Phase-A semantic TTS and BiCodec health",
            "D": "free-running stateful Runtime v2 group-0 cascade",
        },
        "results": rows,
        "worker_reports": [str(path.resolve()) for path in reports],
        "runtime_rollout": str(args.runtime_rollout.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
