#!/usr/bin/env python3
"""Merge immutable fresh-rollout workers without copying audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-workers", type=int, default=8)
    parser.add_argument("--expected-episodes", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = sorted(args.worker_root.glob("worker_*/ROLLOUT.json"))
    if len(paths) != args.expected_workers:
        raise ValueError(f"expected {args.expected_workers} workers, found {len(paths)}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rounds = {int(value["round_index"]) for value in payloads}
    groups = {int(value["group_size"]) for value in payloads}
    if len(rounds) != 1 or len(groups) != 1:
        raise ValueError("worker rollout geometry differs")
    summaries = [row for payload in payloads for row in payload["summaries"]]
    summaries.sort(key=lambda row: str(row["episode_id"]))
    if len(summaries) != args.expected_episodes or len(
        {str(row["episode_id"]) for row in summaries}
    ) != args.expected_episodes:
        raise ValueError("merged episode coverage is incomplete")
    output = {
        "schema_version": "uniss_event_constrained_rollout_merged_v2",
        "status": "complete",
        "round_index": rounds.pop(),
        "workers": len(payloads),
        "group_size": groups.pop(),
        "episodes": len(summaries),
        "trajectory_paths": [str((path.parent / "trajectories.jsonl").resolve()) for path in paths],
        "worker_rollouts": [str(path.resolve()) for path in paths],
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: output[key] for key in ("status", "round_index", "workers", "group_size", "episodes")}, sort_keys=True))


if __name__ == "__main__":
    main()
