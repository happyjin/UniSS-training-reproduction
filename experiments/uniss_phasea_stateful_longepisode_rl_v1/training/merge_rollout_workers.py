#!/usr/bin/env python3
"""Merge isolated free-running rollout workers without changing trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_workers(workers_root: Path, expected_workers: int) -> dict[str, object]:
    reports = sorted(workers_root.glob("worker_*/ROLLOUT.json"))
    if len(reports) != expected_workers:
        raise ValueError(
            f"expected {expected_workers} worker reports, found {len(reports)}"
        )
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    indexes = [int(value["worker_index"]) for value in payloads]
    if sorted(indexes) != list(range(expected_workers)):
        raise ValueError(f"worker indexes are incomplete: {sorted(indexes)}")
    if any(value.get("status") != "complete" for value in payloads):
        raise ValueError("one or more rollout workers are incomplete")
    group_sizes = {int(value["group_size"]) for value in payloads}
    if len(group_sizes) != 1:
        raise ValueError("rollout workers used different group sizes")
    summaries = [row for value in payloads for row in value["summaries"]]
    episode_ids = [str(row["episode_id"]) for row in summaries]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("duplicate episode IDs across workers")
    trajectory_paths = [Path(str(value["trajectory_path"])) for value in payloads]
    for path in trajectory_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    candidates = [candidate for row in summaries for candidate in row["candidates"]]
    rewards = [float(value["reward"]["total"]) for value in candidates]
    first_writes = [float(value["observation"]["first_write_ms"]) for value in candidates]
    spoken = [float(value["observation"]["spoken_text_fraction"]) for value in candidates]
    return {
        "schema_version": "uniss_free_running_episode_grpo_rollout_merged_v1",
        "status": "complete",
        "workers": expected_workers,
        "group_size": group_sizes.pop(),
        "episodes": len(summaries),
        "candidates": len(candidates),
        "trajectory_paths": [str(path.resolve()) for path in trajectory_paths],
        "worker_reports": [str(path.resolve()) for path in reports],
        "aggregate": {
            "mean_reward": sum(rewards) / len(rewards),
            "mean_first_write_ms": sum(first_writes) / len(first_writes),
            "mean_spoken_text_fraction": sum(spoken) / len(spoken),
            "complete_spoken_candidates": sum(value >= 0.999 for value in spoken),
        },
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-root", type=Path, required=True)
    parser.add_argument("--expected-workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = merge_workers(args.workers_root, args.expected_workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["aggregate"], ensure_ascii=False, sort_keys=True))
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
