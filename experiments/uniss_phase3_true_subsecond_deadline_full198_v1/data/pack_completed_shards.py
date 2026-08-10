#!/usr/bin/env python3
"""Continuously pack completed cache shards with bounded CPU parallelism."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.pack_trajectory_cache import (
    pack_cache_part,
)


def _pack_one(values: tuple[Path, Path, Path, Path, int]) -> dict[str, object]:
    return pack_cache_part(*values[:4], seq_length=values[4])


def _paths(cache_root: Path, raw_root: Path, parts_root: Path, shard: int, seq_length: int):
    output_root = parts_root / f"part-{shard:03d}"
    return (
        cache_root / f"part-{shard:03d}",
        raw_root / f"train-{shard:05d}.parquet",
        output_root / "packed_trajectory.jsonl",
        output_root / "PACK_COMPLETE.json",
        seq_length,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--parts-root", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=198)
    parser.add_argument("--seq-length", type=int, default=18_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.workers <= 0 or args.poll_seconds <= 0:
        raise ValueError("workers and poll seconds must be positive")

    while True:
        complete = [
            shard
            for shard in range(args.shard_count)
            if (args.parts_root / f"part-{shard:03d}" / "PACK_COMPLETE.json").is_file()
        ]
        if len(complete) == args.shard_count:
            print(json.dumps({"packed": len(complete), "status": "complete"}), flush=True)
            return
        ready = [
            shard
            for shard in range(args.shard_count)
            if (args.cache_root / f"part-{shard:03d}" / "PART_COMPLETE.json").is_file()
            and shard not in complete
        ]
        if not ready:
            print(
                json.dumps(
                    {"packed": len(complete), "ready": 0, "status": "waiting"},
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(args.poll_seconds)
            continue
        print(
            json.dumps(
                {"packed": len(complete), "ready": ready, "status": "packing"},
                sort_keys=True,
            ),
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=min(args.workers, len(ready))) as executor:
            futures = {
                executor.submit(
                    _pack_one,
                    _paths(
                        args.cache_root,
                        args.raw_root,
                        args.parts_root,
                        shard,
                        args.seq_length,
                    ),
                ): shard
                for shard in ready
            }
            for future in as_completed(futures):
                shard = futures[future]
                result = future.result()
                print(
                    json.dumps(
                        {
                            "shard": shard,
                            "packed_records": result["packed_records"],
                            "trajectory_samples": result["trajectory_samples"],
                            "status": "packed",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
