#!/usr/bin/env python3
"""Validate and summarize completed full198 trajectory cache parts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_trajectory_cache import (
    CACHE_PART_SCHEMA,
)


CACHE_ASSEMBLY_SCHEMA = "uniss_true_subsecond_trajectory_cache_assembly_v1"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_part(root: Path, shard: int) -> dict[str, object]:
    part = root / f"part-{shard:03d}"
    marker_path = part / "PART_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema_version") != CACHE_PART_SCHEMA:
        raise ValueError(f"unexpected cache schema in {marker_path}")
    output = Path(str(marker.get("output")))
    if not output.is_file() or output.resolve() != (part / "trajectory_cache.jsonl").resolve():
        raise ValueError(f"missing or misplaced cache JSONL for shard {shard}")
    if output.stat().st_size != int(marker.get("output_size_bytes", -1)):
        raise ValueError(f"cache JSONL size changed for shard {shard}")
    accepted = int(marker.get("accepted_rows", 0))
    trajectories = int(marker.get("trajectory_count", 0))
    if accepted <= 0 or trajectories != 2 * accepted:
        raise ValueError(f"cache row accounting mismatch for shard {shard}")
    bundles = sorted(part.glob("bundle-*.npz"))
    if len(bundles) != int(marker.get("bundle_count", -1)):
        raise ValueError(f"cache bundle count changed for shard {shard}")
    if sum(path.stat().st_size for path in bundles) != int(
        marker.get("bundle_size_bytes", -1)
    ):
        raise ValueError(f"cache bundle size changed for shard {shard}")
    return marker


def validate_cache(root: Path, output: Path, shard_count: int) -> dict[str, object]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    parts = [validate_part(root, shard) for shard in range(shard_count)]
    thresholds = {float(part["confidence_threshold"]) for part in parts}
    topks = {int(part["topk"]) for part in parts}
    temperatures = {float(part["temperature"]) for part in parts}
    batch_sizes = {int(part["batch_size"]) for part in parts}
    if len(thresholds) != 1 or len(topks) != 1 or len(temperatures) != 1:
        raise ValueError("cache teacher settings differ across shards")
    summary = {
        "schema_version": CACHE_ASSEMBLY_SCHEMA,
        "cache_schema": CACHE_PART_SCHEMA,
        "root": str(root.resolve()),
        "shard_count": shard_count,
        "accepted_rows": sum(int(part["accepted_rows"]) for part in parts),
        "trajectory_count": sum(int(part["trajectory_count"]) for part in parts),
        "natural_write": sum(int(part["natural_write"]) for part in parts),
        "natural_read": sum(int(part["natural_read"]) for part in parts),
        "deadline_forced": sum(int(part["deadline_forced"]) for part in parts),
        "bundle_count": sum(int(part["bundle_count"]) for part in parts),
        "jsonl_size_bytes": sum(int(part["output_size_bytes"]) for part in parts),
        "bundle_size_bytes": sum(int(part["bundle_size_bytes"]) for part in parts),
        "confidence_threshold": next(iter(thresholds)),
        "topk": next(iter(topks)),
        "temperature": next(iter(temperatures)),
        "batch_sizes": sorted(batch_sizes),
        "parts": [str((root / f"part-{shard:03d}" / "PART_COMPLETE.json").resolve()) for shard in range(shard_count)],
    }
    if summary["trajectory_count"] != 2 * summary["accepted_rows"]:
        raise AssertionError("full cache trajectory count does not equal 2x accepted rows")
    _atomic_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=198)
    args = parser.parse_args()
    print(json.dumps(validate_cache(args.root, args.output, args.shard_count), sort_keys=True))


if __name__ == "__main__":
    main()
