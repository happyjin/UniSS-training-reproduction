#!/usr/bin/env python3
"""Build parallel frame-length sidecars for cross-rank duration bucketing."""

from __future__ import annotations

import argparse
import json
import math
from array import array
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def extract_part(
    arguments: tuple[int, dict[str, object], Path, str, Path, Path]
) -> tuple[int, Path, int]:
    index, entry, output_dir, split, source_manifest, source_offsets_path = arguments
    source = Path(str(entry["manifest"]))
    expected = int(entry["records"])
    output = output_dir / f"length-{split}-part-{index:03d}.u32"
    values = array("I")
    source_offsets = np.memmap(source_offsets_path, mode="r", dtype=np.uint64)
    with source.open(encoding="utf-8") as handle, source_manifest.open("rb") as source_rows:
        for line in handle:
            row = json.loads(line)
            source_index = int(row["source_manifest_index"])
            source_rows.seek(int(source_offsets[source_index]))
            source_row = json.loads(source_rows.readline())
            if str(source_row["id"]) != str(row["id"]):
                raise ValueError(f"source/target ID mismatch: {source_row['id']} != {row['id']}")
            values.append(max(1, math.ceil(float(source_row["source_duration_ms"]) / 40.0)))
    if len(values) != expected:
        raise ValueError(f"{source}: read {len(values)}, expected {expected}")
    with output.open("wb") as handle:
        values.tofile(handle)
    return index, output, len(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-offsets", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = json.loads(args.dataset_index.read_text(encoding="utf-8"))
    entries = index["parts"][args.split]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            part_index,
            entry,
            args.output_dir,
            args.split,
            args.source_manifest,
            args.source_offsets,
        )
        for part_index, entry in enumerate(entries)
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
        results = sorted(executor.map(extract_part, jobs))
    destination = args.output_dir / f"{args.split}_lengths.u32"
    total = 0
    with destination.open("wb") as output:
        for _, part_path, count in results:
            with part_path.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
            total += count
    expected = int(index["written"][args.split])
    if total != expected or destination.stat().st_size != expected * 4:
        raise ValueError(f"length index mismatch: total={total}, expected={expected}")
    report = {
        "schema_version": "uniss_streamspeech_stage03_length_index_v1",
        "status": "passed",
        "split": args.split,
        "records": total,
        "dtype": "native_uint32",
        "unit": "ceil(source_duration_ms / 40)",
        "output": str(destination.resolve()),
        "part_files": [str(value[1].resolve()) for value in results],
    }
    (args.output_dir / f"{args.split}_lengths.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
