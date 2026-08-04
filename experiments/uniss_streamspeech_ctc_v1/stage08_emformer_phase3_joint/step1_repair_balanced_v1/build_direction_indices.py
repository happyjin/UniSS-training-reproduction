#!/usr/bin/env python3
"""Build compact global-row indices for exact direction-balanced sampling."""

from __future__ import annotations

import argparse
import json
from array import array
from pathlib import Path

import numpy as np


DIRECTION = {"eng->cmn": 0, "cmn->eng": 1}


def collect_split(index: dict, split: str) -> dict[int, np.ndarray]:
    values = {0: array("I"), 1: array("I")}
    global_index = 0
    for entry in index["parts"][split]:
        manifest = Path(entry["manifest"])
        expected = int(entry["records"])
        observed = 0
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                try:
                    direction = DIRECTION[str(row["direction"])]
                except KeyError as error:
                    raise ValueError(
                        f"unknown direction in {manifest} row {observed}: "
                        f"{row.get('direction')}"
                    ) from error
                values[direction].append(global_index)
                global_index += 1
                observed += 1
        if observed != expected:
            raise ValueError(
                f"record mismatch for {manifest}: expected={expected}, observed={observed}"
            )
    arrays = {
        direction: np.frombuffer(rows, dtype=np.uint32).copy()
        for direction, rows in values.items()
    }
    if sum(len(rows) for rows in arrays.values()) != global_index:
        raise AssertionError("direction index lost rows")
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads(args.dataset_index.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "uniss_streamspeech_stage08_direction_indices_v1",
        "dataset_index": str(args.dataset_index.resolve()),
        "splits": {},
    }
    for split in ("train", "valid"):
        arrays = collect_split(index, split)
        counts = {}
        for direction, rows in arrays.items():
            output = args.output_dir / f"{split}_direction_{direction}.npy"
            if output.exists():
                raise FileExistsError(f"refusing to overwrite direction index: {output}")
            np.save(output, rows, allow_pickle=False)
            counts[str(direction)] = len(rows)
        metadata["splits"][split] = {
            "direction_counts": counts,
            "virtual_balanced_records": 2 * max(len(rows) for rows in arrays.values()),
        }
    metadata_path = args.output_dir / "metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite metadata: {metadata_path}")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
