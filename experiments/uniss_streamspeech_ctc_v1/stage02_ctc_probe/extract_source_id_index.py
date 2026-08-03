#!/usr/bin/env python3
"""Extract worker-local ID to current source-manifest byte-offset rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-offsets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    offsets = np.memmap(args.source_offsets, mode="r", dtype=np.uint64)
    total = min(len(offsets), args.limit) if args.limit else len(offsets)
    start = total * args.worker_index // args.num_workers
    stop = total * (args.worker_index + 1) // args.num_workers
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (
        f"source-id-part-{args.worker_index:03d}-of-{args.num_workers:03d}.tsv"
    )
    with args.source_manifest.open("rb") as source, output.open(
        "w", encoding="utf-8"
    ) as target:
        for record_index in range(start, stop):
            byte_offset = int(offsets[record_index])
            source.seek(byte_offset)
            row = json.loads(source.readline())
            record_id = str(row["id"])
            if "\t" in record_id or "\n" in record_id:
                raise ValueError(f"unsafe record id: {record_id!r}")
            target.write(f"{record_id}\t{record_index}\t{byte_offset}\n")
    summary = {
        "schema_version": "uniss_streamspeech_source_id_index_part_v1",
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "start": start,
        "stop": stop,
        "records": stop - start,
        "output": str(output.resolve()),
    }
    (args.output_dir / f"source-id-part-{args.worker_index:03d}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

