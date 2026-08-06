#!/usr/bin/env python3
"""Build a compact byte-offset index for an immutable packed Phase3 JSONL."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from array import array
from pathlib import Path


def build_replay_index(
    source: str | Path,
    output: str | Path,
    *,
    max_records: int | None = None,
    progress_interval: int = 100_000,
) -> dict[str, object]:
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    offsets = array("Q")
    offset = 0
    with source_path.open("rb", buffering=16 * 1024 * 1024) as handle:
        for line in handle:
            if line.strip():
                offsets.append(offset)
                if progress_interval and len(offsets) % progress_interval == 0:
                    print(json.dumps({"records": len(offsets), "bytes": offset}), flush=True)
                if max_records is not None and len(offsets) >= max_records:
                    break
            offset += len(line)
    if not offsets:
        raise ValueError(f"no packed samples found in {source_path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            offsets.tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    stat = source_path.stat()
    metadata = {
        "schema_version": "uniss_phase3_replay_offsets_v1",
        "source": str(source_path),
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "offsets": str(output_path),
        "records": len(offsets),
        "complete": max_records is None,
        "max_records": max_records,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--progress-interval", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_replay_index(
        args.source,
        args.output,
        max_records=args.max_records,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()
