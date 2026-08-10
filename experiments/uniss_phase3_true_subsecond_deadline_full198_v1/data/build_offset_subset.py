#!/usr/bin/env python3
"""Build an immutable uniform subset of an existing packed offset index.

The packed JSONL is never copied or modified.  Only selected uint64 byte
offsets and matching metadata are written, so short native-Megatron pilots can
exercise data from a large multi-shard assembly without weakening the exact
coverage checks used by formal training.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.assemble_trajectory_packs import (
    OFFSET_SCHEMA,
)


REPLAY_OFFSET_SCHEMA = "uniss_phase3_replay_offsets_v1"


def _fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _metadata_path(offsets: Path) -> Path:
    return offsets.with_suffix(offsets.suffix + ".json")


def _validate_source(kind: str, packed: Path, offsets: Path) -> tuple[dict, int]:
    metadata = json.loads(_metadata_path(offsets).read_text(encoding="utf-8"))
    if kind == "trajectory":
        if metadata.get("schema_version") != OFFSET_SCHEMA:
            raise ValueError("unexpected trajectory source offset schema")
        if metadata.get("source") != _fingerprint(packed):
            raise ValueError("trajectory packed source changed after indexing")
    else:
        if metadata.get("schema_version") != REPLAY_OFFSET_SCHEMA:
            raise ValueError("unexpected replay source offset schema")
        stat = packed.stat()
        if (
            Path(str(metadata.get("source"))).resolve() != packed.resolve()
            or int(metadata.get("source_size_bytes", -1)) != stat.st_size
            or int(metadata.get("source_mtime_ns", -1)) != stat.st_mtime_ns
        ):
            raise ValueError("replay packed source changed after indexing")
    records = int(metadata.get("records", -1))
    if records <= 0 or offsets.stat().st_size != records * 8:
        raise ValueError("source offset count is invalid")
    return metadata, records


def _atomic_json(path: Path, value: object) -> None:
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


def build_subset(
    *,
    kind: str,
    packed: Path,
    source_offsets: Path,
    output_offsets: Path,
    records: int,
    excluded_source_indices: Sequence[int] = (),
) -> dict[str, object]:
    packed = packed.resolve()
    source_offsets = source_offsets.resolve()
    output_offsets = output_offsets.resolve()
    output_metadata = _metadata_path(output_offsets)
    if records <= 0:
        raise ValueError("records must be positive")
    if output_offsets.exists() or output_metadata.exists():
        raise FileExistsError("refusing to overwrite an offset subset")
    source_metadata, source_records = _validate_source(kind, packed, source_offsets)
    excluded = sorted({int(index) for index in excluded_source_indices})
    if any(index < 0 or index >= source_records for index in excluded):
        raise ValueError("excluded source index is outside the source offset range")
    eligible_indices = np.asarray(
        [index for index in range(source_records) if index not in set(excluded)],
        dtype=np.int64,
    )
    if records > len(eligible_indices):
        raise ValueError("subset cannot contain more records than its source")

    if records == 1:
        selected_indices = eligible_indices[[0]]
    else:
        eligible_positions = np.asarray(
            [
                index * (len(eligible_indices) - 1) // (records - 1)
                for index in range(records)
            ],
            dtype=np.int64,
        )
        selected_indices = eligible_indices[eligible_positions]
    if len(np.unique(selected_indices)) != records:
        raise AssertionError("uniform subset selection produced duplicate indices")
    source_values = np.memmap(source_offsets, mode="r", dtype="<u8")
    selected_offsets = np.asarray(source_values[selected_indices], dtype="<u8")

    output_offsets.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_offsets.name}.", dir=output_offsets.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        selected_offsets.tofile(temporary_path)
        with temporary_path.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_offsets)
    finally:
        temporary_path.unlink(missing_ok=True)

    subset = {
        "selection": "uniform_inclusive_endpoints_v1",
        "source_offsets": str(source_offsets),
        "source_records": source_records,
        "first_source_index": int(selected_indices[0]),
        "last_source_index": int(selected_indices[-1]),
        "excluded_source_indices": excluded,
    }
    if kind == "trajectory":
        metadata: dict[str, object] = {
            "schema_version": OFFSET_SCHEMA,
            "source": _fingerprint(packed),
            "offsets": _fingerprint(output_offsets),
            "dtype": "uint64-little-endian",
            "records": records,
            "subset": subset,
        }
    else:
        stat = packed.stat()
        metadata = {
            "schema_version": REPLAY_OFFSET_SCHEMA,
            "source": str(packed),
            "source_size_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "offsets": str(output_offsets),
            "records": records,
            "complete": False,
            "max_records": records,
            "subset": subset,
        }
    _atomic_json(output_metadata, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("trajectory", "replay"), required=True)
    parser.add_argument("--packed", required=True, type=Path)
    parser.add_argument("--source-offsets", required=True, type=Path)
    parser.add_argument("--output-offsets", required=True, type=Path)
    parser.add_argument("--records", required=True, type=int)
    parser.add_argument(
        "--exclude-source-index",
        action="append",
        default=[],
        type=int,
        help="source offset index to exclude; may be supplied more than once",
    )
    args = parser.parse_args()
    result = build_subset(
        kind=args.kind,
        packed=args.packed,
        source_offsets=args.source_offsets,
        output_offsets=args.output_offsets,
        records=args.records,
        excluded_source_indices=args.exclude_source_index,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
