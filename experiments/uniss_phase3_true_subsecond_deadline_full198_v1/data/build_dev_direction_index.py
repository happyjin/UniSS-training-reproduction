#!/usr/bin/env python3
"""Partition the canonical UniST dev split into deterministic GPU work parts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_direction_index import (
    REQUIRED_COLUMNS,
    valid_mask,
)


SCHEMA_VERSION = "uniss_true_subsecond_dev_direction_index_v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _stable_key(sample_id: object, row_index: int, direction: str) -> bytes:
    payload = f"{sample_id}\x1f{row_index}\x1f{direction}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).digest()


def _balanced_parts(
    indices: np.ndarray,
    ids: np.ndarray,
    *,
    direction: str,
    partitions: int,
) -> list[np.ndarray]:
    ordered = sorted(
        (int(index) for index in indices),
        key=lambda index: _stable_key(ids[index], index, direction),
    )
    return [
        np.asarray(ordered[part::partitions], dtype=np.uint32)
        for part in range(partitions)
    ]


def build(source: Path, output_root: Path, partitions: int) -> dict[str, object]:
    source = source.resolve()
    output_root = output_root.resolve()
    marker = output_root / "index.json"
    source_metadata = _metadata(source)
    if marker.is_file():
        current = json.loads(marker.read_text(encoding="utf-8"))
        if (
            current.get("schema_version") == SCHEMA_VERSION
            and current.get("source") == source_metadata
            and int(current.get("shard_count", -1)) == partitions
        ):
            expected_paths = [
                output_root / f"part-{part:03d}.{lang}.npy"
                for part in range(partitions)
                for lang in ("eng", "cmn")
            ]
            if all(path.is_file() for path in expected_paths):
                return current

    parquet = pq.ParquetFile(source)
    missing = sorted(set(REQUIRED_COLUMNS) - set(parquet.schema_arrow.names))
    if missing:
        raise KeyError(f"{source} missing columns: {missing}")
    table = pq.read_table(source, columns=list(REQUIRED_COLUMNS))
    valid = valid_mask(table)
    src = table["src_lang"].to_numpy(zero_copy_only=False)
    ids = table["id"].to_numpy(zero_copy_only=False)
    eng_indices = np.flatnonzero(valid & (src == "eng"))
    cmn_indices = np.flatnonzero(valid & (src == "cmn"))
    eng_parts = _balanced_parts(
        eng_indices, ids, direction="eng->cmn", partitions=partitions
    )
    cmn_parts = _balanced_parts(
        cmn_indices, ids, direction="cmn->eng", partitions=partitions
    )

    output_root.mkdir(parents=True, exist_ok=True)
    shards = []
    for part, (eng, cmn) in enumerate(zip(eng_parts, cmn_parts)):
        eng_path = output_root / f"part-{part:03d}.eng.npy"
        cmn_path = output_root / f"part-{part:03d}.cmn.npy"
        np.save(eng_path, eng, allow_pickle=False)
        np.save(cmn_path, cmn, allow_pickle=False)
        shards.append(
            {
                "shard": part,
                "source": str(source),
                "eng": len(eng),
                "cmn": len(cmn),
                "accepted": len(eng) + len(cmn),
                "eng_index": str(eng_path),
                "cmn_index": str(cmn_path),
            }
        )

    represented = np.concatenate([*eng_parts, *cmn_parts])
    expected = np.flatnonzero(valid).astype(np.uint32)
    if len(represented) != len(expected) or not np.array_equal(
        np.sort(represented), expected
    ):
        raise AssertionError("dev partitioning lost or duplicated accepted rows")

    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source": source_metadata,
        "shards": shards,
        "shard_count": partitions,
        "rows": table.num_rows,
        "accepted": len(expected),
        "rejected": table.num_rows - len(expected),
        "eng": len(eng_indices),
        "cmn": len(cmn_indices),
        "partition_method": "direction-stratified-blake2b-round-robin-v1",
    }
    _atomic_json(marker, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--partitions", type=int, default=8)
    args = parser.parse_args()
    if args.partitions <= 0:
        raise ValueError("partitions must be positive")
    print(json.dumps(build(args.input, args.output_root, args.partitions), sort_keys=True))


if __name__ == "__main__":
    main()
