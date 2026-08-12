#!/usr/bin/env python3
"""Freeze existing packed parts into one global random-access namespace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from experiments.uniss_phase3_event_rollout_joint_pilot15_v1.training.dataset import (
    MANIFEST_SCHEMA,
)
from training.simul_uniss.jsonl_index import load_index


PACK_PART_SCHEMA = "uniss_dense_aligned_streaming_pack_part_v3"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(
    *,
    parts_root: Path,
    output: Path,
    split: str,
    expected_parts: int,
    records_per_part: int | None = None,
) -> dict[str, object]:
    parts_root = parts_root.resolve()
    output = output.resolve()
    directories = sorted(path for path in parts_root.glob("part-*") if path.is_dir())
    if len(directories) != expected_parts:
        raise ValueError(
            f"{parts_root} contains {len(directories)} parts, expected {expected_parts}"
        )
    parts: list[dict[str, object]] = []
    cursor = 0
    for expected_index, directory in enumerate(directories):
        part_id = f"part-{expected_index:03d}"
        if directory.name != part_id:
            raise ValueError(f"expected {part_id}, found {directory.name}")
        marker = directory / "PACK_COMPLETE.json"
        packed = directory / "packed.jsonl"
        offsets = directory / "packed.jsonl.offsets.bin"
        if not marker.is_file() or not packed.is_file() or not offsets.is_file():
            raise FileNotFoundError(f"incomplete packed part: {directory}")
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != PACK_PART_SCHEMA:
            raise ValueError(f"unexpected marker schema in {marker}")
        if metadata.get("status") != "complete" or int(metadata["seq_length"]) != 18_000:
            raise ValueError(f"packed part is not a complete 18k artifact: {marker}")
        indexed_records = int(metadata["counts"]["packed_records"])
        index_values = load_index(packed)
        if index_values is None or len(index_values) != indexed_records:
            raise ValueError(f"packed index count mismatch in {directory}")
        records = (
            indexed_records
            if records_per_part is None
            else min(indexed_records, int(records_per_part))
        )
        if records <= 0:
            raise ValueError("records_per_part must expose at least one record")
        packed_stat = packed.stat()
        index_meta = dict(metadata["index"])
        if int(index_meta["records"]) != indexed_records:
            raise ValueError(f"marker index record count mismatch in {directory}")
        if int(index_meta["data_size_bytes"]) != packed_stat.st_size:
            raise ValueError(f"marker data size mismatch in {directory}")
        parts.append(
            {
                "part_id": part_id,
                "packed": str(packed.resolve()),
                "offsets": str(offsets.resolve()),
                "marker": str(marker.resolve()),
                "records": records,
                "indexed_records": indexed_records,
                "global_start": cursor,
                "global_end": cursor + records,
                "packed_size_bytes": packed_stat.st_size,
                "packed_mtime_ns": packed_stat.st_mtime_ns,
                "offsets_size_bytes": offsets.stat().st_size,
                "offsets_sha256": _sha256(offsets),
                "marker_sha256": _sha256(marker),
                "sessions": int(metadata["counts"]["sessions"]),
                "annotations": int(metadata["counts"]["annotations"]),
                "packing_efficiency": float(metadata["packing_efficiency"]),
            }
        )
        cursor += records
    result = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "complete",
        "split": split,
        "source_scope": "UniST train shards 00000-00014",
        "seq_length": 18_000,
        "part_count": len(parts),
        "total_records": cursor,
        "global_namespace": "prefix_sum_over_complete_pack_ids",
        "view": (
            "all_indexed_records"
            if records_per_part is None
            else f"first_{int(records_per_part)}_records_per_part"
        ),
        "shuffle_unit": "complete_18000_token_pack",
        "session_internal_event_order": "immutable",
        "parts": parts,
    }
    _atomic_json(output, result)
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "valid"))
    parser.add_argument("--expected-parts", required=True, type=int)
    parser.add_argument("--records-per-part", type=int)
    args = parser.parse_args()
    build(
        parts_root=args.parts_root,
        output=args.output,
        split=args.split,
        expected_parts=args.expected_parts,
        records_per_part=args.records_per_part,
    )


if __name__ == "__main__":
    main()
