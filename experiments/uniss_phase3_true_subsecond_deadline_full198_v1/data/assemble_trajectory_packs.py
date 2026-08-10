#!/usr/bin/env python3
"""Assemble immutable shard packs and build a uint64 random-access index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.pack_trajectory_cache import (
    PACK_PART_SCHEMA,
)


ASSEMBLY_SCHEMA = "uniss_true_subsecond_trajectory_pack_assembly_v1"
OFFSET_SCHEMA = "uniss_true_subsecond_trajectory_pack_offsets_v1"


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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _load_part(marker_path: Path, seq_length: int) -> dict[str, object]:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema_version") != PACK_PART_SCHEMA:
        raise ValueError(f"unexpected trajectory pack schema in {marker_path}")
    if int(marker.get("seq_length", -1)) != seq_length:
        raise ValueError(f"sequence length mismatch in {marker_path}")
    output_metadata = marker.get("output")
    if not isinstance(output_metadata, dict):
        raise ValueError(f"missing output metadata in {marker_path}")
    output = Path(str(output_metadata.get("path")))
    if not output.is_file() or output.stat().st_size != int(
        output_metadata.get("size_bytes", -1)
    ):
        raise ValueError(f"missing or changed packed part: {output}")
    if int(marker.get("packed_records", 0)) <= 0:
        raise ValueError(f"empty packed part: {marker_path}")
    return marker


def assemble(
    parts_root: Path,
    output: Path,
    offsets: Path,
    marker_path: Path,
    *,
    shard_count: int,
    seq_length: int,
) -> dict[str, object]:
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("schema_version") != ASSEMBLY_SCHEMA:
            raise ValueError(f"unexpected assembly schema in {marker_path}")
        return marker
    if output.exists() or offsets.exists():
        raise FileExistsError("refusing unmarked trajectory assembly outputs")
    if shard_count <= 0 or seq_length <= 0:
        raise ValueError("shard_count and seq_length must be positive")

    parts = [
        _load_part(parts_root / f"part-{shard:03d}" / "PACK_COMPLETE.json", seq_length)
        for shard in range(shard_count)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    offsets.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    offsets_tmp = offsets.with_name(f".{offsets.name}.tmp.{os.getpid()}")
    output_tmp.unlink(missing_ok=True)
    offsets_tmp.unlink(missing_ok=True)
    digest = hashlib.sha256()
    packed_records = 0
    byte_offset = 0
    try:
        with output_tmp.open("wb") as output_handle, offsets_tmp.open("wb") as offset_handle:
            for part in parts:
                source = Path(str(part["output"]["path"]))  # type: ignore[index]
                part_records = 0
                with source.open("rb") as source_handle:
                    for line in source_handle:
                        if not line.strip():
                            continue
                        offset_handle.write(struct.pack("<Q", byte_offset))
                        output_handle.write(line)
                        digest.update(line)
                        byte_offset += len(line)
                        packed_records += 1
                        part_records += 1
                if part_records != int(part["packed_records"]):
                    raise ValueError(
                        f"packed part line count changed: {source} "
                        f"expected={part['packed_records']} actual={part_records}"
                    )
            output_handle.flush()
            offset_handle.flush()
            os.fsync(output_handle.fileno())
            os.fsync(offset_handle.fileno())
        os.replace(output_tmp, output)
        os.replace(offsets_tmp, offsets)
    finally:
        output_tmp.unlink(missing_ok=True)
        offsets_tmp.unlink(missing_ok=True)

    expected = sum(int(part["packed_records"]) for part in parts)
    if packed_records != expected or offsets.stat().st_size != packed_records * 8:
        raise AssertionError("assembled trajectory pack accounting mismatch")
    offset_metadata = {
        "schema_version": OFFSET_SCHEMA,
        "source": _metadata(output),
        "offsets": _metadata(offsets),
        "dtype": "uint64-little-endian",
        "records": packed_records,
    }
    _atomic_json(offsets.with_suffix(offsets.suffix + ".json"), offset_metadata)
    _atomic_text(output.with_suffix(output.suffix + ".count"), f"{packed_records}\n")
    marker = {
        "schema_version": ASSEMBLY_SCHEMA,
        "seq_length": seq_length,
        "shard_count": shard_count,
        "packed_records": packed_records,
        "trajectory_samples": sum(int(part["trajectory_samples"]) for part in parts),
        "deadline_forced": sum(int(part["deadline_forced"]) for part in parts),
        "supervised_tokens": sum(float(part["supervised_tokens"]) for part in parts),
        "output": _metadata(output),
        "output_sha256": digest.hexdigest(),
        "offset_index": offset_metadata,
        "part_markers": [str((parts_root / f"part-{shard:03d}" / "PACK_COMPLETE.json").resolve()) for shard in range(shard_count)],
    }
    _atomic_json(marker_path, marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--offsets", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=198)
    parser.add_argument("--seq-length", type=int, default=18_000)
    args = parser.parse_args()
    result = assemble(
        args.parts_root,
        args.output,
        args.offsets,
        args.marker,
        shard_count=args.shard_count,
        seq_length=args.seq_length,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
