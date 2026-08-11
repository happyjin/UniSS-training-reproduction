#!/usr/bin/env python3
"""Assemble the 15 isolated v2 shard packs and native uint64 offsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.assemble_trajectory_packs import (
    OFFSET_SCHEMA,
)
from experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.pack_cache import (
    PACK_PART_SCHEMA,
)


ASSEMBLY_SCHEMA = "uniss_true_subsecond_pilot15_pack_assembly_v2"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def assemble(parts_root: Path, output_root: Path, shard_count: int = 15) -> dict[str, object]:
    if shard_count != 15:
        raise ValueError("pilot15 v2 assembly is frozen to shards 0..14")
    marker_path = output_root / "ASSEMBLY_COMPLETE.json"
    output = output_root / "packed_trajectory.jsonl"
    offsets = output_root / "packed_trajectory.offsets.u64"
    if marker_path.is_file() and output.is_file() and offsets.is_file():
        value = json.loads(marker_path.read_text(encoding="utf-8"))
        if value.get("schema_version") == ASSEMBLY_SCHEMA:
            return value
    if output.exists() or offsets.exists():
        raise FileExistsError("refusing unmarked pilot15 v2 assembly output")
    parts = []
    for shard in range(shard_count):
        path = parts_root / f"part-{shard:03d}" / "PACK_COMPLETE.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != PACK_PART_SCHEMA:
            raise ValueError(f"unexpected pack marker: {path}")
        parts.append(value)

    output_root.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    offsets_tmp = offsets.with_name(f".{offsets.name}.tmp.{os.getpid()}")
    byte_offset = 0
    packed_records = 0
    digest = hashlib.sha256()
    try:
        with output_tmp.open("wb") as output_handle, offsets_tmp.open("wb") as offset_handle:
            for part in parts:
                source = Path(str(part["output"]["path"]))
                observed = 0
                with source.open("rb") as source_handle:
                    for line in source_handle:
                        if not line.strip():
                            continue
                        offset_handle.write(struct.pack("<Q", byte_offset))
                        output_handle.write(line)
                        digest.update(line)
                        byte_offset += len(line)
                        packed_records += 1
                        observed += 1
                if observed != int(part["packed_records"]):
                    raise ValueError("packed part line count changed")
            output_handle.flush()
            offset_handle.flush()
            os.fsync(output_handle.fileno())
            os.fsync(offset_handle.fileno())
        os.replace(output_tmp, output)
        os.replace(offsets_tmp, offsets)
    finally:
        output_tmp.unlink(missing_ok=True)
        offsets_tmp.unlink(missing_ok=True)

    offset_metadata = {
        "schema_version": OFFSET_SCHEMA,
        "source": _metadata(output),
        "offsets": _metadata(offsets),
        "dtype": "uint64-little-endian",
        "records": packed_records,
    }
    _atomic_json(offsets.with_suffix(offsets.suffix + ".json"), offset_metadata)
    marker = {
        "schema_version": ASSEMBLY_SCHEMA,
        "shard_count": shard_count,
        "seq_length": 18_000,
        "packed_records": packed_records,
        "sessions": sum(int(value["sessions"]) for value in parts),
        "trajectory_samples": sum(int(value["trajectory_samples"]) for value in parts),
        "natural_write": sum(int(value["natural_write"]) for value in parts),
        "natural_read": sum(int(value["natural_read"]) for value in parts),
        "deadline_forced": sum(int(value["deadline_forced"]) for value in parts),
        "output": _metadata(output),
        "output_sha256": digest.hexdigest(),
        "offset_index": offset_metadata,
    }
    _atomic_json(marker_path, marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=15)
    args = parser.parse_args()
    print(json.dumps(assemble(args.parts_root, args.output_root, args.shard_count), sort_keys=True))


if __name__ == "__main__":
    main()
