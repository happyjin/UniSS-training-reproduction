#!/usr/bin/env python3
"""Assemble independently packed dense parts and build a validated JSONL index."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from array import array
from collections import Counter
from pathlib import Path

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.pack_dense_sessions import (
    PACK_PART_SCHEMA,
)
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    PACK_SCHEMA,
)
from training.simul_uniss.jsonl_index import load_index, write_index


ASSEMBLY_SCHEMA = "uniss_dense_aligned_streaming_pack_assembly_v1"


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


def assemble(args: argparse.Namespace) -> dict[str, object]:
    parts_root = Path(args.parts_root).resolve()
    output = Path(args.output).resolve()
    marker_path = Path(args.marker).resolve()
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("schema_version") != ASSEMBLY_SCHEMA:
            raise ValueError(f"unexpected pack assembly marker: {marker_path}")
        if output.is_file():
            print(json.dumps({"status": "already_complete", **marker}, sort_keys=True))
            return marker
        raise FileNotFoundError(output)
    if output.exists():
        raise FileExistsError(f"refusing unmarked packed output: {output}")

    paths = sorted(parts_root.glob("part-*/PACK_COMPLETE.json"))
    if len(paths) != args.expected_parts:
        raise ValueError(
            f"expected {args.expected_parts} pack markers, found {len(paths)}"
        )
    markers = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for marker in markers:
        if marker.get("schema_version") != PACK_PART_SCHEMA:
            raise ValueError("pack part has the wrong schema")
        if marker.get("pack_schema_version") != PACK_SCHEMA:
            raise ValueError("pack part has the wrong record schema")
        if int(marker["seq_length"]) != args.seq_length:
            raise ValueError("pack part has a different sequence length")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    output_offsets = array("Q")
    byte_offset = 0
    counts: Counter[str] = Counter()
    started = time.time()
    try:
        with temporary.open("wb") as target:
            for marker in markers:
                part = Path(str(marker["output"]))
                offsets = load_index(part)
                if offsets is None:
                    raise ValueError(f"missing index for packed part {part}")
                output_offsets.extend(byte_offset + int(value) for value in offsets)
                with part.open("rb") as source:
                    shutil.copyfileobj(source, target, length=32 * 1024 * 1024)
                byte_offset += part.stat().st_size
                for name, value in dict(marker["counts"]).items():
                    counts[name] += int(value)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    index = write_index(output, output_offsets)
    if len(output_offsets) != counts["packed_records"]:
        raise AssertionError("assembled pack count differs from part accounting")
    marker = {
        "schema_version": ASSEMBLY_SCHEMA,
        "pack_schema_version": PACK_SCHEMA,
        "status": "complete",
        "parts_root": str(parts_root),
        "expected_parts": args.expected_parts,
        "seq_length": args.seq_length,
        "output": str(output),
        "index": index,
        "counts": dict(counts),
        "packing_efficiency": counts["session_tokens"]
        / max(1, counts["packed_records"] * args.seq_length),
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(marker_path, marker)
    print(json.dumps(marker, sort_keys=True))
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--expected-parts", type=int, required=True)
    parser.add_argument("--seq-length", type=int, default=18_000)
    assemble(parser.parse_args())


if __name__ == "__main__":
    main()
