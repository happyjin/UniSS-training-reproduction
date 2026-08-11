#!/usr/bin/env python3
"""Atomically assemble ordered dense-session parts without reparsing JSON."""

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

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.build_dense_sessions import (
    PART_SCHEMA,
)
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.schema import (
    SCHEMA_VERSION,
)
from training.simul_uniss.jsonl_index import load_index, write_index


ASSEMBLY_SCHEMA = "uniss_dense_aligned_streaming_assembly_v2"


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
            raise ValueError(f"unexpected assembly marker: {marker_path}")
        if output.is_file():
            print(json.dumps({"status": "already_complete", **marker}, sort_keys=True))
            return marker
        raise FileNotFoundError(output)
    if output.exists():
        raise FileExistsError(f"refusing unmarked output: {output}")

    marker_paths = sorted(parts_root.glob("part-*/PART_COMPLETE.json"))
    if len(marker_paths) != args.expected_parts:
        raise ValueError(
            f"expected {args.expected_parts} part markers, found {len(marker_paths)}"
        )
    markers = [json.loads(path.read_text(encoding="utf-8")) for path in marker_paths]
    markers.sort(key=lambda value: int(value["part_index"]))
    for expected, marker in enumerate(markers):
        if marker.get("schema_version") != PART_SCHEMA:
            raise ValueError(f"part {expected} has the wrong schema")
        if marker.get("dense_schema_version") != SCHEMA_VERSION:
            raise ValueError(f"part {expected} has the wrong dense schema")
        if int(marker["part_index"]) != expected:
            raise ValueError("part indices are not complete and ordered")
        if int(marker["num_parts"]) != args.expected_parts:
            raise ValueError("part was built with a different num_parts value")
        if expected and int(marker["source_start"]) != int(markers[expected - 1]["source_end"]):
            raise ValueError("part source ranges have a gap or overlap")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    combined_offsets = array("Q")
    byte_offset = 0
    counts: Counter[str] = Counter()
    started = time.time()
    try:
        with temporary.open("wb") as target:
            for marker in markers:
                part = Path(str(marker["output"]))
                offsets = load_index(part)
                if offsets is None:
                    raise ValueError(f"missing index for dense part {part}")
                combined_offsets.extend(byte_offset + int(value) for value in offsets)
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
    index = write_index(output, combined_offsets)
    marker = {
        "schema_version": ASSEMBLY_SCHEMA,
        "dense_schema_version": SCHEMA_VERSION,
        "status": "complete",
        "parts_root": str(parts_root),
        "expected_parts": args.expected_parts,
        "source_start": int(markers[0]["source_start"]),
        "source_end": int(markers[-1]["source_end"]),
        "output": str(output),
        "index": index,
        "counts": dict(counts),
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
    assemble(parser.parse_args())


if __name__ == "__main__":
    main()
