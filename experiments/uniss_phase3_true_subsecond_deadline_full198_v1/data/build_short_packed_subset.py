#!/usr/bin/env python3
"""Materialize a shorter packed pilot from selected 18k JSONL offsets.

Only complete packed samples whose boundaries fit in the requested sequence
length are retained.  The remainder is replaced by the canonical UniSS pad
token, so no partial utterance leaks into the smoke run.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.assemble_trajectory_packs import (
    OFFSET_SCHEMA,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_offset_subset import (
    REPLAY_OFFSET_SCHEMA,
    _validate_source,
)
from training import constants_uniss as c


def _fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def shorten_record(value: Mapping[str, object], *, kind: str, seq_length: int) -> dict:
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")
    result = dict(value)
    boundaries = [
        [int(boundary[0]), int(boundary[1])]
        for boundary in value["sample_boundaries"]  # type: ignore[index]
        if int(boundary[1]) <= seq_length
    ]
    if not boundaries:
        raise ValueError("selected packed record has no complete sample in short sequence")
    keep_samples = len(boundaries)
    keep_end = boundaries[-1][1]

    def shorten(name: str, pad_value: int | float) -> list:
        source = list(value[name])  # type: ignore[index]
        if len(source) < seq_length:
            raise ValueError(f"{name} is shorter than requested sequence length")
        return source[:keep_end] + [pad_value] * (seq_length - keep_end)

    result["tokens"] = shorten("tokens", c.TOKEN_PAD)
    result["labels"] = shorten("labels", c.TOKEN_PAD)
    result["loss_mask"] = shorten("loss_mask", 0)
    result["position_ids"] = shorten("position_ids", 0)
    result["sample_boundaries"] = boundaries
    for name in ("tasks", "source_ids"):
        source = list(value[name])  # type: ignore[index]
        if len(source) < keep_samples:
            raise ValueError(f"{name} is shorter than sample boundaries")
        result[name] = source[:keep_samples]
    if kind == "trajectory":
        result["token_roles"] = shorten("token_roles", 0)
        sidecars = list(value["trajectory_sidecars"])  # type: ignore[index]
        if len(sidecars) < keep_samples:
            raise ValueError("trajectory sidecars are shorter than sample boundaries")
        result["trajectory_sidecars"] = sidecars[:keep_samples]
    return result


def build_short_packed_subset(
    *,
    kind: str,
    source_packed: Path,
    selected_offsets: Path,
    output_packed: Path,
    output_offsets: Path,
    seq_length: int,
) -> dict[str, object]:
    source_packed = source_packed.resolve()
    selected_offsets = selected_offsets.resolve()
    output_packed = output_packed.resolve()
    output_offsets = output_offsets.resolve()
    output_metadata = output_offsets.with_suffix(output_offsets.suffix + ".json")
    if any(path.exists() for path in (output_packed, output_offsets, output_metadata)):
        raise FileExistsError("refusing to overwrite a short packed subset")
    _, records = _validate_source(kind, source_packed, selected_offsets)
    offsets = np.memmap(selected_offsets, mode="r", dtype="<u8")

    output_packed.parent.mkdir(parents=True, exist_ok=True)
    output_offsets.parent.mkdir(parents=True, exist_ok=True)
    packed_fd, packed_name = tempfile.mkstemp(
        prefix=f".{output_packed.name}.", dir=output_packed.parent
    )
    offsets_fd, offsets_name = tempfile.mkstemp(
        prefix=f".{output_offsets.name}.", dir=output_offsets.parent
    )
    packed_tmp, offsets_tmp = Path(packed_name), Path(offsets_name)
    byte_offset = 0
    try:
        with (
            os.fdopen(packed_fd, "wb") as packed_handle,
            os.fdopen(offsets_fd, "wb") as offsets_handle,
            source_packed.open("rb") as source_handle,
        ):
            for source_offset in offsets:
                source_handle.seek(int(source_offset))
                value = json.loads(source_handle.readline())
                shortened = shorten_record(value, kind=kind, seq_length=seq_length)
                encoded = (
                    json.dumps(shortened, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                offsets_handle.write(struct.pack("<Q", byte_offset))
                packed_handle.write(encoded)
                byte_offset += len(encoded)
            packed_handle.flush()
            offsets_handle.flush()
            os.fsync(packed_handle.fileno())
            os.fsync(offsets_handle.fileno())
        os.replace(packed_tmp, output_packed)
        os.replace(offsets_tmp, output_offsets)
    finally:
        packed_tmp.unlink(missing_ok=True)
        offsets_tmp.unlink(missing_ok=True)

    if output_offsets.stat().st_size != records * 8:
        raise AssertionError("short packed offset accounting mismatch")
    if kind == "trajectory":
        metadata: dict[str, object] = {
            "schema_version": OFFSET_SCHEMA,
            "source": _fingerprint(output_packed),
            "offsets": _fingerprint(output_offsets),
            "dtype": "uint64-little-endian",
            "records": records,
            "short_subset": {
                "source_packed": str(source_packed),
                "selected_offsets": str(selected_offsets),
                "seq_length": seq_length,
            },
        }
    else:
        stat = output_packed.stat()
        metadata = {
            "schema_version": REPLAY_OFFSET_SCHEMA,
            "source": str(output_packed),
            "source_size_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "offsets": str(output_offsets),
            "records": records,
            "complete": False,
            "max_records": records,
            "short_subset": {
                "source_packed": str(source_packed),
                "selected_offsets": str(selected_offsets),
                "seq_length": seq_length,
            },
        }
    output_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("trajectory", "replay"), required=True)
    parser.add_argument("--source-packed", required=True, type=Path)
    parser.add_argument("--selected-offsets", required=True, type=Path)
    parser.add_argument("--output-packed", required=True, type=Path)
    parser.add_argument("--output-offsets", required=True, type=Path)
    parser.add_argument("--seq-length", type=int, default=4096)
    args = parser.parse_args()
    result = build_short_packed_subset(
        kind=args.kind,
        source_packed=args.source_packed,
        selected_offsets=args.selected_offsets,
        output_packed=args.output_packed,
        output_offsets=args.output_offsets,
        seq_length=args.seq_length,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
