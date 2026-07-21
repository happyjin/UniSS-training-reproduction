"""Validate the boundary records of a UniSS packed JSONL artifact.

The full line count is intentionally left to ``wc -l`` in the packing runner.
This validator seeks directly to the first and last non-empty records, verifies
their fixed-length arrays and packed-sample boundaries, and converts them with
the same adapter used by Megatron training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import BinaryIO, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.megatron_uniss_dataset import packed_json_to_megatron_item


REQUIRED_ARRAYS = ("tokens", "labels", "loss_mask", "position_ids")


def _first_nonempty_line(handle: BinaryIO) -> bytes:
    handle.seek(0)
    for line in handle:
        if line.strip():
            return line
    raise ValueError("packed JSONL contains no non-empty records")


def _last_nonempty_line(handle: BinaryIO, chunk_size: int = 1024 * 1024) -> bytes:
    handle.seek(0, 2)
    end = handle.tell()
    if end == 0:
        raise ValueError("packed JSONL is empty")

    buffer = b""
    cursor = end
    while cursor > 0:
        read_size = min(chunk_size, cursor)
        cursor -= read_size
        handle.seek(cursor)
        buffer = handle.read(read_size) + buffer
        lines = buffer.splitlines()
        if cursor > 0 and lines:
            lines = lines[1:]
        for line in reversed(lines):
            if line.strip():
                return line
    if buffer.strip():
        return buffer.strip()
    raise ValueError("packed JSONL contains no non-empty records")


def _parse_record(raw_line: bytes, label: str) -> dict[str, object]:
    try:
        record = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON record") from exc
    if not isinstance(record, dict):
        raise TypeError(f"{label} record must be a JSON object")
    return record


def validate_record(record: Mapping[str, object], seq_length: int, label: str) -> None:
    for key in REQUIRED_ARRAYS:
        value = record.get(key)
        if not isinstance(value, list):
            raise TypeError(f"{label}.{key} must be a list")
        if len(value) != seq_length:
            raise ValueError(
                f"{label}.{key} length {len(value)} does not match {seq_length}"
            )

    boundaries = record.get("sample_boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError(f"{label}.sample_boundaries must be a non-empty list")

    previous_end = 0
    for boundary in boundaries:
        if not isinstance(boundary, list) or len(boundary) != 2:
            raise ValueError(f"{label} has invalid boundary {boundary!r}")
        start, end = boundary
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError(f"{label} boundary values must be integers")
        if start != previous_end or end <= start or end > seq_length:
            raise ValueError(f"{label} has invalid boundary {boundary!r}")
        previous_end = end

    # Exercise the exact tensor conversion path used by Megatron training.
    item = packed_json_to_megatron_item(record, seq_length=seq_length)
    if item["tokens"].numel() != seq_length:
        raise AssertionError(f"{label} tensor conversion returned an invalid length")


def validate_file(path: Path, seq_length: int) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"{path} is empty")

    with path.open("rb") as handle:
        handle.seek(-1, 2)
        if handle.read(1) != b"\n":
            raise ValueError(f"{path} does not end with a newline")
        first = _parse_record(_first_nonempty_line(handle), "first")
        last = _parse_record(_last_nonempty_line(handle), "last")

    validate_record(first, seq_length=seq_length, label="first")
    validate_record(last, seq_length=seq_length, label="last")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "seq_length": seq_length,
        "first_tasks": first.get("tasks", []),
        "last_tasks": last.get("tasks", []),
        "status": "ok",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--seq-length", type=int, default=18_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(validate_file(args.input, args.seq_length), sort_keys=True))


if __name__ == "__main__":
    main()
