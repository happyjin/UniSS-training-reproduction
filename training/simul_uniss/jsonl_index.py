"""Compact, validated byte-offset sidecars for large JSONL datasets."""

from __future__ import annotations

import json
import os
import tempfile
from array import array
from pathlib import Path
from typing import Iterable, Sequence


def index_paths(data_path: Path) -> tuple[Path, Path]:
    return (
        data_path.with_name(f"{data_path.name}.offsets.bin"),
        data_path.with_name(f"{data_path.name}.offsets.json"),
    )


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_index(data_path: Path, offsets: Iterable[int]) -> dict[str, object]:
    values = offsets if isinstance(offsets, array) and offsets.typecode == "Q" else array("Q", offsets)
    binary_path, metadata_path = index_paths(data_path)
    _atomic_write_bytes(binary_path, values.tobytes())
    stat = data_path.stat()
    metadata = {
        "schema_version": "simul_uniss_jsonl_offsets_v1",
        "data_path": str(data_path.resolve()),
        "data_size_bytes": stat.st_size,
        "data_mtime_ns": stat.st_mtime_ns,
        "records": len(values),
        "offset_bytes": values.itemsize,
        "binary_path": str(binary_path.resolve()),
        "binary_size_bytes": binary_path.stat().st_size,
    }
    _atomic_write_json(metadata_path, metadata)
    return metadata


def load_index(data_path: Path) -> Sequence[int] | None:
    binary_path, metadata_path = index_paths(data_path)
    if not binary_path.exists() and not metadata_path.exists():
        return None
    if not binary_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"incomplete JSONL offset sidecar for {data_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "simul_uniss_jsonl_offsets_v1":
        raise ValueError(f"unexpected JSONL offset schema in {metadata_path}")
    stat = data_path.stat()
    if stat.st_size != int(metadata.get("data_size_bytes", -1)):
        raise ValueError(f"JSONL size changed after indexing: {data_path}")
    if stat.st_mtime_ns != int(metadata.get("data_mtime_ns", -1)):
        raise ValueError(f"JSONL mtime changed after indexing: {data_path}")
    values = array("Q")
    with binary_path.open("rb") as handle:
        values.fromfile(handle, binary_path.stat().st_size // values.itemsize)
    if len(values) != int(metadata.get("records", -1)):
        raise ValueError(f"JSONL offset count mismatch for {data_path}")
    if values and (values[0] != 0 or values[-1] >= stat.st_size):
        raise ValueError(f"invalid JSONL offsets for {data_path}")
    return values
