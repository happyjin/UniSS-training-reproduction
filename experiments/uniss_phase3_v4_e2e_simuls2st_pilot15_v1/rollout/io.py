"""Lossless indexed JSONL helpers for immutable rollout sidecars."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from array import array
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Sequence

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.schema import (
    V1Rollout,
)
from training.simul_uniss.jsonl_index import load_index


def partition_bounds(total: int, worker_index: int, num_workers: int) -> tuple[int, int]:
    total = int(total)
    worker_index = int(worker_index)
    num_workers = int(num_workers)
    if total <= 0 or num_workers <= 0 or not 0 <= worker_index < num_workers:
        raise ValueError("invalid rollout partition geometry")
    return (
        total * worker_index // num_workers,
        total * (worker_index + 1) // num_workers,
    )


def selected_total(path: Path, limit: int | None) -> tuple[Sequence[int], int]:
    offsets = load_index(path)
    if offsets is None:
        raise ValueError(f"missing JSONL offset index: {path}")
    total = len(offsets)
    if limit is not None:
        total = min(total, max(0, int(limit)))
    if total <= 0:
        raise ValueError("rollout selection is empty")
    return offsets, total


def iter_trajectories(
    path: Path,
    offsets: Sequence[int],
    start: int,
    stop: int,
) -> Iterator[tuple[int, E2ETrajectory]]:
    if not 0 <= start <= stop <= len(offsets):
        raise ValueError("trajectory range is outside its offset index")
    with path.open("rb") as handle:
        for record_index in range(start, stop):
            handle.seek(int(offsets[record_index]))
            yield record_index, E2ETrajectory.from_mapping(json.loads(handle.readline()))


def read_rollout_at(handle: BinaryIO, offsets: Sequence[int], index: int) -> V1Rollout:
    if not 0 <= int(index) < len(offsets):
        raise IndexError("rollout index is outside its offset sidecar")
    handle.seek(int(offsets[int(index)]))
    return V1Rollout.from_mapping(json.loads(handle.readline()))


def atomic_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite rollout report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class IndexedJSONLWriter:
    """Exclusive JSONL writer that hashes bytes and records exact offsets."""

    def __init__(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite rollout part: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("xb")
        self.offsets = array("Q")
        self.bytes = 0
        self.digest = hashlib.sha256()
        self.closed = False

    def write(self, value: str) -> None:
        if self.closed:
            raise RuntimeError("rollout writer is already closed")
        encoded = (value + "\n").encode("utf-8")
        self.offsets.append(self.bytes)
        self.handle.write(encoded)
        self.digest.update(encoded)
        self.bytes += len(encoded)

    def close(self) -> dict[str, object]:
        if not self.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
            self.closed = True
        return {
            "path": str(self.path.resolve()),
            "records": len(self.offsets),
            "bytes": self.bytes,
            "sha256": self.digest.hexdigest(),
        }

    def __enter__(self) -> "IndexedJSONLWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def file_sha256(path: Path, *, block_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def merged_offsets(parts: Iterable[tuple[int, Sequence[int]]]) -> array:
    output = array("Q")
    byte_base = 0
    for part_bytes, offsets in parts:
        output.extend(byte_base + int(value) for value in offsets)
        byte_base += int(part_bytes)
    return output


__all__ = [
    "IndexedJSONLWriter",
    "atomic_json",
    "file_sha256",
    "iter_trajectories",
    "merged_offsets",
    "partition_bounds",
    "read_rollout_at",
    "selected_total",
]
