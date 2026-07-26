"""Deterministic row sharding and crash-safe JSONL merging utilities."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

from evaluation.io_utils import iter_jsonl


def validate_shard(*, num_shards: int, shard_index: int) -> None:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")


def select_shard(
    rows: Iterable[Mapping[str, object]],
    *,
    num_shards: int,
    shard_index: int,
) -> Iterator[Mapping[str, object]]:
    """Yield a stable, disjoint index-modulo partition of ``rows``."""

    validate_shard(num_shards=num_shards, shard_index=shard_index)
    for index, row in enumerate(rows):
        if index % num_shards == shard_index:
            yield row


def row_key(row: Mapping[str, object]) -> tuple[str, str]:
    return str(row["id"]), str(row["mode"])


def load_keys(paths: Sequence[str | Path]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for value in paths:
        path = Path(value)
        if path.is_file():
            keys.update(row_key(row) for row in iter_jsonl(path))
    return keys


def merge_jsonl_by_key(
    inputs: Sequence[str | Path],
    destination: str | Path,
) -> dict[str, int]:
    """Merge JSONL files by id/mode with an atomic destination replacement.

    Existing canonical output can safely be included in ``inputs``. Duplicate
    keys are retained from the first input, which makes interrupted merge and
    resume cycles idempotent.
    """

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.merge-{os.getpid()}.tmp")
    seen: set[tuple[str, str]] = set()
    written = 0
    duplicates = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for value in inputs:
                path = Path(value)
                if not path.is_file():
                    continue
                for row in iter_jsonl(path):
                    key = row_key(row)
                    if key in seen:
                        duplicates += 1
                        continue
                    seen.add(key)
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                    handle.write("\n")
                    written += 1
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"written": written, "duplicates_skipped": duplicates}
