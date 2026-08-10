#!/usr/bin/env python3
"""Materialize one cache part into isolated 18k trajectory training packs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
import numpy as np

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_trajectory_cache import (
    CACHE_PART_SCHEMA,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.schema import (
    TrajectoryRecord,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.trajectory_packing import (
    build_trajectory_token_sample,
    pack_trajectory_samples,
    shift_trajectory_sample,
)
from training import constants_uniss as c


PACK_PART_SCHEMA = "uniss_true_subsecond_trajectory_pack_part_v1"


def _file_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


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


def _load_complete_cache(cache_part: Path) -> tuple[Path, dict[str, object]]:
    marker_path = cache_part / "PART_COMPLETE.json"
    cache_path = cache_part / "trajectory_cache.jsonl"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema_version") != CACHE_PART_SCHEMA:
        raise ValueError(f"unexpected trajectory cache schema in {marker_path}")
    if Path(str(marker.get("output"))).resolve() != cache_path.resolve():
        raise ValueError("trajectory cache marker points to a different output")
    if int(marker.get("trajectory_count", 0)) <= 0:
        raise ValueError("trajectory cache part is empty")
    return cache_path, marker


def _iter_shifted(
    cache_path: Path,
    target_bicodec: list[list[int]],
    counts: Counter[str],
) -> Iterator:
    loaded_bundle_path: Path | None = None
    loaded_bundle: dict[str, np.ndarray] = {}

    def anticipation(record: TrajectoryRecord) -> list[int]:
        nonlocal loaded_bundle_path, loaded_bundle
        if not record.deadline_forced_target:
            return []
        reference = record.teacher_prefix_topk_path
        path_value, suffix = reference.rsplit("::", 1)
        namespace, index_text = suffix.split(":", 1)
        if namespace != "teacher":
            raise ValueError(f"invalid teacher cache reference: {reference}")
        index_value = int(index_text)
        path = Path(path_value)
        if loaded_bundle_path != path:
            with np.load(path) as values:
                loaded_bundle = {name: values[name].copy() for name in values.files}
            loaded_bundle_path = path
        topk = loaded_bundle[f"request_{index_value}_indices"]
        start = record.previous_committed_length
        end = min(len(topk), start + 4)
        values = []
        for candidates in topk[start:end]:
            valid = next(
                (int(value) for value in candidates if 0 <= int(value) < c.VOCAB_SIZE),
                None,
            )
            if valid is not None:
                values.append(valid)
        if not values:
            raise ValueError(f"teacher cache has no valid anticipation token: {reference}")
        return values

    with cache_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = TrajectoryRecord.from_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid trajectory at {cache_path}:{line_number}") from exc
            if record.row_index >= len(target_bicodec):
                raise ValueError(f"row index {record.row_index} exceeds raw parquet")
            sample = shift_trajectory_sample(
                build_trajectory_token_sample(
                    record,
                    target_bicodec[record.row_index],
                    anticipation_ids=anticipation(record),
                )
            )
            counts["trajectory_samples"] += 1
            counts[f"deadline_action:{record.deadline_action_target.value}"] += 1
            counts[f"natural_action:{record.natural_action_target.value}"] += 1
            counts["deadline_forced"] += int(record.deadline_forced_target)
            counts["supervised_tokens"] += sum(sample.loss_mask)
            yield sample


def pack_cache_part(
    cache_part: Path,
    raw_parquet: Path,
    output: Path,
    marker_path: Path,
    *,
    seq_length: int,
) -> dict[str, object]:
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("schema_version") != PACK_PART_SCHEMA:
            raise ValueError(f"unexpected pack marker schema in {marker_path}")
        if not Path(str(marker["output"]["path"])).is_file():  # type: ignore[index]
            raise FileNotFoundError(marker["output"]["path"])  # type: ignore[index]
        return marker
    if output.exists():
        raise FileExistsError(f"refusing unmarked trajectory pack output: {output}")
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")

    cache_path, cache_marker = _load_complete_cache(cache_part)
    table = pq.read_table(raw_parquet, columns=["target_bicodec"])
    target_bicodec = [
        [int(value) for value in row["target_bicodec"]]
        for row in table.to_pylist()
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    packed_records = 0
    represented_samples = 0
    try:
        shifted = _iter_shifted(cache_path, target_bicodec, counts)
        with temporary.open("wb") as handle:
            for item in pack_trajectory_samples(shifted, seq_length):
                represented_samples += len(item["trajectory_sidecars"])
                encoded = (
                    json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                handle.write(encoded)
                digest.update(encoded)
                packed_records += 1
            handle.flush()
            os.fsync(handle.fileno())
        if packed_records <= 0:
            raise ValueError("trajectory packing produced no records")
        if represented_samples != int(cache_marker["trajectory_count"]):
            raise ValueError(
                "trajectory packing accounting mismatch: "
                f"represented={represented_samples}, cache={cache_marker['trajectory_count']}"
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    marker = {
        "schema_version": PACK_PART_SCHEMA,
        "seq_length": seq_length,
        "cache_part": _file_metadata(cache_path),
        "raw_parquet": _file_metadata(raw_parquet),
        "output": _file_metadata(output),
        "output_sha256": digest.hexdigest(),
        "trajectory_samples": represented_samples,
        "packed_records": packed_records,
        "natural_write": counts["natural_action:WRITE"],
        "natural_read": counts["natural_action:READ"],
        "deadline_write": counts["deadline_action:WRITE"],
        "deadline_read": counts["deadline_action:READ"],
        "deadline_forced": counts["deadline_forced"],
        "supervised_tokens": counts["supervised_tokens"],
    }
    _atomic_text(output.with_suffix(output.suffix + ".count"), f"{packed_records}\n")
    _atomic_json(marker_path, marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-part", required=True, type=Path)
    parser.add_argument("--raw-parquet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    parser.add_argument("--seq-length", type=int, default=18_000)
    args = parser.parse_args()
    result = pack_cache_part(
        args.cache_part,
        args.raw_parquet,
        args.output,
        args.marker,
        seq_length=args.seq_length,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
