#!/usr/bin/env python3
"""Pack one completed repaired cache shard without splitting a session."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq

from experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.build_cache import (
    CACHE_PART_SCHEMA,
)
from experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.packing import (
    SessionSamples,
    build_token_sample,
    pack_sessions,
    shift_sample,
)
from experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.schema import TrajectoryRecord
from training import constants_uniss as c


PACK_PART_SCHEMA = "uniss_true_subsecond_pilot15_pack_part_v2"


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


class AnticipationReader:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.bundle: dict[str, np.ndarray] = {}

    def __call__(self, record: TrajectoryRecord) -> list[int]:
        if not record.deadline_forced_target:
            return []
        path_value, suffix = record.teacher_prefix_topk_path.rsplit("::", 1)
        namespace, index_text = suffix.split(":", 1)
        if namespace != "teacher":
            raise ValueError("invalid teacher bundle reference")
        path = Path(path_value)
        if self.path != path:
            with np.load(path, allow_pickle=False) as values:
                self.bundle = {name: values[name].copy() for name in values.files}
            self.path = path
        topk = self.bundle[f"request_{int(index_text)}_indices"]
        start = record.previous_committed_length
        result: list[int] = []
        for candidates in topk[start : min(len(topk), start + 4)]:
            token = next(
                (int(value) for value in candidates if 0 <= int(value) < c.VOCAB_SIZE),
                None,
            )
            if token is not None:
                result.append(token)
        if not result:
            raise ValueError("forced WRITE teacher produced no valid soft token")
        return result


def iter_sessions(
    cache_path: Path,
    target_bicodec: list[list[int]],
    counts: Counter[str],
) -> Iterator[SessionSamples]:
    anticipation = AnticipationReader()
    current_id: str | None = None
    current_records: list[TrajectoryRecord] = []

    def materialize(records: list[TrajectoryRecord]) -> SessionSamples:
        if not records:
            raise ValueError("cannot materialize an empty session")
        if [record.chunk_end_ms for record in records] != sorted(
            record.chunk_end_ms for record in records
        ):
            raise ValueError("session ticks are not monotonic")
        events = []
        for record in records:
            if record.row_index >= len(target_bicodec):
                raise ValueError("row index exceeds raw parquet")
            shifted = shift_sample(
                build_token_sample(
                    record,
                    target_bicodec[record.row_index],
                    anticipation_ids=anticipation(record),
                )
            )
            events.append(shifted)
            counts["trajectory_samples"] += 1
            counts[f"natural_action:{record.natural_action_target.value}"] += 1
            counts["deadline_forced"] += int(record.deadline_forced_target)
            counts["supervised_tokens"] += sum(shifted.loss_mask)
            if record.deadline_forced_target:
                action_losses = [
                    mask
                    for role, mask in zip(shifted.token_roles, shifted.loss_mask)
                    if role == 1
                ]
                if action_losses != [0.0]:
                    raise AssertionError("forced action token received hard CE")
        counts["sessions"] += 1
        return SessionSamples(records[0].sample_id, tuple(events))

    with cache_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = TrajectoryRecord.from_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid trajectory at line {line_number}") from exc
            if current_id is not None and record.sample_id != current_id:
                yield materialize(current_records)
                current_records = []
            current_id = record.sample_id
            current_records.append(record)
    if current_records:
        yield materialize(current_records)


def pack_cache_part(
    cache_part: Path,
    raw_parquet: Path,
    output: Path,
    marker_path: Path,
    *,
    seq_length: int,
) -> dict[str, object]:
    if marker_path.is_file() and output.is_file():
        value = json.loads(marker_path.read_text(encoding="utf-8"))
        if value.get("schema_version") == PACK_PART_SCHEMA:
            return value
    cache_marker = json.loads((cache_part / "PART_COMPLETE.json").read_text(encoding="utf-8"))
    if cache_marker.get("schema_version") != CACHE_PART_SCHEMA:
        raise ValueError("cache part is not a completed pilot15 v2 shard")
    cache_path = cache_part / "trajectory_cache.jsonl"
    table = pq.read_table(raw_parquet, columns=["target_bicodec"])
    targets = [
        [int(value) for value in row["target_bicodec"]] for row in table.to_pylist()
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    packed_records = 0
    represented = 0
    try:
        with temporary.open("wb") as handle:
            for item in pack_sessions(iter_sessions(cache_path, targets, counts), seq_length):
                represented += len(item["trajectory_sidecars"])
                encoded = (json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
                handle.write(encoded)
                digest.update(encoded)
                packed_records += 1
            handle.flush()
            os.fsync(handle.fileno())
        if represented != int(cache_marker["trajectory_count"]):
            raise ValueError("packed trajectory accounting mismatch")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    marker = {
        "schema_version": PACK_PART_SCHEMA,
        "seq_length": seq_length,
        "cache_part": _metadata(cache_path),
        "raw_parquet": _metadata(raw_parquet),
        "output": _metadata(output),
        "output_sha256": digest.hexdigest(),
        "sessions": counts["sessions"],
        "trajectory_samples": represented,
        "packed_records": packed_records,
        "natural_write": counts["natural_action:WRITE"],
        "natural_read": counts["natural_action:READ"],
        "deadline_forced": counts["deadline_forced"],
        "supervised_tokens": counts["supervised_tokens"],
    }
    output.with_suffix(output.suffix + ".count").write_text(f"{packed_records}\n", encoding="utf-8")
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
    print(
        json.dumps(
            pack_cache_part(
                args.cache_part,
                args.raw_parquet,
                args.output,
                args.marker,
                seq_length=args.seq_length,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
