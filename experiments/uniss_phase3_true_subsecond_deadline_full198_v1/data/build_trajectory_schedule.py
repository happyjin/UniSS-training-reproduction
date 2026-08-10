#!/usr/bin/env python3
"""Materialize two real-time trajectory plans for every accepted full198 row."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.schema import (
    PLAN_SCHEMA_VERSION,
    TrajectoryPlan,
)


REQUIRED_COLUMNS = (
    "id",
    "source_glm",
    "source_bicodec",
    "target_bicodec",
    "src_lang",
    "tgt_lang",
)
PART_SCHEMA = "uniss_true_subsecond_trajectory_plan_part_v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def stable_uniform(*values: object) -> float:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    integer = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return integer / float(1 << 64)


def choose_times(sample_id: str, duration_ms: int) -> tuple[int, int]:
    if duration_ms <= 0:
        raise ValueError("duration must be positive")
    eligible_early = [value for value in (320, 480, 640, 800) if value <= duration_ms]
    early = eligible_early[
        min(int(stable_uniform(sample_id, "early") * len(eligible_early)), len(eligible_early) - 1)
    ] if eligible_early else duration_ms
    lower = min(duration_ms, max(960, math.ceil(duration_ms * 0.45 / 160) * 160))
    upper = min(duration_ms, max(lower, math.ceil(duration_ms * 0.85 / 160) * 160))
    candidates = list(range(lower, upper + 1, 160)) or [duration_ms]
    middle_late = candidates[
        min(int(stable_uniform(sample_id, "middle_late") * len(candidates)), len(candidates) - 1)
    ]
    return min(early, duration_ms), min(middle_late, duration_ms)


def _future_times(chunk_end_ms: int, duration_ms: int) -> tuple[int, int]:
    return min(duration_ms, chunk_end_ms + 160), min(duration_ms, chunk_end_ms + 320)


def plans_for_row(shard: int, row_index: int, row: dict[str, Any]) -> tuple[TrajectoryPlan, TrajectoryPlan]:
    sample_id = str(row["id"])
    source_glm = row["source_glm"] or []
    source_bicodec = row["source_bicodec"] or []
    target_bicodec = row["target_bicodec"] or []
    # UniST BiCodec semantic tokens are 50 Hz. This is a physical token clock,
    # not a source/target length ratio proxy.
    duration_ms = len(source_bicodec) * 20
    early, middle_late = choose_times(sample_id, duration_ms)
    result = []
    for kind, tick in (("early", early), ("middle_late", middle_late)):
        future_1, future_2 = _future_times(tick, duration_ms)
        result.append(
            TrajectoryPlan(
                sample_id=sample_id,
                shard=shard,
                row_index=row_index,
                src_lang=str(row["src_lang"]),
                tgt_lang=str(row["tgt_lang"]),
                source_duration_ms=duration_ms,
                chunk_end_ms=tick,
                future_1_end_ms=future_1,
                future_2_end_ms=future_2,
                trajectory_kind=kind,
                source_glm_length=len(source_glm),
                source_bicodec_length=len(source_bicodec),
                target_bicodec_length=len(target_bicodec),
            )
        )
    return result[0], result[1]


def _iter_rows(source: Path, accepted: np.ndarray, batch_size: int = 8192) -> Iterator[tuple[int, dict[str, Any]]]:
    parquet = pq.ParquetFile(source)
    accepted_set = set(int(value) for value in accepted)
    offset = 0
    for batch in parquet.iter_batches(columns=list(REQUIRED_COLUMNS), batch_size=batch_size):
        for local, row in enumerate(batch.to_pylist()):
            row_index = offset + local
            if row_index in accepted_set:
                yield row_index, row
        offset += batch.num_rows


def build_one(payload: tuple[int, str, str, str, str]) -> dict[str, Any]:
    shard, source_name, index_root_name, output_root_name, index_template = payload
    source = Path(source_name).resolve()
    index_root = Path(index_root_name).resolve()
    output_root = Path(output_root_name).resolve()
    output_dir = output_root / f"part-{shard:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "trajectory_plan.jsonl"
    marker = output_dir / "PART_COMPLETE.json"
    if marker.is_file() and output.is_file():
        value = json.loads(marker.read_text(encoding="utf-8"))
        if value.get("schema_version") == PART_SCHEMA and value.get("source") == str(source):
            return value
    eng = np.load(
        index_root / index_template.format(shard=shard, lang="eng"), mmap_mode="r"
    )
    cmn = np.load(
        index_root / index_template.format(shard=shard, lang="cmn"), mmap_mode="r"
    )
    accepted = np.sort(np.concatenate((eng, cmn)))
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as handle:
            for row_index, row in _iter_rows(source, accepted):
                for plan in plans_for_row(shard, row_index, row):
                    encoded = (json.dumps(plan.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                    handle.write(encoded)
                    digest.update(encoded)
                    counts["trajectories"] += 1
                    counts[f"kind:{plan.trajectory_kind}"] += 1
                    counts[f"direction:{plan.src_lang}->{plan.tgt_lang}"] += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    accepted_rows = len(accepted)
    if counts["trajectories"] != 2 * accepted_rows:
        raise AssertionError("every accepted row must produce exactly two trajectories")
    value = {
        "schema_version": PART_SCHEMA,
        "trajectory_schema": PLAN_SCHEMA_VERSION,
        "shard": shard,
        "source": str(source),
        "output": str(output),
        "accepted_rows": accepted_rows,
        "trajectory_count": counts["trajectories"],
        "early": counts["kind:early"],
        "middle_late": counts["kind:middle_late"],
        "directions": {
            key.removeprefix("direction:"): count
            for key, count in counts.items()
            if key.startswith("direction:")
        },
        "sha256": digest.hexdigest(),
    }
    _atomic_json(marker, value)
    return value


def build(
    index_json: Path,
    output_root: Path,
    workers: int,
    *,
    shard_count: int = 198,
    index_template: str = "train-{shard:05d}.{lang}.npy",
) -> dict[str, Any]:
    index = json.loads(index_json.read_text(encoding="utf-8"))
    shards = index["shards"]
    if len(shards) != shard_count:
        raise ValueError(
            f"trajectory schedule requires {shard_count} indexed shards, found {len(shards)}"
        )
    index_root = index_json.parent
    payloads = [
        (
            int(value["shard"]),
            str(value["source"]),
            str(index_root),
            str(output_root),
            index_template,
        )
        for value in shards
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(build_one, payloads))
    summary = {
        "schema_version": "uniss_true_subsecond_trajectory_plan_assembly_v1",
        "parts": parts,
        "shard_count": len(parts),
        "accepted_rows": sum(int(value["accepted_rows"]) for value in parts),
        "trajectory_count": sum(int(value["trajectory_count"]) for value in parts),
        "quality_replay_count": sum(int(value["accepted_rows"]) for value in parts),
        "performance_replay_count": sum(int(value["accepted_rows"]) for value in parts),
        "all_rows_have_four_task_families": True,
        "workers": workers,
    }
    _atomic_json(output_root / "trajectory_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--shard-count", type=int, default=198)
    parser.add_argument(
        "--index-template", default="train-{shard:05d}.{lang}.npy"
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.shard_count <= 0:
        raise ValueError("shard-count must be positive")
    print(
        json.dumps(
            build(
                args.index_json,
                args.output_root,
                args.workers,
                shard_count=args.shard_count,
                index_template=args.index_template,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
