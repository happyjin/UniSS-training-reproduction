#!/usr/bin/env python3
"""Prepare deterministic, indexed, non-overlapping exact-runtime eval shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from training.simul_uniss.jsonl_index import load_index, write_index


SCHEMA = "uniss_event_rollout_fixed15_runtime_manifests_v1"
DIRECTIONS = (("cmn", "eng"), ("eng", "cmn"))


def _row(handle, offset: int) -> dict[str, object]:
    handle.seek(int(offset))
    return json.loads(handle.readline())


def _direction(row: dict[str, object]) -> tuple[str, str]:
    value = (str(row.get("src_lang")), str(row.get("tgt_lang")))
    if value not in DIRECTIONS:
        raise ValueError(f"unexpected fixed15 direction {value}")
    return value


def _balanced_indices(
    source: Path,
    offsets,
    *,
    samples_per_direction: int,
    seed: int,
) -> list[int]:
    if samples_per_direction <= 0:
        raise ValueError("samples_per_direction must be positive")
    target = {direction: samples_per_direction for direction in DIRECTIONS}
    selected: dict[tuple[str, str], list[int]] = {direction: [] for direction in DIRECTIONS}
    rng = random.Random(seed)
    candidates = list(range(len(offsets)))
    rng.shuffle(candidates)
    with source.open("rb") as handle:
        for index in candidates:
            direction = _direction(_row(handle, offsets[index]))
            if len(selected[direction]) < target[direction]:
                selected[direction].append(index)
            if all(len(selected[value]) == target[value] for value in DIRECTIONS):
                break
    missing = {
        f"{source_lang}->{target_lang}": target[direction] - len(selected[direction])
        for direction in DIRECTIONS
        for source_lang, target_lang in (direction,)
        if len(selected[direction]) < target[direction]
    }
    if missing:
        raise ValueError(f"insufficient direction-balanced samples: {missing}")
    return sorted(index for values in selected.values() for index in values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> list[int]:
    offsets: list[int] = []
    with path.open("xb") as handle:
        for row in rows:
            offsets.append(handle.tell())
            handle.write(
                (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
            )
    write_index(path, offsets)
    return offsets


def prepare(
    source: Path,
    output_root: Path,
    *,
    split: str,
    num_shards: int,
    samples_per_direction: int | None,
    seed: int,
) -> dict[str, object]:
    source = source.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite runtime manifests: {output_root}")
    offsets = load_index(source)
    if offsets is None:
        raise ValueError(f"source manifest lacks a validated uint64 index: {source}")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    selected = (
        list(range(len(offsets)))
        if samples_per_direction is None
        else _balanced_indices(
            source,
            offsets,
            samples_per_direction=samples_per_direction,
            seed=seed,
        )
    )
    output_root.mkdir(parents=True)
    per_shard: list[list[int]] = [[] for _ in range(num_shards)]
    for ordinal, source_index in enumerate(selected):
        per_shard[ordinal % num_shards].append(source_index)

    parts = []
    all_ids: set[str] = set()
    aggregate_directions: Counter[str] = Counter()
    with source.open("rb") as handle:
        for shard_index, source_indices in enumerate(per_shard):
            output = output_root / f"part-{shard_index:03d}.jsonl"
            ids: list[str] = []
            directions: Counter[str] = Counter()

            def rows():
                for source_index in source_indices:
                    row = _row(handle, offsets[source_index])
                    sample_id = str(row["id"])
                    if sample_id in all_ids:
                        raise ValueError(f"duplicate evaluation sample ID: {sample_id}")
                    all_ids.add(sample_id)
                    direction = "->".join(_direction(row))
                    directions[direction] += 1
                    aggregate_directions[direction] += 1
                    ids.append(sample_id)
                    yield {
                        **row,
                        "_evaluation_split": split,
                        "_evaluation_source_index": source_index,
                        "_evaluation_shard_index": shard_index,
                    }

            written_offsets = _write_jsonl(output, rows())
            ids_path = output.with_suffix(".sample_ids.json")
            ids_path.write_text(json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            parts.append(
                {
                    "shard_index": shard_index,
                    "manifest": str(output),
                    "sample_ids": str(ids_path),
                    "records": len(written_offsets),
                    "directions": dict(sorted(directions.items())),
                    "sha256": _sha256(output),
                    "first_source_index": source_indices[0] if source_indices else None,
                    "last_source_index": source_indices[-1] if source_indices else None,
                }
            )
    summary = {
        "schema_version": SCHEMA,
        "split": split,
        "source_manifest": str(source),
        "source_records": len(offsets),
        "selection": (
            "all_records"
            if samples_per_direction is None
            else "deterministic_direction_balanced_without_replacement"
        ),
        "samples_per_direction": samples_per_direction,
        "seed": seed,
        "selected_records": len(selected),
        "selected_unique_sample_ids": len(all_ids),
        "directions": dict(sorted(aggregate_directions.items())),
        "num_shards": num_shards,
        "partition": "round_robin_over_selected_source_indices",
        "parts": parts,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--samples-per-direction", type=int)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source,
                args.output_root,
                split=args.split,
                num_shards=args.num_shards,
                samples_per_direction=args.samples_per_direction,
                seed=args.seed,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

