#!/usr/bin/env python3
"""Parallel, order-preserving conversion of aligned records to trajectories."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from training.simul_uniss.jsonl_index import load_index, write_index


BUILD_SCHEMA = "uniss_phase3_v4_e2e_simuls2st_gold_build_v1"


def _ranges(total: int, workers: int) -> list[tuple[int, int]]:
    workers = max(1, min(int(workers), int(total)))
    return [
        (total * part // workers, total * (part + 1) // workers)
        for part in range(workers)
    ]


def _worker(task: tuple[object, ...]) -> dict[str, Any]:
    (
        part,
        manifest_value,
        start,
        stop,
        split,
        v1_sha,
        phase3_sha,
        hash_audio,
        root_value,
    ) = task
    from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.gold_trajectory import (
        build_gold_trajectory,
    )

    manifest = Path(str(manifest_value))
    offsets = load_index(manifest)
    if offsets is None:
        raise ValueError(f"missing source JSONL offset index: {manifest}")
    output = Path(str(root_value)) / f"trajectories.part{int(part):03d}.jsonl"
    byte_offsets = array("Q")
    byte_offset = 0
    counts: Counter[str] = Counter()
    with manifest.open("rb") as source, output.open("xb") as destination:
        for record_index in range(int(start), int(stop)):
            source.seek(int(offsets[record_index]))
            record = json.loads(source.readline())
            trajectory = build_gold_trajectory(
                record,
                split=str(split),
                source_manifest=str(manifest),
                source_manifest_record=record_index,
                v1_checkpoint_sha256=str(v1_sha),
                phase3_teacher_sha256=str(phase3_sha),
                hash_audio=bool(hash_audio),
            )
            encoded = (trajectory.to_json() + "\n").encode("utf-8")
            byte_offsets.append(byte_offset)
            destination.write(encoded)
            byte_offset += len(encoded)
            counts["records"] += 1
            counts["events"] += len(trajectory.events)
            counts["source_glm_tokens"] += trajectory.source_glm_length
            counts["target_semantic_tokens"] += trajectory.target_semantic_length
            counts["prefinal_target_writes"] += sum(
                bool(event.target_semantic_delta) and not event.source_final
                for event in trajectory.events
            )
        destination.flush()
        os.fsync(destination.fileno())
    return {
        "part": int(part),
        "path": str(output),
        "offsets": byte_offsets,
        "bytes": byte_offset,
        "counts": dict(counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--v1-checkpoint-sha256", required=True)
    parser.add_argument("--phase3-teacher-sha256", required=True)
    parser.add_argument("--hash-audio", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite trajectories: {args.output}")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    offsets = load_index(args.manifest)
    if offsets is None:
        raise ValueError(f"missing source JSONL offset index: {args.manifest}")
    total = len(offsets)
    if args.limit is not None:
        total = min(total, max(0, int(args.limit)))
    if total <= 0:
        raise ValueError("source manifest selection is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parts_root = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.parts.", dir=args.output.parent))
    tasks = [
        (
            part,
            str(args.manifest.resolve()),
            start,
            stop,
            args.split,
            args.v1_checkpoint_sha256,
            args.phase3_teacher_sha256,
            args.hash_audio,
            str(parts_root),
        )
        for part, (start, stop) in enumerate(_ranges(total, args.workers))
    ]
    try:
        with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            parts = list(pool.map(_worker, tasks))
        parts.sort(key=lambda item: int(item["part"]))
        merged_offsets = array("Q")
        merged_bytes = 0
        counts: Counter[str] = Counter()
        with args.output.open("xb") as destination:
            for part in parts:
                merged_offsets.extend(
                    merged_bytes + int(offset) for offset in part["offsets"]
                )
                with Path(str(part["path"])).open("rb") as source:
                    shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
                merged_bytes += int(part["bytes"])
                counts.update({str(key): int(value) for key, value in part["counts"].items()})
            destination.flush()
            os.fsync(destination.fileno())
        index = write_index(args.output, merged_offsets)
        report = {
            "schema_version": BUILD_SCHEMA,
            "status": "complete",
            "manifest": str(args.manifest.resolve()),
            "output": str(args.output.resolve()),
            "split": args.split,
            "workers": len(tasks),
            "hash_audio": bool(args.hash_audio),
            "v1_checkpoint_sha256": args.v1_checkpoint_sha256,
            "phase3_teacher_sha256": args.phase3_teacher_sha256,
            "counts": dict(sorted(counts.items())),
            "index": index,
        }
        report_path = args.output.with_name(f"{args.output.name}.build.json")
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        shutil.rmtree(parts_root, ignore_errors=True)


if __name__ == "__main__":
    main()
