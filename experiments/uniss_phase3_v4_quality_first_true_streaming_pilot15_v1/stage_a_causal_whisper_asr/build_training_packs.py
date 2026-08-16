#!/usr/bin/env python3
"""Parallel order-preserving Stage A 18k pack builder."""

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

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.packing import (
    LOSS_KIND_NAMES,
    build_stage_a_sample,
    pack_stage_a_samples,
    supervision_kind,
)
from training.simul_uniss.jsonl_index import load_index, write_index


SCHEMA = "uniss_quality_first_stage_a_pack_build_v1"


def _ranges(total: int, workers: int) -> list[tuple[int, int]]:
    workers = max(1, min(workers, total))
    return [
        (total * part // workers, total * (part + 1) // workers)
        for part in range(workers)
    ]


def _worker(task: tuple[int, str, int, int, str, tuple[int, ...], int, str]) -> dict[str, Any]:
    part, manifest_value, start, stop, model, fixed_speaker, seq_length, root_value = task
    from transformers import AutoTokenizer

    manifest = Path(manifest_value)
    offsets = load_index(manifest)
    if offsets is None:
        raise ValueError(f"missing manifest index: {manifest}")
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        local_files_only=True,
        trust_remote_code=False,
    )
    root = Path(root_value)
    output = root / f"packs.part{part:03d}.jsonl"
    counts: Counter[str] = Counter()
    byte_offsets = array("Q")
    byte_offset = 0

    def records():
        with manifest.open("rb") as handle:
            for index in range(start, stop):
                handle.seek(int(offsets[index]))
                record = json.loads(handle.readline())
                sample = build_stage_a_sample(
                    record,
                    lambda text: tokenizer.encode(text, add_special_tokens=False),
                    fixed_speaker,
                )
                counts["source_records"] += 1
                counts[f"samples:{LOSS_KIND_NAMES[supervision_kind(str(record['id']))]}"] += 1
                yield sample

    with output.open("xb") as handle:
        for packed in pack_stage_a_samples(records(), seq_length=seq_length):
            encoded = (
                json.dumps(packed, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            byte_offsets.append(byte_offset)
            handle.write(encoded)
            byte_offset += len(encoded)
            counts["packs"] += 1
            counts["used_tokens"] += int(packed["used_tokens"])
            counts["acoustic_annotations"] += len(packed["acoustics"])
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "part": part,
        "path": str(output),
        "offsets": byte_offsets,
        "bytes": byte_offset,
        "counts": dict(counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--seq-length", type=int, default=18_000)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage A packs: {args.output}")
    if args.seq_length <= 0 or args.workers <= 0:
        raise ValueError("sequence length and workers must be positive")
    source_offsets = load_index(args.manifest)
    if source_offsets is None:
        raise ValueError(f"missing manifest index: {args.manifest}")
    total = len(source_offsets)
    if args.limit is not None:
        total = min(total, int(args.limit))
    snapshot = json.loads(args.source_snapshot.read_text(encoding="utf-8"))
    fixed_speaker = tuple(
        int(value) for value in snapshot["fixed_system_speaker"]["global_tokens"]
    )
    if len(fixed_speaker) != 32:
        raise ValueError("source snapshot speaker must contain 32 tokens")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    parts_root = Path(
        tempfile.mkdtemp(prefix=f".{args.output.name}.parts.", dir=args.output.parent)
    )
    tasks = [
        (
            part,
            str(args.manifest.resolve()),
            start,
            stop,
            str(args.model.resolve()),
            fixed_speaker,
            args.seq_length,
            str(parts_root),
        )
        for part, (start, stop) in enumerate(_ranges(total, args.workers))
    ]
    try:
        with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            parts = list(pool.map(_worker, tasks))
        parts.sort(key=lambda value: int(value["part"]))
        merged_offsets = array("Q")
        merged_bytes = 0
        counts: Counter[str] = Counter()
        with args.output.open("xb") as destination:
            for part in parts:
                merged_offsets.extend(
                    merged_bytes + int(value) for value in part["offsets"]
                )
                with Path(part["path"]).open("rb") as source:
                    shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
                merged_bytes += int(part["bytes"])
                counts.update(
                    {str(key): int(value) for key, value in part["counts"].items()}
                )
            destination.flush()
            os.fsync(destination.fileno())
        index = write_index(args.output, merged_offsets)
        report = {
            "schema_version": SCHEMA,
            "status": "complete",
            "manifest": str(args.manifest.resolve()),
            "model": str(args.model.resolve()),
            "source_snapshot": str(args.source_snapshot.resolve()),
            "output": str(args.output.resolve()),
            "seq_length": args.seq_length,
            "workers": len(tasks),
            "source_records": total,
            "counts": dict(sorted(counts.items())),
            "fill_ratio": counts["used_tokens"] / max(1, counts["packs"] * args.seq_length),
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
