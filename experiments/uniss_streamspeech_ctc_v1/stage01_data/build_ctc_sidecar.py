#!/usr/bin/env python3
"""Build worker-local CTC target sidecars without changing source data."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ctc_utils import (
    canonical_lang,
    ctc_minimum_frames,
    deterministic_split,
    load_processor,
    normalize_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--offsets", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--valid-basis-points", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker-index outside num-workers")
    processors = {
        language: load_processor(args.tokenizer_dir / f"ctc_{language}.model")
        for language in ("eng", "cmn")
    }
    offsets = np.memmap(args.offsets, mode="r", dtype=np.uint64)
    total = min(len(offsets), args.limit) if args.limit else len(offsets)
    start = total * args.worker_index // args.num_workers
    stop = total * (args.worker_index + 1) // args.num_workers
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"part-{args.worker_index:03d}-of-{args.num_workers:03d}"
    output_paths = {
        "train": args.output_dir / f"train-{stem}.jsonl",
        "valid": args.output_dir / f"valid-{stem}.jsonl",
    }
    counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    invalid: Counter[str] = Counter()
    token_sums: Counter[str] = Counter()
    ratio_samples: dict[str, list[float]] = defaultdict(list)

    with (
        args.manifest.open("rb") as source,
        output_paths["train"].open("w", encoding="utf-8") as train_output,
        output_paths["valid"].open("w", encoding="utf-8") as valid_output,
    ):
        outputs = {"train": train_output, "valid": valid_output}
        for record_index in range(start, stop):
            source_offset = int(offsets[record_index])
            source.seek(source_offset)
            try:
                row = json.loads(source.readline())
                src_lang = canonical_lang(str(row["src_lang"]))
                tgt_lang = canonical_lang(str(row["tgt_lang"]))
                source_text = normalize_text(row["transcription"], src_lang)
                target_text = normalize_text(row["translation"], tgt_lang)
                source_ids = list(processors[src_lang].encode(source_text, out_type=int))
                target_ids = list(processors[tgt_lang].encode(target_text, out_type=int))
                if not source_ids or not target_ids:
                    invalid["empty_token_sequence"] += 1
                    continue
                source_min = ctc_minimum_frames(source_ids)
                target_min = ctc_minimum_frames(target_ids)
                duration_ms = int(round(float(row["source_duration_ms"])))
                frames_25hz = max(1, math.ceil(duration_ms / 40.0))
                frames_12p5hz = len(row.get("source_glm") or []) or max(
                    1, math.ceil(duration_ms / 80.0)
                )
                split = deterministic_split(str(row["id"]), args.valid_basis_points)
                payload = {
                    "schema_version": "uniss_streamspeech_ctc_target_v1",
                    "record_index": record_index,
                    "source_manifest_offset": source_offset,
                    "id": row["id"],
                    "split": split,
                    "direction": f"{src_lang}->{tgt_lang}",
                    "source_head": f"asr_{src_lang}",
                    "target_head": f"nar_s2tt_{tgt_lang}",
                    "source_token_ids": source_ids,
                    "target_token_ids": target_ids,
                    "source_token_count": len(source_ids),
                    "target_token_count": len(target_ids),
                    "source_ctc_min_frames": source_min,
                    "target_ctc_min_frames": target_min,
                    "frames_25hz": frames_25hz,
                    "frames_12p5hz": frames_12p5hz,
                    "valid_25hz": frames_25hz >= max(source_min, target_min),
                    "valid_12p5hz": frames_12p5hz >= max(source_min, target_min),
                }
                outputs[split].write(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                direction = payload["direction"]
                counts[split] += 1
                direction_counts[f"{split}:{direction}"] += 1
                token_sums[f"source:{src_lang}"] += len(source_ids)
                token_sums[f"target:{tgt_lang}"] += len(target_ids)
                for rate, frames in (("25hz", frames_25hz), ("12p5hz", frames_12p5hz)):
                    for role, minimum in (("source", source_min), ("target", target_min)):
                        key = f"{rate}:{role}:{direction}"
                        ratio_samples[key].append(frames / max(1, minimum))
                    if frames < source_min:
                        invalid[f"{rate}:source:{direction}"] += 1
                    if frames < target_min:
                        invalid[f"{rate}:target:{direction}"] += 1
            except Exception as exc:
                invalid[f"exception:{type(exc).__name__}"] += 1

    ratios = {
        key: {
            "p01": float(np.percentile(values, 1)),
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }
        for key, values in sorted(ratio_samples.items())
        if values
    }
    summary = {
        "schema_version": "uniss_streamspeech_ctc_target_part_summary_v1",
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "record_start": start,
        "record_stop": stop,
        "input_records": stop - start,
        "written": dict(counts),
        "directions": dict(direction_counts),
        "invalid": dict(invalid),
        "token_sums": dict(token_sums),
        "ctc_frame_ratios": ratios,
        "outputs": {key: str(value.resolve()) for key, value in output_paths.items()},
    }
    (args.output_dir / f"{stem}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

