#!/usr/bin/env python3
"""Build paired raw-audio/codec subsets for T2S translation benchmarks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--codec-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 16:
        raise ValueError("--num-samples must be at least 16")

    codec_table = pq.read_table(args.codec_parquet)
    codec_rows = codec_table.to_pylist()
    raw_by_source_row = {}
    with args.raw_manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            raw_by_source_row[int(row["source_row"])] = row

    paired = [
        row for row in codec_rows
        if int(row["source_row"]) in raw_by_source_row
    ]
    if len(paired) < args.num_samples:
        raise ValueError(
            f"only {len(paired)} paired rows are available; "
            f"requested {args.num_samples}")

    # Evenly sample the sorted length distribution. This retains short, median,
    # and long examples without biasing the benchmark to a contiguous source
    # region. Shuffle afterwards so every training batch sees a representative
    # mixture while raw and codec modes keep exactly the same row set/order.
    paired.sort(key=lambda row: len(row["target_codec_codes"]))
    n = len(paired)
    indices = [round(i * (n - 1) / (args.num_samples - 1))
               for i in range(args.num_samples)]
    selected_codec = [paired[index] for index in indices]
    random.Random(args.seed).shuffle(selected_codec)
    selected_raw = [raw_by_source_row[int(row["source_row"])]
                    for row in selected_codec]
    # Per-row-group files may have been produced by workers launched before the
    # task name was finalized. The public merged files are canonicalized during
    # finalize; do the same for benchmark subsets built directly from shards.
    for row in selected_codec:
        row["task"] = "text_to_speech_translation"
    for row in selected_raw:
        row["task"] = "text_to_speech_translation"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    codec_output = args.output_dir / "codec.parquet"
    raw_output = args.output_dir / "raw.jsonl"
    pq.write_table(
        pa.Table.from_pylist(selected_codec, schema=codec_table.schema),
        codec_output,
        compression="zstd",
        row_group_size=1024,
    )
    with raw_output.open("w", encoding="utf-8") as handle:
        for row in selected_raw:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    lengths = sorted(len(row["target_codec_codes"]) for row in selected_codec)
    summary = {
        "num_samples": len(lengths),
        "source_row_min": min(int(row["source_row"]) for row in selected_codec),
        "source_row_max": max(int(row["source_row"]) for row in selected_codec),
        "codec_frames_min": lengths[0],
        "codec_frames_p50": lengths[len(lengths) // 2],
        "codec_frames_p95": lengths[int(0.95 * (len(lengths) - 1))],
        "codec_frames_p99": lengths[int(0.99 * (len(lengths) - 1))],
        "codec_frames_max": lengths[-1],
        "raw_jsonl": str(raw_output),
        "codec_parquet": str(codec_output),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
