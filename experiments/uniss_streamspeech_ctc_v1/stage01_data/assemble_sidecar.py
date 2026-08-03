#!/usr/bin/env python3
"""Validate Stage01 worker outputs and create a versioned dataset index."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=1_500_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for path in sorted(args.parts_dir.glob("part-*.summary.json")):
        summaries.append((path, json.loads(path.read_text(encoding="utf-8"))))
    if not summaries:
        raise FileNotFoundError(f"no worker summaries under {args.parts_dir}")
    ranges = sorted((item[1]["record_start"], item[1]["record_stop"]) for item in summaries)
    if ranges[0][0] != 0 or any(left[1] != right[0] for left, right in zip(ranges, ranges[1:])):
        raise ValueError(f"worker ranges are not contiguous: {ranges}")
    if ranges[-1][1] != args.expected_records:
        raise ValueError(
            f"expected {args.expected_records} input records, got {ranges[-1][1]}"
        )
    written: Counter[str] = Counter()
    directions: Counter[str] = Counter()
    invalid: Counter[str] = Counter()
    token_sums: Counter[str] = Counter()
    parts = {"train": [], "valid": []}
    for _, summary in summaries:
        written.update(summary["written"])
        directions.update(summary["directions"])
        invalid.update(summary["invalid"])
        token_sums.update(summary["token_sums"])
        for split in parts:
            path = Path(summary["outputs"][split])
            if not path.is_file():
                raise FileNotFoundError(path)
            parts[split].append(str(path.resolve()))
    if sum(written.values()) != args.expected_records:
        raise ValueError(
            f"written rows {sum(written.values())} != expected {args.expected_records}"
        )
    tokenizer_meta = json.loads(
        (args.tokenizer_dir / "tokenizers.json").read_text(encoding="utf-8")
    )
    report = {
        "schema_version": "uniss_streamspeech_ctc_stage01_dataset_v1",
        "status": "passed",
        "source_manifest": str(args.source_manifest.resolve()),
        "records": args.expected_records,
        "written": dict(written),
        "directions": dict(directions),
        "invalid_ctc_paths": dict(invalid),
        "token_sums": dict(token_sums),
        "tokenizers": tokenizer_meta,
        "parts": parts,
        "worker_summaries": [str(path.resolve()) for path, _ in summaries],
        "notes": {
            "25hz": "planned StreamSpeech-style frontend, one frame per 40 ms",
            "12p5hz": "current frozen WhisperVQ/GLM probe, approximately one frame per 80 ms",
            "immutability": "source manifest was read by byte offset and not modified",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

