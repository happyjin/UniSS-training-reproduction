#!/usr/bin/env python3
"""Assemble and validate Stage02 probe-manifest parts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--latent-manifest", type=Path, required=True)
    parser.add_argument("--expected-input-records", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(args.parts_dir.glob("part-*.summary.json"))
    ]
    if not entries:
        raise FileNotFoundError("no Stage02 part summaries")
    ranges = sorted((row["input_start"], row["input_stop"]) for _, row in entries)
    if ranges[0][0] != 0 or ranges[-1][1] != args.expected_input_records:
        raise ValueError(f"incomplete input coverage: {ranges[0]} .. {ranges[-1]}")
    if any(left[1] != right[0] for left, right in zip(ranges, ranges[1:])):
        raise ValueError("non-contiguous worker input ranges")
    written: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    directions: Counter[str] = Counter()
    parts = {"train": [], "valid": []}
    for _, row in entries:
        written.update(row["written"])
        skipped.update(row["skipped"])
        directions.update(row["directions"])
        for split in parts:
            part = row["outputs"][split]
            manifest = Path(part["manifest"])
            offsets = Path(part["offsets"])
            if not manifest.is_file() or not offsets.is_file():
                raise FileNotFoundError(f"missing output pair: {manifest}, {offsets}")
            if offsets.stat().st_size != int(part["records"]) * 8:
                raise ValueError(f"bad offset size: {offsets}")
            parts[split].append(part)
    accounted = sum(written.values()) + sum(skipped.values())
    if accounted != args.expected_input_records:
        raise ValueError(
            f"accounted {accounted} rows, expected {args.expected_input_records}"
        )
    report = {
        "schema_version": "uniss_streamspeech_ctc_probe_dataset_v1",
        "status": "passed",
        "latent_manifest": str(args.latent_manifest.resolve()),
        "input_records": args.expected_input_records,
        "eligible_records": sum(written.values()),
        "eligibility_rate": sum(written.values()) / args.expected_input_records,
        "written": dict(written),
        "skipped": dict(skipped),
        "directions": dict(directions),
        "tokenizer_dir": str(args.tokenizer_dir.resolve()),
        "parts": parts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

