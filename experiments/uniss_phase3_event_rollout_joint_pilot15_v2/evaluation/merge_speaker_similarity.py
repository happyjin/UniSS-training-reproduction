#!/usr/bin/env python3
"""Merge fixed-speaker WavLM similarity shards with exact coverage checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.io_utils import iter_jsonl, write_json
from evaluation.sharding import merge_jsonl_by_key, row_key
from experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.speaker_similarity import (
    aggregate,
)


def merge(
    input_path: Path,
    parts: list[Path],
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = output_dir / "per_sample_speaker_similarity.jsonl"
    merge_report = merge_jsonl_by_key([canonical, *parts], canonical)
    rows = list(iter_jsonl(canonical))
    expected = {
        row_key(row)
        for row in iter_jsonl(input_path)
        if row.get("audio_path")
        and row.get("fixed_speaker_reference_audio_path")
        and not row.get("error")
    }
    actual = {row_key(row) for row in rows}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "speaker similarity coverage mismatch: "
            f"missing={len(missing)} {missing[:8]}, extra={len(extra)} {extra[:8]}"
        )
    report = {
        **aggregate(rows),
        "coverage": {
            "expected": len(expected),
            "observed": len(actual),
            "complete": True,
        },
        "merge": merge_report,
    }
    write_json(output_dir / "speaker_similarity.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--part", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.input, args.part, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
