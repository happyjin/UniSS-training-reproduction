#!/usr/bin/env python3
"""Merge disjoint Stage A checkpoint diagnosis workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.evaluate_checkpoint import (
    atomic_json,
    markdown_report,
    summarize_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("refusing to overwrite merged Stage A diagnosis")
    values = [json.loads(path.read_text(encoding="utf-8")) for path in args.parts]
    if any(value.get("schema_version") != "uniss_quality_first_stage_a_checkpoint_diagnosis_v1" for value in values):
        raise ValueError("unexpected Stage A diagnosis part schema")
    common_keys = ("checkpoint", "hf_model", "valid_packs", "num_workers")
    for key in common_keys:
        if len({json.dumps(value[key], sort_keys=True) for value in values}) != 1:
            raise ValueError(f"Stage A diagnosis parts disagree on {key}")
    workers = sorted(int(value["worker_index"]) for value in values)
    expected = list(range(int(values[0]["num_workers"])))
    if workers != expected:
        raise ValueError(f"Stage A diagnosis worker coverage differs: {workers} vs {expected}")
    rows = [row for value in values for row in value["samples"]]
    identities = [(row["task"], row["sample_id"], row["chunk_ms"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Stage A diagnosis parts overlap")
    rows.sort(key=lambda row: (str(row["task"]), str(row["sample_id"]), int(row["chunk_ms"])))
    payload = {
        "schema_version": "uniss_quality_first_stage_a_checkpoint_diagnosis_v1",
        "checkpoint": values[0]["checkpoint"],
        "hf_model": values[0]["hf_model"],
        "valid_packs": values[0]["valid_packs"],
        "worker_index": "merged",
        "num_workers": values[0]["num_workers"],
        "parts": [str(path.resolve()) for path in args.parts],
        "summary": summarize_rows(rows),
        "samples": rows,
    }
    atomic_json(args.output_json.resolve(), payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
