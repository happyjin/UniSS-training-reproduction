#!/usr/bin/env python3
"""Merge disjoint cached-runtime Stage A v2 diagnosis workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.evaluate_checkpoint import (
    atomic_json,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.evaluate_checkpoint import (
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
        raise FileExistsError("refusing to overwrite merged Stage A v2 diagnosis")
    values = [json.loads(path.read_text(encoding="utf-8")) for path in args.parts]
    schema = "uniss_quality_first_stage_a_checkpoint_diagnosis_v2"
    if any(value.get("schema_version") != schema for value in values):
        raise ValueError("unexpected Stage A v2 diagnosis part schema")
    common = ("checkpoint", "hf_model", "valid_packs", "num_workers", "max_acoustics_per_pack", "runtime")
    for key in common:
        if len({json.dumps(value[key], sort_keys=True) for value in values}) != 1:
            raise ValueError(f"Stage A v2 diagnosis parts disagree on {key}")
    workers = sorted(int(value["worker_index"]) for value in values)
    expected = list(range(int(values[0]["num_workers"])))
    if workers != expected:
        raise ValueError(f"Stage A v2 worker coverage differs: {workers} vs {expected}")
    rows = [row for value in values for row in value["samples"]]
    identities = [(row["task"], row["sample_id"], row["chunk_ms"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Stage A v2 diagnosis parts overlap")
    rows.sort(key=lambda row: (str(row["task"]), str(row["sample_id"])))
    payload = {
        "schema_version": schema,
        "checkpoint": values[0]["checkpoint"],
        "hf_model": values[0]["hf_model"],
        "valid_packs": values[0]["valid_packs"],
        "worker_index": "merged",
        "num_workers": values[0]["num_workers"],
        "max_acoustics_per_pack": values[0]["max_acoustics_per_pack"],
        "runtime": values[0]["runtime"],
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
