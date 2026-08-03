#!/usr/bin/env python3
"""Merge disjoint Stage05 policy shards into one auditable report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_real_policy import render_markdown, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.parts]
    if not payloads:
        raise ValueError("no reports supplied")
    rows = [row for payload in payloads for row in payload["samples"]]
    rows.sort(key=lambda row: (row["direction"], row["id"]))
    first = payloads[0]
    merged = {
        "schema_version": "uniss_streamspeech_stage05_real_ctc_policy_v1",
        "checkpoint": first["checkpoint"],
        "split": first["split"],
        "confirmations": first["confirmations"],
        "lagging_k": first["lagging_k"],
        "encoder_segment_ms": first["encoder_segment_ms"],
        "encoder_right_context_ms": first["encoder_right_context_ms"],
        "parts": [str(path) for path in args.parts],
        "summary": summarize(rows),
        "samples": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(merged), encoding="utf-8")
    print(json.dumps(merged["summary"], sort_keys=True))


if __name__ == "__main__":
    main()

