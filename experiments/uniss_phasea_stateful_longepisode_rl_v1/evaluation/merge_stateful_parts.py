#!/usr/bin/env python3
"""Merge four isolated stateful long-audio worker reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = sorted(args.parts_root.glob("*/results.json"))
    if len(reports) != 4:
        raise ValueError(f"expected four worker reports, found {len(reports)}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    if any(value.get("status") != "complete" or len(value.get("results", [])) != 1 for value in payloads):
        raise ValueError("one or more worker reports are incomplete")
    results = [value["results"][0] for value in payloads]
    merged = {
        "schema_version": "uniss_phasea_stateful_longepisode_runtime_v2_merged",
        "status": "complete",
        "run_id": args.run_id,
        "runtime_mode": "full-session stateful causal frontend with bounded LLM acoustic ring",
        "decision_chunk_ms": payloads[0]["decision_chunk_ms"],
        "acoustic_rollover_ms": payloads[0]["acoustic_rollover_ms"],
        "adapter_manifest": payloads[0]["adapter_manifest"],
        "results": results,
        "aggregate": {
            "samples": len(results),
            "passed": sum(bool(value["stateful_runtime_passed"]) for value in results),
            "audio_writes": sum(int(value["audio_writes"]) for value in results),
            "pending_unspoken_items": sum(int(value["tts_pending_unspoken_items"]) for value in results),
            "tts_failures": sum(int(value["tts_failures"]) for value in results),
            "rejected_early_end": sum(int(value["rejected_early_end"]) for value in results),
            "semantic_continuations": sum(int(value["semantic_continuations"]) for value in results),
            "mean_rtf": sum(float(value["rtf"]) for value in results) / len(results),
        },
        "part_reports": [str(path.resolve()) for path in reports],
    }
    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(merged["aggregate"], ensure_ascii=False, sort_keys=True))
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()

