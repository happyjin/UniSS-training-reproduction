#!/usr/bin/env python3
"""Merge an arbitrary number of isolated Runtime-v2 sample outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    expected = [str(row["sample_id"]) for row in protocol["records"]]
    reports = [args.parts_root / sample_id / "results.json" for sample_id in expected]
    missing = [str(path) for path in reports if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing sample reports: {missing}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    if any(
        value.get("status") != "complete" or len(value.get("results", [])) != 1
        for value in payloads
    ):
        raise ValueError("one or more sample reports are incomplete")
    results = [value["results"][0] for value in payloads]
    actual = [str(row["sample_id"]) for row in results]
    if actual != expected:
        raise ValueError(f"merged sample order/IDs differ: {actual} != {expected}")
    manifests = [value.get("adapter_manifest") for value in payloads]
    if any(value != manifests[0] for value in manifests[1:]):
        raise ValueError("sample workers loaded different model manifests")
    merged = {
        "schema_version": "uniss_phasea_rl_train_seen_runtime_v2_merged_v1",
        "status": "complete",
        "run_id": args.run_id,
        "decision_chunk_ms": payloads[0]["decision_chunk_ms"],
        "acoustic_rollover_ms": payloads[0]["acoustic_rollover_ms"],
        "adapter_manifest": manifests[0],
        "protocol": str(args.protocol.resolve()),
        "results": results,
        "part_reports": [str(path.resolve()) for path in reports],
    }
    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
