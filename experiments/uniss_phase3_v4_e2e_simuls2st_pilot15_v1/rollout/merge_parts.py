#!/usr/bin/env python3
"""Verify and merge contiguous V1 rollout worker parts without reserialization."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from array import array
from collections import Counter
from pathlib import Path

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.io import (
    atomic_json,
    file_sha256,
)
from training.simul_uniss.jsonl_index import load_index, write_index


MERGE_SCHEMA = "uniss_phase3_v4_e2e_simuls2st_v1_rollout_merge_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def load_reports(paths: list[Path]) -> list[dict[str, object]]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reports.sort(key=lambda value: int(value["global_start"]))
    if not reports:
        raise ValueError("rollout merge has no worker reports")
    invariant_keys = (
        "schema_version",
        "num_workers",
        "global_selected_records",
        "selection_start",
        "selection_stop",
        "input",
        "input_size_bytes",
        "checkpoint",
        "v1_checkpoint_sha256",
        "hf_model",
        "v1_hf_sha256",
        "runtime_sha256",
        "max_event_tokens",
        "max_final_tokens",
    )
    reference = reports[0]
    for report in reports:
        if report.get("status") != "complete":
            raise ValueError("rollout worker report is not complete")
        for key in invariant_keys:
            if report.get(key) != reference.get(key):
                raise ValueError(f"rollout worker invariant differs: {key}")
    if len(reports) != int(reference["num_workers"]):
        raise ValueError("rollout merge does not have every worker report")
    cursor = int(reference["selection_start"])
    for expected_worker, report in enumerate(reports):
        if int(report["worker_index"]) != expected_worker:
            raise ValueError("rollout worker indices are not contiguous")
        if int(report["global_start"]) != cursor:
            raise ValueError("rollout worker ranges contain a gap or overlap")
        cursor = int(report["global_stop"])
    if cursor != int(reference["selection_stop"]):
        raise ValueError("rollout worker ranges do not cover the full selection")
    return reports


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite merged V1 rollout")
    reports = load_reports(args.part_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged = array("Q")
    byte_base = 0
    counts: Counter[str] = Counter()
    weighted_errors: Counter[str] = Counter()
    with args.output.open("xb") as destination:
        for report in reports:
            output = report["output"]
            if not isinstance(output, dict):
                raise ValueError("rollout worker output metadata is malformed")
            path = Path(str(output["path"]))
            if path.stat().st_size != int(output["bytes"]):
                raise ValueError(f"rollout part byte count changed: {path}")
            if file_sha256(path) != output["sha256"]:
                raise ValueError(f"rollout part digest changed: {path}")
            offsets = load_index(path)
            if offsets is None or len(offsets) != int(output["records"]):
                raise ValueError(f"rollout part index is absent or malformed: {path}")
            merged.extend(byte_base + int(value) for value in offsets)
            with path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
            byte_base += int(output["bytes"])
            counts.update({str(key): int(value) for key, value in report["counts"].items()})
            weighted_errors.update(
                {str(key): int(value) for key, value in report["weighted_errors"].items()}
            )
        destination.flush()
        os.fsync(destination.fileno())
    index = write_index(args.output, merged)
    first = reports[0]
    report = {
        "schema_version": MERGE_SCHEMA,
        "status": "complete",
        "input": first["input"],
        "input_size_bytes": first["input_size_bytes"],
        "selection_start": first["selection_start"],
        "selection_stop": first["selection_stop"],
        "checkpoint": first["checkpoint"],
        "v1_checkpoint_sha256": first["v1_checkpoint_sha256"],
        "hf_model": first["hf_model"],
        "v1_hf_sha256": first["v1_hf_sha256"],
        "runtime_sha256": first["runtime_sha256"],
        "max_event_tokens": first["max_event_tokens"],
        "max_final_tokens": first["max_final_tokens"],
        "workers": len(reports),
        "worker_reports": [str(path.resolve()) for path in args.part_report],
        "counts": dict(sorted(counts.items())),
        "weighted_errors": dict(sorted(weighted_errors.items())),
        "elapsed_worker_seconds": sum(float(value["elapsed_seconds"]) for value in reports),
        "maximum_worker_seconds": max(float(value["elapsed_seconds"]) for value in reports),
        "output": {
            "path": str(args.output.resolve()),
            "records": len(merged),
            "bytes": args.output.stat().st_size,
            "sha256": file_sha256(args.output),
        },
        "index": index,
    }
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
