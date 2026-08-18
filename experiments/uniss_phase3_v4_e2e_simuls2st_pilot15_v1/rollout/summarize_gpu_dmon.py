#!/usr/bin/env python3
"""Summarize active GPU utilization/power snapshots from nvidia-smi dmon."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.io import atomic_json


GPU_SUMMARY_SCHEMA = "uniss_phase3_v4_e2e_simuls2st_gpu_dmon_summary_v1"


def parse_rows(path: Path) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0].startswith("#") or len(fields) < 18:
                continue
            try:
                rows.append(
                    {
                        "date": fields[0],
                        "time": fields[1],
                        "gpu": int(fields[2]),
                        "power_watts": int(fields[3]),
                        "sm_percent": int(fields[6]),
                        "memory_percent": int(fields[7]),
                        "framebuffer_mib": int(fields[16]),
                    }
                )
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"GPU dmon log has no parseable samples: {path}")
    return rows


def summarize(rows: list[dict[str, int | str]], minimum_active_memory_mib: int) -> dict[str, object]:
    if minimum_active_memory_mib < 0:
        raise ValueError("minimum active GPU memory cannot be negative")
    grouped: dict[int, list[dict[str, int | str]]] = defaultdict(list)
    for row in rows:
        if int(row["framebuffer_mib"]) >= minimum_active_memory_mib:
            grouped[int(row["gpu"])].append(row)
    if not grouped:
        raise ValueError("GPU dmon log has no active samples")
    devices: dict[str, dict[str, float | int]] = {}
    all_active: list[dict[str, int | str]] = []
    for gpu, values in sorted(grouped.items()):
        all_active.extend(values)
        count = len(values)
        devices[str(gpu)] = {
            "active_samples": count,
            "mean_sm_percent": sum(int(value["sm_percent"]) for value in values) / count,
            "max_sm_percent": max(int(value["sm_percent"]) for value in values),
            "mean_power_watts": sum(int(value["power_watts"]) for value in values) / count,
            "max_power_watts": max(int(value["power_watts"]) for value in values),
            "mean_framebuffer_mib": sum(int(value["framebuffer_mib"]) for value in values) / count,
            "max_framebuffer_mib": max(int(value["framebuffer_mib"]) for value in values),
        }
    count = len(all_active)
    return {
        "active_samples": count,
        "mean_sm_percent": sum(int(value["sm_percent"]) for value in all_active) / count,
        "max_sm_percent": max(int(value["sm_percent"]) for value in all_active),
        "mean_power_watts": sum(int(value["power_watts"]) for value in all_active) / count,
        "max_power_watts": max(int(value["power_watts"]) for value in all_active),
        "mean_framebuffer_mib": sum(int(value["framebuffer_mib"]) for value in all_active) / count,
        "max_framebuffer_mib": max(int(value["framebuffer_mib"]) for value in all_active),
        "devices": devices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-active-memory-mib", type=int, default=512)
    args = parser.parse_args()
    rows = parse_rows(args.input)
    report = {
        "schema_version": GPU_SUMMARY_SCHEMA,
        "status": "complete",
        "input": str(args.input.resolve()),
        "minimum_active_memory_mib": args.minimum_active_memory_mib,
        "all_samples": len(rows),
        "active": summarize(rows, args.minimum_active_memory_mib),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
