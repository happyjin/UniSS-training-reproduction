#!/usr/bin/env python3
"""Summarize steady-state step time and GPU telemetry for the speed test."""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import statistics
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
METRIC_RE = re.compile(r"\{[^\n]*'global_step/max_steps':\s*'[0-9]+/[0-9]+'[^\n]*\}")


def elapsed_seconds(value: str) -> float:
    match = re.fullmatch(
        r"(?:(\d+)d )?(?:(\d+)h )?(?:(\d+)m )?([0-9]+(?:\.[0-9]+)?)s", value
    )
    if not match:
        raise ValueError(f"cannot parse elapsed_time={value!r}")
    days, hours, minutes, seconds = match.groups()
    return (
        int(days or 0) * 86400
        + int(hours or 0) * 3600
        + int(minutes or 0) * 60
        + float(seconds)
    )


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def parse_training(log_path: Path, start_step: int, end_step: int) -> dict:
    text = ANSI_RE.sub("", log_path.read_text(errors="replace")).replace("\r", "\n")
    metrics: dict[int, dict] = {}
    for match in METRIC_RE.finditer(text):
        try:
            item = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            continue
        step = int(str(item["global_step/max_steps"]).split("/", 1)[0])
        if "elapsed_time" in item:
            metrics[step] = item
    selected = [metrics[step] for step in sorted(metrics) if start_step <= step <= end_step]
    if len(selected) < 2:
        raise RuntimeError(f"not enough metric points in steps {start_step}..{end_step}")

    intervals: list[float] = []
    for previous, current in zip(selected, selected[1:]):
        previous_step = int(previous["global_step/max_steps"].split("/", 1)[0])
        current_step = int(current["global_step/max_steps"].split("/", 1)[0])
        seconds = elapsed_seconds(current["elapsed_time"]) - elapsed_seconds(previous["elapsed_time"])
        intervals.append(seconds / (current_step - previous_step))

    first_step = int(selected[0]["global_step/max_steps"].split("/", 1)[0])
    last_step = int(selected[-1]["global_step/max_steps"].split("/", 1)[0])
    window_seconds = elapsed_seconds(selected[-1]["elapsed_time"]) - elapsed_seconds(
        selected[0]["elapsed_time"]
    )
    mean_step_seconds = window_seconds / (last_step - first_step)
    return {
        "window_start_step": first_step,
        "window_end_step": last_step,
        "window_seconds": window_seconds,
        "mean_step_seconds": mean_step_seconds,
        "median_interval_step_seconds": statistics.median(intervals),
        "p95_interval_step_seconds": percentile(intervals, 0.95),
        "effective_samples_per_second": 512 / mean_step_seconds,
        "last_logged_step": max(metrics),
        "cumulative_train_speed_seconds_per_step": metrics[max(metrics)].get(
            "train_speed(s/it)"
        ),
    }


def parse_gpu(csv_path: Path, start_step: int, end_step: int) -> dict:
    rows: list[dict] = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(row["global_step"])
            if start_step <= step <= end_step:
                rows.append(row)
    if not rows:
        return {"samples": 0}
    memory = [float(row["memory_used_mib"].strip()) for row in rows]
    utilization = [float(row["utilization_percent"].strip()) for row in rows]
    power = [float(row["power_watts"].strip()) for row in rows]
    return {
        "samples": len(rows),
        "polls": len(rows) // 8,
        "mean_gpu_utilization_percent": statistics.fmean(utilization),
        "median_gpu_utilization_percent": statistics.median(utilization),
        "peak_memory_mib": max(memory),
        "mean_memory_mib": statistics.fmean(memory),
        "mean_power_watts": statistics.fmean(power),
        "peak_power_watts": max(power),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--gpu-csv", type=Path, required=True)
    parser.add_argument("--start-step", type=int, default=200)
    parser.add_argument("--end-step", type=int, default=900)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = {
        "steady_state": parse_training(args.training_log, args.start_step, args.end_step),
        "gpu": parse_gpu(args.gpu_csv, args.start_step, args.end_step),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
