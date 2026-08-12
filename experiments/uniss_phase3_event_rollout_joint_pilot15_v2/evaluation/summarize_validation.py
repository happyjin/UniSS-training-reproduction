#!/usr/bin/env python3
"""Extract auditable fixed15 validation checkpoint candidates from Megatron logs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


VALIDATION = re.compile(r"validation loss at iteration\s+(\d+)\s*\|(.*)")
VALUE = re.compile(r"(?:^|\|)\s*([A-Za-z0-9_]+) value:\s*([^ |]+)")
NAN_COUNT = re.compile(r"number of nan iterations:\s*(\d+)")
SKIP_COUNT = re.compile(r"number of skipped iterations:\s*(\d+)")


def _number(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"non-finite validation metric: {raw}")
    return value


def parse_log(log_path: Path, checkpoint_root: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        match = VALIDATION.search(line)
        if match is None:
            continue
        iteration = int(match.group(1))
        metrics = {name: _number(raw) for name, raw in VALUE.findall(match.group(2))}
        if not metrics:
            raise ValueError(f"validation row at iteration {iteration} has no metrics")
        checkpoint = checkpoint_root / f"iter_{iteration:07d}"
        rows.append(
            {
                "iteration": iteration,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_exists": (checkpoint / ".metadata").is_file(),
                "metrics": metrics,
            }
        )
    if not rows:
        raise ValueError(f"no Megatron validation rows found in {log_path}")
    duplicate_iterations = sorted(
        iteration
        for iteration in {int(row["iteration"]) for row in rows}
        if sum(int(row["iteration"]) == iteration for row in rows) > 1
    )
    if duplicate_iterations:
        raise ValueError(f"duplicate validation iterations: {duplicate_iterations}")
    nan_counts = [int(value) for value in NAN_COUNT.findall(text)]
    skip_counts = [int(value) for value in SKIP_COUNT.findall(text)]
    return {
        "schema_version": "uniss_event_rollout_fixed15_validation_summary_v1",
        "log": str(log_path.resolve()),
        "checkpoint_root": str(checkpoint_root.resolve()),
        "maximum_nan_iterations": max(nan_counts, default=0),
        "maximum_skipped_iterations": max(skip_counts, default=0),
        "checkpoints": rows,
        "selection_status": "exact_runtime_evaluation_required",
        "selection_rule": (
            "Do not select the last checkpoint or the lowest teacher-forced loss alone. "
            "Shortlist finite validation checkpoints, then select by natural exact-runtime "
            "WRITE, useful-audio latency/quality, EOS, collapse, and Phase3 retention."
        ),
    }


def _fmt(value: object) -> str:
    if value is None:
        return "not_evaluable"
    return f"{float(value):.6g}"


def markdown(summary: dict[str, object]) -> str:
    columns = (
        "interleaved_trajectory",
        "natural_write_fraction",
        "predicted_write_fraction",
        "deadline_forced_fraction",
        "safe_commit_f1",
        "runtime_action_accuracy",
        "runtime_text_token_accuracy",
        "microblock_token_accuracy",
        "runtime_eos_recall",
        "frontend_residual_rms",
    )
    lines = [
        "# Fixed15 V2 validation checkpoint summary",
        "",
        f"- Log: `{summary['log']}`",
        f"- Maximum NaN iterations: {summary['maximum_nan_iterations']}",
        f"- Maximum skipped iterations: {summary['maximum_skipped_iterations']}",
        "- Selection status: `exact_runtime_evaluation_required`",
        "- These are teacher-forced validation diagnostics, not proof of useful audio or subsecond latency.",
        "",
        "| iteration | checkpoint | " + " | ".join(columns) + " |",
        "|---:|:---:|" + "---:|" * len(columns),
    ]
    for row in summary["checkpoints"]:
        metrics = row["metrics"]
        checkpoint_mark = "yes" if row["checkpoint_exists"] else "pending"
        values = [_fmt(metrics.get(column)) for column in columns]
        lines.append(
            f"| {row['iteration']} | {checkpoint_mark} | " + " | ".join(values) + " |"
        )
    lines.extend(
        [
            "",
            "## Selection rule",
            "",
            str(summary["selection_rule"]),
            "",
            "A final best checkpoint must remain `not_selected` until exact-runtime train and validation evaluation verifies natural WRITE, no forced WRITE, valid translated PCM, useful-audio latency, EOS, collapse rate, and Phase3 retention.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    summary = parse_log(args.log, args.checkpoint_root)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()

