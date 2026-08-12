#!/usr/bin/env python3
"""Shortlist finite validation checkpoints without claiming a final winner."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA = "uniss_event_rollout_fixed15_checkpoint_shortlist_v1"
OBJECTIVES = (
    ("interleaved_trajectory", "min"),
    ("runtime_text_token_accuracy", "max"),
    ("runtime_action_accuracy", "max"),
    ("runtime_eos_recall", "max"),
    ("safe_commit_f1", "max"),
)


def _finite(metrics: Mapping[str, object], name: str) -> bool:
    try:
        return math.isfinite(float(metrics[name]))
    except (KeyError, TypeError, ValueError):
        return False


def shortlist(summary: Mapping[str, object], *, maximum_candidates: int) -> dict[str, object]:
    if maximum_candidates <= 0:
        raise ValueError("maximum_candidates must be positive")
    eligible = []
    rejected = []
    required = {name for name, _ in OBJECTIVES} | {
        "natural_write_fraction",
        "deadline_forced_fraction",
        "frontend_residual_rms",
    }
    for raw in summary.get("checkpoints", []):
        row = dict(raw)
        metrics = dict(row.get("metrics", {}))
        reasons = []
        if not bool(row.get("checkpoint_exists")):
            reasons.append("checkpoint_missing")
        missing = sorted(name for name in required if not _finite(metrics, name))
        if missing:
            reasons.append(f"missing_or_nonfinite:{','.join(missing)}")
        if not missing:
            if float(metrics["deadline_forced_fraction"]) != 0.0:
                reasons.append("forced_write_nonzero")
            if float(metrics["natural_write_fraction"]) <= 0.0:
                reasons.append("natural_write_nonpositive")
            if float(metrics["frontend_residual_rms"]) <= 0.0:
                reasons.append("causal_frontend_inactive")
        if reasons:
            rejected.append({"iteration": row.get("iteration"), "reasons": reasons})
        else:
            eligible.append(row)
    if not eligible:
        raise ValueError("no validation checkpoint passes the mechanism prefilter")

    ranks: dict[int, dict[str, int]] = {int(row["iteration"]): {} for row in eligible}
    for metric, direction in OBJECTIVES:
        ordered = sorted(
            eligible,
            key=lambda row: float(row["metrics"][metric]),
            reverse=direction == "max",
        )
        for rank, row in enumerate(ordered, start=1):
            ranks[int(row["iteration"])][metric] = rank

    scored = []
    for row in eligible:
        iteration = int(row["iteration"])
        metric_ranks = ranks[iteration]
        scored.append(
            {
                **row,
                "objective_ranks": metric_ranks,
                "rank_sum": sum(metric_ranks.values()),
                "selection_status": "candidate_only_exact_runtime_required",
            }
        )
    scored.sort(key=lambda row: (int(row["rank_sum"]), -int(row["iteration"])))
    candidates = scored[:maximum_candidates]
    return {
        "schema_version": SCHEMA,
        "maximum_candidates": maximum_candidates,
        "objectives": [
            {"metric": name, "direction": direction} for name, direction in OBJECTIVES
        ],
        "prefilter": {
            "checkpoint_exists": True,
            "all_objectives_finite": True,
            "deadline_forced_fraction": 0.0,
            "natural_write_fraction": ">0",
            "frontend_residual_rms": ">0",
        },
        "eligible_count": len(eligible),
        "rejected": rejected,
        "candidates": candidates,
        "final_selection_status": "not_selected",
        "final_selection_rule": (
            "Teacher-forced rank sum only creates a probe shortlist. Final selection requires "
            "natural exact-runtime WRITE, target-language useful-audio ASR, p50/p90/p95 latency, "
            "valid PCM, EOS, collapse, quality metrics, runtime parity and Phase3 replay retention."
        ),
    }


def markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Fixed15 exact-runtime checkpoint shortlist",
        "",
        "- Final checkpoint: `not_selected`",
        "- This table is a probe shortlist, not a best-checkpoint claim.",
        "",
        "| order | iteration | rank sum | trajectory loss | text acc | action acc | EOS recall | safe-commit F1 | frontend RMS |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for order, row in enumerate(report["candidates"], start=1):
        metrics = row["metrics"]
        lines.append(
            f"| {order} | {row['iteration']} | {row['rank_sum']} | "
            f"{float(metrics['interleaved_trajectory']):.6f} | "
            f"{float(metrics['runtime_text_token_accuracy']):.6f} | "
            f"{float(metrics['runtime_action_accuracy']):.6f} | "
            f"{float(metrics['runtime_eos_recall']):.6f} | "
            f"{float(metrics['safe_commit_f1']):.6f} | "
            f"{float(metrics['frontend_residual_rms']):.6f} |"
        )
    lines.extend(["", str(report["final_selection_rule"]), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--maximum-candidates", type=int, default=3)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.json.exists() or args.markdown.exists():
        raise FileExistsError("refusing to overwrite checkpoint shortlist")
    summary = json.loads(args.validation_summary.read_text(encoding="utf-8"))
    report = shortlist(summary, maximum_candidates=args.maximum_candidates)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"candidates": [row["iteration"] for row in report["candidates"]]}))


if __name__ == "__main__":
    main()
