#!/usr/bin/env python3
"""Build a paired Stage-A-relative comparison across completed arms."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Mapping


def _mean(values) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return statistics.fmean(numbers) if numbers else None


def _mt(metrics: Mapping[str, object], path: str) -> dict[str, float | None]:
    value = metrics["e_mt"][path]  # type: ignore[index]
    directions = value["directions"]  # type: ignore[index]
    return {
        "bleu": _mean(row["candidate_bleu"] for row in directions.values()),
        "chrf": _mean(row["candidate_chrf"] for row in directions.values()),
        "coverage_min": float(value["target_coverage_min"]),
        "coverage_mean": float(value["target_coverage_mean"]),
        "rollback_events": float(value["target_rollback_events"]),
        "commit_conflicts": float(value["commit_conflicts"]),
        "unterminated_generations": float(value["unterminated_generations"]),
    }


def _arm(summary: Mapping[str, object]) -> dict[str, object]:
    candidate = summary["candidate"]
    stage_a = summary["stage_a"]
    s2s = candidate["e_s2s_free"]  # type: ignore[index]
    base_s2s = stage_a["e_s2s_free"]  # type: ignore[index]
    count = int(s2s.get("samples", 0))
    base_count = int(base_s2s.get("samples", 0))
    latency = candidate["latency"]  # type: ignore[index]
    base_latency = stage_a["latency"]  # type: ignore[index]
    current_gold = _mt(candidate, "gold_source")  # type: ignore[arg-type]
    current_free = _mt(candidate, "free_running_source")  # type: ignore[arg-type]
    base_gold = _mt(stage_a, "gold_source")  # type: ignore[arg-type]
    base_free = _mt(stage_a, "free_running_source")  # type: ignore[arg-type]

    quality_pairs = [
        (current_gold["bleu"], base_gold["bleu"]),
        (current_gold["chrf"], base_gold["chrf"]),
        (current_free["bleu"], base_free["bleu"]),
        (current_free["chrf"], base_free["chrf"]),
        (s2s.get("semantic_coverage_mean"), base_s2s.get("semantic_coverage_mean")),
    ]
    ratios = [
        float(current) / max(float(base), 1e-9)
        for current, base in quality_pairs
        if current is not None and base is not None
    ]
    non_silent_rate = (
        float(s2s.get("non_silent_pcm", 0)) / count if count else 0.0
    )
    structure_errors = (
        int(s2s.get("malformed_segments", 0))
        + int(s2s.get("invalid_semantic_tokens", 0))
        + int(current_gold["rollback_events"] or 0)
        + int(current_free["rollback_events"] or 0)
    )
    first_semantic = latency["first_semantic_write_ms"]["p50"]  # type: ignore[index]
    base_first_semantic = base_latency["first_semantic_write_ms"]["p50"]  # type: ignore[index]
    return {
        "gold_source_mt": current_gold,
        "free_source_mt": current_free,
        "stage_a_gold_source_mt": base_gold,
        "stage_a_free_source_mt": base_free,
        "e_s2s": s2s,
        "stage_a_e_s2s": base_s2s,
        "latency": latency,
        "stage_a_latency": base_latency,
        "quality_retention_vs_stage_a_mean": _mean(ratios),
        "non_silent_rate": non_silent_rate,
        "pre_eos_text_rate": (
            float(s2s.get("target_text_before_source_eos", 0)) / count if count else 0.0
        ),
        "pre_eos_semantic_rate": (
            float(s2s.get("target_semantic_before_source_eos", 0)) / count
            if count
            else 0.0
        ),
        "structure_errors": structure_errors,
        "first_semantic_p50_delta_ms": (
            float(first_semantic) - float(base_first_semantic)
            if first_semantic is not None and base_first_semantic is not None
            else None
        ),
        "paired_e_s2s_samples": min(count, base_count),
    }


def ranking_key(value: Mapping[str, object]) -> tuple[float, ...]:
    first_delta = value.get("first_semantic_p50_delta_ms")
    return (
        -float(value["structure_errors"]),
        float(value["non_silent_rate"]),
        float(value["pre_eos_semantic_rate"]),
        float(value.get("quality_retention_vs_stage_a_mean") or 0.0),
        -float(first_delta) if first_delta is not None else -math.inf,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True, help="ARM_ID=SUMMARY.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-output", type=Path, required=True)
    args = parser.parse_args()
    arms: dict[str, object] = {}
    paths: dict[str, str] = {}
    for item in args.arm:
        if "=" not in item:
            raise ValueError("--arm must be ARM_ID=SUMMARY.json")
        arm, raw_path = item.split("=", 1)
        path = Path(raw_path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            raise ValueError(f"incomplete arm summary: {path}")
        arms[arm] = _arm(summary)
        paths[arm] = str(path.resolve())
    if len(arms) < 2:
        raise ValueError("comparison requires at least two arms")
    best = max(arms, key=lambda arm: ranking_key(arms[arm]))
    output = {
        "schema_version": "uniss_stagea_joint_grpo_arm_comparison_v1",
        "status": "complete",
        "selection_rule": (
            "quality-first lexicographic: fewer structure errors, non-silent rate, "
            "pre-EOS semantic rate, paired Stage-A quality retention, then first semantic latency"
        ),
        "best_arm": best,
        "arms": arms,
        "summary_paths": paths,
    }
    if args.output.exists() or args.best_output.exists():
        raise FileExistsError("refusing to overwrite arm comparison")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.best_output.write_text(best + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

