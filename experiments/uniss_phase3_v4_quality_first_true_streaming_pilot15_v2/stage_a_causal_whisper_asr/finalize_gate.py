#!/usr/bin/env python3
"""Finalize Stage A v2 from exact matching content and runtime artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.evaluate_checkpoint import (
    atomic_json,
)


GROUPS = tuple(
    f"{task}:{language}"
    for task in ("streaming_asr", "causal_full_asr")
    for language in ("cmn", "eng")
)


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    errors = sum(int(row["ar_free_running"]["errors"]) for row in values)
    units = sum(int(row["ar_free_running"]["reference_units"]) for row in values)
    return {
        "samples": len(values),
        "errors": errors,
        "reference_units": units,
        "error_rate": errors / max(1, units),
        "empty_rows": sum(not str(row["ar_free_running"]["text"]) for row in values),
        "event_stop_failed_rows": sum(
            not bool(row["ar_free_running"]["all_events_reached_stop"])
            for row in values
        ),
        "ctc_all_blank_rows": sum(
            int(row["ctc"]["collapsed_nonblank_tokens"]) == 0 for row in values
        ),
        "rollback_count": sum(
            int(row["committed_rollback"]["rollback_count"]) for row in values
        ),
    }


def streaming_health(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = [row for row in rows if row["task"] == "streaming_asr"]
    prefinal = 0
    final_only = 0
    for row in values:
        events = list(row["ar_free_running"]["events"])
        has_prefinal = any(event["content_tokens"] for event in events[:-1])
        has_final = bool(events and events[-1]["content_tokens"])
        prefinal += int(has_prefinal)
        final_only += int(len(events) > 1 and not has_prefinal and has_final)
    return {
        "rows": len(values),
        "prefinal_content_rows": prefinal,
        "prefinal_content_rate": prefinal / max(1, len(values)),
        "final_only_rows": final_only,
    }


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Stage A v2 final gate",
        "",
        f"- Decision: **{'PASS' if result['passed'] else 'FAIL'}**",
        f"- Checkpoint: `{result['checkpoint']}`",
        f"- Stage B authorized: **{result['stage_b_authorized']}**",
        f"- Failed checks: {', '.join(result['failed_checks']) if result['failed_checks'] else 'none'}",
        "",
        "| Matching subset | Phase3 offline | Maximum allowed | Stage A v2 | Decision |",
        "|---|---:|---:|---:|---|",
    ]
    for group in GROUPS:
        value = result["content_by_task_language"][group]
        lines.append(
            f"| {group} | {value['offline_error_rate']:.4%} | "
            f"{value['maximum_allowed_error_rate']:.4%} | "
            f"{value['stage_a_error_rate']:.4%} | "
            f"{'PASS' if value['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"- Exact sample coverage: {result['coverage']['unique_ids']}/334 unique IDs.",
            f"- Committed rollback count: {result['runtime_health']['rollback_count']}.",
            f"- Cached/recomputed parity rows: {result['runtime_health']['parity_rows_passed']}/{result['runtime_health']['rows']}.",
            f"- Empty rows: {result['runtime_health']['empty_rows']}.",
            f"- Event-stop failures: {result['runtime_health']['event_stop_failed_rows']}.",
            f"- CTC all-blank rows: {result['runtime_health']['ctc_all_blank_rows']}.",
            f"- Streaming pre-final content rate: {result['streaming_health']['prefinal_content_rate']:.4%}.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(
    diagnosis: dict[str, Any],
    offline: dict[str, Any],
    frontend_gate: dict[str, Any],
    *,
    relative_degradation_limit: float,
) -> dict[str, Any]:
    if diagnosis.get("schema_version") != "uniss_quality_first_stage_a_checkpoint_diagnosis_v2":
        raise ValueError("unexpected Stage A v2 diagnosis schema")
    if offline.get("schema_version") != "uniss_quality_first_stage_a_matching_offline_asr_v1":
        raise ValueError("unexpected matching offline summary schema")
    if frontend_gate.get("schema_version") != "uniss_quality_first_stage_a_checkpoint_frontend_passed_v2":
        raise ValueError("unexpected checkpoint frontend gate schema")
    rows = list(diagnosis["samples"])
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[f"{row['task']}:{row['language']}"] .append(row)

    failed: list[str] = []
    content: dict[str, Any] = {}
    for group in GROUPS:
        current = aggregate(by_group[group])
        anchor = offline["metrics_by_task_language"][group]
        maximum = float(anchor["error_rate"]) * (1.0 + relative_degradation_limit)
        passed = (
            current["samples"] == int(anchor["samples"])
            and current["error_rate"] <= maximum
        )
        if not passed:
            failed.append(f"content_{group.replace(':', '_')}")
        content[group] = {
            "metric": anchor["metric"],
            "samples": current["samples"],
            "offline_samples": int(anchor["samples"]),
            "offline_error_rate": float(anchor["error_rate"]),
            "maximum_allowed_error_rate": maximum,
            "stage_a_error_rate": current["error_rate"],
            "passed": passed,
        }

    identities = [(str(row["task"]), str(row["sample_id"])) for row in rows]
    unique_ids = {sample_id for _, sample_id in identities}
    coverage_passed = (
        len(rows) == int(offline["records"]) == 334
        and len(identities) == len(set(identities))
        and len(unique_ids) == int(offline["unique_ids"]) == 334
    )
    if not coverage_passed:
        failed.append("exact_334_sample_coverage")

    overall = aggregate(rows)
    parity_rows = sum(
        bool(row["cached_recomputed_parity"]["hidden"]["allclose"])
        and bool(row["cached_recomputed_parity"]["tokens"]["exact"])
        and bool(row["cached_recomputed_parity"]["bridge_residual"]["allclose"])
        and bool(row["cached_recomputed_parity"]["free_generation_exact"])
        for row in rows
    )
    runtime_checks = {
        "checkpoint_frontend_gate_passed": bool(frontend_gate.get("passed")),
        "committed_rollback_zero": overall["rollback_count"] == 0,
        "cached_recomputed_all_rows": parity_rows == len(rows),
        "empty_rows_zero": overall["empty_rows"] == 0,
        "event_stop_failures_zero": overall["event_stop_failed_rows"] == 0,
        "ctc_all_blank_rows_zero": overall["ctc_all_blank_rows"] == 0,
    }
    failed.extend(name for name, passed in runtime_checks.items() if not passed)
    stream = streaming_health(rows)
    stream_checks = {
        "streaming_prefinal_content_all_rows": stream["prefinal_content_rows"] == stream["rows"],
        "streaming_final_only_rows_zero": stream["final_only_rows"] == 0,
    }
    failed.extend(name for name, passed in stream_checks.items() if not passed)
    failed = list(dict.fromkeys(failed))
    passed = not failed
    return {
        "schema_version": "uniss_quality_first_stage_a_final_gate_v2",
        "passed": passed,
        "checkpoint": diagnosis["checkpoint"],
        "relative_degradation_limit": relative_degradation_limit,
        "content_by_task_language": content,
        "coverage": {
            "rows": len(rows),
            "unique_ids": len(unique_ids),
            "passed": coverage_passed,
        },
        "runtime_health": {
            "rows": len(rows),
            "rollback_count": overall["rollback_count"],
            "parity_rows_passed": parity_rows,
            "empty_rows": overall["empty_rows"],
            "event_stop_failed_rows": overall["event_stop_failed_rows"],
            "ctc_all_blank_rows": overall["ctc_all_blank_rows"],
            "checks": runtime_checks,
        },
        "streaming_health": {**stream, "checks": stream_checks},
        "failed_checks": failed,
        "stage_b_authorized": passed,
        "blocked_next_stage": None if passed else "stage_b_incremental_mt",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--matching-offline-summary", type=Path, required=True)
    parser.add_argument("--frontend-gate", type=Path, required=True)
    parser.add_argument("--relative-degradation-limit", type=float, default=0.15)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--selection-artifact", type=Path, required=True)
    args = parser.parse_args()
    if any(path.exists() for path in (args.output_json, args.output_md, args.selection_artifact)):
        raise FileExistsError("refusing to overwrite Stage A v2 final gate")
    result = finalize(
        json.loads(args.diagnosis.read_text(encoding="utf-8")),
        json.loads(args.matching_offline_summary.read_text(encoding="utf-8")),
        json.loads(args.frontend_gate.read_text(encoding="utf-8")),
        relative_degradation_limit=args.relative_degradation_limit,
    )
    atomic_json(args.output_json.resolve(), result)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_report(result), encoding="utf-8")
    if result["passed"]:
        atomic_json(
            args.selection_artifact.resolve(),
            {
                "schema_version": "uniss_quality_first_stage_a_selected_v2",
                "passed": True,
                "checkpoint": result["checkpoint"],
                "gate": str(args.output_json.resolve()),
                "stage_b_authorized": True,
            },
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
