#!/usr/bin/env python3
"""Finalize the Stage A quality gate from immutable evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable


LANGUAGE_TO_OFFLINE_DIRECTION = {"cmn": "cmn->eng", "eng": "eng->cmn"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Stage A gate output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def aggregate(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    selected = list(rows)
    errors = sum(int(row["ar_free_running"]["errors"]) for row in selected)
    reference_units = sum(
        int(row["ar_free_running"]["reference_units"]) for row in selected
    )
    teacher_correct = sum(
        int(row["ar_teacher_forced"]["correct_tokens"]) for row in selected
    )
    teacher_tokens = sum(
        int(row["ar_teacher_forced"]["target_tokens"]) for row in selected
    )
    input_frames = sum(int(row["ctc"]["input_frames"]) for row in selected)
    nonblank_frames = sum(int(row["ctc"]["raw_nonblank_frames"]) for row in selected)
    return {
        "evaluations": len(selected),
        "errors": errors,
        "reference_units": reference_units,
        "error_rate": errors / max(1, reference_units),
        "teacher_token_accuracy": teacher_correct / max(1, teacher_tokens),
        "ctc_blank_ratio": (input_frames - nonblank_frames) / max(1, input_frames),
        "ar_empty_rows": sum(
            not str(row["ar_free_running"]["text"]) for row in selected
        ),
        "ctc_all_blank_rows": sum(
            int(row["ctc"]["collapsed_nonblank_tokens"]) == 0 for row in selected
        ),
        "ar_event_stop_failed_rows": sum(
            not bool(row["ar_free_running"]["all_events_reached_stop"])
            for row in selected
        ),
    }


def streaming_event_health(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    selected = [row for row in rows if row["task"] == "streaming_asr"]
    prefinal_content = 0
    final_only = 0
    structure_sum = 0.0
    for row in selected:
        free = row["ar_free_running"]
        events = list(free["events"])
        has_prefinal = any(event["content_tokens"] for event in events[:-1])
        has_final = bool(events and events[-1]["content_tokens"])
        prefinal_content += has_prefinal
        final_only += bool(len(events) > 1 and not has_prefinal and has_final)
        structure_sum += float(free["write_structure_rate"])
    return {
        "evaluations": len(selected),
        "prefinal_content_rows": prefinal_content,
        "prefinal_content_rate": prefinal_content / max(1, len(selected)),
        "final_only_rows": final_only,
        "mean_write_structure_rate": structure_sum / max(1, len(selected)),
    }


def grouped(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    buckets: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["task"]), str(row["language"]), int(row["chunk_ms"]))].append(
            row
        )
    return {
        f"{task}:{language}:{chunk_ms}": aggregate(values)
        for (task, language, chunk_ms), values in sorted(buckets.items())
    }


def finalize(
    diagnosis: dict[str, object],
    baseline: dict[str, object],
    *,
    relative_degradation_limit: float,
) -> tuple[dict[str, object], dict[str, object]]:
    rows = list(diagnosis["samples"])
    by_group = grouped(rows)
    by_task_language: dict[str, dict[str, object]] = {}
    for task in ("causal_full_asr", "streaming_asr"):
        for language in ("cmn", "eng"):
            by_task_language[f"{task}:{language}"] = aggregate(
                row
                for row in rows
                if row["task"] == task and row["language"] == language
            )

    offline: dict[str, dict[str, object]] = {}
    for language, direction in LANGUAGE_TO_OFFLINE_DIRECTION.items():
        anchor = baseline["quality_asr_error"][direction]
        error_rate = float(anchor["error_rate"])
        offline[language] = {
            "direction": direction,
            "metric": str(anchor["metric"]).lower(),
            "samples": int(anchor["samples"]),
            "errors": int(anchor["edits"]),
            "reference_units": int(anchor["reference_units"]),
            "error_rate": error_rate,
            "maximum_allowed_error_rate": error_rate
            * (1.0 + relative_degradation_limit),
        }

    failed_checks: list[str] = []
    for language in ("cmn", "eng"):
        current = float(by_task_language[f"streaming_asr:{language}"]["error_rate"])
        if current > float(offline[language]["maximum_allowed_error_rate"]):
            failed_checks.append(
                f"streaming_{language}_error_exceeds_offline_plus_"
                f"{relative_degradation_limit:.0%}"
            )
    overall = aggregate(rows)
    if int(overall["ctc_all_blank_rows"]):
        failed_checks.append("ctc_sample_level_all_blank_rows_nonzero")
    if int(overall["ar_event_stop_failed_rows"]):
        failed_checks.append("ar_event_stop_failures_nonzero")
    if int(overall["ar_empty_rows"]):
        failed_checks.append("ar_empty_rows_nonzero")

    # The Stage00 anchor is a fixed pilot15 set, not the exact 334 Stage A rows.
    # A strict matching-sample offline comparison and checkpoint-level cached
    # rollback/parity audit remain mandatory even if content metrics improve.
    failed_checks.extend(
        [
            "matching_sample_offline_reference_not_evaluated",
            "checkpoint_level_cached_rollback_not_evaluated",
            "checkpoint_level_cached_full_parity_not_evaluated",
        ]
    )

    summary = {
        "schema_version": "uniss_quality_first_stage_a_final_summary_v1",
        "diagnosis_checkpoint": diagnosis["checkpoint"],
        "diagnosis_summary": diagnosis["summary"],
        "overall": overall,
        "streaming_event_health": streaming_event_health(rows),
        "by_task_language": by_task_language,
        "by_task_language_chunk": by_group,
        "offline_quality_asr_anchor": offline,
        "relative_degradation_limit": relative_degradation_limit,
        "validation_protocol_note": (
            "Stage00 offline anchors and Stage A formal diagnosis are fixed pilot15 "
            "sets but do not contain identical sample IDs; the large gap is diagnostic, "
            "while a strict pass still requires a matching-sample offline rerun."
        ),
    }
    gate = {
        "schema_version": "uniss_quality_first_stage_gate_v1",
        "stage": "stage_a",
        "passed": False,
        "checkpoint": diagnosis["checkpoint"],
        "metrics": {
            "streaming_error_rate_cmn": by_task_language["streaming_asr:cmn"][
                "error_rate"
            ],
            "streaming_error_rate_eng": by_task_language["streaming_asr:eng"][
                "error_rate"
            ],
            "causal_full_error_rate_cmn": by_task_language["causal_full_asr:cmn"][
                "error_rate"
            ],
            "causal_full_error_rate_eng": by_task_language["causal_full_asr:eng"][
                "error_rate"
            ],
            "teacher_token_accuracy": overall["teacher_token_accuracy"],
            "ctc_blank_ratio": overall["ctc_blank_ratio"],
            "ctc_all_blank_rows": overall["ctc_all_blank_rows"],
            "ar_event_stop_failed_rows": overall["ar_event_stop_failed_rows"],
            "ar_empty_rows": overall["ar_empty_rows"],
            "prefinal_content_rate": summary["streaming_event_health"][
                "prefinal_content_rate"
            ],
            "final_only_rows": summary["streaming_event_health"]["final_only_rows"],
        },
        "failed_checks": failed_checks,
        "blocked_next_stage": "stage_b_incremental_mt",
        "selection_artifact_created": False,
    }
    return summary, gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--offline-baseline", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-commit", required=True)
    parser.add_argument("--created-at-utc", required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-gate", type=Path, required=True)
    parser.add_argument("--relative-degradation-limit", type=float, default=0.15)
    args = parser.parse_args()
    if not 0.0 <= args.relative_degradation_limit < 1.0:
        raise ValueError("relative degradation limit must be in [0, 1)")
    diagnosis = json.loads(args.diagnosis.read_text(encoding="utf-8"))
    baseline = json.loads(args.offline_baseline.read_text(encoding="utf-8"))
    summary, gate = finalize(
        diagnosis,
        baseline,
        relative_degradation_limit=args.relative_degradation_limit,
    )
    common = {
        "created_at_utc": args.created_at_utc,
        "diagnosis": str(args.diagnosis.resolve()),
        "diagnosis_sha256": sha256(args.diagnosis),
        "offline_baseline": str(args.offline_baseline.resolve()),
        "offline_baseline_sha256": sha256(args.offline_baseline),
        "data_manifest": str(args.data_manifest.resolve()),
        "data_manifest_sha256": sha256(args.data_manifest),
        "evaluator_commit": args.evaluator_commit,
    }
    summary.update(common)
    gate.update(common)
    atomic_json(args.output_summary.resolve(), summary)
    atomic_json(args.output_gate.resolve(), gate)
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
