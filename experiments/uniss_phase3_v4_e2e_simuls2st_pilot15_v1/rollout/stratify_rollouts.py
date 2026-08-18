#!/usr/bin/env python3
"""Strictly audit and stratify immutable V1 free-running rollouts.

Content errors are retained as realistic noisy-prefix supervision.  Protocol
errors are quarantined so they cannot teach streaming ASR, V1-history MT, or
interleaved E2E control behavior.  Every rollout is represented exactly once
in the output manifest, making the filtering decision independently auditable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.audit_rollouts import (
    _audit_pair,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.io import (
    IndexedJSONLWriter,
    atomic_json,
    file_sha256,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.schema import (
    V1Rollout,
)
from training.simul_uniss.jsonl_index import load_index, write_index


STRATA_SCHEMA = "uniss_phase3_v4_e2e_simuls2st_rollout_stratum_v1"
PART_SCHEMA = "uniss_phase3_v4_e2e_simuls2st_rollout_strata_part_v1"
QUALITY_GATE_SCHEMA = "uniss_phase3_v4_e2e_simuls2st_rollout_quality_gate_v1"

STRATUM_CLEAN = "clean"
STRATUM_NOISY = "noisy_content"
STRATUM_QUARANTINE = "quarantine"
STRATA = (STRATUM_CLEAN, STRATUM_NOISY, STRATUM_QUARANTINE)


def clean_threshold(language: str, *, english_wer: float, chinese_cer: float) -> float:
    return float(chinese_cer if language == "cmn" else english_wer)


def classify_rollout(
    rollout: V1Rollout,
    *,
    english_clean_wer: float,
    chinese_clean_cer: float,
) -> tuple[str, tuple[str, ...]]:
    """Return a mutually exclusive quality stratum and explicit reasons."""

    reasons: list[str] = []
    if rollout.malformed_write_events:
        reasons.append("malformed_write")
    if rollout.early_eos_events:
        reasons.append("early_eos")
    if not rollout.final_reached_eos:
        reasons.append("missing_final_eos")
    if reasons:
        return STRATUM_QUARANTINE, tuple(reasons)
    threshold = clean_threshold(
        rollout.src_lang,
        english_wer=english_clean_wer,
        chinese_cer=chinese_clean_cer,
    )
    if rollout.error_rate <= threshold:
        return STRATUM_CLEAN, ()
    return STRATUM_NOISY, ("content_error_above_clean_threshold",)


def validate_stratum_row(value: Mapping[str, object]) -> None:
    if value.get("schema_version") != STRATA_SCHEMA:
        raise ValueError("unexpected rollout stratum schema")
    if value.get("stratum") not in STRATA:
        raise ValueError("unknown rollout quality stratum")
    if not str(value.get("sample_id", "")):
        raise ValueError("rollout stratum row has no sample ID")
    if int(value.get("source_manifest_record", -1)) < 0:
        raise ValueError("rollout stratum row has an invalid source record")
    if int(value.get("rollout_ordinal", -1)) < 0:
        raise ValueError("rollout stratum row has an invalid rollout ordinal")
    reasons = value.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
        raise ValueError("rollout stratum reasons are malformed")
    structural = value.get("structural")
    content = value.get("content")
    if not isinstance(structural, dict) or not isinstance(content, dict):
        raise ValueError("rollout stratum metrics are malformed")
    is_quarantine = value["stratum"] == STRATUM_QUARANTINE
    has_protocol_error = bool(
        int(structural.get("malformed_write_events", 0))
        or int(structural.get("early_eos_events", 0))
        or not bool(structural.get("final_reached_eos"))
    )
    if is_quarantine != has_protocol_error:
        raise ValueError("rollout stratum disagrees with structural flags")


def _ranges(total: int, workers: int) -> list[tuple[int, int]]:
    workers = max(1, min(int(workers), int(total)))
    return [
        (total * rank // workers, total * (rank + 1) // workers)
        for rank in range(workers)
    ]


def _read_at(handle, offsets, index: int, cls):
    handle.seek(int(offsets[index]))
    return cls.from_mapping(json.loads(handle.readline()))


def _worker(task: tuple[object, ...]) -> dict[str, Any]:
    (
        rank,
        gold_value,
        rollout_value,
        start,
        stop,
        output_value,
        english_clean_wer,
        chinese_clean_cer,
    ) = task
    rank = int(rank)
    start = int(start)
    stop = int(stop)
    gold = Path(str(gold_value))
    rollouts = Path(str(rollout_value))
    output_root = Path(str(output_value))
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"part_{rank:03d}.jsonl"
    report_path = output_root / f"part_{rank:03d}.json"
    gold_offsets = load_index(gold)
    rollout_offsets = load_index(rollouts)
    if gold_offsets is None or rollout_offsets is None:
        raise ValueError("stratification input is missing its offset index")
    counts: Counter[str] = Counter()
    language: dict[str, Counter[str]] = {}
    writer = IndexedJSONLWriter(output)
    try:
        with gold.open("rb") as gold_handle, rollouts.open("rb") as rollout_handle:
            for rollout_ordinal in range(start, stop):
                rollout = _read_at(
                    rollout_handle, rollout_offsets, rollout_ordinal, V1Rollout
                )
                record_index = int(rollout.source_manifest_record)
                if not 0 <= record_index < len(gold_offsets):
                    raise ValueError("rollout source record is outside gold data")
                gold_row = _read_at(gold_handle, gold_offsets, record_index, E2ETrajectory)
                _audit_pair(gold_row, rollout)
                stratum, reasons = classify_rollout(
                    rollout,
                    english_clean_wer=float(english_clean_wer),
                    chinese_clean_cer=float(chinese_clean_cer),
                )
                row = {
                    "schema_version": STRATA_SCHEMA,
                    "sample_id": rollout.sample_id,
                    "split": rollout.split,
                    "src_lang": rollout.src_lang,
                    "source_manifest_record": record_index,
                    "rollout_ordinal": rollout_ordinal,
                    "stratum": stratum,
                    "reasons": list(reasons),
                    "structural": {
                        "malformed_write_events": rollout.malformed_write_events,
                        "early_eos_events": rollout.early_eos_events,
                        "final_reached_eos": rollout.final_reached_eos,
                    },
                    "content": {
                        "metric": rollout.metric,
                        "errors": rollout.errors,
                        "reference_units": rollout.reference_units,
                        "error_rate": rollout.error_rate,
                        "empty_events": rollout.empty_events,
                        "events": len(rollout.events),
                    },
                }
                validate_stratum_row(row)
                writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                counts["records"] += 1
                counts[f"stratum:{stratum}"] += 1
                counts["accepted_rollout_records"] += int(stratum != STRATUM_QUARANTINE)
                counts["events"] += len(rollout.events)
                counts["malformed_write_events"] += rollout.malformed_write_events
                counts["early_eos_events"] += rollout.early_eos_events
                counts["final_eos_samples"] += int(rollout.final_reached_eos)
                values = language.setdefault(rollout.src_lang, Counter())
                values["records"] += 1
                values[f"stratum:{stratum}"] += 1
                values["errors"] += rollout.errors
                values["reference_units"] += rollout.reference_units
    finally:
        output_report = writer.close()
    index = write_index(output, writer.offsets)
    report = {
        "schema_version": PART_SCHEMA,
        "status": "complete",
        "rank": rank,
        "start": start,
        "stop": stop,
        "gold": str(gold.resolve()),
        "rollouts": str(rollouts.resolve()),
        "counts": dict(sorted(counts.items())),
        "by_language": {
            name: dict(sorted(values.items()))
            for name, values in sorted(language.items())
        },
        "output": output_report,
        "index": index,
    }
    atomic_json(report_path, report)
    return report


def _merge_parts(parts: list[dict[str, Any]], output: Path) -> dict[str, object]:
    offsets = array("Q")
    byte_base = 0
    counts: Counter[str] = Counter()
    by_language: dict[str, Counter[str]] = {}
    with output.open("xb") as destination:
        for part in parts:
            metadata = part["output"]
            path = Path(str(metadata["path"]))
            if path.stat().st_size != int(metadata["bytes"]):
                raise ValueError("rollout stratum part byte count changed")
            if file_sha256(path) != metadata["sha256"]:
                raise ValueError("rollout stratum part digest changed")
            part_offsets = load_index(path)
            if part_offsets is None or len(part_offsets) != int(metadata["records"]):
                raise ValueError("rollout stratum part offset index differs")
            offsets.extend(byte_base + int(value) for value in part_offsets)
            with path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
            byte_base += int(metadata["bytes"])
            counts.update({str(key): int(value) for key, value in part["counts"].items()})
            for name, raw in part["by_language"].items():
                by_language.setdefault(str(name), Counter()).update(
                    {str(key): int(value) for key, value in raw.items()}
                )
        destination.flush()
        os.fsync(destination.fileno())
    index = write_index(output, offsets)
    return {
        "path": str(output.resolve()),
        "records": len(offsets),
        "bytes": output.stat().st_size,
        "sha256": file_sha256(output),
        "index": index,
        "counts": dict(sorted(counts.items())),
        "by_language": {
            name: dict(sorted(values.items()))
            for name, values in sorted(by_language.items())
        },
    }


def markdown(report: Mapping[str, object]) -> str:
    summary = report["summary"]
    policy = report["policy"]
    checks = report["checks"]
    by_language = report["by_language"]
    assert isinstance(summary, dict)
    assert isinstance(policy, dict)
    assert isinstance(checks, dict)
    assert isinstance(by_language, dict)
    lines = [
        "# V1 rollout strict quality gate and strata",
        "",
        f"- Status: **{report['status']}**",
        f"- Records: **{summary['records']:,}**",
        f"- Clean: **{summary['clean_records']:,}** ({summary['clean_rate']:.2%})",
        f"- Noisy content retained: **{summary['noisy_content_records']:,}** ({summary['noisy_content_rate']:.2%})",
        f"- Quarantined protocol errors: **{summary['quarantine_records']:,}** ({summary['quarantine_rate']:.2%})",
        f"- Rollout-dependent supervision retained: **{summary['accepted_rollout_records']:,}** ({summary['accepted_rollout_rate']:.2%})",
        f"- Final EOS sample rate: **{summary['final_eos_rate']:.4f}**",
        "",
        "## Policy",
        "",
        f"- clean English WER <= {policy['english_clean_wer']:.2f}",
        f"- clean Chinese CER <= {policy['chinese_clean_cer']:.2f}",
        "- noisy_content keeps structurally valid free-running errors for robustness training",
        "- quarantine means malformed WRITE, early EOS, or missing final EOS",
        "- quarantine remains eligible only for gold-source incremental MT and Phase3 replay",
        "",
        "## Hard checks",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- {name}: **{'PASS' if value else 'FAIL'}**")
    lines.extend(
        [
            "",
            "| source language | samples | clean | noisy | quarantine | weighted WER/CER |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, raw in sorted(by_language.items()):
        lines.append(
            f"| {name} | {raw['records']:,} | {raw['clean_records']:,} | "
            f"{raw['noisy_content_records']:,} | {raw['quarantine_records']:,} | "
            f"{raw['error_rate']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--merge-report", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--english-clean-wer", type=float, default=0.30)
    parser.add_argument("--chinese-clean-cer", type=float, default=0.20)
    parser.add_argument("--maximum-quarantine-rate", type=float, default=0.40)
    parser.add_argument("--minimum-accepted-rate", type=float, default=0.60)
    parser.add_argument("--minimum-final-eos-rate", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = (args.output_manifest, args.output_json, args.output_md, args.parts_root)
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite rollout stratification outputs")
    if args.workers <= 0:
        raise ValueError("rollout stratification workers must be positive")
    for value in (
        args.english_clean_wer,
        args.chinese_clean_cer,
        args.maximum_quarantine_rate,
        args.minimum_accepted_rate,
        args.minimum_final_eos_rate,
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("rollout quality thresholds must be in [0,1]")
    merge = json.loads(args.merge_report.read_text(encoding="utf-8"))
    if merge.get("status") != "complete":
        raise ValueError("V1 rollout merge is not complete")
    gold_offsets = load_index(args.gold)
    rollout_offsets = load_index(args.rollouts)
    if gold_offsets is None or rollout_offsets is None:
        raise ValueError("gold or rollout JSONL is missing its offset index")
    total = len(rollout_offsets)
    if total != int(merge["output"]["records"]):
        raise ValueError("rollout count differs from merge report")
    if file_sha256(args.rollouts) != str(merge["output"]["sha256"]):
        raise ValueError("rollout digest differs from merge report")
    args.parts_root.mkdir(parents=True)
    tasks = [
        (
            rank,
            str(args.gold.resolve()),
            str(args.rollouts.resolve()),
            start,
            stop,
            str(args.parts_root.resolve()),
            args.english_clean_wer,
            args.chinese_clean_cer,
        )
        for rank, (start, stop) in enumerate(_ranges(total, args.workers))
    ]
    with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
        parts = list(pool.map(_worker, tasks))
    parts.sort(key=lambda value: int(value["rank"]))
    cursor = 0
    for rank, part in enumerate(parts):
        if part.get("schema_version") != PART_SCHEMA or part.get("status") != "complete":
            raise ValueError("rollout stratum part is incomplete")
        if int(part["rank"]) != rank or int(part["start"]) != cursor:
            raise ValueError("rollout stratum parts contain a rank/range gap")
        cursor = int(part["stop"])
    if cursor != total:
        raise ValueError("rollout stratum parts do not cover every rollout")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    merged = _merge_parts(parts, args.output_manifest)
    counts = merged["counts"]
    assert isinstance(counts, dict)
    records = int(counts.get("records", 0))
    clean = int(counts.get(f"stratum:{STRATUM_CLEAN}", 0))
    noisy = int(counts.get(f"stratum:{STRATUM_NOISY}", 0))
    quarantine = int(counts.get(f"stratum:{STRATUM_QUARANTINE}", 0))
    accepted = int(counts.get("accepted_rollout_records", 0))
    final_eos = int(counts.get("final_eos_samples", 0))
    if clean + noisy + quarantine != records or records != total:
        raise ValueError("rollout strata are not mutually exclusive and exhaustive")
    by_language: dict[str, dict[str, object]] = {}
    raw_language = merged["by_language"]
    assert isinstance(raw_language, dict)
    for name, raw in sorted(raw_language.items()):
        values = dict(raw)
        units = int(values.get("reference_units", 0))
        language_records = int(values.get("records", 0))
        language_accepted = int(values.get(f"stratum:{STRATUM_CLEAN}", 0)) + int(
            values.get(f"stratum:{STRATUM_NOISY}", 0)
        )
        by_language[str(name)] = {
            "metric": "cer" if name == "cmn" else "wer",
            "records": language_records,
            "clean_records": int(values.get(f"stratum:{STRATUM_CLEAN}", 0)),
            "noisy_content_records": int(values.get(f"stratum:{STRATUM_NOISY}", 0)),
            "quarantine_records": int(values.get(f"stratum:{STRATUM_QUARANTINE}", 0)),
            "accepted_records": language_accepted,
            "accepted_rate": language_accepted / max(1, language_records),
            "errors": int(values.get("errors", 0)),
            "reference_units": units,
            "error_rate": int(values.get("errors", 0)) / max(1, units),
        }
    accepted_rate = accepted / max(1, records)
    quarantine_rate = quarantine / max(1, records)
    final_eos_rate = final_eos / max(1, records)
    checks = {
        "all_records_classified_once": records == total,
        "quarantine_rate_within_limit": quarantine_rate <= args.maximum_quarantine_rate,
        "accepted_rollout_rate_meets_minimum": accepted_rate >= args.minimum_accepted_rate,
        "final_eos_rate_meets_minimum": final_eos_rate >= args.minimum_final_eos_rate,
        "every_language_retains_rollout_supervision": all(
            int(value["accepted_records"]) > 0 for value in by_language.values()
        ),
    }
    status = "passed" if all(checks.values()) else "failed"
    report = {
        "schema_version": QUALITY_GATE_SCHEMA,
        "status": status,
        "formal_training_authorized": False,
        "gold": str(args.gold.resolve()),
        "rollouts": str(args.rollouts.resolve()),
        "merge_report": str(args.merge_report.resolve()),
        "manifest": merged,
        "policy": {
            "english_clean_wer": args.english_clean_wer,
            "chinese_clean_cer": args.chinese_clean_cer,
            "maximum_quarantine_rate": args.maximum_quarantine_rate,
            "minimum_accepted_rate": args.minimum_accepted_rate,
            "minimum_final_eos_rate": args.minimum_final_eos_rate,
            "quarantine_protocol_errors": [
                "malformed_write",
                "early_eos",
                "missing_final_eos",
            ],
            "quarantine_allowed_task_families": [
                "incremental_mt_event:gold_source_only",
                "phase3_quality_replay",
                "phase3_performance_replay",
            ],
        },
        "summary": {
            "records": records,
            "clean_records": clean,
            "clean_rate": clean / max(1, records),
            "noisy_content_records": noisy,
            "noisy_content_rate": noisy / max(1, records),
            "quarantine_records": quarantine,
            "quarantine_rate": quarantine_rate,
            "accepted_rollout_records": accepted,
            "accepted_rollout_rate": accepted_rate,
            "events": int(counts.get("events", 0)),
            "malformed_write_events": int(counts.get("malformed_write_events", 0)),
            "early_eos_events": int(counts.get("early_eos_events", 0)),
            "final_eos_samples": final_eos,
            "final_eos_rate": final_eos_rate,
        },
        "by_language": by_language,
        "checks": checks,
    }
    atomic_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with args.output_md.open("x", encoding="utf-8") as handle:
        handle.write(markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if status != "passed":
        raise SystemExit(4)


if __name__ == "__main__":
    main()

