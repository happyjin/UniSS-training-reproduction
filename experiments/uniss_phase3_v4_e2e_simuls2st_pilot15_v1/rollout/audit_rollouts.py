#!/usr/bin/env python3
"""Align every V1 rollout with immutable gold events and aggregate quality gates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
    validate_trajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.io import atomic_json
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.schema import (
    V1Rollout,
    validate_rollout,
)
from training.simul_uniss.jsonl_index import load_index


AUDIT_SCHEMA = "uniss_phase3_v4_e2e_simuls2st_v1_rollout_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--merge-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def _read_at(handle, offsets, index: int, cls):
    handle.seek(int(offsets[index]))
    return cls.from_mapping(json.loads(handle.readline()))


def _audit_pair(gold: E2ETrajectory, rollout: V1Rollout) -> None:
    validate_trajectory(gold, require_audio_hash=True, require_audio_audit=True)
    validate_rollout(rollout, expected_events=len(gold.events))
    invariants = {
        "sample ID": (gold.sample_id, rollout.sample_id),
        "split": (gold.split, rollout.split),
        "source language": (gold.src_lang, rollout.src_lang),
        "source manifest record": (
            gold.source_manifest_record,
            rollout.source_manifest_record,
        ),
        "V1 checkpoint SHA256": (
            gold.v1_checkpoint_sha256,
            rollout.v1_checkpoint_sha256,
        ),
        "source audio SHA256": (gold.source_audio_sha256, rollout.source_audio_sha256),
    }
    for label, (expected, actual) in invariants.items():
        if expected != actual:
            raise ValueError(f"gold/rollout {label} differs for {gold.sample_id}")
    visible_glm = 0
    for gold_event, actual in zip(gold.events, rollout.events):
        if gold_event.event_index != actual.event_index:
            raise ValueError("gold/rollout event index differs")
        if gold_event.source_end_ms != actual.source_end_ms:
            raise ValueError("gold/rollout source event time differs")
        if gold_event.gold_source_delta:
            visible_glm = gold_event.source_glm_end
            if not actual.generated_tokens:
                raise ValueError("V1 rollout omitted a gold text event query")
        elif actual.generated_tokens or actual.content_tokens or actual.v1_source_delta:
            raise ValueError("V1 rollout generated on an unqueried empty gold event")
        if actual.visible_glm_tokens != visible_glm:
            raise ValueError("V1 rollout visible GLM boundary differs from gold protocol")
    if rollout.final_visible_glm_tokens != gold.source_glm_length:
        raise ValueError("V1 rollout did not consume the complete source GLM")


def markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    assert isinstance(summary, dict)
    language = report["by_language"]
    assert isinstance(language, dict)
    lines = [
        "# V1 append-only ASR rollout audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Gold: `{report['gold']}`",
        f"- Rollouts: `{report['rollouts']}`",
        f"- Samples: **{summary['records']:,}**",
        f"- Events: **{summary['events']:,}**",
        f"- Append-only rollback count: **0**",
        f"- Empty event rate: **{summary['empty_event_rate']:.4f}**",
        f"- Malformed WRITE rate: **{summary['malformed_write_rate']:.4f}**",
        f"- Early EOS rate: **{summary['early_eos_rate']:.4f}**",
        f"- Final EOS sample rate: **{summary['final_eos_rate']:.4f}**",
        "",
        "| source language | metric | samples | errors | reference units | weighted error rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, value in sorted(language.items()):
        lines.append(
            f"| {name} | {value['metric']} | {value['samples']:,} | "
            f"{value['errors']:,} | {value['reference_units']:,} | "
            f"{value['error_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The rollout is an immutable sidecar. It uses the gold event clock only to decide when the trained V1 ASR is queried; generated text is fully free-running and every accepted delta is append-only. Empty gold text events are deliberately not queried because that is the exact Stage A training protocol.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("refusing to overwrite V1 rollout audit")
    merge = json.loads(args.merge_report.read_text(encoding="utf-8"))
    if merge.get("status") != "complete":
        raise ValueError("V1 rollout merge is not complete")
    gold_offsets = load_index(args.gold)
    rollout_offsets = load_index(args.rollouts)
    if gold_offsets is None or rollout_offsets is None:
        raise ValueError("gold or rollout JSONL is missing its offset index")
    if len(rollout_offsets) != int(merge["output"]["records"]):
        raise ValueError("rollout count differs from merge report")
    counts: Counter[str] = Counter()
    severity: Counter[str] = Counter()
    by_language: dict[str, Counter[str]] = {}
    runtime_hash: str | None = None
    hf_hash: str | None = None
    with args.gold.open("rb") as gold_handle, args.rollouts.open("rb") as rollout_handle:
        for rollout_index in range(len(rollout_offsets)):
            rollout = _read_at(rollout_handle, rollout_offsets, rollout_index, V1Rollout)
            record_index = int(rollout.source_manifest_record)
            if not 0 <= record_index < len(gold_offsets):
                raise ValueError("rollout source manifest record is outside gold data")
            gold = _read_at(gold_handle, gold_offsets, record_index, E2ETrajectory)
            _audit_pair(gold, rollout)
            if runtime_hash is None:
                runtime_hash = rollout.runtime_sha256
                hf_hash = rollout.v1_hf_sha256
            elif runtime_hash != rollout.runtime_sha256 or hf_hash != rollout.v1_hf_sha256:
                raise ValueError("runtime or V1 HF fingerprint changed within rollout")
            counts["records"] += 1
            counts["events"] += len(rollout.events)
            counts["empty_events"] += rollout.empty_events
            counts["early_eos_events"] += rollout.early_eos_events
            counts["malformed_write_events"] += rollout.malformed_write_events
            counts["final_eos_samples"] += int(rollout.final_reached_eos)
            severity.update(event.noise_severity for event in rollout.events)
            values = by_language.setdefault(rollout.src_lang, Counter())
            values["samples"] += 1
            values["errors"] += rollout.errors
            values["reference_units"] += rollout.reference_units
    language_report = {
        name: {
            "metric": "cer" if name == "cmn" else "wer",
            "samples": values["samples"],
            "errors": values["errors"],
            "reference_units": values["reference_units"],
            "error_rate": values["errors"] / max(1, values["reference_units"]),
        }
        for name, values in sorted(by_language.items())
    }
    summary = dict(sorted(counts.items()))
    summary.update(
        {
            "empty_event_rate": counts["empty_events"] / max(1, counts["events"]),
            "early_eos_rate": counts["early_eos_events"] / max(1, counts["events"]),
            "malformed_write_rate": counts["malformed_write_events"]
            / max(1, counts["events"] - counts["empty_events"]),
            "final_eos_rate": counts["final_eos_samples"] / max(1, counts["records"]),
        }
    )
    report = {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed",
        "gold": str(args.gold.resolve()),
        "rollouts": str(args.rollouts.resolve()),
        "merge_report": str(args.merge_report.resolve()),
        "runtime_sha256": runtime_hash,
        "v1_hf_sha256": hf_hash,
        "summary": summary,
        "noise_severity": dict(sorted(severity.items())),
        "by_language": language_report,
    }
    atomic_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with args.output_md.open("x", encoding="utf-8") as handle:
        handle.write(markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
