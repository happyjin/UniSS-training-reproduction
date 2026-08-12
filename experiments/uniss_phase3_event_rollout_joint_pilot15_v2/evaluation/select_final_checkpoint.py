#!/usr/bin/env python3
"""Select a fixed15 checkpoint only after all exact-runtime hard gates pass.

Teacher-forced validation creates the shortlist.  This selector consumes the
real train/validation runtime, prefix-ASR, quality, parity and paired Phase3
retention artifacts produced by the eight-GPU evaluation scripts.  Missing or
non-finite evidence is a rejection, never an implicit pass.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Mapping, Sequence


SCHEMA = "uniss_event_rollout_fixed15_final_checkpoint_selection_v1"
QUALITY_FILES = (
    "text_bleu.json",
    "speech_bleu.json",
    "slc.json",
    "utmos.json",
    "autopcp.json",
    "speaker_similarity.json",
)
RETENTION_QUALITY_FILES = (
    "text_bleu.json",
    "speech_bleu.json",
    "slc.json",
    "utmos.json",
    "autopcp.json",
)


@dataclass(frozen=True)
class GateThresholds:
    useful_audio_p50_ms: float = 1000.0
    minimum_useful_audio_recall: float = 0.50
    minimum_natural_write_rate: float = 0.50
    maximum_post_source_first_write_rate: float = 0.50
    maximum_premature_first_write_rate: float = 0.50
    minimum_playable_pcm_rate: float = 0.80
    minimum_finite_pcm_rate: float = 0.95
    maximum_collapse_rate: float = 0.20
    minimum_natural_eos_rate: float = 0.50
    minimum_retention_output_rate: float = 0.80
    minimum_retention_text_bleu_ratio: float = 0.50
    minimum_retention_speech_bleu_ratio: float = 0.50
    minimum_retention_slc_ratio: float = 0.50
    maximum_retention_utmos_drop: float = 1.00
    maximum_retention_autopcp_drop: float = 1.00


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _value(mapping: Mapping[str, object], key: str, reasons: list[str], prefix: str) -> float | None:
    value = mapping.get(key)
    if not _finite(value):
        reasons.append(f"missing_or_nonfinite:{prefix}.{key}")
        return None
    return float(value)


def _weighted_group_mean(
    report: Mapping[str, object],
    field: str,
    *,
    mode: str | None = None,
) -> float | None:
    weighted = []
    for name, raw in dict(report.get("groups", {})).items():
        if mode is not None and not str(name).startswith(f"{mode}:"):
            continue
        group = dict(raw)
        value = group.get(field)
        if not _finite(value):
            continue
        count = int(group.get("sample_count", group.get("samples", 0)) or 0)
        if count <= 0:
            continue
        weighted.extend([float(value)] * count)
    return fmean(weighted) if weighted else None


def _quality_summary(reports: Mapping[str, Mapping[str, object]]) -> dict[str, float | None]:
    return {
        "text_bleu": _weighted_group_mean(reports.get("text_bleu.json", {}), "score"),
        "speech_bleu": _weighted_group_mean(reports.get("speech_bleu.json", {}), "score"),
        "slc_0_4": _weighted_group_mean(reports.get("slc.json", {}), "slc_0_4"),
        "utmos": _weighted_group_mean(reports.get("utmos.json", {}), "mean"),
        "autopcp": _weighted_group_mean(reports.get("autopcp.json", {}), "mean"),
        "speaker_similarity": _weighted_group_mean(
            reports.get("speaker_similarity.json", {}), "mean"
        ),
    }


def _retention_quality_summary(
    reports: Mapping[str, Mapping[str, object]], mode: str
) -> dict[str, float | None]:
    return {
        "text_bleu": _weighted_group_mean(
            reports.get("text_bleu.json", {}), "score", mode=mode
        ),
        "speech_bleu": _weighted_group_mean(
            reports.get("speech_bleu.json", {}), "score", mode=mode
        ),
        "slc_0_4": _weighted_group_mean(
            reports.get("slc.json", {}), "slc_0_4", mode=mode
        ),
        "utmos": _weighted_group_mean(
            reports.get("utmos.json", {}), "mean", mode=mode
        ),
        "autopcp": _weighted_group_mean(
            reports.get("autopcp.json", {}), "mean", mode=mode
        ),
    }


def _ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline <= 0:
        return None
    return candidate / baseline


def _runtime_gate(
    report: Mapping[str, object],
    *,
    split: str,
    thresholds: GateThresholds,
    reasons: list[str],
) -> dict[str, object]:
    group = dict(dict(report.get("groups", {})).get("all", {}))
    if not bool(dict(report.get("coverage", {})).get("complete")):
        reasons.append(f"{split}:incomplete_runtime_coverage")
    if int(group.get("samples", 0) or 0) <= 0:
        reasons.append(f"{split}:no_runtime_samples")

    natural_write = _value(group, "natural_write_sample_rate", reasons, split)
    all_wait = _value(group, "all_wait_rate", reasons, split)
    post_source = _value(group, "post_source_eos_first_write_rate", reasons, split)
    premature = _value(group, "first_write_premature_rate", reasons, split)
    playable = _value(group, "playable_pcm_rate", reasons, split)
    finite = _value(group, "finite_pcm_rate", reasons, split)
    collapse = _value(group, "severe_semantic_collapse_rate", reasons, split)
    eos = _value(group, "natural_eos_rate", reasons, split)
    forced = _value(group, "forced_writes", reasons, split)
    revisions = _value(group, "committed_revision_violations", reasons, split)

    checks = (
        (natural_write is not None and natural_write >= thresholds.minimum_natural_write_rate,
         f"{split}:natural_write_rate_below_threshold"),
        (all_wait is not None and all_wait <= 1.0 - thresholds.minimum_natural_write_rate,
         f"{split}:all_wait_rate_above_threshold"),
        (post_source is not None and post_source <= thresholds.maximum_post_source_first_write_rate,
         f"{split}:mostly_waits_until_source_end"),
        (premature is not None and premature <= thresholds.maximum_premature_first_write_rate,
         f"{split}:premature_write_rate_above_threshold"),
        (playable is not None and playable >= thresholds.minimum_playable_pcm_rate,
         f"{split}:playable_pcm_rate_below_threshold"),
        (finite is not None and finite >= thresholds.minimum_finite_pcm_rate,
         f"{split}:finite_pcm_rate_below_threshold"),
        (collapse is not None and collapse <= thresholds.maximum_collapse_rate,
         f"{split}:semantic_or_audio_collapse"),
        (eos is not None and eos >= thresholds.minimum_natural_eos_rate,
         f"{split}:natural_eos_rate_below_threshold"),
        (forced is not None and forced == 0.0, f"{split}:forced_write_nonzero"),
        (revisions is not None and revisions == 0.0, f"{split}:committed_revision_nonzero"),
    )
    for passed, reason in checks:
        if not passed and reason not in reasons:
            reasons.append(reason)
    return group


def evaluate_candidate(
    candidate: Mapping[str, object], thresholds: GateThresholds
) -> dict[str, object]:
    reasons: list[str] = []
    train_group = _runtime_gate(
        dict(candidate.get("train_runtime", {})),
        split="train",
        thresholds=thresholds,
        reasons=reasons,
    )
    valid_group = _runtime_gate(
        dict(candidate.get("valid_runtime", {})),
        split="valid",
        thresholds=thresholds,
        reasons=reasons,
    )

    useful_all = dict(dict(candidate.get("useful_audio", {})).get("groups", {})).get("all", {})
    useful_recall = _value(useful_all, "useful_audio_recall", reasons, "valid.useful_audio")
    useful_latency = dict(useful_all.get("first_useful_audio_wall_ms", {}))
    useful_p50 = _value(useful_latency, "p50", reasons, "valid.first_useful_audio_wall_ms")
    if useful_recall is None or useful_recall < thresholds.minimum_useful_audio_recall:
        reasons.append("valid:useful_audio_recall_below_threshold")
    if useful_p50 is None or useful_p50 >= thresholds.useful_audio_p50_ms:
        reasons.append("valid:first_useful_audio_p50_not_subsecond")

    parity = dict(candidate.get("parity", {}))
    if not bool(parity.get("passed")):
        reasons.append("runtime_parity_failed_or_missing")

    quality_reports = dict(candidate.get("quality", {}))
    quality = _quality_summary(quality_reports)
    for name, value in quality.items():
        if value is None:
            reasons.append(f"missing_or_nonfinite:valid_quality.{name}")

    retention = dict(candidate.get("retention", {}))
    if not bool(retention.get("paired_complete")):
        reasons.append("phase3_retention_pairing_failed_or_missing")
    adapter = dict(dict(retention.get("groups", {})).get("streaming_adapter", {}))
    for field in ("generated_text_rate", "semantic_output_rate", "playable_audio_rate", "finite_audio_rate", "non_silent_audio_rate"):
        value = _value(adapter, field, reasons, "phase3_retention.streaming_adapter")
        if value is None or value < thresholds.minimum_retention_output_rate:
            reasons.append(f"phase3_retention:{field}_below_threshold")

    retention_reports = dict(candidate.get("retention_quality", {}))
    phase3_quality = _retention_quality_summary(retention_reports, "phase3_v4")
    adapter_quality = _retention_quality_summary(retention_reports, "streaming_adapter")
    retention_ratios = {
        "text_bleu": _ratio(adapter_quality["text_bleu"], phase3_quality["text_bleu"]),
        "speech_bleu": _ratio(adapter_quality["speech_bleu"], phase3_quality["speech_bleu"]),
        "slc_0_4": _ratio(adapter_quality["slc_0_4"], phase3_quality["slc_0_4"]),
    }
    ratio_thresholds = {
        "text_bleu": thresholds.minimum_retention_text_bleu_ratio,
        "speech_bleu": thresholds.minimum_retention_speech_bleu_ratio,
        "slc_0_4": thresholds.minimum_retention_slc_ratio,
    }
    for name, minimum in ratio_thresholds.items():
        value = retention_ratios[name]
        if value is None:
            reasons.append(f"missing_or_nonfinite:phase3_retention.{name}_ratio")
        elif value < minimum:
            reasons.append(f"phase3_retention:{name}_ratio_below_threshold")
    for name, maximum_drop in (
        ("utmos", thresholds.maximum_retention_utmos_drop),
        ("autopcp", thresholds.maximum_retention_autopcp_drop),
    ):
        base = phase3_quality[name]
        adapted = adapter_quality[name]
        if base is None or adapted is None:
            reasons.append(f"missing_or_nonfinite:phase3_retention.{name}")
        elif base - adapted > maximum_drop:
            reasons.append(f"phase3_retention:{name}_drop_above_threshold")

    reasons = list(dict.fromkeys(reasons))
    return {
        "iteration": int(candidate["iteration"]),
        "evaluation_root": str(candidate.get("evaluation_root", "")),
        "retention_root": str(candidate.get("retention_root", "")),
        "passed": not reasons,
        "rejection_reasons": reasons,
        "runtime": {"train": train_group, "valid": valid_group},
        "useful_audio": {
            "recall": useful_recall,
            "p50_ms": useful_p50,
            "p90_ms": useful_latency.get("p90"),
            "p95_ms": useful_latency.get("p95"),
        },
        "quality": quality,
        "runtime_parity": {
            "passed": bool(parity.get("passed")),
            "failure_count": parity.get("failure_count"),
        },
        "phase3_retention": {
            "aggregate": retention,
            "phase3_v4_quality": phase3_quality,
            "streaming_adapter_quality": adapter_quality,
            "quality_ratios": retention_ratios,
        },
    }


def select(candidates: Sequence[Mapping[str, object]], thresholds: GateThresholds) -> dict[str, object]:
    if not candidates:
        raise ValueError("at least one checkpoint candidate is required")
    evaluated = [evaluate_candidate(candidate, thresholds) for candidate in candidates]
    iterations = [int(row["iteration"]) for row in evaluated]
    if len(iterations) != len(set(iterations)):
        raise ValueError(f"duplicate checkpoint iterations: {iterations}")
    eligible = [row for row in evaluated if row["passed"]]
    eligible.sort(
        key=lambda row: (
            float(row["useful_audio"]["p50_ms"]),
            -float(row["useful_audio"]["recall"]),
            -float(row["quality"]["speech_bleu"]),
            -float(row["quality"]["text_bleu"]),
            float(row["runtime"]["valid"]["severe_semantic_collapse_rate"]),
            -int(row["iteration"]),
        )
    )
    winner = eligible[0] if eligible else None
    return {
        "schema_version": SCHEMA,
        "thresholds": asdict(thresholds),
        "candidate_count": len(evaluated),
        "eligible_count": len(eligible),
        "selection_status": "selected" if winner is not None else "no_checkpoint_passed",
        "selected_iteration": None if winner is None else winner["iteration"],
        "selected_evaluation_root": None if winner is None else winner["evaluation_root"],
        "selected_retention_root": None if winner is None else winner["retention_root"],
        "selection_order": [row["iteration"] for row in eligible],
        "selection_rule": (
            "Reject missing evidence and every mechanism, useful-audio, parity, collapse or "
            "Phase3-retention failure. Among passing checkpoints, minimize useful-audio p50, "
            "then maximize useful-audio recall, Speech BLEU and Text BLEU."
        ),
        "candidates": evaluated,
    }


def _read(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_candidate(iteration: int, evaluation_root: Path, retention_root: Path) -> dict[str, object]:
    quality_root = evaluation_root / "valid_aggregate" / "metrics"
    retention_quality_root = retention_root / "aggregate" / "metrics"
    if not (quality_root / "complete.json").is_file():
        raise FileNotFoundError(quality_root / "complete.json")
    if not (retention_quality_root / "complete.json").is_file():
        raise FileNotFoundError(retention_quality_root / "complete.json")
    return {
        "iteration": iteration,
        "evaluation_root": str(evaluation_root.resolve()),
        "retention_root": str(retention_root.resolve()),
        "train_runtime": _read(evaluation_root / "train_aggregate" / "aggregate.json"),
        "valid_runtime": _read(evaluation_root / "valid_aggregate" / "aggregate.json"),
        "useful_audio": _read(quality_root / "prefix_asr" / "useful_audio.json"),
        "parity": _read(evaluation_root / "parity" / "report.json"),
        "quality": {name: _read(quality_root / name) for name in QUALITY_FILES},
        "retention": _read(retention_root / "aggregate" / "aggregate.json"),
        "retention_quality": {
            name: _read(retention_quality_root / name) for name in RETENTION_QUALITY_FILES
        },
    }


def markdown(report: Mapping[str, object]) -> str:
    selected = report.get("selected_iteration")
    lines = [
        "# Fixed15 final checkpoint selection",
        "",
        f"- Status: `{report['selection_status']}`",
        f"- Selected iteration: `{selected if selected is not None else 'none'}`",
        f"- Eligible candidates: {report['eligible_count']}/{report['candidate_count']}",
        "- Any missing or non-finite hard-gate evidence is a rejection.",
        "",
        "| iteration | pass | useful recall | useful p50/p95 ms | natural WRITE valid | all-WAIT valid | collapse valid | Speech BLEU | Text BLEU | parity | rejection reasons |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in report["candidates"]:
        useful = row["useful_audio"]
        valid = row["runtime"]["valid"]
        quality = row["quality"]
        reasons = "; ".join(row["rejection_reasons"]) or "none"
        lines.append(
            f"| {row['iteration']} | {'yes' if row['passed'] else 'no'} | "
            f"{useful['recall']} | {useful['p50_ms']}/{useful['p95_ms']} | "
            f"{valid.get('natural_write_sample_rate')} | {valid.get('all_wait_rate')} | "
            f"{valid.get('severe_semantic_collapse_rate')} | {quality['speech_bleu']} | "
            f"{quality['text_bleu']} | {'yes' if row['runtime_parity']['passed'] else 'no'} | "
            f"{reasons} |"
        )
    lines.extend(["", "## Selection rule", "", str(report["selection_rule"]), ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        nargs=3,
        action="append",
        metavar=("ITERATION", "CHECKPOINT_EVALUATION_ROOT", "PHASE3_RETENTION_ROOT"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite final selection: {args.output_root}")
    candidates = [
        load_candidate(int(iteration), Path(evaluation), Path(retention))
        for iteration, evaluation, retention in args.candidate
    ]
    report = select(candidates, GateThresholds())
    args.output_root.mkdir(parents=True)
    (args.output_root / "checkpoint_selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "checkpoint_selection.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(json.dumps({"status": report["selection_status"], "iteration": report["selected_iteration"]}))


if __name__ == "__main__":
    main()
