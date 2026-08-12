"""Three-tier mechanism, generalization and subsecond dataset gates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA = "uniss_runtime_parity_dataset_gate_v1"
VALID_STATUS = {"pass", "fail", "not_evaluable"}


@dataclass(frozen=True)
class GateResult:
    status: str
    failures: tuple[str, ...]
    observed: Mapping[str, object]
    thresholds: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUS:
            raise ValueError(f"invalid gate status {self.status!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "failures": list(self.failures),
            "observed": dict(self.observed),
            "thresholds": dict(self.thresholds),
        }


def percentile(values: Sequence[float], quantile: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * float(quantile)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    fraction = position - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def _rows(summary: Mapping[str, object]) -> list[dict[str, object]]:
    samples = summary.get("samples")
    if not isinstance(samples, list):
        raise ValueError("runtime summary has no samples list")
    return [dict(value) for value in samples]


def _finite_pcm(row: Mapping[str, object]) -> bool | None:
    for key in ("pcm_finite", "translation_audio_finite"):
        if key in row:
            return bool(row[key])
    return None


def mechanism_gate(
    summaries: Sequence[Mapping[str, object]],
    *,
    causality_audit: Mapping[str, object] | None = None,
    cache_parity: Mapping[str, object] | None = None,
) -> GateResult:
    rows = [row for summary in summaries for row in _rows(summary)]
    thresholds = {
        "strict_pass_rate_min": 0.95,
        "forced_writes": 0,
        "revision_violations": 0,
        "source_eos_before_first_write_rate_max": 0.05,
        "playable_pcm_rate": 1.0,
        "natural_eos_rate_min": 0.98,
        "rtf_p50_max": 1.0,
        "rtf_p95_max": 1.0,
        "severe_semantic_collapse_rate": 0.0,
    }
    if not rows:
        return GateResult("not_evaluable", ("no_runtime_samples",), {}, thresholds)
    missing: list[str] = []
    if causality_audit is None:
        missing.append("missing_causality_audit")
    if cache_parity is None:
        missing.append("missing_cache_parity_audit")
    finite_values = [_finite_pcm(row) for row in rows]
    if any(value is None for value in finite_values):
        missing.append("missing_pcm_finite_check")
    collapse_values = [row.get("severe_semantic_collapse") for row in rows]
    if any(value is None for value in collapse_values):
        missing.append("missing_semantic_collapse_check")

    strict = [bool(row.get("quality_passed", False)) for row in rows]
    forced = sum(int(row.get("forced_writes", 0) or 0) for row in rows)
    revisions = sum(int(row.get("committed_revision_violations", 0) or 0) for row in rows)
    eos_before = sum(bool(row.get("source_finished_before_first_write", False)) for row in rows)
    playable = sum(int(row.get("translation_audio_samples", 0) or 0) > 0 for row in rows)
    natural_eos = sum(bool(row.get("natural_eos", False)) for row in rows)
    rtf = [float(row["rtf"]) for row in rows if row.get("rtf") is not None]
    observed = {
        "samples": len(rows),
        "legacy_single_sample_quality_pass_rate": statistics.fmean(strict),
        "forced_writes": forced,
        "revision_violations": revisions,
        "source_eos_before_first_write_rate": eos_before / len(rows),
        "playable_pcm_rate": playable / len(rows),
        "pcm_finite_rate": (
            None if any(value is None for value in finite_values) else statistics.fmean(bool(value) for value in finite_values)
        ),
        "natural_eos_rate": natural_eos / len(rows),
        "rtf_p50": percentile(rtf, 0.50),
        "rtf_p95": percentile(rtf, 0.95),
        "severe_semantic_collapse_rate": (
            None
            if any(value is None for value in collapse_values)
            else statistics.fmean(bool(value) for value in collapse_values)
        ),
        "causality_audit_passed": None if causality_audit is None else bool(causality_audit.get("passed")),
        "cache_parity_passed": None if cache_parity is None else bool(cache_parity.get("passed")),
    }
    if missing:
        return GateResult("not_evaluable", tuple(missing), observed, thresholds)
    failures: list[str] = []
    checks = (
        (observed["causality_audit_passed"] is True, "causality_audit_failed"),
        (observed["cache_parity_passed"] is True, "cache_parity_failed"),
        (observed["legacy_single_sample_quality_pass_rate"] >= 0.95, "strict_pass_rate_below_0.95"),
        (forced == 0, "forced_write_nonzero"),
        (revisions == 0, "revision_nonzero"),
        (observed["source_eos_before_first_write_rate"] < 0.05, "too_many_post_eos_first_writes"),
        (observed["playable_pcm_rate"] == 1.0, "unplayable_pcm"),
        (observed["pcm_finite_rate"] == 1.0, "nonfinite_pcm"),
        (observed["natural_eos_rate"] >= 0.98, "natural_eos_rate_below_0.98"),
        (observed["rtf_p50"] is not None and observed["rtf_p50"] < 1.0, "rtf_p50_not_realtime"),
        (observed["rtf_p95"] is not None and observed["rtf_p95"] < 1.0, "rtf_p95_not_realtime"),
        (observed["severe_semantic_collapse_rate"] == 0.0, "semantic_collapse_nonzero"),
    )
    failures.extend(reason for passed, reason in checks if not passed)
    return GateResult("fail" if failures else "pass", tuple(failures), observed, thresholds)


def generalization_gate(
    metrics: Mapping[str, object] | None,
    *,
    mechanism_status: str,
) -> GateResult:
    thresholds = {
        "minimum_held_out_samples": 256,
        "failure_rate_max": 0.01,
        "text_bleu_retention_min": 0.80,
        "speech_bleu_retention_min": 0.75,
        "utmos_drop_max": 0.30,
        "autopcp_drop_max": 0.10,
        "worst_direction_decides": True,
    }
    if mechanism_status != "pass":
        return GateResult(
            "not_evaluable",
            ("mechanism_gate_not_passed",),
            {},
            thresholds,
        )
    if metrics is None:
        return GateResult("not_evaluable", ("missing_generalization_metrics",), {}, thresholds)
    directions = metrics.get("directions")
    if not isinstance(directions, Mapping) or not directions:
        return GateResult("not_evaluable", ("missing_direction_metrics",), dict(metrics), thresholds)
    missing: list[str] = []
    failures: list[str] = []
    for direction, raw in directions.items():
        value = dict(raw)
        required = (
            "samples",
            "failure_rate",
            "text_bleu_retention",
            "speech_bleu_retention",
            "utmos_drop",
            "autopcp_drop",
        )
        absent = [key for key in required if value.get(key) is None]
        missing.extend(f"{direction}:{key}" for key in absent)
        if absent:
            continue
        checks = (
            (int(value["samples"]) >= 256, "samples_below_256"),
            (float(value["failure_rate"]) <= 0.01, "failure_rate_above_0.01"),
            (float(value["text_bleu_retention"]) >= 0.80, "text_bleu_retention_below_0.80"),
            (float(value["speech_bleu_retention"]) >= 0.75, "speech_bleu_retention_below_0.75"),
            (float(value["utmos_drop"]) <= 0.30, "utmos_drop_above_0.30"),
            (float(value["autopcp_drop"]) <= 0.10, "autopcp_drop_above_0.10"),
        )
        failures.extend(f"{direction}:{reason}" for passed, reason in checks if not passed)
    if missing:
        return GateResult("not_evaluable", tuple(missing), dict(metrics), thresholds)
    return GateResult("fail" if failures else "pass", tuple(failures), dict(metrics), thresholds)


def subsecond_gate(
    metrics: Mapping[str, object] | None,
    *,
    formal_status: str,
) -> GateResult:
    thresholds = {
        "first_write_ca_p95_ms_max": 800.0,
        "first_useful_audio_ca_p95_ms_max": 1000.0,
        "write_to_pcm_p95_ms_max": 200.0,
        "premature_write_rate_max": 0.05,
        "rtf_p95_max": 1.0,
    }
    if formal_status != "pass":
        return GateResult("not_evaluable", ("formal_gate_not_passed",), {}, thresholds)
    if metrics is None:
        return GateResult("not_evaluable", ("missing_subsecond_metrics",), {}, thresholds)
    required = tuple(thresholds)
    missing = [key for key in required if metrics.get(key) is None]
    if missing:
        return GateResult(
            "not_evaluable",
            tuple(f"missing_{key}" for key in missing),
            dict(metrics),
            thresholds,
        )
    checks = (
        (float(metrics["first_write_ca_p95_ms_max"]) <= 800.0, "first_write_ca_p95_above_800ms"),
        (float(metrics["first_useful_audio_ca_p95_ms_max"]) <= 1000.0, "first_useful_audio_ca_p95_above_1000ms"),
        (float(metrics["write_to_pcm_p95_ms_max"]) <= 200.0, "write_to_pcm_p95_above_200ms"),
        (float(metrics["premature_write_rate_max"]) <= 0.05, "premature_write_rate_above_0.05"),
        (float(metrics["rtf_p95_max"]) < 1.0, "rtf_p95_not_realtime"),
    )
    failures = tuple(reason for passed, reason in checks if not passed)
    return GateResult("fail" if failures else "pass", failures, dict(metrics), thresholds)


def build_report(
    summaries: Sequence[Mapping[str, object]],
    *,
    causality_audit: Mapping[str, object] | None = None,
    cache_parity: Mapping[str, object] | None = None,
    generalization_metrics: Mapping[str, object] | None = None,
    subsecond_metrics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    mechanism = mechanism_gate(
        summaries, causality_audit=causality_audit, cache_parity=cache_parity
    )
    generalization = generalization_gate(
        generalization_metrics, mechanism_status=mechanism.status
    )
    formal_status = (
        "pass"
        if mechanism.status == generalization.status == "pass"
        else "not_evaluable"
        if "not_evaluable" in {mechanism.status, generalization.status}
        else "fail"
    )
    subsecond = subsecond_gate(subsecond_metrics, formal_status=formal_status)
    return {
        "schema_version": SCHEMA,
        "legacy_single_sample_quality_passed": all(
            bool(row.get("quality_passed", False))
            for summary in summaries
            for row in _rows(summary)
        ),
        "mechanism_gate": mechanism.to_dict(),
        "generalization_gate": generalization.to_dict(),
        "formal_status": formal_status,
        "subsecond_gate": subsecond.to_dict(),
        "subsecond_status": "pass" if formal_status == subsecond.status == "pass" else subsecond.status,
    }


def _load(path: str | None) -> dict[str, object] | None:
    return None if not path else json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--causality-audit")
    parser.add_argument("--cache-parity")
    parser.add_argument("--generalization-metrics")
    parser.add_argument("--subsecond-metrics")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite gate report: {output}")
    value = build_report(
        [_load(path) for path in args.summary],  # type: ignore[list-item]
        causality_audit=_load(args.causality_audit),
        cache_parity=_load(args.cache_parity),
        generalization_metrics=_load(args.generalization_metrics),
        subsecond_metrics=_load(args.subsecond_metrics),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()

