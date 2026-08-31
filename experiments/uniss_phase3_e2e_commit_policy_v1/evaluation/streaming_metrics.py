#!/usr/bin/env python3
"""Simultaneous-S2ST metrics over E2E free-running gate worker reports.

Covers the three capabilities separately (ASR, incremental MT, speech output)
plus the simultaneous-translation latency family, so a streaming run can be put
next to the offline Phase-3 baseline
(``stage00_phase3_offline_20260816T031129Z/baseline_summary.json``).

Latency definitions follow the speech formulation used by SimulEval, with the
source axis measured in milliseconds of input audio rather than source tokens:

    d_i   = source time already consumed when target unit i is emitted
    AL    = (1/tau) * sum_{i=1..tau} ( d_i - (i-1) * T_src / |Y*| )
            tau = min{ i : d_i >= T_src }
    LAAL  = same, with |Y*| replaced by max(|Y|, |Y*|)
    AP    = sum_i d_i / (T_src * |Y|)
    DAL   = (1/|Y|) * sum_i ( d'_i - (i-1) * T_src / |Y*| ),
            d'_i = max(d_i, d'_{i-1} + T_src / |Y*|)

LAAL is not an alias for AL.  ``training/simul_uniss/latency_metrics.py`` reports
``laal_glm_tokens = al``, which is only valid when the hypothesis and reference
have the same length.  This model over-generates speech by five to twelve times,
so the distinction is load-bearing here and LAAL is computed from its own
definition.

Every metric is source-timeline and therefore *not* computation aware: it says
when a unit became emittable given the audio consumed, not when a listener
heard it.  Wall-clock cost is reported separately as RTF.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from experiments.uniss_phase3_e2e_commit_policy_v1.runtime.local_agreement import (
    display_units,
)


SCHEMA = "uniss_e2e_streaming_s2st_metrics_v1"


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize(values: Sequence[float]) -> dict[str, float]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"n": 0}
    return {
        "n": len(numbers),
        "mean": statistics.fmean(numbers),
        "p50": percentile(numbers, 0.50),
        "p90": percentile(numbers, 0.90),
        "min": min(numbers),
        "max": max(numbers),
    }


def emission_times(
    events: Sequence[Mapping[str, object]], language: str
) -> tuple[list[float], list[float]]:
    """Source time in ms at which each target text unit / speech token appears."""

    text_times: list[float] = []
    speech_times: list[float] = []
    for event in events:
        moment = float(event["source_end_ms"])
        for delta in event.get("mt_deltas", []):  # type: ignore[union-attr]
            text_times.extend([moment] * len(display_units(str(delta), language)))
        count = int(event.get("semantic_tokens", 0) or 0)
        speech_times.extend([moment] * count)
    return text_times, speech_times


def latency_family(
    times: Sequence[float],
    *,
    source_duration_ms: float,
    reference_units: int,
) -> dict[str, object]:
    """AL / LAAL / AP / DAL plus first emission, all on the source timeline."""

    hypothesis_units = len(times)
    if not hypothesis_units or source_duration_ms <= 0:
        return {
            "hypothesis_units": hypothesis_units,
            "reference_units": int(reference_units),
            "emitted": False,
        }
    reference = max(1, int(reference_units))
    al_step = source_duration_ms / reference
    laal_step = source_duration_ms / max(hypothesis_units, reference)

    tau = next(
        (
            index + 1
            for index, value in enumerate(times)
            if float(value) >= source_duration_ms
        ),
        hypothesis_units,
    )
    al = statistics.fmean(
        float(times[index]) - index * al_step for index in range(tau)
    )
    laal = statistics.fmean(
        float(times[index]) - index * laal_step for index in range(tau)
    )
    average_proportion = sum(float(value) for value in times) / (
        source_duration_ms * hypothesis_units
    )
    delayed: list[float] = []
    for index, value in enumerate(times):
        floor = delayed[-1] + al_step if delayed else 0.0
        delayed.append(max(float(value), floor))
    dal = statistics.fmean(
        delayed[index] - index * al_step for index in range(hypothesis_units)
    )
    gaps = [
        float(right) - float(left)
        for left, right in zip(times, times[1:])
        if float(right) > float(left)
    ]
    return {
        "hypothesis_units": hypothesis_units,
        "reference_units": reference,
        "emitted": True,
        "first_emission_ms": float(times[0]),
        "last_emission_ms": float(times[-1]),
        "al_ms": al,
        "laal_ms": laal,
        "average_proportion": average_proportion,
        "dal_ms": dal,
        "tau": tau,
        "maximum_gap_ms": max(gaps) if gaps else 0.0,
        "length_ratio": hypothesis_units / reference,
    }


def action_counts(events: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        for value in event.get("chosen_continuations", []):  # type: ignore[union-attr]
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
    return counts


def sample_metrics(row: Mapping[str, object]) -> dict[str, object]:
    direction = f"{row['src_lang']}->{row['tgt_lang']}"
    source_ms = float(row["source_duration_ms"])
    asr = dict(row["e_asr"])  # type: ignore[arg-type]
    output: dict[str, object] = {
        "sample_id": str(row["sample_id"]),
        "direction": direction,
        "source_duration_ms": source_ms,
        "asr": {
            "metric": str(asr.get("metric")),
            "error_rate": float(asr.get("error_rate", 0.0)),
            "errors": int(asr.get("errors", 0)),
            "reference_units": int(asr.get("reference_units", 0)),
            "empty_events": int(asr.get("empty_events", 0)),
            "early_eos_events": int(asr.get("early_eos_events", 0)),
            "malformed_write_events": int(asr.get("malformed_write_events", 0)),
            "source_rollbacks": int(asr.get("source_rollbacks", 0)),
            "final_reached_eos": bool(asr.get("final_reached_eos", False)),
        },
    }
    for name, key in (("mt_gold", "e_mt_gold"), ("mt_free", "e_mt_free")):
        value = row.get(key)
        if not isinstance(value, Mapping):
            continue
        hypothesis = int(value.get("hypothesis_units", 0))
        reference = max(1, int(value.get("reference_units", 1)))
        output[name] = {
            "coverage": float(value.get("coverage", 0.0)),
            "hypothesis_units": hypothesis,
            "reference_units": reference,
            "length_ratio": hypothesis / reference,
            "commit_conflicts": int(value.get("commit_conflicts", 0)),
            "rollback_events": int(value.get("rollback_events", 0)),
            "events": int(value.get("events", 0)),
            "nonempty_events": int(value.get("nonempty_events", 0)),
            "final_hypothesis": str(value.get("final_hypothesis", "")),
        }
    s2s = row.get("e_s2s_free")
    if isinstance(s2s, Mapping):
        events = list(s2s.get("events", []))  # type: ignore[arg-type]
        audio = dict(s2s.get("audio") or {})
        text_times, speech_times = emission_times(events, str(row["tgt_lang"]))
        reference_text_units = max(
            1, int((row.get("e_mt_gold") or {}).get("reference_units", 1))  # type: ignore[union-attr]
        )
        reference_speech = max(1, int(s2s.get("semantic_reference_tokens", 1)))
        generated_speech = int(s2s.get("semantic_tokens", 0))
        duration_seconds = float(audio.get("duration_seconds") or 0.0)
        output["s2s"] = {
            "semantic_tokens": generated_speech,
            "semantic_reference_tokens": reference_speech,
            # The gate reports min(1, generated/reference), which cannot show
            # over-generation.  This ratio is deliberately unclamped.
            "semantic_length_ratio": generated_speech / reference_speech,
            "semantic_coverage_clamped": float(s2s.get("semantic_coverage", 0.0)),
            "malformed_segments": int(s2s.get("malformed_segments", 0)),
            "invalid_semantic_tokens": int(s2s.get("invalid_semantic_tokens", 0)),
            "natural_eos": bool(s2s.get("natural_eos", False)),
            "non_silent": bool(audio.get("non_silent", False)),
            "audio_duration_seconds": duration_seconds,
            "audio_rms": float(audio.get("rms") or 0.0),
            "audio_peak": float(audio.get("peak") or 0.0),
            "audio_over_source_ratio": (
                duration_seconds * 1000.0 / source_ms if source_ms else 0.0
            ),
            "target_text_before_source_eos": bool(
                s2s.get("target_text_before_source_eos", False)
            ),
            "target_semantic_before_source_eos": bool(
                s2s.get("target_semantic_before_source_eos", False)
            ),
            "events": len(events),
            "actions": action_counts(events),
            "latency_text": latency_family(
                text_times,
                source_duration_ms=source_ms,
                reference_units=reference_text_units,
            ),
            "latency_speech": latency_family(
                speech_times,
                source_duration_ms=source_ms,
                reference_units=reference_speech,
            ),
        }
    return output


def load_worker_rows(run_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((run_root / "workers").glob("worker_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["samples"])
    if not rows:
        raise RuntimeError(f"no worker samples under {run_root}")
    return rows


def _collect(values: Iterable[Mapping[str, object]], *path: str) -> list[float]:
    output: list[float] = []
    for value in values:
        cursor: object = value
        for key in path:
            if not isinstance(cursor, Mapping) or key not in cursor:
                cursor = None
                break
            cursor = cursor[key]
        if isinstance(cursor, bool):
            output.append(1.0 if cursor else 0.0)
        elif isinstance(cursor, (int, float)):
            output.append(float(cursor))
    return output


AGGREGATES = (
    ("asr.error_rate", ("asr", "error_rate")),
    ("asr.empty_events", ("asr", "empty_events")),
    ("asr.early_eos_events", ("asr", "early_eos_events")),
    ("mt_gold.coverage", ("mt_gold", "coverage")),
    ("mt_gold.length_ratio", ("mt_gold", "length_ratio")),
    ("mt_gold.commit_conflicts", ("mt_gold", "commit_conflicts")),
    ("mt_free.coverage", ("mt_free", "coverage")),
    ("mt_free.length_ratio", ("mt_free", "length_ratio")),
    ("mt_free.commit_conflicts", ("mt_free", "commit_conflicts")),
    ("s2s.semantic_length_ratio", ("s2s", "semantic_length_ratio")),
    ("s2s.malformed_segments", ("s2s", "malformed_segments")),
    ("s2s.audio_over_source_ratio", ("s2s", "audio_over_source_ratio")),
    ("s2s.non_silent", ("s2s", "non_silent")),
    ("s2s.natural_eos", ("s2s", "natural_eos")),
    ("s2s.pre_eos_speech", ("s2s", "target_semantic_before_source_eos")),
    ("latency_text.first_emission_ms", ("s2s", "latency_text", "first_emission_ms")),
    ("latency_text.al_ms", ("s2s", "latency_text", "al_ms")),
    ("latency_text.laal_ms", ("s2s", "latency_text", "laal_ms")),
    ("latency_text.dal_ms", ("s2s", "latency_text", "dal_ms")),
    ("latency_text.average_proportion", ("s2s", "latency_text", "average_proportion")),
    ("latency_speech.first_emission_ms", ("s2s", "latency_speech", "first_emission_ms")),
    ("latency_speech.al_ms", ("s2s", "latency_speech", "al_ms")),
    ("latency_speech.laal_ms", ("s2s", "latency_speech", "laal_ms")),
    ("latency_speech.dal_ms", ("s2s", "latency_speech", "dal_ms")),
    ("latency_speech.maximum_gap_ms", ("s2s", "latency_speech", "maximum_gap_ms")),
    ("latency_speech.average_proportion", ("s2s", "latency_speech", "average_proportion")),
)


def aggregate(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    def block(subset: Sequence[Mapping[str, object]]) -> dict[str, object]:
        return {
            "samples": len(subset),
            **{
                name: summarize(_collect(subset, *path))
                for name, path in AGGREGATES
            },
        }

    directions = sorted({str(value["direction"]) for value in samples})
    return {
        "all": block(samples),
        "by_direction": {
            direction: block(
                [value for value in samples if value["direction"] == direction]
            )
            for direction in directions
        },
    }


def evaluate_run(run_root: Path, label: str) -> dict[str, object]:
    rows = load_worker_rows(run_root)
    samples = [sample_metrics(row) for row in rows]
    gate_path = run_root / "E2E_FREE_RUNNING_GATE.json"
    gate = (
        json.loads(gate_path.read_text(encoding="utf-8"))
        if gate_path.is_file()
        else None
    )
    return {
        "label": label,
        "run_root": str(run_root),
        "gate_status": (gate or {}).get("status"),
        "gate_failed_checks": sorted(
            key for key, value in (gate or {}).get("checks", {}).items() if not value
        ),
        "candidate": (gate or {}).get("candidate"),
        "aggregate": aggregate(samples),
        "samples": samples,
    }


def offline_reference(path: Path) -> dict[str, object]:
    baseline = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "asr_error": baseline.get("quality_asr_error"),
        "text_bleu": {
            key: value["score"]
            for key, value in baseline["text_translation_bleu"]["groups"].items()
        },
        "audio_subset_bleu": {
            key: value["score"]
            for key, value in baseline["audio_subset_translation_bleu"]["groups"].items()
        },
        "audio_duration_ratio": {
            key: {
                "mean": value["duration_ratio_mean"],
                "slc_0_2": value["slc_0_2"],
                "slc_0_4": value["slc_0_4"],
            }
            for key, value in baseline["audio_slc"]["groups"].items()
        },
        "semantic_tokens_mean": baseline["audio_health"]["semantic_tokens_mean"],
        "missing_eos": baseline["audio_health"]["counts"]["missing_eos"],
        "rows": baseline["audio_health"]["counts"]["rows"],
        "generation": baseline.get("generation"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=RUN_ROOT",
        help="a free-running gate run directory to score",
    )
    parser.add_argument("--offline-baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for value in args.run:
        if "=" not in value:
            raise ValueError(f"--run needs LABEL=RUN_ROOT, got {value}")
        label, _, root = value.partition("=")
        runs.append(evaluate_run(Path(root), label))
    report: dict[str, object] = {
        "schema_version": SCHEMA,
        "claim_scope": "frozen_fixed16_selection_train_seen",
        "latency_note": (
            "source-timeline, not computation aware; LAAL uses "
            "max(hypothesis, reference) length per Papi et al. 2022"
        ),
        "runs": runs,
    }
    if args.offline_baseline:
        report["offline_reference"] = offline_reference(args.offline_baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    for run in runs:
        block = run["aggregate"]["all"]  # type: ignore[index]
        print(
            json.dumps(
                {
                    "label": run["label"],
                    "samples": block["samples"],
                    "asr_error_rate": block["asr.error_rate"].get("mean"),
                    "mt_gold_coverage": block["mt_gold.coverage"].get("mean"),
                    "speech_laal_ms": block["latency_speech.laal_ms"].get("mean"),
                    "speech_first_ms": block["latency_speech.first_emission_ms"].get("mean"),
                    "semantic_length_ratio": block["s2s.semantic_length_ratio"].get("mean"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
