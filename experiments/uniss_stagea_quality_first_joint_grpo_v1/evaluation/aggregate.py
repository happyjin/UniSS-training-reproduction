#!/usr/bin/env python3
"""Aggregate routed workers without using a quality gate to stop training."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Mapping, Sequence

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation import gate
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.gate import (
    WORKER_SCHEMA,
    text_units,
    write_new_json,
)


SCHEMA = "uniss_stagea_quality_first_joint_grpo_eval_summary_v1"


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _latency(samples: Sequence[Mapping[str, object]], key: str) -> dict[str, object]:
    first_text: list[float] = []
    first_semantic: list[float] = []
    event_compute: list[float] = []
    ap: list[float] = []
    al: list[float] = []
    dal: list[float] = []
    wait = write_mt = write_semantic = write_asr = 0
    total_compute_ms = 0.0
    total_source_ms = 0.0
    valid = 0
    for sample in samples:
        value = sample.get(key)
        if not isinstance(value, Mapping):
            continue
        valid += 1
        duration = float(sample["source_duration_ms"])
        total_source_ms += duration
        timestamps: list[float] = []
        events = value.get("events")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, Mapping):
                continue
            source_ms = float(event["source_end_ms"])
            compute_ms = float(event.get("compute_ms", 0.0))
            event_compute.append(compute_ms)
            total_compute_ms += compute_ms
            continuations = [str(item) for item in event.get("chosen_continuations", [])]
            wait += sum(item == "WAIT" for item in continuations)
            write_mt += sum(item == "WRITE_MT" for item in continuations)
            write_semantic += sum(item == "WRITE_SEMANTIC" for item in continuations)
            write_asr += sum(item == "WRITE_ASR" for item in continuations)
            deltas = [str(item) for item in event.get("mt_deltas", [])]
            units = sum(
                len(text_units(delta, str(sample["tgt_lang"]))) for delta in deltas
            )
            timestamps.extend([source_ms] * units)
        text_times = [
            float(event["source_end_ms"])
            for event in events
            if isinstance(event, Mapping) and event.get("mt_deltas")
        ]
        semantic_times = [
            float(event["source_end_ms"])
            for event in events
            if isinstance(event, Mapping) and int(event.get("semantic_tokens", 0)) > 0
        ]
        if text_times:
            first_text.append(min(text_times))
        if semantic_times:
            first_semantic.append(min(semantic_times))
        if timestamps and duration > 0:
            target = len(timestamps)
            ideal = duration / target
            ap.append(sum(timestamps) / (target * duration))
            al.append(
                sum(value - index * ideal for index, value in enumerate(timestamps))
                / target
            )
            delayed: list[float] = []
            for index, value in enumerate(timestamps):
                delayed.append(
                    value
                    if index == 0
                    else max(value, delayed[-1] + ideal)
                )
            dal.append(
                sum(value - index * ideal for index, value in enumerate(delayed))
                / target
            )
    return {
        "samples": valid,
        "first_text_write_ms": {
            "mean": statistics.fmean(first_text) if first_text else None,
            "p50": _percentile(first_text, 0.50),
            "p95": _percentile(first_text, 0.95),
            "observed": len(first_text),
        },
        "first_semantic_write_ms": {
            "mean": statistics.fmean(first_semantic) if first_semantic else None,
            "p50": _percentile(first_semantic, 0.50),
            "p95": _percentile(first_semantic, 0.95),
            "observed": len(first_semantic),
        },
        "average_proportion": statistics.fmean(ap) if ap else None,
        "average_lagging_ms": statistics.fmean(al) if al else None,
        "differentiable_average_lagging_ms": statistics.fmean(dal) if dal else None,
        "event_compute_ms": {
            "mean": statistics.fmean(event_compute) if event_compute else None,
            "p50": _percentile(event_compute, 0.50),
            "p95": _percentile(event_compute, 0.95),
        },
        "generation_rtf": (
            total_compute_ms / total_source_ms if total_source_ms > 0 else None
        ),
        "actions": {
            "wait": wait,
            "write_asr": write_asr,
            "write_mt": write_mt,
            "write_semantic": write_semantic,
        },
    }


def _as_stage_a(samples: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for sample in samples:
        value = dict(sample)
        value["e_mt_gold"] = sample["stage_a_mt_gold"]
        value["e_mt_free"] = sample["stage_a_mt_free"]
        value["e_s2s_free"] = sample.get("stage_a_e_s2s_free")
        output.append(value)
    return output


def build_summary(
    worker_reports: Sequence[Path], selection: Path
) -> dict[str, object]:
    workers = [json.loads(path.read_text(encoding="utf-8")) for path in worker_reports]
    if not workers or any(
        value.get("schema_version") != WORKER_SCHEMA
        or value.get("status") != "complete"
        for value in workers
    ):
        raise ValueError("worker report schema/status differs")
    expected = int(workers[0]["num_workers"])
    indices = sorted(int(value["worker_index"]) for value in workers)
    if len(workers) != expected or indices != list(range(expected)):
        raise ValueError("worker set is incomplete")
    run_ids = {str(value["run_id"]) for value in workers}
    if len(run_ids) != 1:
        raise ValueError("worker run IDs differ")
    samples = [sample for worker in workers for sample in worker["samples"]]
    selected = json.loads(selection.read_text(encoding="utf-8"))
    selected_ids = {str(value["sample_id"]) for value in selected["records"]}
    observed_ids = {str(value["sample_id"]) for value in samples}
    if len(samples) != len(selected_ids) or observed_ids != selected_ids:
        raise ValueError("workers do not exactly cover fixed selection")
    stage_a = _as_stage_a(samples)
    candidate_metrics = {
        "e_asr": gate._weighted_asr(samples),
        "e_mt": gate._mt_summary(samples),
        "e_s2s_free": gate._s2s_summary(samples),
        "latency": _latency(samples, "e_s2s_free"),
    }
    stage_a_metrics = {
        "e_asr": gate._weighted_asr(stage_a),
        "e_mt": gate._mt_summary(stage_a),
        "e_s2s_free": gate._s2s_summary(stage_a),
        "latency": _latency(stage_a, "e_s2s_free"),
    }
    return {
        "schema_version": SCHEMA,
        "status": "complete",
        "run_id": next(iter(run_ids)),
        "selection": str(selection.resolve()),
        "samples": len(samples),
        "candidate": candidate_metrics,
        "stage_a": stage_a_metrics,
        "adapter_manifest": workers[0]["adapter_manifest"],
        "worker_reports": [str(path.resolve()) for path in worker_reports],
        "comparison_note": (
            "ASR is expected to be identical because the adapter is disabled on "
            "ASR routes. MT/TTS/control use the adapter; Stage A disables it globally."
        ),
        "route_limit": (
            "Training uses oracle next-token loss-family masks. Free-running evaluation "
            "uses a deterministic state-machine approximation: off inside ASR, on for "
            "MT, semantic TTS and external control decisions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--worker-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.worker_report, args.selection)
    write_new_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
