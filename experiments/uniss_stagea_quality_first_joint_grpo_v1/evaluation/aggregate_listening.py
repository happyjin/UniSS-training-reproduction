#!/usr/bin/env python3
"""Aggregate strict-cascade listening outputs without imposing a stop gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import sacrebleu


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


def summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    referenced = [row for row in rows if row.get("reference_translation")]
    first_audio = [
        float(row["first_audio_source_ms"])
        for row in rows
        if row.get("first_audio_source_ms") is not None
    ]
    source_ms = sum(float(row["source_duration_ms"]) for row in rows)
    processing_ms = sum(float(row["processing_seconds"]) * 1000.0 for row in rows)
    errors = sum(int(row["asr_errors"]) for row in referenced)
    units = sum(int(row["asr_reference_units"]) for row in referenced)
    by_direction: dict[str, dict[str, object]] = {}
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in referenced:
        groups[f"{row['src_lang']}->{row['tgt_lang']}"] .append(row)
    for direction, values in sorted(groups.items()):
        hypotheses = [str(value["generated_streaming_translation"]) for value in values]
        references = [str(value["reference_translation"]) for value in values]
        target = str(values[0]["tgt_lang"])
        by_direction[direction] = {
            "samples": len(values),
            "bleu": float(
                sacrebleu.corpus_bleu(
                    hypotheses,
                    [references],
                    tokenize="zh" if target == "cmn" else "13a",
                ).score
            ),
            "chrf": float(sacrebleu.corpus_chrf(hypotheses, [references]).score),
        }
    return {
        "samples": len(rows),
        "referenced_samples": len(referenced),
        "weighted_asr_error_rate": errors / units if units else None,
        "translation": by_direction,
        "strict_streaming_pass_rate": (
            sum(bool(row["strict_streaming_runtime_passed"]) for row in rows)
            / len(rows)
            if rows
            else None
        ),
        "prefinal_audio_rate": (
            sum(bool(row["prefinal_audio_emitted"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "healthy_audio_rate": (
            sum(bool(row["audio_audit"]["healthy"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "first_audio_source_ms": {
            "observed": len(first_audio),
            "mean": statistics.fmean(first_audio) if first_audio else None,
            "p50": _percentile(first_audio, 0.50),
            "p95": _percentile(first_audio, 0.95),
        },
        "mean_audio_writes": (
            statistics.fmean(float(row["audio_writes"]) for row in rows)
            if rows
            else None
        ),
        "mean_semantic_tokens": (
            statistics.fmean(float(row["semantic_tokens"]) for row in rows)
            if rows
            else None
        ),
        "runtime_rtf": processing_ms / source_ms if source_ms else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.root.glob("chunk_*ms/results.json"))
    if not paths:
        raise ValueError(f"no strict listening results under {args.root}")
    chunks: dict[str, object] = {}
    all_rows: list[Mapping[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete strict cascade output: {path}")
        rows = list(payload["results"])
        chunk = str(int(payload["decision_chunk_ms"]))
        chunks[chunk] = summarize(rows)
        all_rows.extend(rows)
    output = {
        "schema_version": "uniss_stagea_joint_grpo_listening_summary_v1",
        "status": "complete",
        "root": str(args.root.resolve()),
        "chunks_ms": chunks,
        "all_observations": summarize(all_rows),
        "result_paths": [str(path.resolve()) for path in paths],
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

