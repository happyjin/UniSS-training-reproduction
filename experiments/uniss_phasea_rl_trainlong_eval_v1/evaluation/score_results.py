#!/usr/bin/env python3
"""Reference and runtime scoring for train-seen long-episode inference."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import sacrebleu
import soundfile as sf


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mean(values: Iterable[float]) -> float | None:
    materialized = [float(value) for value in values]
    return statistics.fmean(materialized) if materialized else None


def edit_distance(reference: Sequence[object], hypothesis: Sequence[object]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, left in enumerate(reference, 1):
        current = [row]
        for column, right in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + int(left != right),
                )
            )
        previous = current
    return previous[-1]


def asr_units(text: str, language: str) -> list[str]:
    if language == "cmn":
        return list("".join(text.split()))
    return text.lower().split()


def content_units(text: str, language: str) -> list[str]:
    """Units used for coverage/repetition, excluding punctuation and spacing."""
    if language == "cmn":
        chunks = re.findall(r"[\u3400-\u9fffA-Za-z0-9]+", text.lower())
        return list("".join(chunks))
    return re.findall(r"[\w']+", text.lower(), flags=re.UNICODE)


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for lvalue in left:
        current = [0]
        for column, rvalue in enumerate(right, 1):
            current.append(
                previous[column - 1] + 1
                if lvalue == rvalue
                else max(previous[column], current[-1])
            )
        previous = current
    return previous[-1]


def ngram_repetition(units: Sequence[str], order: int = 4) -> dict[str, float | int]:
    total = max(0, len(units) - order + 1)
    if total == 0:
        return {
            "order": order,
            "total": 0,
            "unique": 0,
            "repeated_occurrences": 0,
            "repetition_rate": 0.0,
            "maximum_frequency": 0,
        }
    counts = Counter(tuple(units[index : index + order]) for index in range(total))
    repeated = sum(max(0, count - 1) for count in counts.values())
    return {
        "order": order,
        "total": total,
        "unique": len(counts),
        "repeated_occurrences": repeated,
        "repetition_rate": repeated / total,
        "maximum_frequency": max(counts.values()),
    }


def audit_wav(path: str, expected_channels: int) -> dict[str, Any]:
    audio_path = Path(path)
    if not audio_path.is_file():
        return {"path": str(audio_path), "exists": False, "healthy": False}
    values, rate = sf.read(audio_path, dtype="float32", always_2d=True)
    finite = bool(np.isfinite(values).all())
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))) if values.size else 0.0
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    non_silent = float(np.mean(np.abs(values) >= 1.0e-4)) if values.size else 0.0
    healthy = bool(
        len(values)
        and int(rate) == 16_000
        and values.shape[1] == expected_channels
        and finite
        and rms >= 1.0e-5
        and non_silent >= 0.01
        and peak < 1.2
    )
    return {
        "path": str(audio_path.resolve()),
        "exists": True,
        "sample_rate": int(rate),
        "channels": int(values.shape[1]),
        "frames": int(len(values)),
        "duration_seconds": len(values) / max(1, int(rate)),
        "finite": finite,
        "rms": rms,
        "peak": peak,
        "non_silent_fraction": non_silent,
        "healthy": healthy,
    }


def write_gaps(row: dict[str, Any]) -> list[float]:
    source_times = [
        float(value["source_available_ms"]) for value in row.get("playback_schedule", [])
    ]
    return [right - left for left, right in zip(source_times, source_times[1:])]


def score_row(result: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    src_lang, tgt_lang = str(result["src_lang"]), str(result["tgt_lang"])
    if (src_lang, tgt_lang) != (
        str(reference["src_lang"]),
        str(reference["tgt_lang"]),
    ):
        raise ValueError(f"language mismatch for {result['sample_id']}")
    asr_reference = str(reference["reference_transcription"])
    asr_hypothesis = str(result["generated_streaming_transcription"])
    left, right = asr_units(asr_reference, src_lang), asr_units(asr_hypothesis, src_lang)
    asr_errors = edit_distance(left, right)

    mt_reference = str(reference["reference_translation"])
    mt_hypothesis = str(result["generated_streaming_translation"])
    mt_left, mt_right = content_units(mt_reference, tgt_lang), content_units(
        mt_hypothesis, tgt_lang
    )
    coverage = lcs_length(mt_left, mt_right) / max(1, len(mt_left))
    repetition = ngram_repetition(mt_right, order=4)
    tokenization = "zh" if tgt_lang == "cmn" else "13a"
    gaps = write_gaps(result)
    metrics = {
        "asr_metric": "cer" if src_lang == "cmn" else "wer",
        "asr_errors": asr_errors,
        "asr_reference_units": len(left),
        "asr_error_rate": asr_errors / max(1, len(left)),
        "asr_normalized_similarity": max(0.0, 1.0 - asr_errors / max(1, len(left))),
        "mt_sentence_bleu": float(
            sacrebleu.sentence_bleu(
                mt_hypothesis,
                [mt_reference],
                tokenize=tokenization,
                use_effective_order=True,
            ).score
        ),
        "mt_sentence_chrf": float(
            sacrebleu.sentence_chrf(mt_hypothesis, [mt_reference]).score
        ),
        "final_translation_lcs_coverage": coverage,
        "translation_length_ratio": len(mt_right) / max(1, len(mt_left)),
        "translation_4gram_repetition": repetition,
        "write_gaps_ms": gaps,
        "independent_wav_audit": {
            "continuous": audit_wav(str(result["continuous_audio_path"]), 1),
            "timeline": audit_wav(str(result["timeline_audio_path"]), 1),
            "stereo": audit_wav(str(result["stereo_audio_path"]), 2),
        },
    }
    return {
        **result,
        "reference_transcription": asr_reference,
        "reference_translation": mt_reference,
        "train_seen_provenance": {
            "rl_train_seen": bool(reference["rl_train_seen"]),
            "formal_rollout_seen": bool(reference["formal_rollout_seen"]),
            "validation_overlap": bool(reference["validation_overlap"]),
            "component_count": int(reference["component_count"]),
            "source_audio_sha256": str(reference["source_audio_sha256"]),
        },
        "reference_metrics": metrics,
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    errors = sum(int(row["reference_metrics"]["asr_errors"]) for row in rows)
    units = sum(int(row["reference_metrics"]["asr_reference_units"]) for row in rows)
    first = [
        float(row["first_audio_source_ms"])
        for row in rows
        if row.get("first_audio_source_ms") is not None
    ]
    gaps = [
        float(value)
        for row in rows
        for value in row["reference_metrics"]["write_gaps_ms"]
    ]
    row_gap_summaries = [
        row.get("inter_write_gap_ms", {})
        for row in rows
        if isinstance(row.get("inter_write_gap_ms"), dict)
    ]
    metric_names = {
        str(row["reference_metrics"]["asr_metric"]) for row in rows
    }
    hypotheses = [str(row["generated_streaming_translation"]) for row in rows]
    references = [str(row["reference_translation"]) for row in rows]
    target = str(rows[0]["tgt_lang"]) if rows else "eng"
    return {
        "samples": len(rows),
        "source_duration_seconds": sum(float(row["source_duration_ms"]) for row in rows)
        / 1000.0,
        "asr_metric": (
            next(iter(metric_names))
            if len(metric_names) == 1
            else "mixed_cer_wer" if metric_names else None
        ),
        "asr_errors": errors,
        "asr_reference_units": units,
        "asr_error_rate": errors / units if units else None,
        "asr_normalized_similarity_mean": mean(
            row["reference_metrics"]["asr_normalized_similarity"] for row in rows
        ),
        "mt_corpus_bleu": (
            float(
                sacrebleu.corpus_bleu(
                    hypotheses,
                    [references],
                    tokenize="zh" if target == "cmn" else "13a",
                ).score
            )
            if rows
            else None
        ),
        "mt_corpus_chrf": (
            float(sacrebleu.corpus_chrf(hypotheses, [references]).score)
            if rows
            else None
        ),
        "final_translation_lcs_coverage_mean": mean(
            row["reference_metrics"]["final_translation_lcs_coverage"] for row in rows
        ),
        "translation_length_ratio_mean": mean(
            row["reference_metrics"]["translation_length_ratio"] for row in rows
        ),
        "translation_4gram_repetition_rate_mean": mean(
            row["reference_metrics"]["translation_4gram_repetition"]["repetition_rate"]
            for row in rows
        ),
        "translation_4gram_maximum_frequency": max(
            (
                int(row["reference_metrics"]["translation_4gram_repetition"]["maximum_frequency"])
                for row in rows
            ),
            default=0,
        ),
        "first_audio_source_ms": {
            "observed": len(first),
            "mean": mean(first),
            "p50": percentile(first, 0.50),
            "p95": percentile(first, 0.95),
            "maximum": max(first) if first else None,
        },
        "write_gap_ms": {
            "observed": (
                len(gaps)
                if gaps
                else sum(max(0, int(row["audio_writes"]) - 1) for row in rows)
            ),
            "mean": (
                mean(gaps)
                if gaps
                else mean(
                    value["mean"]
                    for value in row_gap_summaries
                    if value.get("mean") is not None
                )
            ),
            "p50": (
                percentile(gaps, 0.50)
                if gaps
                else mean(
                    value["p50"]
                    for value in row_gap_summaries
                    if value.get("p50") is not None
                )
            ),
            "p95": (
                percentile(gaps, 0.95)
                if gaps
                else mean(
                    value["p95"]
                    for value in row_gap_summaries
                    if value.get("p95") is not None
                )
            ),
            "maximum": (
                max(gaps)
                if gaps
                else max(
                    (
                        float(value["maximum"])
                        for value in row_gap_summaries
                        if value.get("maximum") is not None
                    ),
                    default=None,
                )
            ),
        },
        "maximum_internal_timeline_silence_ms_mean": mean(
            row["maximum_internal_timeline_silence_ms"] for row in rows
        ),
        "maximum_internal_timeline_silence_ms_max": max(
            (float(row["maximum_internal_timeline_silence_ms"]) for row in rows),
            default=None,
        ),
        "translation_audio_to_source_duration_ratio_mean": mean(
            row["translation_audio_to_source_duration_ratio"] for row in rows
        ),
        "audio_writes_total": sum(int(row["audio_writes"]) for row in rows),
        "audio_writes_mean": mean(row["audio_writes"] for row in rows),
        "prefinal_audio_rate": mean(float(bool(row["prefinal_audio_emitted"])) for row in rows),
        "pending_unspoken_total": sum(int(row["tts_pending_unspoken_items"]) for row in rows),
        "tts_failures_total": sum(int(row["tts_failures"]) for row in rows),
        "rejected_early_end_total": sum(int(row["rejected_early_end"]) for row in rows),
        "semantic_continuations_total": sum(int(row["semantic_continuations"]) for row in rows),
        "stateful_runtime_pass_rate": mean(
            float(bool(row["stateful_runtime_passed"])) for row in rows
        ),
        "continuous_wav_health_rate": mean(
            float(bool(row["reference_metrics"]["independent_wav_audit"]["continuous"]["healthy"]))
            for row in rows
        ),
        "timeline_wav_health_rate": mean(
            float(bool(row["reference_metrics"]["independent_wav_audit"]["timeline"]["healthy"]))
            for row in rows
        ),
        "stereo_wav_health_rate": mean(
            float(bool(row["reference_metrics"]["independent_wav_audit"]["stereo"]["healthy"]))
            for row in rows
        ),
        "rtf_mean": mean(row["rtf"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    raw = json.loads(args.results.read_text(encoding="utf-8"))
    references = {str(row["sample_id"]): row for row in protocol["records"]}
    result_ids = [str(row["sample_id"]) for row in raw["results"]]
    if result_ids != list(references):
        raise ValueError("result IDs/order differ from the immutable protocol")
    rows = [score_row(row, references[str(row["sample_id"])]) for row in raw["results"]]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['src_lang']}->{row['tgt_lang']}"].append(row)
    output = {
        "schema_version": "uniss_phasea_rl_train_seen_reference_score_v1",
        "status": "complete",
        "run_id": args.run_id,
        "claim_boundary": protocol["claim_boundary"],
        "protocol": str(args.protocol.resolve()),
        "raw_results": str(args.results.resolve()),
        "model_manifest": raw.get("adapter_manifest"),
        "results": rows,
        "aggregate": {
            "overall": summarize(rows),
            "by_direction": {
                direction: summarize(values) for direction, values in sorted(grouped.items())
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["aggregate"], ensure_ascii=False, indent=2))
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
