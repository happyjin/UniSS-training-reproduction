#!/usr/bin/env python3
"""Strictly merge matching Phase3 ASR workers and compute WER/CER."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean

from evaluation.text_metrics import normalize_for_bleu


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for left in reference:
        current = [previous[0] + 1]
        for index, right in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[index] + 1,
                    previous[index - 1] + int(left != right),
                )
            )
        previous = current
    return previous[-1]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite matching offline merge: {path}")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=8)
    parser.add_argument("--expected-records", type=int, default=334)
    args = parser.parse_args()
    manifest = read_jsonl(args.manifest)
    manifest_by_id = {str(row["id"]): row for row in manifest}
    if len(manifest) != args.expected_records or len(manifest_by_id) != len(manifest):
        raise ValueError("matching offline manifest count or uniqueness differs")
    rows: list[dict[str, object]] = []
    for rank in range(args.parts):
        path = args.root / "parts" / f"part_{rank:02d}" / "results.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            sample_id = str(row["id"])
            if sample_id not in manifest_by_id:
                raise ValueError(f"worker returned foreign sample: {sample_id}")
            expected_rank = int(manifest_by_id[sample_id]["worker_rank"])
            if expected_rank != rank:
                raise ValueError(f"worker rank differs for {sample_id}: {rank} vs {expected_rank}")
            rows.append({**manifest_by_id[sample_id], **row, "worker_rank": rank})
    ids = [str(row["id"]) for row in rows]
    if len(rows) != args.expected_records or len(set(ids)) != len(rows):
        raise ValueError("matching offline worker coverage has duplicates or omissions")
    if set(ids) != set(manifest_by_id):
        raise ValueError("matching offline worker IDs differ from manifest")
    rows.sort(key=lambda row: str(row["id"]))
    merged = args.root / "merged_results.jsonl"
    if merged.exists():
        raise FileExistsError(f"refusing to overwrite matching offline merge: {merged}")
    with merged.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["src_lang"]))].append(row)
    metrics: dict[str, dict[str, object]] = {}
    for (task, language), values in sorted(grouped.items()):
        errors = units = empty = stop_failures = 0
        for row in values:
            reference = normalize_for_bleu(str(row["transcription_ref"]), language).split()
            hypothesis = normalize_for_bleu(
                str(row["generated_transcription"]), language
            ).split()
            errors += edit_distance(reference, hypothesis)
            units += len(reference)
            empty += int(not hypothesis)
            stop_failures += int(not bool(row["reached_end_content"]))
        metrics[f"{task}:{language}"] = {
            "metric": "wer" if language == "eng" else "cer",
            "samples": len(values),
            "errors": errors,
            "reference_units": units,
            "error_rate": errors / max(1, units),
            "empty_hypotheses": empty,
            "stop_failures": stop_failures,
        }
    overall_by_language: dict[str, dict[str, object]] = {}
    for language in ("cmn", "eng"):
        values = [row for row in rows if row["src_lang"] == language]
        errors = units = 0
        for row in values:
            reference = normalize_for_bleu(str(row["transcription_ref"]), language).split()
            hypothesis = normalize_for_bleu(
                str(row["generated_transcription"]), language
            ).split()
            errors += edit_distance(reference, hypothesis)
            units += len(reference)
        overall_by_language[language] = {
            "metric": "wer" if language == "eng" else "cer",
            "samples": len(values),
            "errors": errors,
            "reference_units": units,
            "error_rate": errors / max(1, units),
        }
    summary = {
        "schema_version": "uniss_quality_first_stage_a_matching_offline_asr_v1",
        "passed": all(bool(row["reached_end_content"]) for row in rows),
        "manifest": str(args.manifest.resolve()),
        "records": len(rows),
        "unique_ids": len(set(ids)),
        "metrics_by_task_language": metrics,
        "metrics_by_language": overall_by_language,
        "empty_hypotheses": sum(not str(row["generated_transcription"]) for row in rows),
        "stop_failures": sum(not bool(row["reached_end_content"]) for row in rows),
        "generation_seconds_mean": mean(float(row["generation_seconds"]) for row in rows),
        "merged_results": str(merged.resolve()),
    }
    if any(math.isnan(float(value["error_rate"])) for value in overall_by_language.values()):
        raise ValueError("matching offline ASR produced NaN error rate")
    atomic_json(args.root / "matching_offline_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
