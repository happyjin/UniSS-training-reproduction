#!/usr/bin/env python3
"""Merge eight Phase3 baseline workers and compute deterministic gate metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from evaluation.slc_metrics import aggregate_slc, compute_slc_rows
from evaluation.text_metrics import compute_grouped_bleu, normalize_for_bleu


def _atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {path}")
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


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _merge(root: Path, kind: str, expected_parts: int) -> list[dict[str, Any]]:
    rows = []
    for rank in range(expected_parts):
        path = root / kind / f"part{rank:02d}" / "results.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing baseline worker output: {path}")
        for row in _iter_jsonl(path):
            row["worker_rank"] = rank
            rows.append(row)
    seen = Counter((str(row.get("id")), str(row.get("mode"))) for row in rows)
    duplicates = [list(key) for key, value in seen.items() if value != 1]
    if duplicates:
        raise ValueError(f"duplicate or missing id/mode pairs: {duplicates[:10]}")
    rows.sort(key=lambda row: (str(row.get("id")), str(row.get("mode"))))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite merged baseline: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            row = {"global_index": index, **row}
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref in reference:
        current = [previous[0] + 1]
        for index, hyp in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[index] + 1,
                    previous[index - 1] + (ref != hyp),
                )
            )
        previous = current
    return previous[-1]


def _asr_error(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("mode") != "quality":
            continue
        groups[f"{row.get('src_lang')}->{row.get('tgt_lang')}"].append(row)
    output = {}
    for direction, values in sorted(groups.items()):
        edits = units = empty = 0
        for row in values:
            language = str(row["src_lang"])
            reference = normalize_for_bleu(str(row.get("transcription_ref") or ""), language).split()
            hypothesis = normalize_for_bleu(str(row.get("generated_transcription") or ""), language).split()
            empty += int(not hypothesis)
            edits += _edit_distance(reference, hypothesis)
            units += len(reference)
        output[direction] = {
            "metric": "WER" if direction.startswith("eng") else "CER",
            "error_rate": edits / units if units else math.nan,
            "edits": edits,
            "reference_units": units,
            "samples": len(values),
            "empty_hypotheses": empty,
        }
    return output


def _health(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    generation = []
    semantic = []
    for row in rows:
        counts["rows"] += 1
        counts[f"mode:{row.get('mode')}"] += 1
        counts[f"direction:{row.get('src_lang')}->{row.get('tgt_lang')}"] += 1
        counts["errors"] += int(bool(row.get("error")))
        counts["missing_eos"] += int(not bool(row.get("has_eos")))
        counts["missing_semantic"] += int(int(row.get("semantic_token_count") or 0) == 0)
        generation.append(float(row.get("generation_seconds") or 0.0))
        semantic.append(int(row.get("semantic_token_count") or 0))
    return {
        "counts": dict(sorted(counts.items())),
        "generation_seconds_mean": mean(generation),
        "generation_seconds_p50": float(np.percentile(generation, 50)),
        "generation_seconds_p95": float(np.percentile(generation, 95)),
        "semantic_tokens_mean": mean(semantic),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--parts", type=int, default=8)
    parser.add_argument("--expected-text-records", type=int, default=256)
    parser.add_argument("--expected-audio-records", type=int, default=64)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    text_rows = _merge(root, "text", args.parts)
    audio_rows = _merge(root, "audio", args.parts)
    expected_text_rows = args.expected_text_records * 2
    expected_audio_rows = args.expected_audio_records * 4
    if len(text_rows) != expected_text_rows or len(audio_rows) != expected_audio_rows:
        raise ValueError(
            f"baseline row counts differ: text={len(text_rows)}/{expected_text_rows}, "
            f"audio={len(audio_rows)}/{expected_audio_rows}"
        )
    merged = root / "merged"
    text_path = merged / "text_results.jsonl"
    audio_path = merged / "audio_results.jsonl"
    _write_jsonl(text_path, text_rows)
    _write_jsonl(audio_path, audio_rows)
    text_bleu = compute_grouped_bleu(
        text_rows,
        hypothesis_field="generated_translation",
        reference_field="translation_ref",
        score_empty_hypotheses=True,
    )
    audio_translation_bleu = compute_grouped_bleu(
        [row for row in audio_rows if row["mode"] in {"quality", "performance"}],
        hypothesis_field="generated_translation",
        reference_field="translation_ref",
        score_empty_hypotheses=True,
    )
    slc_rows, slc_skipped = compute_slc_rows(audio_rows, results_path=audio_path)
    slc = aggregate_slc(slc_rows, slc_skipped)
    summary = {
        "schema_version": "uniss_stage00_phase3_offline_baseline_v1",
        "passed": True,
        "model": str(
            text_rows[0].get("checkpoint") if text_rows else ""
        ),
        "generation": {
            "temperature": 0.0,
            "repetition_penalty": 1.1,
            "max_new_tokens": 1500,
        },
        "text_records": args.expected_text_records,
        "audio_records": args.expected_audio_records,
        "text_health": _health(text_rows),
        "audio_health": _health(audio_rows),
        "text_translation_bleu": text_bleu,
        "audio_subset_translation_bleu": audio_translation_bleu,
        "quality_asr_error": _asr_error(text_rows),
        "audio_slc": slc,
        "merged_text_results": str(text_path),
        "merged_audio_results": str(audio_path),
    }
    _atomic_json(root / "baseline_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
