#!/usr/bin/env python3
"""Read-only audit for the UniST input used by StreamSpeech-CTC v1."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / (
    "data/processed/simul_uniss_subsecond_v1/pilot_15shard/"
    "stage_a_source/stage_a_source_manifest.jsonl"
)
REQUIRED_FIELDS = {
    "id",
    "src_lang",
    "tgt_lang",
    "transcription",
    "translation",
    "source_glm",
    "source_audio",
    "source_duration_ms",
    "target_bicodec",
    "bicodec_global",
}
VALID_DIRECTIONS = {("eng", "cmn"), ("cmn", "eng")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--offsets", type=Path)
    parser.add_argument("--sample-records", type=int, default=50_000)
    parser.add_argument("--audio-path-checks", type=int, default=1_024)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def even_indices(total: int, requested: int) -> np.ndarray:
    count = min(total, max(1, requested))
    if count == total:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, num=count, dtype=np.int64)


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    offsets_path = (
        args.offsets.resolve()
        if args.offsets
        else Path(str(manifest) + ".offsets.bin")
    )
    if not manifest.is_file() or not offsets_path.is_file():
        raise FileNotFoundError(f"missing manifest/index: {manifest}, {offsets_path}")
    if offsets_path.stat().st_size % 8:
        raise ValueError("offset index size is not divisible by uint64 width")

    offsets = np.memmap(offsets_path, mode="r", dtype=np.uint64)
    total = int(offsets.shape[0])
    indices = even_indices(total, args.sample_records)
    directions: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    empty_text: Counter[str] = Counter()
    malformed = 0
    invalid_direction = 0
    invalid_duration = 0
    invalid_glm_geometry = 0
    audio_missing = 0
    durations: list[float] = []
    glm_lengths: list[float] = []
    src_chars: list[float] = []
    tgt_chars: list[float] = []
    audio_stride = max(1, len(indices) // max(1, args.audio_path_checks))
    audio_checked = 0

    with manifest.open("rb") as handle:
        for sampled_position, record_index in enumerate(indices.tolist()):
            handle.seek(int(offsets[record_index]))
            raw = handle.readline()
            try:
                row: dict[str, Any] = json.loads(raw)
            except Exception:
                malformed += 1
                continue
            absent = REQUIRED_FIELDS.difference(row)
            missing_fields.update(absent)
            src = str(row.get("src_lang", ""))
            tgt = str(row.get("tgt_lang", ""))
            directions[f"{src}->{tgt}"] += 1
            if (src, tgt) not in VALID_DIRECTIONS:
                invalid_direction += 1
            transcription = str(row.get("transcription", "")).strip()
            translation = str(row.get("translation", "")).strip()
            if not transcription:
                empty_text["transcription"] += 1
            if not translation:
                empty_text["translation"] += 1
            src_chars.append(float(len(transcription)))
            tgt_chars.append(float(len(translation)))
            duration = float(row.get("source_duration_ms", 0) or 0)
            durations.append(duration)
            if not math.isfinite(duration) or duration <= 0:
                invalid_duration += 1
            glm = row.get("source_glm") or []
            glm_lengths.append(float(len(glm)))
            if duration > 0 and abs(len(glm) * 80 - duration) > 240:
                invalid_glm_geometry += 1
            if sampled_position % audio_stride == 0 and audio_checked < args.audio_path_checks:
                audio_checked += 1
                if not Path(str(row.get("source_audio", ""))).is_file():
                    audio_missing += 1

    stat = manifest.stat()
    first_offset = int(offsets[0]) if total else None
    last_offset = int(offsets[-1]) if total else None
    index_monotonic_sample = bool(
        np.all(np.diff(np.asarray(offsets[indices], dtype=np.int64)) >= 0)
    )
    report = {
        "schema_version": "uniss_streamspeech_ctc_stage00_audit_v1",
        "status": "passed"
        if not any(
            [malformed, invalid_direction, invalid_duration, audio_missing]
        )
        else "failed",
        "manifest": {
            "path": str(manifest),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "records_from_offsets": total,
        },
        "offset_index": {
            "path": str(offsets_path),
            "size_bytes": offsets_path.stat().st_size,
            "first_offset": first_offset,
            "last_offset": last_offset,
            "sample_monotonic": index_monotonic_sample,
        },
        "sample": {
            "records": len(indices),
            "directions": dict(sorted(directions.items())),
            "malformed_json": malformed,
            "missing_fields": dict(sorted(missing_fields.items())),
            "empty_text": dict(sorted(empty_text.items())),
            "invalid_direction": invalid_direction,
            "invalid_duration": invalid_duration,
            "invalid_glm_geometry_gt_240ms": invalid_glm_geometry,
            "audio_paths_checked": audio_checked,
            "audio_paths_missing": audio_missing,
        },
        "distributions": {
            "source_duration_ms": {
                "mean": mean(durations),
                "p01": percentile(durations, 1),
                "p50": percentile(durations, 50),
                "p95": percentile(durations, 95),
                "p99": percentile(durations, 99),
            },
            "source_glm_tokens_12p5hz": {
                "mean": mean(glm_lengths),
                "p01": percentile(glm_lengths, 1),
                "p50": percentile(glm_lengths, 50),
                "p95": percentile(glm_lengths, 95),
            },
            "transcription_characters": {
                "mean": mean(src_chars),
                "p50": percentile(src_chars, 50),
                "p95": percentile(src_chars, 95),
            },
            "translation_characters": {
                "mean": mean(tgt_chars),
                "p50": percentile(tgt_chars, 50),
                "p95": percentile(tgt_chars, 95),
            },
        },
        "environment": {
            "python": os.sys.executable,
            "numpy": np.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

