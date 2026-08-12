#!/usr/bin/env python3
"""Materialize cumulative natural-WRITE audio prefixes for useful-audio ASR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf


SCHEMA = "uniss_event_rollout_fixed15_prefix_asr_manifest_v1"


def prefix_candidates(row: Mapping[str, object]) -> list[dict[str, object]]:
    cumulative_samples = 0
    candidates: list[dict[str, object]] = []
    for event in row.get("events", []):
        emitted = int(event.get("emitted_audio_samples", 0) or 0)
        if emitted <= 0:
            continue
        cumulative_samples += emitted
        candidates.append(
            {
                "event_index": int(event["event_index"]),
                "source_end_ms": int(event["source_end_ms"]),
                "wall_end_ms": float(event["wall_end_ms"]),
                "cumulative_audio_samples": cumulative_samples,
            }
        )
    return candidates


def build(rows: Sequence[Mapping[str, object]], output_root: Path) -> list[dict[str, object]]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite prefix ASR artifacts: {output_root}")
    output_root.mkdir(parents=True)
    audio_root = output_root / "audio"
    audio_root.mkdir()
    output: list[dict[str, object]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        audio_path = Path(str(row["audio_path"])).resolve()
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        value = np.asarray(audio, dtype=np.float32)
        if value.ndim == 2:
            value = value.mean(axis=1)
        value = value.reshape(-1)
        for candidate_index, candidate in enumerate(prefix_candidates(row)):
            end = min(len(value), int(candidate["cumulative_audio_samples"]))
            if end <= 0:
                continue
            prefix_path = audio_root / f"{sample_id}.event-{int(candidate['event_index']):04d}.wav"
            sf.write(prefix_path, value[:end], sample_rate, subtype="PCM_16")
            output.append(
                {
                    "schema_version": SCHEMA,
                    "id": f"{sample_id}:event-{int(candidate['event_index']):04d}",
                    "mode": "exact_runtime_prefix_asr",
                    "parent_sample_id": sample_id,
                    "candidate_index": candidate_index,
                    "src_lang": row["src_lang"],
                    "tgt_lang": row["tgt_lang"],
                    "translation_ref": row["target_text"],
                    "oracle_target_text_prefixes": row["oracle_target_text_prefixes"],
                    "audio_path": str(prefix_path.resolve()),
                    "audio_duration_seconds": end / float(sample_rate),
                    "source_end_ms": candidate["source_end_ms"],
                    "wall_end_ms": candidate["wall_end_ms"],
                    "cumulative_audio_samples": end,
                    "natural_writes": row["natural_writes"],
                    "forced_writes": row["forced_writes"],
                }
            )
    return output


def _rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = build(_rows(args.results), args.output_root)
    _write(args.output_root / "prefix_asr_manifest.jsonl", rows)
    summary = {
        "schema_version": SCHEMA,
        "source_results": str(args.results.resolve()),
        "parent_samples": len({str(row["parent_sample_id"]) for row in rows}),
        "prefix_candidates": len(rows),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

