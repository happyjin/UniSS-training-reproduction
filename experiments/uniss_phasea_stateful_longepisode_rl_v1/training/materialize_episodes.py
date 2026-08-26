#!/usr/bin/env python3
"""Materialize episode waveforms in parallel without touching source audio."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16_000


def materialize(row: dict[str, object], output_root: Path, gap_ms: int) -> dict[str, object]:
    parts: list[np.ndarray] = []
    gap = np.zeros(int(round(gap_ms * SAMPLE_RATE / 1000.0)), dtype=np.float32)
    components = list(row["components"])
    for index, component in enumerate(components):
        audio, rate = sf.read(component["source_audio"], dtype="float32", always_2d=True)
        if int(rate) != SAMPLE_RATE:
            raise ValueError(f"component is not 16 kHz: {component['source_audio']}")
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        if not len(mono) or not np.isfinite(mono).all():
            raise ValueError(f"component is empty/non-finite: {component['source_audio']}")
        if index:
            parts.append(gap)
        parts.append(mono)
    waveform = np.concatenate(parts)
    path = output_root / "audio" / f"{row['episode_id']}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, SAMPLE_RATE, subtype="PCM_16")
    stored, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.asarray(stored.mean(axis=1), dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    result = dict(row)
    result.update(
        {
            "source_audio": str(path.resolve()),
            "source_audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_sample_rate": int(rate),
            "source_frames": len(mono),
            "source_duration_ms": int(round(len(mono) * 1000 / rate)),
            "source_rms": rms,
            "source_finite": bool(np.isfinite(mono).all()),
            "speaker_global": [int(value) for value in components[0]["speaker_global"]],
            "gap_ms": int(gap_ms),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--gap-ms", type=int, default=160)
    args = parser.parse_args()
    output_manifest = args.output_root / "episodes.jsonl"
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    args.output_root.mkdir(parents=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(materialize, row, args.output_root, args.gap_ms) for row in rows]
        results = [future.result() for future in futures]
    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    audit = {
        "schema_version": "uniss_stateful_longepisode_audio_v1",
        "status": "passed",
        "input_manifest": str(args.manifest.resolve()),
        "output_manifest": str(output_manifest.resolve()),
        "episodes": len(results),
        "audio_healthy": sum(
            int(row["source_sample_rate"]) == SAMPLE_RATE
            and bool(row["source_finite"])
            and float(row["source_rms"]) >= 1e-5
            for row in results
        ),
        "duration_ms": {
            "minimum": min(int(row["source_duration_ms"]) for row in results),
            "maximum": max(int(row["source_duration_ms"]) for row in results),
            "total": sum(int(row["source_duration_ms"]) for row in results),
        },
        "sha256": hashlib.sha256(output_manifest.read_bytes()).hexdigest(),
    }
    if audit["audio_healthy"] != len(results):
        raise RuntimeError("one or more materialized episodes failed the audio audit")
    (args.output_root / "AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

