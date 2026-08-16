#!/usr/bin/env python3
"""Freeze balanced pilot15 validation subsets without copying source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.simul_uniss.jsonl_index import load_index


DURATION_BOUNDS_MS = (4_000, 8_000, 15_000)


def _duration_bin(duration_ms: int) -> str:
    if duration_ms < DURATION_BOUNDS_MS[0]:
        return "lt4s"
    if duration_ms < DURATION_BOUNDS_MS[1]:
        return "4to8s"
    if duration_ms < DURATION_BOUNDS_MS[2]:
        return "8to15s"
    return "ge15s"


def _stable_score(seed: int, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
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


def _read_at(handle, offset: int) -> dict[str, Any]:
    handle.seek(offset)
    return json.loads(handle.readline())


def _entry(record: dict[str, Any], formal_index: int) -> dict[str, Any]:
    return {
        "id": str(record["id"]),
        "parquet_path": str(Path(record["source_parquet"]).resolve()),
        "row_index": int(record["source_row_index"]),
        "src_lang": str(record["src_lang"]),
        "tgt_lang": str(record["tgt_lang"]),
        "source_duration_ms": int(record["source_duration_ms"]),
        "duration_bin": _duration_bin(int(record["source_duration_ms"])),
        "formal_index": formal_index,
    }


def select(
    manifest: Path, *, count: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    offsets = load_index(manifest)
    if offsets is None:
        raise FileNotFoundError(f"missing immutable JSONL index for {manifest}")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    with manifest.open("rb") as handle:
        for formal_index, offset in enumerate(offsets):
            record = _read_at(handle, int(offset))
            entry = _entry(record, formal_index)
            key = (entry["src_lang"], entry["tgt_lang"], entry["duration_bin"])
            groups[key].append(entry)
    for values in groups.values():
        values.sort(key=lambda item: _stable_score(seed, str(item["id"])))

    keys = sorted(groups)
    if count < len(keys):
        raise ValueError("requested validation count cannot cover every stratum")
    quota, remainder = divmod(count, len(keys))
    chosen: list[dict[str, Any]] = []
    for index, key in enumerate(keys):
        take = quota + (1 if index < remainder else 0)
        chosen.extend(groups[key][:take])
    if len(chosen) < count:
        selected_ids = {item["id"] for item in chosen}
        remainder_pool = [
            item
            for values in groups.values()
            for item in values
            if item["id"] not in selected_ids
        ]
        remainder_pool.sort(key=lambda item: _stable_score(seed + 1, str(item["id"])))
        chosen.extend(remainder_pool[: count - len(chosen)])
    if len(chosen) != count or len({item["id"] for item in chosen}) != count:
        raise ValueError("failed to build a unique fixed validation subset")
    chosen.sort(key=lambda item: _stable_score(seed + 2, str(item["id"])))
    population = {"/".join(key): len(value) for key, value in sorted(groups.items())}
    return chosen, population


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite validation manifest: {path}")
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _write_parts(root: Path, name: str, rows: list[dict[str, Any]], parts: int) -> None:
    buckets = [rows[index::parts] for index in range(parts)]
    for index, bucket in enumerate(buckets):
        _write_jsonl(root / f"{name}.part{index:02d}.jsonl", bucket)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-count", type=int, default=256)
    parser.add_argument("--audio-count", type=int, default=64)
    parser.add_argument("--parts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    manifest = Path(args.formal_manifest).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse fixed validation directory: {output}")
    output.mkdir(parents=True)
    text_rows, population = select(manifest, count=args.text_count, seed=args.seed)
    audio_candidates, _ = select(manifest, count=args.audio_count, seed=args.seed + 10_000)
    text_path = output / f"pilot15_text_{args.text_count}.jsonl"
    audio_path = output / f"pilot15_audio_{args.audio_count}.jsonl"
    _write_jsonl(text_path, text_rows)
    _write_jsonl(audio_path, audio_candidates)
    _write_parts(output, f"pilot15_text_{args.text_count}", text_rows, args.parts)
    _write_parts(output, f"pilot15_audio_{args.audio_count}", audio_candidates, args.parts)
    selected_counts: Counter[str] = Counter()
    for row in text_rows:
        selected_counts[f"{row['src_lang']}/{row['tgt_lang']}/{row['duration_bin']}"] += 1
    audit = {
        "schema_version": "uniss_stage00_fixed_pilot15_validation_v1",
        "formal_manifest": str(manifest),
        "formal_manifest_sha256": _sha256(manifest),
        "seed": args.seed,
        "parts": args.parts,
        "text": {
            "path": str(text_path),
            "records": len(text_rows),
            "sha256": _sha256(text_path),
        },
        "audio": {
            "path": str(audio_path),
            "records": len(audio_candidates),
            "sha256": _sha256(audio_path),
        },
        "population_by_stratum": population,
        "text_selected_by_stratum": dict(sorted(selected_counts.items())),
        "unique_text_ids": len({row["id"] for row in text_rows}),
        "unique_audio_ids": len({row["id"] for row in audio_candidates}),
    }
    _atomic_json(output / "validation_manifest_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

