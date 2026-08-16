#!/usr/bin/env python3
"""Build immutable Phase3 manifests for the exact Stage A formal selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.dataset import (
    rotated_acoustic_indices,
)


TASKS = {"streaming_asr", "causal_full_asr"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_acoustics(packs: Path, max_acoustics_per_pack: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    with packs.open(encoding="utf-8") as handle:
        for pack_index, line in enumerate(handle):
            pack = json.loads(line)
            acoustics = list(pack.get("acoustics", []))
            for position in rotated_acoustic_indices(
                len(acoustics), max_acoustics_per_pack, 0, pack_index
            ):
                acoustic = acoustics[position]
                task = str(acoustic["task"])
                if task not in TASKS:
                    continue
                selected.append(
                    {
                        "id": str(acoustic["sample_id"]),
                        "task": task,
                        "src_lang": str(acoustic["src_lang"]),
                        "transcription": str(acoustic["canonical_transcript"]),
                        "source_audio": str(acoustic["source_audio"]),
                        "pack_index": pack_index,
                        "acoustic_position": int(position),
                    }
                )
    identities = [(row["task"], row["id"]) for row in selected]
    if len(identities) != len(set(identities)):
        raise ValueError("Stage A formal selection contains duplicate task/sample IDs")
    return selected


def source_rows(formal_manifest: Path, wanted: set[str]) -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    with formal_manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            sample_id = str(value["id"])
            if sample_id not in wanted:
                continue
            if sample_id in found:
                raise ValueError(f"formal validation contains duplicate ID: {sample_id}")
            found[sample_id] = {
                "parquet_path": str(value["source_parquet"]),
                "row_index": int(value["source_row_index"]),
                "formal_input_index": int(value["formal_input_index"]),
                "src_lang": str(value["src_lang"]),
                "tgt_lang": str(value["tgt_lang"]),
                "transcription": str(value["transcription"]),
                "source_audio": str(value["source_audio"]),
            }
            if len(found) == len(wanted):
                break
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(f"formal validation is missing selected IDs: {missing[:20]}")
    return found


def assemble(
    selected: Iterable[dict[str, object]],
    sources: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for occurrence, acoustic in enumerate(selected):
        sample_id = str(acoustic["id"])
        source = sources[sample_id]
        if str(source["src_lang"]) != str(acoustic["src_lang"]):
            raise ValueError(f"source language differs for {sample_id}")
        source_transcription = " ".join(str(source["transcription"]).split())
        canonical_transcription = " ".join(str(acoustic["transcription"]).split())
        if Path(str(source["source_audio"])).resolve() != Path(
            str(acoustic["source_audio"])
        ).resolve():
            raise ValueError(f"source audio differs for {sample_id}")
        rows.append(
            {
                "id": sample_id,
                "parquet_path": str(source["parquet_path"]),
                "row_index": int(source["row_index"]),
                "task": str(acoustic["task"]),
                "src_lang": str(source["src_lang"]),
                "tgt_lang": str(source["tgt_lang"]),
                # Score against the exact canonical reference used by Stage A.
                # The raw parquet transcription is retained for provenance.  It
                # does not affect the Quality prompt, which ends before target
                # transcription tokens begin.
                "transcription": canonical_transcription,
                "source_transcription": source_transcription,
                "source_audio": str(source["source_audio"]),
                "formal_input_index": int(source["formal_input_index"]),
                "pack_index": int(acoustic["pack_index"]),
                "acoustic_position": int(acoustic["acoustic_position"]),
                "worker_rank": occurrence % 8,
            }
        )
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite matching manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite matching manifest audit: {path}")
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
    parser.add_argument("--valid-packs", type=Path, required=True)
    parser.add_argument("--formal-valid-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-acoustics-per-pack", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite matching manifest directory: {args.output_dir}")
    if args.max_acoustics_per_pack <= 0:
        raise ValueError("max acoustics per pack must be positive")
    selected = selected_acoustics(args.valid_packs, args.max_acoustics_per_pack)
    sources = source_rows(args.formal_valid_manifest, {str(row["id"]) for row in selected})
    rows = assemble(selected, sources)
    if len(rows) != 334:
        raise ValueError(f"matching offline selection must contain 334 rows, got {len(rows)}")
    args.output_dir.mkdir(parents=True)
    merged = args.output_dir / "matching_stage_a_334.jsonl"
    write_jsonl(merged, rows)
    part_paths = []
    for rank in range(8):
        path = args.output_dir / "parts" / f"part_{rank:02d}.jsonl"
        write_jsonl(path, (row for row in rows if int(row["worker_rank"]) == rank))
        part_paths.append(path)
    counts = Counter((str(row["task"]), str(row["src_lang"])) for row in rows)
    audit = {
        "schema_version": "uniss_quality_first_stage_a_matching_offline_manifest_v1",
        "status": "complete",
        "valid_packs": str(args.valid_packs.resolve()),
        "valid_packs_sha256": sha256(args.valid_packs),
        "formal_valid_manifest": str(args.formal_valid_manifest.resolve()),
        "formal_valid_manifest_sha256": sha256(args.formal_valid_manifest),
        "output": str(merged.resolve()),
        "output_sha256": sha256(merged),
        "records": len(rows),
        "unique_ids": len({str(row["id"]) for row in rows}),
        "canonical_source_transcription_exact_matches": sum(
            str(row["transcription"]) == str(row["source_transcription"]) for row in rows
        ),
        "counts": {f"{task}:{language}": value for (task, language), value in sorted(counts.items())},
        "parts": [
            {"path": str(path.resolve()), "sha256": sha256(path), "records": sum(1 for _ in path.open(encoding="utf-8"))}
            for path in part_paths
        ],
    }
    atomic_json(args.output_dir / "MANIFEST_AUDIT.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
