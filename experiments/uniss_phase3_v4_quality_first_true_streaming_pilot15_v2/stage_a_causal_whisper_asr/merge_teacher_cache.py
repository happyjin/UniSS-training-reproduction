#!/usr/bin/env python3
"""Merge and validate disjoint Stage A same-prefix teacher cache parts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.build_teacher_cache import (
    CACHE_SCHEMA,
    partition,
    selected_acoustic_indices,
    sha256,
)
from training.simul_uniss.jsonl_index import load_index, write_index


def expected_keys(
    packs: Path,
    *,
    total: int,
    coverage_epochs: int,
    max_acoustics_per_pack: int,
) -> set[tuple[int, int]]:
    offsets = load_index(packs)
    if offsets is None:
        raise ValueError(f"missing Stage A pack index: {packs}")
    keys: set[tuple[int, int]] = set()
    with packs.open("rb") as handle:
        for pack_index in range(total):
            handle.seek(int(offsets[pack_index]))
            pack = json.loads(handle.readline())
            for acoustic_index in selected_acoustic_indices(
                len(pack.get("acoustics", [])),
                pack_index=pack_index,
                coverage_epochs=coverage_epochs,
                max_acoustics_per_pack=max_acoustics_per_pack,
            ):
                keys.add((pack_index, acoustic_index))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--coverage-epochs", type=int, default=3)
    parser.add_argument("--max-acoustics-per-pack", type=int, default=2)
    parser.add_argument("--limit-packs", type=int)
    args = parser.parse_args()
    packs = args.packs.resolve()
    offsets = load_index(packs)
    if offsets is None:
        raise ValueError(f"missing Stage A pack index: {packs}")
    total = len(offsets)
    if args.limit_packs is not None:
        total = min(total, int(args.limit_packs))
    output = args.parts_root.resolve() / "teacher_cache.jsonl"
    audit_path = args.parts_root.resolve() / "TEACHER_CACHE_AUDIT.json"
    if output.exists() or audit_path.exists():
        raise FileExistsError("refusing to overwrite merged teacher cache")
    rows: list[dict[str, object]] = []
    markers: list[dict[str, object]] = []
    shared_metadata = (
        "schema_version",
        "status",
        "world_size",
        "packs",
        "packs_sha256",
        "model",
        "speaker_source",
        "coverage_epochs",
        "max_acoustics_per_pack",
        "topk",
        "temperature",
        "require_reference_in_topk",
        "reference_anchor",
    )
    expected_pack_sha = sha256(packs)
    for rank in range(args.world_size):
        root = args.parts_root / f"part_{rank:02d}"
        marker = json.loads((root / "PART_COMPLETE.json").read_text(encoding="utf-8"))
        if marker.get("schema_version") != CACHE_SCHEMA or marker.get("status") != "complete":
            raise ValueError(f"teacher cache part {rank} is incomplete")
        expected_start, expected_stop = partition(total, rank, args.world_size)
        expected_geometry = {
            "rank": rank,
            "world_size": args.world_size,
            "assigned_start": expected_start,
            "assigned_stop": expected_stop,
            "assigned_packs": expected_stop - expected_start,
            "coverage_epochs": args.coverage_epochs,
            "max_acoustics_per_pack": args.max_acoustics_per_pack,
        }
        for name, expected_value in expected_geometry.items():
            if marker.get(name) != expected_value:
                raise ValueError(
                    f"teacher cache part {rank} has inconsistent {name}: "
                    f"{marker.get(name)!r} != {expected_value!r}"
                )
        if Path(str(marker.get("packs"))).resolve() != packs:
            raise ValueError(f"teacher cache part {rank} names a different pack file")
        if marker.get("packs_sha256") != expected_pack_sha:
            raise ValueError(f"teacher cache part {rank} pack digest differs")
        if marker.get("speaker_source") != "stage_a_pack_prompt":
            raise ValueError(f"teacher cache part {rank} used an unsafe speaker source")
        if markers:
            for name in shared_metadata:
                if marker.get(name) != markers[0].get(name):
                    raise ValueError(
                        f"teacher cache part {rank} metadata differs for {name}"
                    )
        manifest = root / "teacher_cache.jsonl"
        if marker.get("manifest_sha256") != sha256(manifest):
            raise ValueError(f"teacher cache part {rank} manifest digest differs")
        markers.append(marker)
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema_version") != CACHE_SCHEMA:
                    raise ValueError(f"teacher cache part {rank} row schema differs")
                bundle = Path(str(row.get("bundle_path"))).resolve()
                if root.resolve() not in bundle.parents or not bundle.is_file():
                    raise ValueError(
                        f"teacher cache part {rank} references a foreign/missing bundle"
                    )
                rows.append(row)
    rows.sort(key=lambda row: (int(row["pack_index"]), int(row["acoustic_index"])))
    actual = {(int(row["pack_index"]), int(row["acoustic_index"])) for row in rows}
    expected = expected_keys(
        packs,
        total=total,
        coverage_epochs=args.coverage_epochs,
        max_acoustics_per_pack=args.max_acoustics_per_pack,
    )
    if len(rows) != len(actual) or actual != expected:
        raise ValueError(
            f"teacher cache coverage differs: rows={len(rows)} unique={len(actual)} "
            f"expected={len(expected)} missing={sorted(expected-actual)[:10]} "
            f"foreign={sorted(actual-expected)[:10]}"
        )
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(name)
    byte_offsets = []
    byte_offset = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for row in rows:
                encoded = (
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                byte_offsets.append(byte_offset)
                handle.write(encoded)
                byte_offset += len(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    index = write_index(output, byte_offsets)
    candidates = sum(int(row["teacher_candidate_positions"]) for row in rows)
    positions = sum(int(row["teacher_positions"]) for row in rows)
    correct = sum(int(row["teacher_top1_correct"]) for row in rows)
    audit = {
        "schema_version": CACHE_SCHEMA,
        "status": "complete",
        "packs": str(packs),
        "packs_sha256": expected_pack_sha,
        "total_packs": total,
        "coverage_epochs": args.coverage_epochs,
        "max_acoustics_per_pack": args.max_acoustics_per_pack,
        "records": len(rows),
        "teacher_candidate_positions": candidates,
        "teacher_positions": positions,
        "teacher_top1_correct": correct,
        "teacher_top1_accuracy": correct / max(1, candidates),
        "teacher_selection_rate": positions / max(1, candidates),
        "topk": markers[0]["topk"],
        "temperature": markers[0]["temperature"],
        "require_reference_in_topk": markers[0]["require_reference_in_topk"],
        "reference_anchor": markers[0]["reference_anchor"],
        "model": markers[0]["model"],
        "speaker_source": markers[0]["speaker_source"],
        "output": str(output),
        "output_sha256": sha256(output),
        "index": index,
        "parts": [marker["manifest_sha256"] for marker in markers],
    }
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
