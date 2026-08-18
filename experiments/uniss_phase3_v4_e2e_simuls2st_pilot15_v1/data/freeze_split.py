#!/usr/bin/env python3
"""Freeze and independently verify immutable 15-shard train/valid inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from training.simul_uniss.jsonl_index import index_paths, load_index


SPLIT_SCHEMA = "uniss_phase3_v4_e2e_simuls2st_frozen_split_v1"


def sha256_file(path: Path, block_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def freeze_split(
    source_snapshot_path: Path,
    stage_a_audit_path: Path,
    checkpoint_fingerprints_path: Path,
) -> dict[str, object]:
    snapshot = json.loads(source_snapshot_path.read_text(encoding="utf-8"))
    audit = json.loads(stage_a_audit_path.read_text(encoding="utf-8"))
    fingerprints = json.loads(checkpoint_fingerprints_path.read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != "uniss_quality_first_stage_a_source_snapshot_v2":
        raise ValueError("unexpected Stage-A source snapshot schema")
    if audit.get("passed") is not True or int(audit.get("train_valid_id_overlap", -1)) != 0:
        raise ValueError("Stage-A source audit did not pass the disjoint split gate")
    if fingerprints.get("status") != "complete":
        raise ValueError("checkpoint fingerprint report is incomplete")

    split_result: dict[str, object] = {}
    id_arrays: dict[str, np.ndarray] = {}
    for split in ("train", "valid"):
        manifest = Path(str(snapshot[split]["path"]))
        offsets = load_index(manifest)
        if offsets is None:
            raise ValueError(f"missing source offset index: {manifest}")
        expected_records = int(snapshot[split]["records"])
        if len(offsets) != expected_records:
            raise ValueError(f"{split} offset count differs from snapshot")
        audit_split = audit[split]
        if int(audit_split["counters"]["records"]) != expected_records:
            raise ValueError(f"{split} audit count differs from snapshot")
        ids_path = Path(str(audit_split["sorted_id_hashes"]))
        ids = np.load(ids_path, mmap_mode="r")
        if ids.ndim != 1 or len(ids) != expected_records:
            raise ValueError(f"{split} ID hash array shape differs from snapshot")
        if len(ids) > 1 and bool(np.any(ids[:-1] >= ids[1:])):
            raise ValueError(f"{split} ID hashes are not sorted and unique")
        id_arrays[split] = ids
        binary_index, metadata_index = index_paths(manifest)
        split_result[split] = {
            "records": expected_records,
            "manifest": _file_identity(manifest),
            "offset_index_binary": _file_identity(binary_index),
            "offset_index_metadata": _file_identity(metadata_index),
            "sorted_id_hashes": _file_identity(ids_path),
            "directions": {
                key.removeprefix("direction:"): int(value)
                for key, value in audit_split["counters"].items()
                if str(key).startswith("direction:")
            },
            "duration_ms": int(audit_split["counters"]["duration_ms"]),
        }
    overlap = np.intersect1d(id_arrays["train"], id_arrays["valid"], assume_unique=True)
    if len(overlap):
        raise ValueError("train/validation ID hash overlap is non-zero")
    checkpoints = fingerprints["checkpoints"]
    if checkpoints["v1"]["sha256"] == checkpoints["phase3"]["sha256"]:
        raise ValueError("V1 and Phase3 checkpoint tree fingerprints are identical")
    return {
        "schema_version": SPLIT_SCHEMA,
        "status": "frozen",
        "train_valid_id_overlap": 0,
        "source_snapshot": _file_identity(source_snapshot_path),
        "stage_a_data_audit": _file_identity(stage_a_audit_path),
        "checkpoint_fingerprints": _file_identity(checkpoint_fingerprints_path),
        "checkpoint_tree_sha256": {
            "v1": checkpoints["v1"]["sha256"],
            "phase3": checkpoints["phase3"]["sha256"],
        },
        "splits": split_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--stage-a-audit", type=Path, required=True)
    parser.add_argument("--checkpoint-fingerprints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen split: {args.output}")
    result = freeze_split(
        args.source_snapshot,
        args.stage_a_audit,
        args.checkpoint_fingerprints,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
