#!/usr/bin/env python3
"""Create a tiny isolated direction index for real trajectory-cache smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


SMOKE_INDEX_SCHEMA = "uniss_true_subsecond_cache_smoke_index_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_smoke_rows(eng: np.ndarray, cmn: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    eng = np.asarray(eng, dtype=np.int64)
    cmn = np.asarray(cmn, dtype=np.int64)
    merged = np.sort(np.concatenate((eng, cmn)))[:limit]
    selected_eng = np.intersect1d(merged, eng, assume_unique=True)
    selected_cmn = np.intersect1d(merged, cmn, assume_unique=True)
    if len(selected_eng) + len(selected_cmn) != min(limit, len(eng) + len(cmn)):
        raise AssertionError("smoke selection lost or duplicated row indices")
    return selected_eng, selected_cmn


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def build(source_root: Path, output_root: Path, shard: int, limit: int) -> dict[str, object]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty smoke index: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    stem = f"train-{shard:05d}"
    eng = np.load(source_root / f"{stem}.eng.npy", mmap_mode="r")
    cmn = np.load(source_root / f"{stem}.cmn.npy", mmap_mode="r")
    selected_eng, selected_cmn = select_smoke_rows(eng, cmn, limit)
    eng_output = output_root / f"{stem}.eng.npy"
    cmn_output = output_root / f"{stem}.cmn.npy"
    np.save(eng_output, selected_eng)
    np.save(cmn_output, selected_cmn)
    summary = {
        "schema_version": SMOKE_INDEX_SCHEMA,
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "shard": shard,
        "limit": limit,
        "eng_rows": int(len(selected_eng)),
        "cmn_rows": int(len(selected_cmn)),
        "total_rows": int(len(selected_eng) + len(selected_cmn)),
        "files": {
            eng_output.name: {"sha256": _sha256(eng_output), "bytes": eng_output.stat().st_size},
            cmn_output.name: {"sha256": _sha256(cmn_output), "bytes": cmn_output.stat().st_size},
        },
    }
    _atomic_json(output_root / "smoke_index.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(build(args.source_root, args.output_root, args.shard, args.limit), sort_keys=True))


if __name__ == "__main__":
    main()
