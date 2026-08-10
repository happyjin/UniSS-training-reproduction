#!/usr/bin/env python3
"""Build resumable full198 row indices with parallel shard workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


SCHEMA_VERSION = "uniss_true_subsecond_direction_index_v1"
REQUIRED_COLUMNS = (
    "id",
    "transcription",
    "translation",
    "source_glm",
    "source_bicodec",
    "target_bicodec",
    "bicodec_global",
    "src_lang",
    "tgt_lang",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_mask(table) -> np.ndarray:
    mask = pc.greater(pc.fill_null(pc.list_value_length(table["source_glm"]), 0), 0)
    for name in ("source_bicodec", "target_bicodec"):
        mask = pc.and_(
            mask,
            pc.greater_equal(pc.fill_null(pc.list_value_length(table[name]), 0), 16),
        )
    mask = pc.and_(
        mask,
        pc.equal(pc.fill_null(pc.list_value_length(table["bicodec_global"]), 0), 32),
    )
    for name in ("transcription", "translation"):
        present = pc.greater(
            pc.utf8_length(pc.utf8_trim_whitespace(pc.fill_null(table[name], ""))), 0
        )
        mask = pc.and_(mask, present)
    src = table["src_lang"]
    tgt = table["tgt_lang"]
    direction = pc.or_(
        pc.and_(pc.equal(src, "eng"), pc.equal(tgt, "cmn")),
        pc.and_(pc.equal(src, "cmn"), pc.equal(tgt, "eng")),
    )
    return pc.fill_null(pc.and_(mask, direction), False).to_numpy(zero_copy_only=False)


def _part_current(marker: Path, source: Path) -> dict[str, Any] | None:
    if not marker.is_file():
        return None
    value = json.loads(marker.read_text(encoding="utf-8"))
    stat = source.stat()
    if (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("source") == str(source.resolve())
        and int(value.get("source_size", -1)) == stat.st_size
        and int(value.get("source_mtime_ns", -1)) == stat.st_mtime_ns
    ):
        for key in ("eng_index", "cmn_index"):
            if not Path(str(value.get(key, ""))).is_file():
                return None
        return value
    return None


def build_one(payload: tuple[int, str, str, bool]) -> dict[str, Any]:
    shard, source_name, output_name, checksum = payload
    source = Path(source_name).resolve()
    output = Path(output_name).resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / f"part-{shard:03d}.json"
    current = _part_current(marker, source)
    if current is not None:
        return current
    parquet = pq.ParquetFile(source)
    missing = sorted(set(REQUIRED_COLUMNS) - set(parquet.schema_arrow.names))
    if missing:
        raise KeyError(f"{source} missing columns: {missing}")
    table = pq.read_table(source, columns=list(REQUIRED_COLUMNS))
    valid = valid_mask(table)
    src = table["src_lang"].to_numpy(zero_copy_only=False)
    eng = np.flatnonzero(valid & (src == "eng")).astype(np.uint32)
    cmn = np.flatnonzero(valid & (src == "cmn")).astype(np.uint32)
    eng_path = output / f"train-{shard:05d}.eng.npy"
    cmn_path = output / f"train-{shard:05d}.cmn.npy"
    np.save(eng_path, eng, allow_pickle=False)
    np.save(cmn_path, cmn, allow_pickle=False)
    stat = source.stat()
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "shard": shard,
        "source": str(source),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "rows": table.num_rows,
        "eng": len(eng),
        "cmn": len(cmn),
        "accepted": len(eng) + len(cmn),
        "rejected": table.num_rows - len(eng) - len(cmn),
        "eng_index": str(eng_path),
        "cmn_index": str(cmn_path),
    }
    if checksum:
        value["source_sha256"] = _sha256(source)
    _atomic_json(marker, value)
    return value


def build(input_dir: Path, output_dir: Path, workers: int, checksum: bool) -> dict[str, Any]:
    sources = sorted(input_dir.glob("train-*.parquet"))
    if len(sources) != 198:
        raise ValueError(f"expected 198 train shards, found {len(sources)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [(index, str(path), str(output_dir), checksum) for index, path in enumerate(sources)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(build_one, payloads))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "shards": parts,
        "shard_count": len(parts),
        "rows": sum(int(value["rows"]) for value in parts),
        "accepted": sum(int(value["accepted"]) for value in parts),
        "rejected": sum(int(value["rejected"]) for value in parts),
        "eng": sum(int(value["eng"]) for value in parts),
        "cmn": sum(int(value["cmn"]) for value in parts),
        "workers": workers,
    }
    _atomic_json(output_dir / "index.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--checksum", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    print(json.dumps(build(args.input_dir, args.output_dir, args.workers, args.checksum), sort_keys=True))


if __name__ == "__main__":
    main()
