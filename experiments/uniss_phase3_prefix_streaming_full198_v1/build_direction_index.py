#!/usr/bin/env python3
"""Build compact per-shard direction indices without modifying UniST parquet."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc


SCHEMA_VERSION = "uniss_phase3_prefix_streaming_direction_index_v3"


def _nonempty_mask(table) -> np.ndarray:
    mask = None
    source_present = pc.greater(pc.fill_null(pc.list_value_length(table["source_glm"]), 0), 0)
    semantic_usable = pc.greater_equal(
        pc.fill_null(pc.list_value_length(table["target_bicodec"]), 0), 2
    )
    global_valid = pc.equal(
        pc.fill_null(pc.list_value_length(table["bicodec_global"]), 0), 32
    )
    mask = pc.and_(pc.and_(source_present, semantic_usable), global_valid)
    for name in ("transcription", "translation"):
        text = pc.fill_null(table[name], "")
        present = pc.greater(pc.utf8_length(pc.utf8_trim_whitespace(text)), 0)
        mask = pc.and_(mask, present)
    return pc.fill_null(mask, False).to_numpy(zero_copy_only=False)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _one(payload: tuple[str, str]) -> dict[str, object]:
    source_name, output_name = payload
    source = Path(source_name)
    output = Path(output_name)
    eng_path = output / f"{source.stem}.eng.npy"
    cmn_path = output / f"{source.stem}.cmn.npy"
    if eng_path.is_file() and cmn_path.is_file():
        rows = int(pq.ParquetFile(source).metadata.num_rows)
        eng = int(np.load(eng_path, mmap_mode="r").shape[0])
        cmn = int(np.load(cmn_path, mmap_mode="r").shape[0])
        return {
            "file": str(source.resolve()),
            "rows": rows,
            "eng": eng,
            "cmn": cmn,
            "rejected": rows - eng - cmn,
            "eng_index": str(eng_path.resolve()),
            "cmn_index": str(cmn_path.resolve()),
        }
    table = pq.read_table(
        source,
        columns=[
            "src_lang",
            "transcription",
            "translation",
            "source_glm",
            "target_bicodec",
            "bicodec_global",
        ],
    )
    values = table.column("src_lang").to_numpy(zero_copy_only=False)
    valid = _nonempty_mask(table)
    eng = np.flatnonzero((values == "eng") & valid).astype(np.uint32)
    cmn = np.flatnonzero((values == "cmn") & valid).astype(np.uint32)
    output.mkdir(parents=True, exist_ok=True)
    np.save(eng_path, eng, allow_pickle=False)
    np.save(cmn_path, cmn, allow_pickle=False)
    return {
        "file": str(source.resolve()),
        "rows": int(len(values)),
        "eng": int(len(eng)),
        "cmn": int(len(cmn)),
        "rejected": int(len(values) - len(eng) - len(cmn)),
        "eng_index": str(eng_path.resolve()),
        "cmn_index": str(cmn_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    sources = sorted(Path(args.input_dir).glob("train-*.parquet"))
    if len(sources) != 198:
        raise ValueError(f"expected 198 train shards, found {len(sources)}")
    output = Path(args.output_dir)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        shards = list(pool.map(_one, [(str(path), str(output)) for path in sources]))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "shards": shards,
        "rows": sum(int(value["rows"]) for value in shards),
        "eng": sum(int(value["eng"]) for value in shards),
        "cmn": sum(int(value["cmn"]) for value in shards),
        "rejected": sum(int(value["rejected"]) for value in shards),
    }
    _atomic_json(output / "index.json", summary)
    print(json.dumps(summary | {"shards": len(shards)}, sort_keys=True))


if __name__ == "__main__":
    main()
