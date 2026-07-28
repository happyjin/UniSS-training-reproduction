"""Load full CVSS-T parquet rows while preserving canonical waveform paths."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from evaluation.io_utils import iter_jsonl
from training.generate_unist_eval_audio import resolve_manifest_parquet_path
from training.prepare_unist_s2st import normalize_unist_record


def iter_cvss_manifest_records(manifest_path: Path) -> Iterator[dict[str, object]]:
    tables: dict[Path, object] = {}
    for entry in iter_jsonl(manifest_path):
        parquet_path = resolve_manifest_parquet_path(entry["parquet_path"], manifest_path=manifest_path)
        table = tables.get(parquet_path)
        if table is None:
            if not parquet_path.is_file():
                raise FileNotFoundError(f"CVSS manifest parquet does not exist: {parquet_path}")
            table = pq.read_table(parquet_path)
            tables[parquet_path] = table
        row_index = int(entry["row_index"])
        if row_index < 0 or row_index >= table.num_rows:  # type: ignore[union-attr]
            raise IndexError(f"CVSS manifest row_index {row_index} is invalid for {parquet_path}")
        raw = table.slice(row_index, 1).to_pylist()[0]  # type: ignore[union-attr]
        if str(raw.get("id")) != str(entry.get("id")):
            raise ValueError(
                f"CVSS manifest ID mismatch at {parquet_path}:{row_index}: "
                f"expected {entry.get('id')!r}, found {raw.get('id')!r}"
            )
        normalized = normalize_unist_record(raw)
        yield {**raw, **normalized}


def cvss_record_map(manifest_path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for record in iter_cvss_manifest_records(manifest_path):
        sample_id = str(record["id"])
        if sample_id in records:
            raise ValueError(f"Duplicate CVSS manifest ID: {sample_id}")
        for field in ("source_audio_path", "reference_audio_path"):
            path = Path(str(record.get(field) or ""))
            if not path.is_file():
                raise FileNotFoundError(f"CVSS {sample_id} missing {field}: {path}")
        records[sample_id] = record
    return records
