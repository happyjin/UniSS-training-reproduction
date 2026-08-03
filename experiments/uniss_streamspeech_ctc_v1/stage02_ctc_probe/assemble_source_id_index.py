#!/usr/bin/env python3
"""Assemble immutable SQLite lookup for stable UniST IDs."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    return parser.parse_args()


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record_id, record_index, byte_offset = line.rstrip("\n").split("\t")
            yield record_id, int(record_index), int(byte_offset)


def main() -> None:
    args = parse_args()
    parts = sorted(args.parts_dir.glob("source-id-part-*-of-*.tsv"))
    if not parts:
        raise FileNotFoundError("no source-ID index parts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite source-ID index; use a new version: {args.output}"
        )
    connection = sqlite3.connect(args.output)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute(
            "CREATE TABLE source_offsets ("
            "id TEXT PRIMARY KEY, record_index INTEGER NOT NULL, byte_offset INTEGER NOT NULL"
            ") WITHOUT ROWID"
        )
        inserted = 0
        for path in parts:
            before = connection.total_changes
            connection.executemany(
                "INSERT INTO source_offsets(id, record_index, byte_offset) VALUES (?, ?, ?)",
                rows(path),
            )
            inserted += connection.total_changes - before
            connection.commit()
        if inserted != args.expected_records:
            raise ValueError(f"inserted {inserted}, expected {args.expected_records}")
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("schema_version", "uniss_streamspeech_source_id_index_v1"),
                ("records", str(inserted)),
            ],
        )
        connection.commit()
        result = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {result}")
    finally:
        connection.close()
    report = {
        "schema_version": "uniss_streamspeech_source_id_index_v1",
        "status": "passed",
        "records": args.expected_records,
        "parts": [str(path.resolve()) for path in parts],
        "sqlite": str(args.output.resolve()),
        "size_bytes": args.output.stat().st_size,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

