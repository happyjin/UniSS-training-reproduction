#!/usr/bin/env python3
"""Extract the exact CVSS-T zh/en test source clips from mirrored CoVoST2 parquet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_cvss_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if len(row) != 2:
                raise ValueError(f"{path}:{line_number} must contain filename and translation")
            ids.append(row[0])
    if len(ids) != 4897 or len(set(ids)) != 4897:
        raise ValueError(f"Expected 4,897 unique CVSS test IDs, found {len(ids)} rows and {len(set(ids))} unique")
    return ids


def normalized_filename(row: dict[str, object]) -> str:
    audio = row.get("audio")
    if isinstance(audio, dict) and audio.get("path"):
        return Path(str(audio["path"])).name
    value = row.get("id") or row.get("file")
    name = Path(str(value)).name
    return name if name.endswith(".mp3") else f"{name}.mp3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--cvss-test-tsv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-repo", default="fixie-ai/covost2")
    parser.add_argument("--source-revision", default="17c8c81e331e7a6929118121771a58c7ef7331d8")
    args = parser.parse_args()

    parquet_paths = sorted(args.input_dir.glob("test-*.parquet"))
    if len(parquet_paths) != 8:
        raise FileNotFoundError(f"Expected 8 test parquet shards in {args.input_dir}, found {len(parquet_paths)}")
    expected_ids = read_cvss_ids(args.cvss_test_tsv)
    expected = set(expected_ids)
    rows_by_id: dict[str, dict[str, object]] = {}
    mirror_ids: set[str] = set()
    duplicate_ids: list[str] = []

    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path)
        required = {"audio", "sentence", "translation"}
        missing_columns = required - set(table.column_names)
        if missing_columns:
            raise ValueError(f"{parquet_path} is missing columns: {sorted(missing_columns)}")
        for row in table.to_pylist():
            filename = normalized_filename(row)
            if filename in mirror_ids:
                duplicate_ids.append(filename)
            mirror_ids.add(filename)
            if filename in expected:
                rows_by_id[filename] = row

    missing_ids = sorted(expected - set(rows_by_id))
    extra_ids = sorted(mirror_ids - expected)
    if duplicate_ids:
        raise ValueError(f"Mirror contains duplicate IDs: {duplicate_ids[:10]}")
    if missing_ids:
        raise ValueError(f"Mirror is missing {len(missing_ids)} CVSS IDs: {missing_ids[:10]}")

    clips_dir = args.output_root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows: list[dict[str, str]] = []
    audio_hashes: dict[str, str] = {}

    for filename in expected_ids:
        row = rows_by_id[filename]
        audio = row.get("audio")
        if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
            raise ValueError(f"Missing embedded audio bytes for {filename}")
        audio_bytes = audio["bytes"]
        output_path = clips_dir / filename
        digest = sha256_bytes(audio_bytes)
        if output_path.exists():
            existing_digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
            if existing_digest != digest:
                raise ValueError(f"Existing clip differs from mirror: {output_path}")
        else:
            output_path.write_bytes(audio_bytes)
        sentence = str(row.get("sentence") or "").strip()
        if not sentence:
            raise ValueError(f"Missing Chinese source sentence for {filename}")
        metadata_rows.append(
            {
                "client_id": str(row.get("client_id") or ""),
                "path": filename,
                "sentence": sentence,
            }
        )
        audio_hashes[filename] = digest

    metadata_path = args.output_root / "test.tsv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["client_id", "path", "sentence"], delimiter="\t")
        writer.writeheader()
        writer.writerows(metadata_rows)

    hashes_path = args.output_root / "audio_sha256.json"
    hashes_path.write_text(json.dumps(audio_hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "source_repo": args.source_repo,
        "source_revision": args.source_revision,
        "mirror_row_count": len(mirror_ids),
        "cvss_expected_count": len(expected_ids),
        "matched_count": len(rows_by_id),
        "missing_count": len(missing_ids),
        "extra_count": len(extra_ids),
        "extra_ids": extra_ids,
        "output_root": str(args.output_root.resolve()),
        "metadata_path": str(metadata_path.resolve()),
    }
    (args.output_root / "extraction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "SOURCE.txt").write_text(
        "\n".join(
            [
                f"source_repo={args.source_repo}",
                f"source_revision={args.source_revision}",
                "source_config=zh-CN_en",
                "source_split=test",
                "note=Public secondary mirror of Common Voice v4 / CoVoST2 test audio; not a complete official Common Voice v4 archive.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
