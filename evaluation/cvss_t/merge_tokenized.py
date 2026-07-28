"""Merge and validate sharded CVSS-T UniSS token parquets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from evaluation.io_utils import write_json
from evaluation.unist_manifest import create_manifests
from training.prepare_unist_s2st import normalize_unist_record


def validate_rows(rows: Sequence[Mapping[str, object]], *, expected_pairs: int, direction: str) -> None:
    if len(rows) != expected_pairs:
        raise ValueError(f"{direction} row count={len(rows)}, expected={expected_pairs}")
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{direction} contains duplicate IDs")
    pair_indices = [int(row["pair_index"]) for row in rows]
    if pair_indices != list(range(expected_pairs)):
        raise ValueError(f"{direction} pair indices are not the complete ordered range")
    for row in rows:
        normalized = normalize_unist_record(row)
        if len(normalized["bicodec_global"]) != 32:  # type: ignore[arg-type]
            raise ValueError(f"{direction}:{row['id']} global token count is not 32")
        if not normalized.get("source_bicodec") or not normalized["target_bicodec"]:
            raise ValueError(f"{direction}:{row['id']} has empty semantic tokens")
        for field in ("source_audio_path", "reference_audio_path"):
            path = Path(str(row[field]))
            if not path.is_file():
                raise FileNotFoundError(f"{direction}:{row['id']} missing {field}: {path}")


def write_final_parquet(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite final tokenized parquet: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        pq.write_table(pa.Table.from_pylist(list(rows)), temporary, compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def merge_direction(part_paths: Sequence[Path], *, expected_pairs: int, direction: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in part_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {direction} tokenization part: {path}")
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: int(row["pair_index"]))
    validate_rows(rows, expected_pairs=expected_pairs, direction=direction)
    return rows


def merge_tokenized(args: argparse.Namespace) -> dict[str, object]:
    part_root = Path(args.part_root)
    output_dir = Path(args.output_dir)
    zh_en_parts = [
        part_root / "zh_en" / f"part_{index:03d}-of-{args.num_shards:03d}.parquet"
        for index in range(args.num_shards)
    ]
    en_zh_parts = [
        part_root / "en_zh" / f"part_{index:03d}-of-{args.num_shards:03d}.parquet"
        for index in range(args.num_shards)
    ]
    zh_en_rows = merge_direction(zh_en_parts, expected_pairs=args.expected_pairs, direction="cmn->eng")
    en_zh_rows = merge_direction(en_zh_parts, expected_pairs=args.expected_pairs, direction="eng->cmn")
    zh_en_path = output_dir / "cvss_t_zh_en_test.parquet"
    en_zh_path = output_dir / "cvss_t_en_zh_test.parquet"
    write_final_parquet(zh_en_path, zh_en_rows)
    write_final_parquet(en_zh_path, en_zh_rows)

    manifest_root = output_dir / "manifests"
    zh_en_manifests = create_manifests(
        zh_en_path,
        manifest_root / "zh_en",
        split_name="test",
        seed=args.seed,
        smoke_count=args.smoke_count,
        listen_count=args.listen_count,
    )
    en_zh_manifests = create_manifests(
        en_zh_path,
        manifest_root / "en_zh",
        split_name="test",
        seed=args.seed,
        smoke_count=args.smoke_count,
        listen_count=args.listen_count,
    )
    summary = {
        "expected_pairs": args.expected_pairs,
        "num_shards": args.num_shards,
        "zh_en_parquet": str(zh_en_path.resolve()),
        "en_zh_parquet": str(en_zh_path.resolve()),
        "zh_en_manifests": zh_en_manifests,
        "en_zh_manifests": en_zh_manifests,
    }
    write_json(output_dir / "tokenization_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--expected-pairs", type=int, default=4897)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--smoke-count", type=int, default=10)
    parser.add_argument("--listen-count", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(merge_tokenized(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
