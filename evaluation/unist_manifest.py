"""Build deterministic, stratified UniST evaluation manifests."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pyarrow.parquet as pq

from evaluation.io_utils import write_json, write_jsonl


MANIFEST_COLUMNS = (
    "id",
    "dataset_name",
    "src_lang",
    "tgt_lang",
    "transcription",
    "translation",
    "source_glm",
    "source_bicodec",
    "target_bicodec",
    "bicodec_global",
    "duration_ratio",
)


def stable_score(seed: int, sample_id: object) -> str:
    payload = f"{seed}:{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parquet_manifest_rows(path: Path, *, repo_root: Path | None = None) -> list[dict[str, object]]:
    parquet_file = pq.ParquetFile(path)
    available = set(parquet_file.schema_arrow.names)
    columns = [column for column in MANIFEST_COLUMNS if column in available]
    required = {"id", "dataset_name", "src_lang", "tgt_lang", "transcription", "translation"}
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"{path} is missing required manifest columns: {missing}")

    resolved = path.resolve()
    if repo_root is not None:
        try:
            path_text = str(resolved.relative_to(repo_root.resolve()))
        except ValueError:
            path_text = str(resolved)
    else:
        path_text = str(resolved)

    rows: list[dict[str, object]] = []
    row_index = 0
    for batch in parquet_file.iter_batches(columns=columns, batch_size=1024):
        for raw in batch.to_pylist():
            row = {
                "id": str(raw["id"]),
                "parquet_path": path_text,
                "row_index": row_index,
                "dataset_name": str(raw["dataset_name"]),
                "src_lang": str(raw["src_lang"]),
                "tgt_lang": str(raw["tgt_lang"]),
                "transcription": str(raw["transcription"]),
                "translation": str(raw["translation"]),
                "source_glm_length": len(raw.get("source_glm") or []),
                "source_bicodec_length": len(raw.get("source_bicodec") or []),
                "target_bicodec_length": len(raw.get("target_bicodec") or []),
                "bicodec_global_length": len(raw.get("bicodec_global") or []),
                "duration_ratio": raw.get("duration_ratio"),
            }
            rows.append(row)
            row_index += 1
    return rows


def _round_robin(groups: Mapping[tuple[str, str, str], list[dict[str, object]]]) -> Iterable[dict[str, object]]:
    keys = sorted(groups)
    offsets = {key: 0 for key in keys}
    while True:
        emitted = False
        for key in keys:
            offset = offsets[key]
            values = groups[key]
            if offset >= len(values):
                continue
            yield values[offset]
            offsets[key] += 1
            emitted = True
        if not emitted:
            return


def stratified_sample(rows: Sequence[dict[str, object]], count: int, *, seed: int) -> list[dict[str, object]]:
    if count < 1:
        raise ValueError("sample count must be positive")
    if count > len(rows):
        raise ValueError(f"requested {count} rows from a split containing {len(rows)} rows")

    directions: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        directions[(str(row["src_lang"]), str(row["tgt_lang"]))].append(row)

    direction_keys = sorted(directions)
    if count >= len(direction_keys):
        quotas = {key: count // len(direction_keys) for key in direction_keys}
        for key in direction_keys[: count % len(direction_keys)]:
            quotas[key] += 1
    else:
        quotas = {key: int(index < count) for index, key in enumerate(direction_keys)}

    selected: list[dict[str, object]] = []
    for direction in direction_keys:
        grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
        for row in directions[direction]:
            key = (str(row["src_lang"]), str(row["tgt_lang"]), str(row["dataset_name"]))
            grouped[key].append(row)
        for values in grouped.values():
            values.sort(key=lambda row: stable_score(seed, row["id"]))
        selected.extend(list(_round_robin(grouped))[: quotas[direction]])

    selected.sort(key=lambda row: (stable_score(seed, row["id"]), str(row["id"])))
    return selected


def manifest_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    direction_counts = Counter(f"{row['src_lang']}->{row['tgt_lang']}" for row in rows)
    dataset_counts = Counter(str(row["dataset_name"]) for row in rows)
    ids = [str(row["id"]) for row in rows]
    return {
        "count": len(rows),
        "unique_id_count": len(set(ids)),
        "direction_counts": dict(sorted(direction_counts.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
    }


def create_manifests(
    input_path: Path,
    output_dir: Path,
    *,
    split_name: str,
    seed: int,
    smoke_count: int,
    listen_count: int,
    repo_root: Path | None = None,
) -> dict[str, object]:
    rows = parquet_manifest_rows(input_path, repo_root=repo_root)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{input_path} contains duplicate ids")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / f"unist_{split_name}_all.jsonl"
    smoke_path = output_dir / f"unist_{split_name}_smoke_{smoke_count}.jsonl"
    listen_path = output_dir / f"unist_{split_name}_listen_{listen_count}.jsonl"

    smoke_rows = stratified_sample(rows, smoke_count, seed=seed)
    listen_rows = stratified_sample(rows, listen_count, seed=seed)
    write_jsonl(all_path, rows)
    write_jsonl(smoke_path, smoke_rows)
    write_jsonl(listen_path, listen_rows)

    summary = {
        "input": str(input_path.resolve()),
        "split": split_name,
        "seed": seed,
        "all": {"path": str(all_path), **manifest_summary(rows)},
        "smoke": {"path": str(smoke_path), **manifest_summary(smoke_rows)},
        "listen": {"path": str(listen_path), **manifest_summary(listen_rows)},
    }
    write_json(output_dir / f"unist_{split_name}_manifest_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--smoke-count", type=int, default=3)
    parser.add_argument("--listen-count", type=int, default=50)
    parser.add_argument("--repo-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = create_manifests(
        args.input,
        args.output_dir,
        split_name=args.split_name,
        seed=args.seed,
        smoke_count=args.smoke_count,
        listen_count=args.listen_count,
        repo_root=args.repo_root,
    )
    print(summary)


if __name__ == "__main__":
    main()
