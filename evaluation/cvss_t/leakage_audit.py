"""Audit exact ID and paper-normalized text overlap with UniST training data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow.parquet as pq

from evaluation.io_utils import iter_jsonl, write_json
from evaluation.text_metrics import normalize_chinese, normalize_english, opencc_simplifier
from training.constants_uniss import normalize_language


@dataclass(frozen=True)
class ReferenceTexts:
    ids: frozenset[str]
    ids_without_extension: frozenset[str]
    chinese: frozenset[str]
    english: frozenset[str]


def build_references(rows: Sequence[Mapping[str, object]]) -> ReferenceTexts:
    simplifier = opencc_simplifier()
    ids = {str(row["id"]) for row in rows}
    chinese = {
        normalize_chinese(str(row["source_zh_text"]), simplify=simplifier)
        for row in rows
        if str(row.get("source_zh_text") or "").strip()
    }
    english = {
        normalize_english(str(row["target_en_text"]))
        for row in rows
        if str(row.get("target_en_text") or "").strip()
    }
    return ReferenceTexts(
        ids=frozenset(ids),
        ids_without_extension=frozenset(Path(value).stem for value in ids),
        chinese=frozenset(chinese),
        english=frozenset(english),
    )


def normalize_text(text: object, language: object, *, simplifier) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    if normalize_language(str(language)) == "eng":
        return normalize_english(value)
    return normalize_chinese(value, simplify=simplifier)


def audit_shard(
    path_text: str,
    references: ReferenceTexts,
    *,
    max_examples: int,
) -> dict[str, object]:
    path = Path(path_text)
    parquet_file = pq.ParquetFile(path)
    required = {"id", "dataset_name", "src_lang", "tgt_lang", "transcription", "translation"}
    available = set(parquet_file.schema_arrow.names)
    missing = required - available
    if missing:
        raise ValueError(f"{path} is missing leakage-audit columns: {sorted(missing)}")

    simplifier = opencc_simplifier()
    row_count = 0
    matched_record_keys: set[tuple[str, int]] = set()
    id_match_count = 0
    text_match_counts: Counter[str] = Counter()
    dataset_match_counts: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    columns = ["id", "dataset_name", "src_lang", "tgt_lang", "transcription", "translation"]
    for batch in parquet_file.iter_batches(columns=columns, batch_size=4096):
        for row in batch.to_pylist():
            row_index = row_count
            row_count += 1
            sample_id = str(row["id"])
            dataset_name = str(row["dataset_name"])
            fields: list[tuple[str, str, str]] = []
            if sample_id in references.ids or sample_id in references.ids_without_extension:
                id_match_count += 1
                fields.append(("id", "id", sample_id))

            for field_name, language_field in (("transcription", "src_lang"), ("translation", "tgt_lang")):
                language = normalize_language(str(row[language_field]))
                normalized = normalize_text(row[field_name], language, simplifier=simplifier)
                reference_set = references.english if language == "eng" else references.chinese
                if normalized and normalized in reference_set:
                    key = f"{field_name}:{language}"
                    text_match_counts[key] += 1
                    fields.append((field_name, language, normalized))

            if fields:
                matched_record_keys.add((path.name, row_index))
                dataset_match_counts[dataset_name] += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "shard": str(path.resolve()),
                            "row_index": row_index,
                            "id": sample_id,
                            "dataset_name": dataset_name,
                            "src_lang": row["src_lang"],
                            "tgt_lang": row["tgt_lang"],
                            "matches": [
                                {"field": field, "language": language, "normalized_value": value}
                                for field, language, value in fields
                            ],
                        }
                    )
    return {
        "path": str(path.resolve()),
        "row_count": row_count,
        "matched_record_count": len(matched_record_keys),
        "id_match_count": id_match_count,
        "text_match_counts": dict(text_match_counts),
        "dataset_match_counts": dict(dataset_match_counts),
        "examples": examples,
    }


def merge_results(
    shard_results: Sequence[Mapping[str, object]],
    *,
    pair_manifest: Path,
    train_paths: Sequence[Path],
    max_examples: int,
) -> dict[str, object]:
    text_match_counts: Counter[str] = Counter()
    dataset_match_counts: Counter[str] = Counter()
    examples: list[object] = []
    for result in shard_results:
        text_match_counts.update(result["text_match_counts"])  # type: ignore[arg-type]
        dataset_match_counts.update(result["dataset_match_counts"])  # type: ignore[arg-type]
        remaining = max_examples - len(examples)
        if remaining > 0:
            examples.extend(list(result["examples"])[:remaining])  # type: ignore[arg-type]
    return {
        "pair_manifest": str(pair_manifest.resolve()),
        "train_shard_count": len(train_paths),
        "train_row_count": sum(int(result["row_count"]) for result in shard_results),
        "matched_train_record_count": sum(int(result["matched_record_count"]) for result in shard_results),
        "id_match_count": sum(int(result["id_match_count"]) for result in shard_results),
        "text_match_counts": dict(sorted(text_match_counts.items())),
        "dataset_match_counts": dict(sorted(dataset_match_counts.items())),
        "audio_exact_overlap_status": "deferred_until_cvss_tokenization",
        "audio_exact_overlap_reason": "UniST training parquet stores speech tokens but no original audio path or audio hash",
        "examples": examples,
        "shards": list(shard_results),
    }


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    pair_manifest = Path(args.pair_manifest)
    train_paths = sorted(Path().glob(args.train_glob) if not Path(args.train_glob).is_absolute() else Path("/").glob(args.train_glob[1:]))
    if not train_paths:
        raise FileNotFoundError(f"No training parquet matched: {args.train_glob}")
    references = build_references(list(iter_jsonl(pair_manifest)))
    if len(references.ids) != 4897:
        raise ValueError(f"Expected 4,897 CVSS reference IDs, found {len(references.ids)}")

    if args.workers == 1:
        shard_results = [audit_shard(str(path), references, max_examples=args.examples_per_shard) for path in train_paths]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    audit_shard,
                    str(path),
                    references,
                    max_examples=args.examples_per_shard,
                )
                for path in train_paths
            ]
            shard_results = [future.result() for future in futures]
    report = merge_results(
        shard_results,
        pair_manifest=pair_manifest,
        train_paths=train_paths,
        max_examples=args.max_examples,
    )
    write_json(Path(args.output), report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--train-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--examples-per-shard", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=200)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.examples_per_shard < 0 or args.max_examples < 0:
        parser.error("worker and example counts must be non-negative, with at least one worker")
    if args.output.exists():
        parser.error(f"Refusing to overwrite leakage report: {args.output}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    report = run_audit(parse_args(argv))
    summary = {key: value for key, value in report.items() if key not in {"examples", "shards"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
