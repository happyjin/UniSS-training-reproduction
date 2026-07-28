"""Merge sharded vLLM generation/audio outputs into canonical evaluation files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from evaluation.io_utils import iter_jsonl, write_json
from evaluation.sharding import merge_jsonl_by_key, row_key


def missing_text(value: object) -> bool:
    return not isinstance(value, str) or not value.strip()


def expected_keys(manifest: Path, modes: Sequence[str]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in iter_jsonl(manifest):
        sample_id = str(row["id"])
        for mode in modes:
            key = sample_id, mode
            if key in keys:
                raise ValueError(f"Duplicate expected evaluation key: {key}")
            keys.add(key)
    return keys


def validate_keys(path: Path, expected: set[tuple[str, str]], *, kind: str) -> list[dict[str, object]]:
    rows = list(iter_jsonl(path))
    actual = {row_key(row) for row in rows}
    if len(actual) != len(rows):
        raise ValueError(f"{kind} output contains duplicate id/mode keys")
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            json.dumps(
                {
                    "kind": kind,
                    "expected": len(expected),
                    "actual": len(actual),
                    "missing_count": len(missing),
                    "unexpected_count": len(unexpected),
                    "missing_examples": missing[:20],
                    "unexpected_examples": unexpected[:20],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return rows


def merge_shards(args: argparse.Namespace) -> dict[str, object]:
    manifest = Path(args.manifest)
    output_root = Path(args.output_root)
    shard_root = Path(args.shard_root)
    expected = expected_keys(manifest, args.modes)
    generation_output = output_root / "vllm" / "generation_results.jsonl"
    results_output = output_root / "results.jsonl"
    generation_parts = [
        shard_root / f"shard_{index:03d}" / "vllm" / "generation_results.jsonl"
        for index in range(args.num_shards)
    ]
    results_parts = [shard_root / f"shard_{index:03d}" / "results.jsonl" for index in range(args.num_shards)]
    generation_merge = merge_jsonl_by_key([generation_output, *generation_parts], generation_output)
    results_merge = merge_jsonl_by_key([results_output, *results_parts], results_output)
    generation_rows = validate_keys(generation_output, expected, kind="generation")
    result_rows = validate_keys(results_output, expected, kind="decoded_audio")

    generation_summary = {
        "completed_before_resume": 0,
        "generated": len(generation_rows),
        "no_semantic_tokens": sum(int(row.get("semantic_token_count", 0)) == 0 for row in generation_rows),
        "missing_translation": sum(missing_text(row.get("generated_translation")) for row in generation_rows),
        "dummy_generated_tokens": sum(int(row.get("dummy_token_count", 0)) for row in generation_rows),
        "total_results": len(generation_rows),
        "num_data_parallel_shards": args.num_shards,
    }
    write_json(output_root / "vllm" / "generation_summary.json", generation_summary)
    decode_summary = {
        "decoded": len(result_rows),
        "failed": sum(
            bool(row.get("error") or row.get("source_audio_error") or row.get("reference_audio_error"))
            for row in result_rows
        ),
        "source_audio": sum(bool(row.get("source_audio_path")) for row in result_rows),
        "reference_audio": sum(bool(row.get("reference_audio_path")) for row in result_rows),
        "no_semantic_tokens": sum(int(row.get("semantic_token_count", 0)) == 0 for row in result_rows),
        "num_data_parallel_shards": args.num_shards,
    }
    write_json(output_root / "summary.json", decode_summary)
    return {
        "expected": len(expected),
        "generation_merge": generation_merge,
        "results_merge": results_merge,
        "generation_summary": generation_summary,
        "decode_summary": decode_summary,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--modes", nargs="+", default=["quality", "performance"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(merge_shards(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
