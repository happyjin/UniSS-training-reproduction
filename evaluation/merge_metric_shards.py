"""Merge independently written metric shards into canonical evaluation files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from evaluation import autopcp_metrics, utmos_metrics
from evaluation.io_utils import iter_jsonl, write_json
from evaluation.sharding import merge_jsonl_by_key, row_key


def eligible_keys(results_path: Path) -> set[tuple[str, str]]:
    return {
        row_key(row)
        for row in iter_jsonl(results_path)
        if row.get("audio_path") and not row.get("error")
    }


def metric_paths(metric: str, metric_dir: Path, shard_root: Path, num_shards: int) -> tuple[Path, list[Path]]:
    if metric == "asr":
        canonical = metric_dir / "asr_results.jsonl"
        parts = [shard_root / "asr" / f"part_{index:03d}.jsonl" for index in range(num_shards)]
    elif metric == "utmos":
        canonical = metric_dir / "per_sample_utmos.jsonl"
        parts = [
            shard_root / "utmos" / f"part_{index:03d}" / "per_sample_utmos.jsonl"
            for index in range(num_shards)
        ]
    elif metric == "autopcp":
        canonical = metric_dir / "per_sample_autopcp.jsonl"
        parts = [
            shard_root / "autopcp" / f"part_{index:03d}" / "per_sample_autopcp.jsonl"
            for index in range(num_shards)
        ]
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return canonical, parts


def merge_metric(args: argparse.Namespace) -> Mapping[str, object]:
    results_path = Path(args.input)
    metric_dir = Path(args.metric_dir)
    shard_root = Path(args.shard_root)
    canonical, parts = metric_paths(args.metric, metric_dir, shard_root, args.num_shards)
    merge = merge_jsonl_by_key([canonical, *parts], canonical)
    rows = list(iter_jsonl(canonical))
    actual = {row_key(row) for row in rows}
    expected = eligible_keys(results_path)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            json.dumps(
                {
                    "metric": args.metric,
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

    if args.metric == "asr":
        report: dict[str, object] = {
            "transcribed": len(rows),
            "empty": sum(not str(row.get("asr_text", "")).strip() for row in rows),
            "expected": len(expected),
        }
        write_json(canonical.with_suffix(".summary.json"), report)
    elif args.metric == "utmos":
        report = {**utmos_metrics.aggregate_scores(rows), "failure_count": 0}
        write_json(metric_dir / "utmos.json", report)
    else:
        report = autopcp_metrics.aggregate_scores(rows)
        write_json(metric_dir / "autopcp.json", report)
    return {"metric": args.metric, **merge, **report}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=("asr", "utmos", "autopcp"), required=True)
    parser.add_argument("--input", required=True, help="Canonical decoded results.jsonl")
    parser.add_argument("--metric-dir", required=True)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(merge_metric(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
