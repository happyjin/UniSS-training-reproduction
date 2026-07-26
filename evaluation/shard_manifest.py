"""Create deterministic index-modulo JSONL manifest shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from evaluation.io_utils import iter_jsonl, write_jsonl


def shard_manifest(input_path: Path, output_dir: Path, num_shards: int) -> dict[str, object]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    rows = list(iter_jsonl(input_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    counts = []
    for shard_index in range(num_shards):
        path = output_dir / f"manifest.part_{shard_index:03d}-of-{num_shards:03d}.jsonl"
        count = write_jsonl(path, (row for index, row in enumerate(rows) if index % num_shards == shard_index))
        paths.append(str(path.resolve()))
        counts.append(count)
    return {"input_count": len(rows), "num_shards": num_shards, "counts": counts, "paths": paths}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(shard_manifest(args.input, args.output_dir, args.num_shards), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
