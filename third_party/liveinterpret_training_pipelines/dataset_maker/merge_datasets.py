#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge multiple JSONL datasets by sampling each input with given ratios.

Example:
  python3 merge_datasets.py \
      --inputs speech_to_speech_translation.jsonl text_to_speech.jsonl \
      --ratios 1 1 \
      --output-name bridge_alignment_joint
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Iterable, List, Sequence, Set

import shardio


# Output directory for the merged jsonl manifest. Mirrors the worker's
# ${BASE_PATH}/datasets/jsonl/ so the argparse default never NameErrors
# (the worker still passes --output-dir explicitly).
DEFAULT_BASE_PATH = os.environ.get(
    "BASE_PATH",
    "/local/speech-language-model/liveinterpret_training_pipelines",
)
DEFAULT_JSONL_DIR = os.environ.get(
    "DEFAULT_JSONL_DIR",
    os.path.join(DEFAULT_BASE_PATH, "datasets", "jsonl"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample multiple JSONL datasets by ratio and concatenate them into one JSONL."
    )
    parser.add_argument(
        "--inputs",
        "-i",
        nargs="+",
        required=True,
        help="Input JSONL paths. Provide one or more files.",
    )
    parser.add_argument(
        "--ratios",
        "-r",
        nargs="+",
        type=float,
        required=True,
        help="Sampling ratios for inputs, e.g. --ratios 1 1 0.5.",
    )
    parser.add_argument(
        "--output-name",
        "-n",
        required=True,
        help="Output dataset name. '.jsonl' is appended if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        default=DEFAULT_JSONL_DIR,
        help=f"Output directory. Default: {DEFAULT_JSONL_DIR}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling. Default: 42.",
    )
    parser.add_argument(
        "--no-shuffle-sample",
        action="store_true",
        help="Use the first N samples from each input instead of random sampling.",
    )
    parser.add_argument(
        "--shuffle-output",
        action="store_true",
        help="Shuffle sampled records across datasets before writing. Default keeps per-input order.",
    )
    parser.add_argument(
        "--no-validate-json",
        action="store_true",
        help="Do not validate each non-empty line as JSON before counting/writing.",
    )
    parser.add_argument(
        "--shard-max-lines",
        type=int,
        default=0,
        help=(
            "If > 0, split the merged output into multiple shards of at most this "
            "many lines each (written to <output_dir>/<name>/<name>_NNNNN.jsonl) so "
            "the training preprocess can read them in parallel. 0 = single file "
            "(backward compatible). Default: 0"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(args.inputs) != len(args.ratios):
        raise ValueError("--inputs and --ratios must have the same length.")
    if not args.inputs:
        raise ValueError("At least one input JSONL is required.")
    for path in args.inputs:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file/directory not found: {path}")
    for ratio in args.ratios:
        if ratio <= 0:
            raise ValueError(f"Ratios must be positive numbers, got: {ratio}")


def output_path(output_dir: str, output_name: str) -> str:
    filename = output_name if output_name.endswith(".jsonl") else f"{output_name}.jsonl"
    return os.path.abspath(os.path.join(output_dir, filename))


def iter_valid_lines(path: str, validate_json: bool) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            if validate_json:
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"[WARN] Skipping malformed JSON in {path}:{line_no}: {exc}",
                        file=sys.stderr,
                    )
                    continue
            yield line


def count_valid_lines(path: str, validate_json: bool) -> int:
    return sum(1 for _ in iter_valid_lines(path, validate_json))


def compute_sample_counts(sizes: Sequence[int], ratios: Sequence[float]) -> List[int]:
    scale = min(size / ratio for size, ratio in zip(sizes, ratios))
    counts = [min(size, int(math.floor(scale * ratio))) for size, ratio in zip(sizes, ratios)]
    return counts


def choose_indices(size: int, count: int, rng: random.Random, random_sample: bool) -> Set[int]:
    if count <= 0:
        return set()
    if count >= size:
        return set(range(size))
    if random_sample:
        return set(rng.sample(range(size), count))
    return set(range(count))


def collect_sampled_lines(
    inputs_files: Sequence[Sequence[str]],
    counts: Sequence[int],
    validate_json: bool,
    random_sample: bool,
    seed: int,
) -> List[List[str]]:
    rng = random.Random(seed)
    sampled: List[List[str]] = []
    for file_list, count in zip(inputs_files, counts):
        size = shardio.count_valid_lines_multi(file_list, validate_json)
        indices = choose_indices(size, count, rng, random_sample)
        lines: List[str] = []
        for idx, line in enumerate(shardio.iter_valid_lines_multi(file_list, validate_json)):
            if idx in indices:
                lines.append(line)
        sampled.append(lines)
    return sampled


def write_output(
    path: str,
    grouped_lines: Sequence[Sequence[str]],
    shuffle_output: bool,
    seed: int,
    shard_max_lines: int = 0,
) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if shuffle_output:
        merged = [line for lines in grouped_lines for line in lines]
        random.Random(seed).shuffle(merged)
    else:
        merged = []
        for lines in grouped_lines:
            merged.extend(lines)

    writer = shardio.ShardWriter(path, shard_max_lines)
    try:
        for line in merged:
            writer.write(line)
    finally:
        writer.close()
    return len(merged)


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)

        validate_json = not args.no_validate_json
        random_sample = not args.no_shuffle_sample
        inputs_files = [shardio.resolve_input_paths([p]) for p in args.inputs]
        sizes = [shardio.count_valid_lines_multi(fl, validate_json) for fl in inputs_files]
        counts = compute_sample_counts(sizes, args.ratios)
        if sum(counts) == 0:
            raise ValueError("Computed zero samples. Check input sizes and ratios.")

        grouped_lines = collect_sampled_lines(
            inputs_files,
            counts,
            validate_json=validate_json,
            random_sample=random_sample,
            seed=args.seed,
        )
        out_path = output_path(args.output_dir, args.output_name)
        written = write_output(
            out_path, grouped_lines, args.shuffle_output, args.seed, args.shard_max_lines
        )

        print("[DONE] merged datasets", file=sys.stderr)
        for path, size, ratio, count in zip(args.inputs, sizes, args.ratios, counts):
            print(
                f"       input={path} size={size} ratio={ratio:g} sampled={count}",
                file=sys.stderr,
            )
        print(f"       total={written}", file=sys.stderr)
        if args.shard_max_lines and args.shard_max_lines > 0:
            out_desc = output_path(args.output_dir, args.output_name).replace(
                ".jsonl", ""
            ) + "/ (shards)"
        else:
            out_desc = out_path
        print(f"       output -> {out_desc}", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
