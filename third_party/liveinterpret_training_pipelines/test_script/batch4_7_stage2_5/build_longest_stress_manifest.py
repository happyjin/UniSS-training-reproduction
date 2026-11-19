#!/usr/bin/env python3
"""Build a deterministic long-sample stress manifest from legal UniST rows."""
from __future__ import annotations

import argparse
import glob
import heapq
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codec-len", type=int, default=737)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument(
        "--unique-longest",
        action="store_true",
        help="select distinct longest rows instead of repeating exact codec-len matches",
    )
    parser.add_argument("--min-codec-len", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(str(args.dataset_root / "manifests/manifest-*.jsonl")))
    if args.unique_longest:
        # Stream candidate rows and retain only the top N in memory. The regex
        # is deliberately broad; parsed codec_len remains the source of truth.
        pattern = r'"codec_len":[6-9][0-9][0-9]'
        process = subprocess.Popen(
            ["rg", "--threads", "16", "--no-filename", "-e", pattern, *paths],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        longest = []
        serial = 0
        for line in process.stdout:
            row = json.loads(line)
            codec_len = int(row.get("codec_len", -1))
            if not args.min_codec_len <= codec_len <= args.codec_len:
                continue
            item = (codec_len, serial, row)
            serial += 1
            if len(longest) < args.rows:
                heapq.heappush(longest, item)
            elif item[:2] > longest[0][:2]:
                heapq.heapreplace(longest, item)
        returncode = process.wait()
        if returncode not in {0, 1}:
            raise RuntimeError(f"rg failed with exit code {returncode}")
        if len(longest) < args.rows:
            raise RuntimeError(
                f"only {len(longest)} unique rows found in codec range "
                f"[{args.min_codec_len}, {args.codec_len}]"
            )
        matches = [item[2] for item in sorted(longest, reverse=True)]
    else:
        matches = []
        needle = f'"codec_len":{args.codec_len}'
        # The manifests are ~140 GiB. ripgrep scans shards concurrently and
        # emits only the handful of exact-length rows.
        completed = subprocess.run(
            ["rg", "--threads", "16", "--no-filename", "--fixed-strings", needle, *paths],
            check=False, stdout=subprocess.PIPE, text=True)
        if completed.returncode not in {0, 1}:
            raise RuntimeError(f"rg failed with exit code {completed.returncode}")
        for line in completed.stdout.splitlines():
            row = json.loads(line)
            if int(row.get("codec_len", -1)) == args.codec_len:
                matches.append(row)
        if not matches:
            raise RuntimeError(f"no legal codec_len={args.codec_len} rows found")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "manifest-longest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for index in range(args.rows):
            row = dict(matches[index] if args.unique_longest else matches[index % len(matches)])
            if not args.unique_longest:
                row["stress_repeat_index"] = index
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    committed = {
        "schema_version": "ms-swift-speech-simul-v1",
        "dataset_version": "unist_qwen3_s2s_v2_longest_stress",
        "rows": args.rows,
        "source_unique_rows": args.rows if args.unique_longest else len(matches),
        "selection": "unique_longest" if args.unique_longest else "repeat_exact_max",
        "codec_len_max": args.codec_len,
        "codec_len_min": min(int(row["codec_len"]) for row in matches[:args.rows]),
        "manifest_glob": str(manifest.resolve()),
    }
    (args.output_root / "_COMMITTED.json").write_text(
        json.dumps(committed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(committed, ensure_ascii=False))


if __name__ == "__main__":
    main()
