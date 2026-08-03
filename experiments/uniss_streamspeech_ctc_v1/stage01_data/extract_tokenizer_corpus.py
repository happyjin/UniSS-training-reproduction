#!/usr/bin/env python3
"""Extract deterministic, worker-local English/Chinese tokenizer corpora."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ctc_utils import canonical_lang, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--offsets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker-index outside num-workers")
    offsets = np.memmap(args.offsets, mode="r", dtype=np.uint64)
    total = min(len(offsets), args.limit) if args.limit else len(offsets)
    start = total * args.worker_index // args.num_workers
    stop = total * (args.worker_index + 1) // args.num_workers
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"part-{args.worker_index:03d}-of-{args.num_workers:03d}"
    paths = {
        "eng": args.output_dir / f"corpus-eng-{stem}.txt",
        "cmn": args.output_dir / f"corpus-cmn-{stem}.txt",
    }
    counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    with (
        args.manifest.open("rb") as source,
        paths["eng"].open("w", encoding="utf-8") as english,
        paths["cmn"].open("w", encoding="utf-8") as chinese,
    ):
        outputs = {"eng": english, "cmn": chinese}
        for record_index in range(start, stop):
            source.seek(int(offsets[record_index]))
            try:
                row = json.loads(source.readline())
                pairs = (
                    (canonical_lang(str(row["src_lang"])), row["transcription"]),
                    (canonical_lang(str(row["tgt_lang"])), row["translation"]),
                )
                for language, raw_text in pairs:
                    text = normalize_text(str(raw_text), language)
                    if not text:
                        rejected[f"empty_{language}"] += 1
                        continue
                    outputs[language].write(text.replace("\n", " ") + "\n")
                    counts[language] += 1
            except Exception as exc:
                rejected[type(exc).__name__] += 1
    summary = {
        "schema_version": "uniss_streamspeech_ctc_tokenizer_corpus_part_v1",
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "record_start": start,
        "record_stop": stop,
        "records": stop - start,
        "lines": dict(counts),
        "rejected": dict(rejected),
        "outputs": {key: str(value.resolve()) for key, value in paths.items()},
    }
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

