#!/usr/bin/env python3
"""Join causal pre-VQ latent references with Stage01 CTC text targets."""

from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np


STAGE01 = Path(__file__).resolve().parents[1] / "stage01_data"
sys.path.insert(0, str(STAGE01))
from ctc_utils import (  # noqa: E402
    canonical_lang,
    ctc_minimum_frames,
    deterministic_split,
    load_processor,
    normalize_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-manifest", type=Path, required=True)
    parser.add_argument("--latent-offsets", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-id-index", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--valid-basis-points", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker-index outside num-workers")
    processors = {
        language: load_processor(args.tokenizer_dir / f"ctc_{language}.model")
        for language in ("eng", "cmn")
    }
    offsets = np.memmap(args.latent_offsets, mode="r", dtype=np.uint64)
    total = min(len(offsets), args.limit) if args.limit else len(offsets)
    start = total * args.worker_index // args.num_workers
    stop = total * (args.worker_index + 1) // args.num_workers
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"part-{args.worker_index:03d}-of-{args.num_workers:03d}"
    output_paths = {
        split: args.output_dir / f"{split}-{stem}.jsonl" for split in ("train", "valid")
    }
    offset_paths = {
        split: Path(str(path) + ".offsets.bin") for split, path in output_paths.items()
    }
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    directions: Counter[str] = Counter()

    source_lookup = sqlite3.connect(
        f"file:{args.source_id_index.resolve()}?mode=ro&immutable=1", uri=True
    )
    with (
        args.latent_manifest.open("rb") as latent_source,
        args.source_manifest.open("rb") as text_source,
        output_paths["train"].open("wb") as train_output,
        output_paths["valid"].open("wb") as valid_output,
        offset_paths["train"].open("wb") as train_offsets,
        offset_paths["valid"].open("wb") as valid_offsets,
    ):
        outputs = {"train": train_output, "valid": valid_output}
        output_offsets = {"train": train_offsets, "valid": valid_offsets}
        for latent_index in range(start, stop):
            latent_source.seek(int(offsets[latent_index]))
            try:
                latent = json.loads(latent_source.readline())
                match = source_lookup.execute(
                    "SELECT record_index, byte_offset FROM source_offsets WHERE id = ?",
                    (str(latent["id"]),),
                ).fetchone()
                if match is None:
                    raise KeyError(f"source ID not found: {latent['id']}")
                source_index, source_offset = map(int, match)
                text_source.seek(source_offset)
                row = json.loads(text_source.readline())
                if str(latent["id"]) != str(row["id"]):
                    raise ValueError("latent/source id mismatch")
                src_lang = canonical_lang(str(row["src_lang"]))
                tgt_lang = canonical_lang(str(row["tgt_lang"]))
                source_ids = list(
                    processors[src_lang].encode(
                        normalize_text(row["transcription"], src_lang), out_type=int
                    )
                )
                target_ids = list(
                    processors[tgt_lang].encode(
                        normalize_text(row["translation"], tgt_lang), out_type=int
                    )
                )
                hidden_start = int(latent["target_start"])
                hidden_end = int(latent["target_end"])
                hidden_frames = hidden_end - hidden_start
                source_min = ctc_minimum_frames(source_ids)
                target_min = ctc_minimum_frames(target_ids)
                direction = f"{src_lang}->{tgt_lang}"
                if not source_ids or not target_ids:
                    skipped[f"empty_target:{direction}"] += 1
                    continue
                if hidden_frames < source_min:
                    skipped[f"source_path_too_short:{direction}"] += 1
                    continue
                if hidden_frames < target_min:
                    skipped[f"target_path_too_short:{direction}"] += 1
                    continue
                split = deterministic_split(str(row["id"]), args.valid_basis_points)
                payload = {
                    "schema_version": "uniss_streamspeech_ctc_probe_row_v1",
                    "id": row["id"],
                    "direction": direction,
                    "source_head": f"asr_{src_lang}",
                    "target_head": f"nar_s2tt_{tgt_lang}",
                    "shard_path": latent["shard_path"],
                    "shard_row": latent["shard_row"],
                    "source_manifest_index": source_index,
                    "hidden_start": hidden_start,
                    "hidden_end": hidden_end,
                    "hidden_frames": hidden_frames,
                    "source_token_ids": source_ids,
                    "target_token_ids": target_ids,
                }
                encoded = (
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                output_offsets[split].write(struct.pack("<Q", outputs[split].tell()))
                outputs[split].write(encoded)
                counts[split] += 1
                directions[f"{split}:{direction}"] += 1
            except Exception as exc:
                skipped[f"exception:{type(exc).__name__}"] += 1

    summary = {
        "schema_version": "uniss_streamspeech_ctc_probe_part_summary_v1",
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "input_start": start,
        "input_stop": stop,
        "input_records": stop - start,
        "written": dict(counts),
        "skipped": dict(skipped),
        "directions": dict(directions),
        "outputs": {
            split: {
                "manifest": str(output_paths[split].resolve()),
                "offsets": str(offset_paths[split].resolve()),
                "records": counts[split],
            }
            for split in ("train", "valid")
        },
    }
    (args.output_dir / f"{stem}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    source_lookup.close()


if __name__ == "__main__":
    main()
