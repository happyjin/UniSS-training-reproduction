#!/usr/bin/env python3
"""Validate generated Stage01 sidecar parts and tokenizer round trips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctc_utils import load_processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--sample-per-part", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = json.loads(args.dataset_index.read_text(encoding="utf-8"))
    tokenizer_entries = index["tokenizers"]["tokenizers"]
    processors = {
        entry["language"]: load_processor(entry["model"])
        for entry in tokenizer_entries
    }
    checked = 0
    errors: list[str] = []
    split_counts = {"train": 0, "valid": 0}
    for split, parts in index["parts"].items():
        for part in parts:
            with Path(part).open(encoding="utf-8") as handle:
                for line_index, line in enumerate(handle):
                    split_counts[split] += 1
                    if line_index >= args.sample_per_part:
                        continue
                    row = json.loads(line)
                    checked += 1
                    src_lang = row["source_head"].split("_")[-1]
                    tgt_lang = row["target_head"].split("_")[-1]
                    for role, language in (("source", src_lang), ("target", tgt_lang)):
                        ids = row[f"{role}_token_ids"]
                        if not ids:
                            errors.append(f"{row['id']}:{role}:empty")
                        if max(ids) >= processors[language].vocab_size() or min(ids) < 0:
                            errors.append(f"{row['id']}:{role}:out_of_vocab")
                    expected_25 = row["frames_25hz"] >= max(
                        row["source_ctc_min_frames"], row["target_ctc_min_frames"]
                    )
                    if bool(row["valid_25hz"]) != expected_25:
                        errors.append(f"{row['id']}:bad_25hz_flag")
    report = {
        "schema_version": "uniss_streamspeech_ctc_stage01_validation_v1",
        "status": "passed" if not errors else "failed",
        "sample_records_checked": checked,
        "full_line_counts": split_counts,
        "expected_line_counts": index["written"],
        "errors": errors[:100],
    }
    if split_counts != index["written"]:
        report["status"] = "failed"
        report["errors"].append("sidecar line counts do not match dataset index")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

