#!/usr/bin/env python3
"""Merge B2 text probes and compute paper-aligned corpus BLEU."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.text_metrics import corpus_bleu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.parts]
    rows = [row for payload in payloads for row in payload["samples"]]
    scores = {}
    for direction, target_lang in (("eng->cmn", "cmn"), ("cmn->eng", "eng")):
        group = [row for row in rows if row["direction"] == direction]
        scores[direction] = corpus_bleu(
            [row["generated_translation"] for row in group],
            [row["translation_ref"] for row in group],
            language=target_lang,
        )
    summary = {
        "samples": len(rows),
        "nonempty": sum(bool(row["generated_translation"]) for row in rows),
        "end_content_rate": sum(bool(row["generated_end_content"]) for row in rows) / len(rows),
        "generation_seconds_mean": statistics.fmean(float(row["generation_seconds"]) for row in rows),
        "bleu": scores,
    }
    output = {
        "schema_version": "uniss_streamspeech_stage04_b2_text_probe_v1",
        "parts": [str(path) for path in args.parts],
        "summary": summary,
        "samples": rows,
    }
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    args.output_md.write_text(
        "# Stage04 B2 frozen-Phase3 text probe\n\n"
        "| Metric | EN→ZH | ZH→EN |\n| --- | ---: | ---: |\n"
        f"| Text-BLEU | {scores['eng->cmn']['score']:.4f} | {scores['cmn->eng']['score']:.4f} |\n\n"
        f"Samples: {len(rows)}; non-empty: {summary['nonempty']}; "
        f"END_CONTENT rate: {summary['end_content_rate']:.4f}.\n\n"
        "This is a direction-balanced greedy text probe on the 15-shard validation sidecar. "
        "It does not yet include semantic-token audio decoding.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

