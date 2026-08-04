#!/usr/bin/env python3
"""Merge fixed Stage08 Step1 text-probe partitions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.uniss_streamspeech_ctc_v1.stage07_end_to_end_eval.merge_text_probe import (
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_json, args.output_md):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite merged probe: {output}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.parts]
    summary, rows = summarize(payloads)
    output = {
        "schema_version": "uniss_streamspeech_stage08_step1_text_probe_v1",
        "parts": [str(path) for path in args.parts],
        "summary": summary,
        "samples": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    en_zh = summary["bleu"]["eng->cmn"]["score"]
    zh_en = summary["bleu"]["cmn->eng"]["score"]
    args.output_md.write_text(
        f"# Stage08 Step1 checkpoint {summary['checkpoint_iteration']} text gate\n\n"
        "| Metric | EN→ZH | ZH→EN |\n|---|---:|---:|\n"
        f"| Text-BLEU | {en_zh:.4f} | {zh_en:.4f} |\n\n"
        f"Samples: {summary['samples']}; non-empty: {summary['nonempty']}; "
        f"END_CONTENT rate: {summary['end_content_rate']:.4f}.\n\n"
        f"Compute RTF/source mean/p95: {summary['compute_rtf_source_mean']:.4f} / "
        f"{summary['compute_rtf_source_p95']:.4f}; mean first text token wall time: "
        f"{summary['first_text_token_seconds_mean']:.4f} s; max peak memory: "
        f"{summary['peak_memory_mib_max']:.1f} MiB.\n\n"
        "This is the same fixed direction-balanced 32-row full-utterance probe used "
        "for Stage04 and Stage07 comparisons.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
