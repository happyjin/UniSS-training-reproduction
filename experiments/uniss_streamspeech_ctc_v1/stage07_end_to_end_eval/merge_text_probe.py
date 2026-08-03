#!/usr/bin/env python3
"""Merge Stage07 B1 probe partitions and compute fixed-sample metrics."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.text_metrics import corpus_bleu


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute percentile of empty values")
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def summarize(payloads: list[dict]) -> tuple[dict, list[dict]]:
    rows = [row for payload in payloads for row in payload["samples"]]
    iterations = {int(payload["checkpoint"]["iteration"]) for payload in payloads}
    if len(iterations) != 1:
        raise ValueError(f"probe parts mix checkpoint iterations: {sorted(iterations)}")
    scores = {}
    for direction, target_lang in (("eng->cmn", "cmn"), ("cmn->eng", "eng")):
        group = [row for row in rows if row["direction"] == direction]
        scores[direction] = corpus_bleu(
            [row["generated_translation"] for row in group],
            [row["translation_ref"] for row in group],
            language=target_lang,
        )
    rtfs = [float(row["compute_rtf_source"]) for row in rows]
    total_seconds = [float(row["total_seconds"]) for row in rows]
    first_tokens = [
        float(row["bridge_seconds"]) + float(row["qwen_first_token_seconds"])
        for row in rows
    ]
    summary = {
        "checkpoint_iteration": iterations.pop(),
        "samples": len(rows),
        "nonempty": sum(bool(row["generated_translation"]) for row in rows),
        "end_content_rate": sum(bool(row["generated_end_content"]) for row in rows)
        / len(rows),
        "bleu": scores,
        "compute_rtf_source_mean": statistics.fmean(rtfs),
        "compute_rtf_source_p95": percentile(rtfs, 0.95),
        "total_seconds_mean": statistics.fmean(total_seconds),
        "first_text_token_seconds_mean": statistics.fmean(first_tokens),
        "peak_memory_mib_max": max(float(payload["peak_memory_mib"]) for payload in payloads),
    }
    return summary, rows


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
        "schema_version": "uniss_streamspeech_stage07_b1_text_probe_v1",
        "parts": [str(path) for path in args.parts],
        "summary": summary,
        "samples": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    en_zh = summary["bleu"]["eng->cmn"]["score"]
    zh_en = summary["bleu"]["cmn->eng"]["score"]
    args.output_md.write_text(
        f"# Stage07 B1 checkpoint {summary['checkpoint_iteration']} text gate\n\n"
        "| Metric | EN→ZH | ZH→EN |\n| --- | ---: | ---: |\n"
        f"| Text-BLEU | {en_zh:.4f} | {zh_en:.4f} |\n\n"
        f"Samples: {summary['samples']}; non-empty: {summary['nonempty']}; "
        f"END_CONTENT rate: {summary['end_content_rate']:.4f}.\n\n"
        f"Compute RTF/source mean/p95: {summary['compute_rtf_source_mean']:.4f} / "
        f"{summary['compute_rtf_source_p95']:.4f}; mean first text token wall time: "
        f"{summary['first_text_token_seconds_mean']:.4f} s; max peak memory: "
        f"{summary['peak_memory_mib_max']:.1f} MiB.\n\n"
        "This is a fixed direction-balanced full-utterance checkpoint-selection probe. "
        "It is not yet a streaming latency or generated-audio result.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
