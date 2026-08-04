#!/usr/bin/env python3
"""Compare research-only Step2 checkpoints with the selected Step1-R model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def scores(path: Path) -> dict[str, float | int]:
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
    en_zh = float(summary["bleu"]["eng->cmn"]["score"])
    zh_en = float(summary["bleu"]["cmn->eng"]["score"])
    return {
        "iteration": int(summary["checkpoint_iteration"]),
        "en_zh_bleu": en_zh,
        "zh_en_bleu": zh_en,
        "mean_bleu": (en_zh + zh_en) / 2,
        "compute_rtf_source_mean": float(summary["compute_rtf_source_mean"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_json, args.output_md):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite comparison: {output}")
    reference = scores(args.reference_json)
    candidates = [scores(path) for path in args.candidate_json]
    for row in candidates:
        row["delta_en_zh"] = row["en_zh_bleu"] - reference["en_zh_bleu"]
        row["delta_zh_en"] = row["zh_en_bleu"] - reference["zh_en_bleu"]
        row["delta_mean"] = row["mean_bleu"] - reference["mean_bleu"]
    selected = max(candidates, key=lambda row: (row["mean_bleu"], -row["iteration"]))
    payload = {
        "schema_version": "uniss_streamspeech_stage08_step2_research_comparison_v1",
        "research_only": True,
        "reference": reference,
        "candidates": candidates,
        "selected": selected,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Stage08 Step2 research-only comparison",
        "",
        "> Step1-R did not pass the formal hard gate. These results validate the pipeline and hypothesis only.",
        "",
        "| Model | EN→ZH BLEU | ZH→EN BLEU | Mean | Δ Mean vs Step1-R | RTF |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Step1-R iter350 | {reference['en_zh_bleu']:.4f} | {reference['zh_en_bleu']:.4f} | {reference['mean_bleu']:.4f} | — | {reference['compute_rtf_source_mean']:.4f} |",
    ]
    for row in sorted(candidates, key=lambda value: value["iteration"]):
        lines.append(
            f"| Step2 iter{row['iteration']} | {row['en_zh_bleu']:.4f} | "
            f"{row['zh_en_bleu']:.4f} | {row['mean_bleu']:.4f} | "
            f"{row['delta_mean']:+.4f} | {row['compute_rtf_source_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Selected research checkpoint: iter{selected['iteration']} by bidirectional mean BLEU.",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(selected, sort_keys=True))


if __name__ == "__main__":
    main()
