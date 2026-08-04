#!/usr/bin/env python3
"""Select a Stage08 Step1 checkpoint using the fixed bidirectional BLEU gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def candidate_row(payload: dict) -> dict:
    summary = payload["summary"]
    en_zh = float(summary["bleu"]["eng->cmn"]["score"])
    zh_en = float(summary["bleu"]["cmn->eng"]["score"])
    return {
        "name": f"Step1 iter {summary['checkpoint_iteration']}",
        "iteration": int(summary["checkpoint_iteration"]),
        "en_zh_bleu": en_zh,
        "zh_en_bleu": zh_en,
        "average_bleu": (en_zh + zh_en) / 2,
        "compute_rtf_source_mean": float(summary["compute_rtf_source_mean"]),
        "compute_rtf_source_p95": float(summary["compute_rtf_source_p95"]),
        "first_text_token_seconds_mean": float(summary["first_text_token_seconds_mean"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, nargs="+", required=True)
    parser.add_argument("--en-zh-gate", type=float, default=22.95)
    parser.add_argument("--zh-en-gate", type=float, default=22.46)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_json, args.output_md):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint gate: {output}")
    baseline_payload = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    baseline_summary = baseline_payload["summary"]
    baseline = {
        "name": "Stage04 B2",
        "iteration": None,
        "en_zh_bleu": float(baseline_summary["bleu"]["eng->cmn"]["score"]),
        "zh_en_bleu": float(baseline_summary["bleu"]["cmn->eng"]["score"]),
    }
    baseline["average_bleu"] = (
        baseline["en_zh_bleu"] + baseline["zh_en_bleu"]
    ) / 2
    candidates = [
        candidate_row(json.loads(path.read_text(encoding="utf-8")))
        for path in args.candidate_json
    ]
    for row in candidates:
        row["delta_en_zh_vs_b2"] = row["en_zh_bleu"] - baseline["en_zh_bleu"]
        row["delta_zh_en_vs_b2"] = row["zh_en_bleu"] - baseline["zh_en_bleu"]
        row["passes_step2_gate"] = (
            row["en_zh_bleu"] > args.en_zh_gate
            and row["zh_en_bleu"] > args.zh_en_gate
        )
    selected = max(candidates, key=lambda row: (row["average_bleu"], -row["iteration"]))
    output = {
        "schema_version": "uniss_streamspeech_stage08_step1_checkpoint_gate_v1",
        "baseline": baseline,
        "candidates": candidates,
        "gate": {"en_zh_bleu": args.en_zh_gate, "zh_en_bleu": args.zh_en_gate},
        "selected": selected,
        "any_candidate_passes": any(row["passes_step2_gate"] for row in candidates),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Stage08 Step1 checkpoint-selection gate",
        "",
        "| Model | EN→ZH BLEU | ZH→EN BLEU | Δ EN→ZH vs B2 | Δ ZH→EN vs B2 | Step2 gate | Mean source RTF |",
        "|---|---:|---:|---:|---:|:---:|---:|",
        f"| Stage04 B2 | {baseline['en_zh_bleu']:.4f} | {baseline['zh_en_bleu']:.4f} | — | — | — | — |",
    ]
    for row in sorted(candidates, key=lambda value: value["iteration"]):
        lines.append(
            f"| Step1 iter {row['iteration']} | {row['en_zh_bleu']:.4f} | "
            f"{row['zh_en_bleu']:.4f} | {row['delta_en_zh_vs_b2']:+.4f} | "
            f"{row['delta_zh_en_vs_b2']:+.4f} | "
            f"{'PASS' if row['passes_step2_gate'] else 'FAIL'} | "
            f"{row['compute_rtf_source_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Step2 gate: EN→ZH > {args.en_zh_gate:.2f} and ZH→EN > {args.zh_en_gate:.2f}.",
            "",
            f"Selected checkpoint: iteration {selected['iteration']} by mean bidirectional BLEU.",
            "",
            "Qwen LoRA and offline Phase3 replay must not start unless this fixed probe passes both directions.",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output["selected"], sort_keys=True))


if __name__ == "__main__":
    main()
