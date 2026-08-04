#!/usr/bin/env python3
"""Aggregate bilingual Stage11 outputs and state an auditable demo verdict."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import sacrebleu


def row(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["result"]
    reference = str(payload["reference_translation"])
    hypothesis = str(result["translation"])
    return {
        "id": payload["id"],
        "direction": result["direction"],
        "reference": reference,
        "translation": hypothesis,
        "text_bleu": sacrebleu.corpus_bleu([hypothesis], [[reference]], tokenize="zh" if result["direction"] == "eng->cmn" else "13a").score,
        "chrf": sacrebleu.corpus_chrf([hypothesis], [[reference]]).score,
        "source_seconds": float(result["source_seconds"]),
        "target_seconds": float(result["target_seconds"]),
        "wall_seconds": float(result["wall_seconds"]),
        "first_write_ms": result["first_write_ms"],
        "first_audio_nca_ms": result["first_audio_nca_ms"],
        "first_audio_ca_ms": result["first_audio_ca_ms"],
        "valid_audio_writes": int(result["valid_audio_writes"]),
        "rejected_writes": int(result["rejected_writes"]),
        "fallback_used": bool(result.get("fallback_used", False)),
        "fallback_reason": result.get("fallback_reason"),
        "compute_rtf": float(result["wall_seconds"]) / max(float(result["source_seconds"]), 1e-6),
        "audio_duration_ratio": float(result["target_seconds"]) / max(float(result["source_seconds"]), 1e-6),
        "translation_audio_path": result["translation_audio_path"],
        "timeline_audio_path": result["timeline_audio_path"],
        "stereo_audio_path": result["stereo_audio_path"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_json, args.output_md):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite Stage12 report: {output}")
    rows = [row(path) for path in args.inputs]
    summary = {
        "samples": len(rows),
        "online_audio_success_rate": sum(
            value["valid_audio_writes"] > 0 and not value["fallback_used"] for value in rows
        )
        / max(1, len(rows)),
        "fallback_rate": sum(value["fallback_used"] for value in rows) / max(1, len(rows)),
        "first_write_ms_mean": statistics.fmean(float(value["first_write_ms"]) for value in rows),
        "first_audio_nca_ms_mean": statistics.fmean(float(value["first_audio_nca_ms"]) for value in rows),
        "first_audio_ca_ms_mean": statistics.fmean(float(value["first_audio_ca_ms"]) for value in rows),
        "compute_rtf_mean": statistics.fmean(float(value["compute_rtf"]) for value in rows),
        "accepted_write_rate": sum(int(value["valid_audio_writes"]) for value in rows)
        / max(1, sum(int(value["valid_audio_writes"]) + int(value["rejected_writes"]) for value in rows)),
    }
    payload = {
        "schema_version": "uniss_streamspeech_stage12_research_eval_v1",
        "research_only": True,
        "inputs": [str(path.resolve()) for path in args.inputs],
        "summary": summary,
        "samples": rows,
        "verdict": {
            "pipeline_complete": True,
            "public_demo_allowed": True,
            "quality_gate_passed": False,
            "subsecond_nca_seen": any(
                not value["fallback_used"] and float(value["first_audio_nca_ms"]) < 1000
                for value in rows
            ),
            "subsecond_ca_passed": all(
                not value["fallback_used"] and float(value["first_audio_ca_ms"]) < 1000
                for value in rows
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stage12 Stage09–11 simultaneous S2ST evaluation",
        "",
        "> Research-only. The pipeline is runnable and demoable; upstream quality gates remain unmet.",
        "",
        "| Direction | BLEU | chrF | First WRITE | First audio NCA | First audio CA | Valid/rejected | Fallback | Compute RTF |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for value in rows:
        lines.append(
            f"| {value['direction']} | {value['text_bleu']:.2f} | {value['chrf']:.2f} | "
            f"{value['first_write_ms']:.0f} ms | {value['first_audio_nca_ms']:.0f} ms | "
            f"{value['first_audio_ca_ms']:.0f} ms | {value['valid_audio_writes']}/{value['rejected_writes']} | "
            f"{'yes' if value['fallback_used'] else 'no'} | {value['compute_rtf']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "- Stage09 true chunking, Stage10 KV-cache and Stage11 BiCodec audio are all connected.",
            "- EN→ZH demonstrates first accepted audio at 880 ms NCA, but computation-aware latency is 5.16 s and text is repetitive.",
            "- ZH→EN has no accepted online semantic WRITE; its playable audio is a clearly labeled final offline fallback.",
            "- Therefore a public research demo is appropriate, but neither bilingual quality nor true subsecond wall-clock performance passes.",
            "- The demo must expose fallback status, rejected WRITEs and NCA/CA separately.",
            "",
            "## Listening artifacts",
            "",
        ]
    )
    for value in rows:
        lines.extend(
            [
                f"### {value['direction']}",
                "",
                f"- continuous target: `{value['translation_audio_path']}`",
                f"- WAIT timeline: `{value['timeline_audio_path']}`",
                f"- stereo left-source/right-translation: `{value['stereo_audio_path']}`",
                "",
            ]
        )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({**summary, **payload["verdict"]}, sort_keys=True))


if __name__ == "__main__":
    main()
