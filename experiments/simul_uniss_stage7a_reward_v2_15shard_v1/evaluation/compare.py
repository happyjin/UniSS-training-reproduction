"""Build the four-way Reward-v2 full-dev comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LABELS = {
    "r0_e3_v1_bias": "R0 E3-v1 + WRITE bias",
    "r1_rebalanced_coverage": "R1 rebalanced + coverage",
    "r2_explicit_latency": "R2 explicit latency",
    "r3_bilingual_adaptive": "R3 bilingual + adaptive KL",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def group(metric: dict[str, Any], field: str) -> dict[str, float]:
    result = {}
    for key, values in metric.get("groups", {}).items():
        direction = key.rsplit(":", 1)[-1]
        if field in values:
            result[direction] = float(values[field])
    return result


def extract(path: Path) -> dict[str, Any]:
    payload = read_json(path / "aggregate_metrics.json")
    common = payload["common_metrics"]
    means = payload["streaming_metrics"]["overall"]["means"]
    return {
        "path": str(path.resolve()),
        "samples": payload["streaming_metrics"]["overall"]["samples"],
        "text_bleu": group(common["text_bleu"], "score"),
        "speech_bleu": group(common["speech_bleu"], "score"),
        "utmos": group(common["utmos"], "mean"),
        "autopcp": group(common["autopcp"], "mean"),
        "streaming": {
            key: means.get(key)
            for key in (
                "first_write_ms_proxy",
                "atd_ms_proxy",
                "laal_glm_tokens_proxy",
                "premature_write_given_wait",
                "unnecessary_wait_given_write",
                "forced_actions",
                "write_f1",
                "rtf_source_audio",
            )
        },
        "gpu": payload.get("gpu_monitor", {}),
    }


def fmt(value: Any, digits: int = 3) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    results = {label: extract(root / label) for label in LABELS}
    r1 = results["r1_rebalanced_coverage"]
    deltas = {}
    for label in ("r2_explicit_latency", "r3_bilingual_adaptive"):
        value = results[label]
        deltas[label] = {
            "first_write_ms": value["streaming"]["first_write_ms_proxy"]
            - r1["streaming"]["first_write_ms_proxy"],
            "atd_ms": value["streaming"]["atd_ms_proxy"]
            - r1["streaming"]["atd_ms_proxy"],
            "text_bleu_cmn_eng": value["text_bleu"].get("cmn->eng", 0.0)
            - r1["text_bleu"].get("cmn->eng", 0.0),
            "text_bleu_eng_cmn": value["text_bleu"].get("eng->cmn", 0.0)
            - r1["text_bleu"].get("eng->cmn", 0.0),
        }
    comparison = {
        "schema_version": "simul_uniss_stage7a_reward_v2_full_dev_comparison_v1",
        "results": results,
        "deltas_vs_r1": deltas,
    }
    Path(args.output_json).write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    quality_rows = []
    policy_rows = []
    gpu_rows = []
    for label, title in LABELS.items():
        value = results[label]
        quality_rows.append(
            f"| {title} | {fmt(value['text_bleu'].get('cmn->eng'))} | "
            f"{fmt(value['text_bleu'].get('eng->cmn'))} | "
            f"{fmt(value['speech_bleu'].get('cmn->eng'))} | "
            f"{fmt(value['speech_bleu'].get('eng->cmn'))} | "
            f"{fmt(value['utmos'].get('cmn->eng'))} | "
            f"{fmt(value['utmos'].get('eng->cmn'))} |"
        )
        stream = value["streaming"]
        policy_rows.append(
            f"| {title} | {fmt(stream['first_write_ms_proxy'], 1)} | "
            f"{fmt(stream['atd_ms_proxy'], 1)} | {fmt(stream['laal_glm_tokens_proxy'], 2)} | "
            f"{fmt(stream['premature_write_given_wait'])} | "
            f"{fmt(stream['unnecessary_wait_given_write'])} | "
            f"{fmt(stream['forced_actions'])} | {fmt(stream['rtf_source_audio'])} |"
        )
        gpu = value["gpu"]
        gpu_rows.append(
            f"| {title} | {fmt(gpu.get('utilization_mean'), 1)}% | "
            f"{fmt(gpu.get('utilization_p95'), 1)}% | "
            f"{fmt(gpu.get('power_mean_w'), 1)} W | {fmt(gpu.get('power_p95_w'), 1)} W |"
        )
    conclusions = []
    for label in ("r2_explicit_latency", "r3_bilingual_adaptive"):
        item = deltas[label]
        conclusions.append(
            f"- {LABELS[label]} vs R1: First WRITE {item['first_write_ms']:+.1f} ms, "
            f"ATD {item['atd_ms']:+.1f} ms, zh→en Text BLEU "
            f"{item['text_bleu_cmn_eng']:+.3f}, en→zh Text BLEU "
            f"{item['text_bleu_eng_cmn']:+.3f}."
        )
    report = "\n".join(
        [
            "# Stage7A Reward-v2 four-way full-dev report",
            "",
            "## Quality",
            "",
            "| Experiment | Text BLEU zh→en | Text BLEU en→zh | Speech BLEU zh→en | Speech BLEU en→zh | UTMOS zh→en | UTMOS en→zh |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *quality_rows,
            "",
            "## Streaming policy and latency",
            "",
            "| Experiment | First WRITE ms | ATD ms | LAAL proxy | Premature | Unnecessary WAIT | Forced actions/sample | Source RTF |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *policy_rows,
            "",
            "## GPU profile",
            "",
            "| Experiment | Util mean | Util p95 | Power mean | Power p95 |",
            "|---|---:|---:|---:|---:|",
            *gpu_rows,
            "",
            "## Direct comparison",
            "",
            *conclusions,
            "",
            "R2/R3 must beat R1 on latency while retaining both language directions. GPU utilization is diagnostic only.",
            "The test split remains locked; this report uses dev for reward and operating-point selection.",
            "",
        ]
    )
    Path(args.report).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
