"""Build a detailed four-way Stage7A free-running test comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LABELS = {
    "e0_stage6": "E0 Stage6",
    "e1_continued_sft": "E1 continued SFT",
    "e2_grpo_g4": "E2 GRPO G4",
    "e3_grpo_g8": "E3 GRPO G8",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_by_direction(metric: dict[str, Any], field: str) -> dict[str, float]:
    result = {}
    for key, values in metric.get("groups", {}).items():
        direction = key.rsplit(":", 1)[-1]
        if field in values:
            result[direction] = float(values[field])
    return result


def extract(run_dir: Path) -> dict[str, Any]:
    aggregate = read_json(run_dir / "aggregate_metrics.json")
    latency_path = run_dir / "latency_batch1" / "aggregate_metrics.json"
    latency = read_json(latency_path) if latency_path.is_file() else aggregate
    common = aggregate["common_metrics"]
    means = aggregate["streaming_metrics"]["overall"]["means"]
    latency_means = latency["streaming_metrics"]["overall"]["means"]
    return {
        "run_dir": str(run_dir.resolve()),
        "samples": aggregate["streaming_metrics"]["overall"]["samples"],
        "text_bleu": group_by_direction(common["text_bleu"], "score"),
        "speech_bleu": group_by_direction(common["speech_bleu"], "score"),
        "utmos": group_by_direction(common["utmos"], "mean"),
        "autopcp": group_by_direction(common["autopcp"], "mean"),
        "streaming": {name: means.get(name) for name in (
            "first_write_ms_proxy", "start_offset_nca_ms", "start_offset_ca_ms",
            "atd_ms_proxy", "al_glm_tokens_proxy", "laal_glm_tokens_proxy",
            "dal_glm_tokens_proxy", "ap_proxy", "premature_write_given_wait",
            "unnecessary_wait_given_write", "write_f1", "final_flush_success",
            "audio_chunks", "playback_gap_sum_nca_ms", "structural_recoveries",
            "rtf_source_audio", "rtf_generated_audio", "nonempty_text",
            "nonempty_semantic",
        )},
        "latency_batch1": {name: latency_means.get(name) for name in (
            "action_ttft_seconds_mean", "write_ttft_seconds_mean",
            "rtf_source_audio", "rtf_generated_audio", "first_write_ms_proxy",
            "start_offset_nca_ms", "atd_ms_proxy",
        )},
        "gpu": aggregate.get("gpu_monitor", {}),
        "offline_comparisons": aggregate.get("offline_comparisons", []),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def delta(value: float | None, reference: float | None) -> str:
    if value is None or reference is None:
        return "N/A"
    return f"{value - reference:+.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", default="full_test_v1")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    results = {
        label: extract(root / label / args.run_id)
        for label in LABELS
    }
    payload = {
        "schema_version": "simul_uniss_stage7a_full_test_comparison_v1",
        "results": results,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sample_count = int(next(iter(results.values()))["samples"])

    quality_rows = []
    for label, name in LABELS.items():
        value = results[label]
        quality_rows.append(
            f"| {name} | {fmt(value['text_bleu'].get('cmn->eng'))} | "
            f"{fmt(value['text_bleu'].get('eng->cmn'))} | "
            f"{fmt(value['speech_bleu'].get('cmn->eng'))} | "
            f"{fmt(value['speech_bleu'].get('eng->cmn'))} | "
            f"{fmt(value['utmos'].get('cmn->eng'))} | {fmt(value['utmos'].get('eng->cmn'))} | "
            f"{fmt(value['autopcp'].get('cmn->eng'))} | {fmt(value['autopcp'].get('eng->cmn'))} |"
        )
    streaming_rows = []
    for label, name in LABELS.items():
        values = results[label]["streaming"]
        streaming_rows.append(
            f"| {name} | {fmt(values['first_write_ms_proxy'], 1)} | "
            f"{fmt(values['start_offset_nca_ms'], 1)} | {fmt(values['atd_ms_proxy'], 1)} | "
            f"{fmt(values['laal_glm_tokens_proxy'], 2)} | {fmt(values['write_f1'])} | "
            f"{fmt(values['premature_write_given_wait'])} | "
            f"{fmt(values['unnecessary_wait_given_write'])} | "
            f"{fmt(values['final_flush_success'])} | {fmt(values['rtf_source_audio'])} |"
        )
    latency_rows = []
    for label, name in LABELS.items():
        values = results[label]["latency_batch1"]
        latency_rows.append(
            f"| {name} | {fmt(values['action_ttft_seconds_mean'])} | "
            f"{fmt(values['write_ttft_seconds_mean'])} | {fmt(values['rtf_source_audio'])} | "
            f"{fmt(values['first_write_ms_proxy'], 1)} | {fmt(values['atd_ms_proxy'], 1)} |"
        )
    gpu_rows = []
    for label, name in LABELS.items():
        values = results[label]["gpu"]
        gpu_rows.append(
            f"| {name} | {fmt(values.get('utilization_mean'), 1)}% | "
            f"{fmt(values.get('utilization_p95'), 1)}% | "
            f"{fmt(values.get('power_mean_w'), 1)} W | {fmt(values.get('power_p95_w'), 1)} W |"
        )

    e1 = results["e1_continued_sft"]
    conclusions = [
        "- E1 是必须超过的 matched-training control；只有 E2/E3 在质量不下降时显著降低延迟，才能支持 GRPO 独立贡献。",
        "- 以下 delta 均为相对 E1；负的延迟 delta 更好，正的 BLEU delta 更好。",
    ]
    for label in ("e2_grpo_g4", "e3_grpo_g8"):
        value = results[label]
        conclusions.append(
            f"- {LABELS[label]} vs E1: first-WRITE "
            f"{delta(value['streaming']['first_write_ms_proxy'], e1['streaming']['first_write_ms_proxy'])} ms; "
            f"ATD {delta(value['streaming']['atd_ms_proxy'], e1['streaming']['atd_ms_proxy'])} ms; "
            f"cmn->eng Text BLEU {delta(value['text_bleu'].get('cmn->eng'), e1['text_bleu'].get('cmn->eng'))}; "
            f"eng->cmn Text BLEU {delta(value['text_bleu'].get('eng->cmn'), e1['text_bleu'].get('eng->cmn'))}."
        )
    report_text = "\n".join([
        "# Simul-UniSS Stage7A four-way full test report", "",
        f"> Scope: {sample_count:,}-sample free-running streaming S2ST test; each experiment uses two fixed H200 GPUs.",
        "> E0/E1/E2/E3 use identical schedules, greedy generation, BiCodec decode, metrics, and batch-one latency audit.", "",
        "## 1. Quality and audio metrics", "",
        "| Experiment | Text BLEU zh→en | Text BLEU en→zh | Speech BLEU zh→en | Speech BLEU en→zh | UTMOS zh→en | UTMOS en→zh | AutoPCP zh→en | AutoPCP en→zh |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", *quality_rows, "",
        "## 2. Streaming policy and latency", "",
        "| Experiment | First WRITE ms | StartOffset NCA ms | ATD ms | LAAL token proxy | WRITE F1 | Premature WRITE | Unnecessary WAIT | Final flush | Source RTF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", *streaming_rows, "",
        "## 3. Batch-one deployable latency", "",
        "| Experiment | Action TTFT s | WRITE TTFT s | Source RTF | First WRITE ms | ATD ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |", *latency_rows, "",
        "## 4. GPU utilization and power", "",
        "| Experiment | Util mean | Util p95 | Power mean | Power p95 |",
        "| --- | ---: | ---: | ---: | ---: |", *gpu_rows, "",
        "GPU utilization is a throughput diagnostic, not a model-quality score. No dummy computation or invalid padding is used to inflate power.", "",
        "## 5. GRPO comparison and conclusion", "", *conclusions, "",
        "The final recommendation must jointly consider Text/Speech BLEU, UTMOS/AutoPCP, first-WRITE, ATD/LAAL, premature WRITE, unnecessary WAIT, final flush, and batch-one RTF.",
        "If GRPO does not beat E1 on the quality-latency Pareto frontier, do not expand this reward to full198; revise rollout conditioning and reward first.", "",
        "## 6. Reproducibility", "", f"- Machine-readable comparison: `{output}`",
        *[f"- {LABELS[label]}: `{results[label]['run_dir']}`" for label in LABELS], "",
    ])
    Path(args.report).write_text(report_text, encoding="utf-8")
    print(json.dumps({"output": str(output), "report": args.report}, sort_keys=True))


if __name__ == "__main__":
    main()
