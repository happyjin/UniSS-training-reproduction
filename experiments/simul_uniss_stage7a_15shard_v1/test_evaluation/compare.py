"""Build a detailed four-way Stage7A free-running test comparison report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

LABELS = {
    "e0_stage6": "E0 Stage6",
    "e1_continued_sft": "E1 continued SFT",
    "e2_grpo_g4": "E2 GRPO G4",
    "e3_grpo_g8": "E3 GRPO G8",
}

DIRECTIONS = ("cmn->eng", "eng->cmn")
EXPERIMENT_DESCRIPTIONS = {
    "e0_stage6": "Stage6 frozen action-policy baseline",
    "e1_continued_sft": "Matched continued action SFT control",
    "e2_grpo_g4": "Action-only GRPO, group size 4",
    "e3_grpo_g8": "Action-only GRPO, group size 8",
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
    manifest_path = run_dir / "environment" / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    git_commit_path = run_dir / "environment" / "git_commit.txt"
    git_commit = (
        git_commit_path.read_text(encoding="utf-8").strip()
        if git_commit_path.is_file()
        else None
    )
    return {
        "run_dir": str(run_dir.resolve()),
        "manifest": manifest,
        "git_commit": git_commit,
        "samples": aggregate["streaming_metrics"]["overall"]["samples"],
        "text_bleu": group_by_direction(common["text_bleu"], "score"),
        "speech_bleu": group_by_direction(common["speech_bleu"], "score"),
        "utmos": group_by_direction(common["utmos"], "mean"),
        "autopcp": group_by_direction(common["autopcp"], "mean"),
        "streaming": {
            name: means.get(name)
            for name in (
                "first_write_ms_proxy",
                "start_offset_nca_ms",
                "start_offset_ca_ms",
                "atd_ms_proxy",
                "al_glm_tokens_proxy",
                "laal_glm_tokens_proxy",
                "dal_glm_tokens_proxy",
                "ap_proxy",
                "premature_write_given_wait",
                "unnecessary_wait_given_write",
                "write_f1",
                "final_flush_success",
                "audio_chunks",
                "playback_gap_sum_nca_ms",
                "structural_recoveries",
                "rtf_source_audio",
                "rtf_generated_audio",
                "nonempty_text",
                "nonempty_semantic",
            )
        },
        "latency_batch1": {
            name: latency_means.get(name)
            for name in (
                "action_ttft_seconds_mean",
                "write_ttft_seconds_mean",
                "rtf_source_audio",
                "rtf_generated_audio",
                "first_write_ms_proxy",
                "start_offset_nca_ms",
                "atd_ms_proxy",
            )
        },
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


def mean_metric(values: dict[str, float]) -> float | None:
    available = [float(values[item]) for item in DIRECTIONS if item in values]
    return statistics.fmean(available) if available else None


def reduction(value: float | None, reference: float | None) -> float | None:
    if value is None or reference in (None, 0):
        return None
    return (float(reference) - float(value)) / abs(float(reference))


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:+.{digits}f}%"


def status(value: bool | None) -> str:
    if value is None:
        return "N/A"
    return "PASS" if value else "FAIL"


def offline_rows(label: str, result: dict[str, Any]) -> list[str]:
    rows = []
    for item in result["offline_comparisons"]:
        if item.get("offline_mode") != "quality":
            continue
        if item.get("metric") not in {"text_bleu", "speech_bleu", "utmos", "autopcp"}:
            continue
        rows.append(
            f"| {LABELS[label]} | {item['direction']} | {item['metric']} | "
            f"{fmt(item.get('streaming_value'))} | {fmt(item.get('offline_value'))} | "
            f"{fmt(item.get('delta_streaming_minus_offline'))} |"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", default="full_test_v1")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    results = {label: extract(root / label / args.run_id) for label in LABELS}
    payload = {
        "schema_version": "simul_uniss_stage7a_full_test_comparison_v1",
        "results": results,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
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

    design_rows = []
    for label, name in LABELS.items():
        manifest = results[label]["manifest"]
        commit = results[label]["git_commit"]
        design_rows.append(
            f"| {name} | {EXPERIMENT_DESCRIPTIONS[label]} | {manifest.get('best_step', 'N/A')} | "
            f"{manifest.get('gpus', 'N/A')} | `{commit[:12] if commit else 'N/A'}` | "
            f"`{manifest.get('model', 'N/A')}` |"
        )

    e1 = results["e1_continued_sft"]
    e0 = results["e0_stage6"]
    direct_delta_rows = []
    direct_metrics = (
        ("Text BLEU zh→en", "text_bleu", "cmn->eng", "higher"),
        ("Text BLEU en→zh", "text_bleu", "eng->cmn", "higher"),
        ("Speech BLEU zh→en", "speech_bleu", "cmn->eng", "higher"),
        ("Speech BLEU en→zh", "speech_bleu", "eng->cmn", "higher"),
        ("UTMOS mean", "utmos", None, "higher"),
        ("AutoPCP mean", "autopcp", None, "higher"),
        ("First WRITE ms", "streaming", "first_write_ms_proxy", "lower"),
        ("ATD ms", "streaming", "atd_ms_proxy", "lower"),
        ("LAAL token proxy", "streaming", "laal_glm_tokens_proxy", "lower"),
        ("Premature WRITE", "streaming", "premature_write_given_wait", "lower"),
        ("Unnecessary WAIT", "streaming", "unnecessary_wait_given_write", "lower"),
        ("Final flush", "streaming", "final_flush_success", "higher"),
        ("Batch-one source RTF", "latency_batch1", "rtf_source_audio", "lower"),
    )
    for title, family, key, preferred in direct_metrics:
        reference_family = e1[family]
        reference = (
            mean_metric(reference_family) if key is None else reference_family.get(key)
        )
        row = [title, "↑" if preferred == "higher" else "↓"]
        for label in ("e2_grpo_g4", "e3_grpo_g8"):
            candidate_family = results[label][family]
            candidate = (
                mean_metric(candidate_family)
                if key is None
                else candidate_family.get(key)
            )
            row.append(delta(candidate, reference))
        direct_delta_rows.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    gate_rows = []
    gate_results: dict[str, dict[str, bool]] = {}
    for label in ("e1_continued_sft", "e2_grpo_g4", "e3_grpo_g8"):
        value = results[label]
        first_reduction = reduction(
            value["streaming"]["first_write_ms_proxy"],
            e0["streaming"]["first_write_ms_proxy"],
        )
        first_absolute = (
            e0["streaming"]["first_write_ms_proxy"]
            - value["streaming"]["first_write_ms_proxy"]
        )
        start_reduction = reduction(
            value["streaming"]["start_offset_nca_ms"],
            e0["streaming"]["start_offset_nca_ms"],
        )
        atd_reduction = reduction(
            value["streaming"]["atd_ms_proxy"], e0["streaming"]["atd_ms_proxy"]
        )
        laal_reduction = reduction(
            value["streaming"]["laal_glm_tokens_proxy"],
            e0["streaming"]["laal_glm_tokens_proxy"],
        )
        text_deltas = [
            value["text_bleu"].get(direction, float("-inf"))
            - e0["text_bleu"].get(direction, float("inf"))
            for direction in DIRECTIONS
        ]
        premature_delta = (
            value["streaming"]["premature_write_given_wait"]
            - e0["streaming"]["premature_write_given_wait"]
        )
        gates = {
            "first_write": first_reduction is not None
            and (first_reduction >= 0.15 or first_absolute >= 500),
            "start_offset": start_reduction is not None and start_reduction >= 0.10,
            "atd": atd_reduction is not None and atd_reduction >= 0.10,
            "laal": laal_reduction is not None and laal_reduction >= 0.10,
            "text_bleu": min(text_deltas) >= -0.5,
            "premature": premature_delta <= 0.01,
            "unnecessary_wait": value["streaming"]["unnecessary_wait_given_write"]
            <= 0.12,
            "rtf": value["latency_batch1"]["rtf_source_audio"] < 1.0,
            "final_flush": value["streaming"]["final_flush_success"] >= 0.999,
        }
        gate_results[label] = gates
        gate_rows.append(
            f"| {LABELS[label]} | {status(gates['first_write'])} ({fmt_pct(first_reduction)}) | "
            f"{status(gates['start_offset'])} ({fmt_pct(start_reduction)}) | "
            f"{status(gates['atd'])} ({fmt_pct(atd_reduction)}) | "
            f"{status(gates['laal'])} ({fmt_pct(laal_reduction)}) | "
            f"{status(gates['text_bleu'])} (worst {min(text_deltas):+.3f}) | "
            f"{status(gates['premature'])} ({premature_delta:+.3f}) | "
            f"{status(gates['unnecessary_wait'])} | {status(gates['rtf'])} | "
            f"{status(gates['final_flush'])} |"
        )

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
    supported = []
    for label in ("e2_grpo_g4", "e3_grpo_g8"):
        value = results[label]
        quality_retained = all(
            value["text_bleu"].get(direction, float("-inf"))
            >= e1["text_bleu"].get(direction, float("inf")) - 0.5
            for direction in DIRECTIONS
        )
        policy_safe = (
            value["streaming"]["premature_write_given_wait"]
            <= e1["streaming"]["premature_write_given_wait"] + 0.01
            and value["streaming"]["final_flush_success"] >= 0.999
        )
        latency_better = (
            value["streaming"]["first_write_ms_proxy"]
            < e1["streaming"]["first_write_ms_proxy"]
            and value["streaming"]["atd_ms_proxy"] < e1["streaming"]["atd_ms_proxy"]
        )
        if quality_retained and policy_safe and latency_better:
            supported.append(label)

    if not supported:
        decision = (
            "本次单种子 full-test 没有 GRPO 实验同时满足：相对 E1 保持双向 Text BLEU、"
            "不恶化 premature/final-flush，并同时降低 first-WRITE 与 ATD。当前结果不支持扩大到 full198。"
        )
    elif len(supported) == 1:
        decision = (
            f"{LABELS[supported[0]]} 是本次唯一通过 matched-E1 质量保持与双延迟改善检查的 GRPO 候选；"
            "它应进入重复种子和 paired bootstrap，而不是直接宣称已完成 full198 验证。"
        )
    else:
        a, b = supported
        a_quality = mean_metric(results[a]["text_bleu"])
        b_quality = mean_metric(results[b]["text_bleu"])
        a_streaming = results[a]["streaming"]
        b_streaming = results[b]["streaming"]
        a_dominates = (
            a_quality >= b_quality
            and a_streaming["first_write_ms_proxy"]
            <= b_streaming["first_write_ms_proxy"]
            and a_streaming["atd_ms_proxy"] <= b_streaming["atd_ms_proxy"]
        )
        b_dominates = (
            b_quality >= a_quality
            and b_streaming["first_write_ms_proxy"]
            <= a_streaming["first_write_ms_proxy"]
            and b_streaming["atd_ms_proxy"] <= a_streaming["atd_ms_proxy"]
        )
        if a_dominates and not b_dominates:
            decision = f"{LABELS[a]} 在平均 Text BLEU、first-WRITE 与 ATD 三轴上支配 {LABELS[b]}，是优先复现实验。"
        elif b_dominates and not a_dominates:
            decision = f"{LABELS[b]} 在平均 Text BLEU、first-WRITE 与 ATD 三轴上支配 {LABELS[a]}，是优先复现实验。"
        else:
            decision = (
                f"{LABELS[a]} 与 {LABELS[b]} 都超过 matched E1，但互不支配；"
                "应把二者保留为 quality-latency Pareto 候选，而不能用任意加权总分强行选一个。"
            )
    conclusions.extend(
        [
            f"- 自动判定：{decision}",
            (
                "- 正式进入 full198 仍需补齐 fixed wait-k frontier、10,000 次 paired bootstrap、"
                "至少 3 个随机种子；本次只有一个训练种子，因此只能给出候选排序，不能给出统计显著性结论。"
            ),
        ]
    )

    all_offline_rows = [
        row for label in LABELS for row in offline_rows(label, results[label])
    ]
    payload["analysis"] = {
        "grpo_candidates_passing_matched_e1": supported,
        "stage6_gate_results": gate_results,
        "decision": decision,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report_text = "\n".join(
        [
            "# Simul-UniSS Stage7A four-way full test report",
            "",
            f"> Scope: {sample_count:,}-sample free-running streaming S2ST test; each experiment uses two fixed H200 GPUs.",
            "> E0/E1/E2/E3 use identical schedules, greedy generation, BiCodec decode, metrics, and batch-one latency audit.",
            "",
            "## 1. 实验设计与可归因性",
            "",
            "| Experiment | Role | Best step | GPUs | Eval commit | Exported model |",
            "| --- | --- | ---: | --- | --- | --- |",
            *design_rows,
            "",
            "E0 是原 Stage6 基线；E1 控制额外训练步数和 action-head 继续训练；E2/E3 与 E1 的差异才是 GRPO 与 group size。",
            f"四组使用完全相同的 {sample_count:,} 条 test schedules、greedy decode、BiCodec 和指标实现，因此表内差值是 matched comparison。",
            "",
            "## 2. Quality and audio metrics",
            "",
            "| Experiment | Text BLEU zh→en | Text BLEU en→zh | Speech BLEU zh→en | Speech BLEU en→zh | UTMOS zh→en | UTMOS en→zh | AutoPCP zh→en | AutoPCP en→zh |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *quality_rows,
            "",
            "Text BLEU 衡量文本翻译；Speech BLEU 在实际生成音频经 ASR 后衡量端到端内容；UTMOS/AutoPCP 分别反映感知音质和与参考音频的表示相似度。",
            "",
            "## 3. Streaming policy and latency",
            "",
            "| Experiment | First WRITE ms | StartOffset NCA ms | ATD ms | LAAL token proxy | WRITE F1 | Premature WRITE | Unnecessary WAIT | Final flush | Source RTF |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *streaming_rows,
            "",
            "带 `_proxy` 的指标来自当前 pseudo-alignment/capacity gate，不能表述为真实 CTC 对齐指标。First WRITE、ATD、LAAL、unnecessary WAIT 越低越好；Final flush 应为 1。",
            "",
            "## 4. Batch-one deployable latency",
            "",
            "| Experiment | Action TTFT s | WRITE TTFT s | Source RTF | First WRITE ms | ATD ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *latency_rows,
            "",
            "batch-one 审计固定为每组 200 条，用于估计真实部署延迟；大 batch 的全量评估吞吐不能替代这里的 RTF/TTFT。",
            "",
            "## 5. GPU utilization and power",
            "",
            "| Experiment | Util mean | Util p95 | Power mean | Power p95 |",
            "| --- | ---: | ---: | ---: | ---: |",
            *gpu_rows,
            "",
            "GPU utilization 是吞吐诊断而非模型质量分数。mean 覆盖模型加载、CPU 聚合和阶段切换，p95 更接近 GPU-heavy 稳态。没有使用 dummy computation 或无效 padding 抬高功率。",
            "",
            "## 6. GRPO 相对 matched E1 的直接差值",
            "",
            "| Metric | Better | E2−E1 | E3−E1 |",
            "| --- | :---: | ---: | ---: |",
            *direct_delta_rows,
            "",
            "这里必须优先比较 E2/E3 与 E1，而不是只比较 E0；如果 E1 与 GRPO 同样改善，收益可能只是继续训练，而不是 GRPO。",
            "",
            "## 7. 相对 Stage6 的首轮门槛审计",
            "",
            "| Experiment | First WRITE | StartOffset | ATD | LAAL | Text BLEU retention | Premature Δ | Unnecessary WAIT | RTF<1 | Final flush |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            *gate_rows,
            "",
            "门槛来自 Stage7A plan：first-WRITE 至少 -15% 或 -500 ms，StartOffset/ATD/LAAL 至少 -10%，双向 Text BLEU 最差下降不超过 0.5，premature 增加不超过 0.01，unnecessary WAIT ≤0.12，batch-one RTF<1，final flush≈100%。",
            "",
            "## 8. 与 offline Phase3 quality 模式的同指标差值",
            "",
            "| Streaming experiment | Direction | Metric | Streaming | Offline Phase3 | Streaming−offline |",
            "| --- | --- | --- | ---: | ---: | ---: |",
            *all_offline_rows,
            "",
            "该表说明 streaming 相对 offline upper bound 的质量代价；它不能用于 E2/E3 的训练因果判断，但可以判断 simultaneous 延迟收益是否以不可接受的离线质量损失换取。",
            "",
            "## 9. 结论与下一步",
            "",
            *conclusions,
            "",
            "最终判断必须联合 Text/Speech BLEU、UTMOS/AutoPCP、first-WRITE、ATD/LAAL、premature WRITE、unnecessary WAIT、final flush 和 batch-one RTF。",
            "如果 GRPO 没有超过 E1 的 quality-latency Pareto 点，不应扩大该 reward 到 full198；应先修改 rollout conditioning、真实 alignment 和 reward。",
            "",
            "## 10. 结论边界",
            "",
            "- 这是 full test 集上的单种子、单 operating-point 比较；没有执行 3-seed mean±std。",
            "- 本轮没有生成 fixed wait-k=1/2/3/5 的完整 test frontier，因此不能声称已超过 fixed wait-k Pareto frontier。",
            "- 本轮保存逐样本输出，但没有在此自动报告中执行 10,000 次 paired bootstrap；数值差异不能直接等同统计显著。",
            "- 当前是 frozen Stage6 backbone 上的 action-only GRPO；它没有训练 text/semantic/BiCodec 权重，不能称为 full-Qwen 或 semantic-token GRPO。",
            "- 本轮报告 Text BLEU、Speech BLEU、UTMOS、AutoPCP 和 streaming 指标；chrF/COMET/ASR-COMET 不在当前可复现链路中，不用缺失指标替代或伪造结论。",
            "",
            "## 11. Reproducibility",
            "",
            f"- Machine-readable comparison: `{output}`",
            *[f"- {LABELS[label]}: `{results[label]['run_dir']}`" for label in LABELS],
            "",
        ]
    )
    Path(args.report).write_text(report_text, encoding="utf-8")
    print(json.dumps({"output": str(output), "report": args.report}, sort_keys=True))


if __name__ == "__main__":
    main()
