#!/usr/bin/env python3
"""Write the final Chinese Runtime-v1/v2, A3 and long-episode RL report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def old_summary(payload: dict[str, Any]) -> dict[str, float]:
    rows = list(payload["results"])
    return {
        "first_write": mean([float(row["first_audio_global_ms"]) for row in rows]),
        "coverage": mean(
            [
                float(row["translation_duration_seconds"])
                / max(1e-9, float(row["source_duration_seconds"]))
                for row in rows
            ]
        ),
        "silence": mean(
            [float(row["timeline_silence"]["maximum_internal_silence_ms"]) for row in rows]
        ),
        "rtf": mean([float(row["rtf"]) for row in rows]),
        "writes": float("nan"),
        "pending": float("nan"),
        "early_end": float("nan"),
        "tts_failures": float("nan"),
    }


def stateful_summary(payload: dict[str, Any]) -> dict[str, float]:
    rows = list(payload["results"])
    return {
        "first_write": mean([float(row["first_audio_source_ms"]) for row in rows]),
        "coverage": mean(
            [float(row["translation_audio_to_source_duration_ratio"]) for row in rows]
        ),
        "silence": mean(
            [float(row["maximum_internal_timeline_silence_ms"]) for row in rows]
        ),
        "rtf": mean([float(row["rtf"]) for row in rows]),
        "writes": sum(float(row["audio_writes"]) for row in rows),
        "pending": sum(float(row["tts_pending_unspoken_items"]) for row in rows),
        "early_end": sum(float(row["rejected_early_end"]) for row in rows),
        "tts_failures": sum(float(row["tts_failures"]) for row in rows),
    }


def result_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in payload["results"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-v1", type=Path, required=True)
    parser.add_argument("--runtime-v2", type=Path, required=True)
    parser.add_argument("--a3-v2", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--epoch-result", type=Path, action="append", required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--train-rollout", type=Path, required=True)
    parser.add_argument("--valid-rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    runtime_v1 = load(args.runtime_v1)
    runtime_v2 = load(args.runtime_v2)
    a3_v2 = load(args.a3_v2)
    selection = load(args.selection)
    attribution = load(args.attribution)
    train_rollout = load(args.train_rollout)
    valid_rollout = load(args.valid_rollout)
    epochs = [load(path) for path in args.epoch_result]
    epoch_by_checkpoint = {
        str(payload["adapter_manifest"].get("checkpoint")): payload for payload in epochs
    }
    selected_checkpoint = str(selection["selected_checkpoint"])
    if selected_checkpoint not in epoch_by_checkpoint:
        raise ValueError("selected checkpoint has no long-audio result")
    selected = epoch_by_checkpoint[selected_checkpoint]

    arms = [
        ("C0 Phase A + runtime v1", runtime_v1, old_summary(runtime_v1)),
        ("C1 Phase A + runtime v2", runtime_v2, stateful_summary(runtime_v2)),
        ("C2 旧 A3 + runtime v2", a3_v2, stateful_summary(a3_v2)),
    ]
    for payload in epochs:
        checkpoint = str(payload["adapter_manifest"]["checkpoint"])
        iteration = int(Path(checkpoint).name.split("_")[-1])
        name = f"RL epoch checkpoint iter {iteration} + runtime v2"
        if checkpoint == selected_checkpoint:
            name += "（C3 选中）"
        arms.append((name, payload, stateful_summary(payload)))

    attr_rows = list(attribution["results"])
    attr = {
        "mean_offline_asr_similarity": mean(
            [float(row["offline_asr_similarity"]) for row in attr_rows]
        ),
        "mean_gold_source_mt_chrf": mean(
            [float(row["gold_source_mt_chrf"]) for row in attr_rows]
        ),
        "healthy_gold_target_tts_phrase_fraction": mean(
            [
                float(row["gold_target_tts_phrase_success_fraction"])
                for row in attr_rows
            ]
        ),
    }
    lines = [
        "# Phase A 长时 Stateful Runtime 与 Free-running Episode RL 最终报告",
        "",
        "## 1. 直接结论",
        "",
        "本报告使用同一组四条长音频和 640 ms 决策间隔，对 Runtime 修复、旧 A3 与新长 episode RL 进行拆分比较。质量门只用于记录和 checkpoint 选优，未中断 rollout、训练、逐 epoch 试听或最终对照。C0 是历史 bounded-window pseudo-streaming；C1/C2/C3 才使用完整会话 stateful runtime v2，因此 C0→C1 代表运行时修复收益，C1→C3 才是新 RL 的净训练收益。",
        "",
        "## 2. Rollout 与 A/B/C/D 归因",
        "",
        f"- train rollout：{train_rollout['episodes']} episodes / {train_rollout['candidates']} candidates，平均 reward={fmt(train_rollout['aggregate']['mean_reward'],4)}，平均首次 WRITE={fmt(train_rollout['aggregate']['mean_first_write_ms'],0)} ms。",
        f"- valid rollout：{valid_rollout['episodes']} episodes / {valid_rollout['candidates']} candidates，平均 reward={fmt(valid_rollout['aggregate']['mean_reward'],4)}，平均首次 WRITE={fmt(valid_rollout['aggregate']['mean_first_write_ms'],0)} ms。",
        f"- A（full-context ASR）平均相似度={fmt(attr.get('mean_offline_asr_similarity'),4)}。",
        f"- B（gold-source MT）平均 chrF={fmt(attr.get('mean_gold_source_mt_chrf'),2)}。",
        f"- C（gold-target TTS）健康短语比例={fmt(attr.get('healthy_gold_target_tts_phrase_fraction'),4)}。",
        "- D 是真实 free-running runtime v2；它同时承受 ASR 前缀错误、增量 MT 决策、WRITE 时机和 TTS 队列误差。A/B/C 与 D 的差距用于定位瓶颈，不能把所有错误归因给 RL。",
        "",
        "## 3. C0–C3 与逐 epoch 总表",
        "",
        "| 系统 | 平均首次发声 ms | 平均译音/源音覆盖 | 平均最大内部静音 ms | 总 WRITE | pending | early-END 拒绝 | TTS失败 | 平均RTF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _payload, summary in arms:
        lines.append(
            f"| {name} | {fmt(summary['first_write'],0)} | {fmt(summary['coverage'],3)} | "
            f"{fmt(summary['silence'],0)} | {fmt(summary['writes'],0)} | "
            f"{fmt(summary['pending'],0)} | {fmt(summary['early_end'],0)} | "
            f"{fmt(summary['tts_failures'],0)} | {fmt(summary['rtf'],3)} |"
        )

    base = stateful_summary(runtime_v2)
    best = stateful_summary(selected)
    lines.extend(
        [
            "",
            "## 4. Runtime 修复收益与 RL 净收益",
            "",
            "Runtime v2 修复跨窗口状态重置、人工 final、pending TTS 文本丢失和 320 semantic-token 截断；它不直接修复错误 ASR、错误翻译或过晚 WRITE。",
            "",
            f"C3 相对 C1：平均首次发声变化 {fmt(best['first_write']-base['first_write'],0)} ms；覆盖率变化 {fmt(best['coverage']-base['coverage'],3)}；平均最大内部静音变化 {fmt(best['silence']-base['silence'],0)} ms；总 WRITE 变化 {fmt(best['writes']-base['writes'],0)}；pending 变化 {fmt(best['pending']-base['pending'],0)}；RTF 变化 {fmt(best['rtf']-base['rtf'],3)}。负的时延、静音和 RTF 差值代表改善，正的覆盖和 WRITE 差值通常代表更完整，但仍需结合译文内容试听。",
            "",
            "## 5. 分音频试听路径与内容诊断",
            "",
        ]
    )
    c0 = result_by_id(runtime_v1)
    c1 = result_by_id(runtime_v2)
    c2 = result_by_id(a3_v2)
    c3 = result_by_id(selected)
    for sample_id in sorted(c3):
        bounded, before, old_rl, after = (
            c0[sample_id],
            c1[sample_id],
            c2[sample_id],
            c3[sample_id],
        )
        lines.extend(
            [
                f"### {sample_id}",
                "",
                f"- C0 Phase A runtime v1 立体声：`{bounded['stereo_path']}`",
                f"- C0 Phase A runtime v1 连续译音：`{bounded['translation_path']}`",
                f"- C0 Phase A runtime v1 全局时间轴：`{bounded['timeline_path']}`",
                f"- C1 Phase A 立体声：`{before['stereo_audio_path']}`",
                f"- C2 旧 A3 立体声：`{old_rl['stereo_audio_path']}`",
                f"- C3 新 RL 立体声：`{after['stereo_audio_path']}`",
                f"- C3 连续译音：`{after['continuous_audio_path']}`",
                f"- C3 全局时间轴：`{after['timeline_audio_path']}`",
                f"- 首次发声 C1→C3：{fmt(before['first_audio_source_ms'],0)}→{fmt(after['first_audio_source_ms'],0)} ms；最大内部静音 {fmt(before['maximum_internal_timeline_silence_ms'],0)}→{fmt(after['maximum_internal_timeline_silence_ms'],0)} ms。",
                f"- WRITE C1→C3：{before['audio_writes']}→{after['audio_writes']}；pending {before['tts_pending_unspoken_items']}→{after['tts_pending_unspoken_items']}；TTS failure {before['tts_failures']}→{after['tts_failures']}。",
                f"- C1 增量译文：{before['generated_streaming_translation']}",
                f"- C3 增量译文：{after['generated_streaming_translation']}",
                "",
            ]
        )

    lines.extend(
        [
            "## 6. Checkpoint 选择与声明边界",
            "",
            f"选中 checkpoint：`{selected_checkpoint}`。",
            f"选择规则：{selection['selection_rule']}。",
            f"记录型质量注释：{selection['selected_quality_annotations'] or ['无']}。即使有注释，所有 epoch 和 C0–C3 仍全部执行。",
            "",
            "### 6.1 三个 epoch 的 validation 轨迹",
            "",
            "| iteration | total | policy | reference KL | Phase3 replay | ratio mean | clipped fraction | update RMS | 记录型注释 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for candidate in selection["candidates"]:
        metrics = candidate["metrics"]
        lines.append(
            f"| {candidate['iteration']} | {fmt(metrics.get('loss/total'),6)} | "
            f"{fmt(metrics.get('loss/policy'),6)} | "
            f"{fmt(metrics.get('loss/reference_kl'),6)} | "
            f"{fmt(metrics.get('loss/phase3_replay'),6)} | "
            f"{fmt(metrics.get('diagnostic/ratio_mean'),6)} | "
            f"{fmt(metrics.get('diagnostic/ratio_clipped_fraction'),6)} | "
            f"{fmt(metrics.get('diagnostic/policy_update_rms'),8)} | "
            f"{candidate['quality_annotations'] or ['无']} |"
        )
    lines.extend(
        [
            "",
            "### 6.2 训练、TensorBoard 与逐 epoch 报告",
            "",
            f"- 训练日志：`{selection['training_log']}`",
            f"- checkpoint 根目录：`{selection['checkpoint_root']}`",
            "- TensorBoard：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/runs/uniss_phasea_stateful_longepisode_rl_v1/tensorboard/episode_grpo_formal_8gpu_v1`",
            f"- 逐 epoch/C2 试听报告目录：`{(args.output.parent / 'stages').resolve()}`",
            "- Runtime v2 基线报告：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/reports/uniss_phasea_stateful_longepisode_rl_v1/stage1_runtime_v2/REPORT.zh-CN.md`",
            "- A/B/C/D 归因报告：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/reports/uniss_phasea_stateful_longepisode_rl_v1/attribution/reference_attribution_valid16_v1/REPORT.zh-CN.md`",
            "",
            "### 6.3 声明边界",
            "",
            "这里的 stateful runtime 保留完整会话前端、提交文本、TTS 队列和播放时钟，但 LLM acoustic prompt 仍采用 24 秒有界 ring 重算；因此它是严格因果、跨窗口有状态的研究推理，不等同于已经实现端到端 KV-cache 的生产级实时系统。四条外部长音频没有人工逐句 reference，内容判断结合生成文本与主观试听；正式 A/B/C 指标来自有 teacher/reference 的 validation episodes。",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
