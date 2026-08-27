#!/usr/bin/env python3
"""Write a train-seen Chinese comparison report with listening paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_arm(value: str) -> tuple[str, dict[str, object]]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise ValueError("arm must be LABEL=PATH")
    return label, json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(value, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    arms = [load_arm(value) for value in args.arm]
    if len(arms) < 2:
        raise ValueError("expected Phase A and at least one trained arm")
    lines = [
        "# Phase A 路由一致约束式 GRPO：训练内长音频评估",
        "",
        "## 1. 声明边界",
        "",
        "本报告只评估训练流程明确使用的 8 条 train-seen 长 episode，目标是判断小数据条件下方法本身能否改善，不用于宣称 validation 或外部泛化。所有 arm 固定使用 640 ms decision chunk、160 ms 物理声学 block、24 s acoustic ring、同一 speaker token 和 Runtime v2。",
        "",
        "本实验与旧 RL 的关键区别是：adapter 在 ASR、MT、semantic TTS 和 control 路由全部启用；reward 只有在 ASR、MT、完整性和音频健康保持时才奖励低 first-WRITE 和较短静音。",
        "",
        "## 2. 内容质量",
        "",
        "| 系统 | 中→英 CER↓ | 英→中 WER↓ | 中→英 BLEU/chrF↑ | 英→中 BLEU/chrF↑ | LCS覆盖↑ | 4-gram重复↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, payload in arms:
        aggregate = payload["aggregate"]
        cmn = aggregate["by_direction"]["cmn->eng"]
        eng = aggregate["by_direction"]["eng->cmn"]
        overall = aggregate["overall"]
        lines.append(
            f"| {label} | {fmt(cmn['asr_error_rate'])} | {fmt(eng['asr_error_rate'])} | "
            f"{fmt(cmn['mt_corpus_bleu'],2)}/{fmt(cmn['mt_corpus_chrf'],2)} | "
            f"{fmt(eng['mt_corpus_bleu'],2)}/{fmt(eng['mt_corpus_chrf'],2)} | "
            f"{fmt(overall['final_translation_lcs_coverage_mean'])} | "
            f"{fmt(overall['translation_4gram_repetition_rate_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## 3. WRITE、静音和音频",
            "",
            "| 系统 | 首次发声p50/p95 ms↓ | WRITE gap p95 ms↓ | 最大静音均值 ms↓ | 音频覆盖↑ | WRITE | pending/TTS失败 | WAV健康 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, payload in arms:
        value = payload["aggregate"]["overall"]
        first = value["first_audio_source_ms"]
        gap = value["write_gap_ms"]
        healthy = min(
            float(value["continuous_wav_health_rate"]),
            float(value["timeline_wav_health_rate"]),
            float(value["stereo_wav_health_rate"]),
        )
        lines.append(
            f"| {label} | {fmt(first['p50'],0)}/{fmt(first['p95'],0)} | "
            f"{fmt(gap['p95'],0)} | {fmt(value['maximum_internal_timeline_silence_ms_mean'],0)} | "
            f"{fmt(value['translation_audio_to_source_duration_ratio_mean'])} | "
            f"{value['audio_writes_total']} | {value['pending_unspoken_total']}/{value['tts_failures_total']} | "
            f"{fmt(healthy)} |"
        )
    baseline = arms[0][1]["aggregate"]
    base_cmn = baseline["by_direction"]["cmn->eng"]
    base_eng = baseline["by_direction"]["eng->cmn"]
    base_overall = baseline["overall"]
    lines.extend(
        [
            "",
            "## 4. 相对 Phase A 的自动审计",
            "",
            "| 系统 | CER变化↓ | WER变化↓ | 中→英chrF变化↑ | 英→中chrF变化↑ | 覆盖变化↑ | 首次p50变化ms↓ | 最大静音变化ms↓ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, payload in arms[1:]:
        value = payload["aggregate"]
        cmn, eng, overall = (
            value["by_direction"]["cmn->eng"],
            value["by_direction"]["eng->cmn"],
            value["overall"],
        )
        lines.append(
            f"| {label} | {fmt(float(cmn['asr_error_rate'])-float(base_cmn['asr_error_rate']))} | "
            f"{fmt(float(eng['asr_error_rate'])-float(base_eng['asr_error_rate']))} | "
            f"{fmt(float(cmn['mt_corpus_chrf'])-float(base_cmn['mt_corpus_chrf']),2)} | "
            f"{fmt(float(eng['mt_corpus_chrf'])-float(base_eng['mt_corpus_chrf']),2)} | "
            f"{fmt(float(overall['final_translation_lcs_coverage_mean'])-float(base_overall['final_translation_lcs_coverage_mean']))} | "
            f"{fmt(float(overall['first_audio_source_ms']['p50'])-float(base_overall['first_audio_source_ms']['p50']),0)} | "
            f"{fmt(float(overall['maximum_internal_timeline_silence_ms_mean'])-float(base_overall['maximum_internal_timeline_silence_ms_mean']),0)} |"
        )
    by_label = {label: payload for label, payload in arms}
    if {"SFT64", "RL epoch1", "RL epoch2", "RL epoch3"}.issubset(by_label):
        sft = by_label["SFT64"]["aggregate"]["overall"]
        lines.extend(
            [
                "",
                "## 5. 相对 SFT64 的 RL 增益与 checkpoint 选择",
                "",
                "| 系统 | ASR error变化↓ | BLEU变化↑ | chrF变化↑ | LCS覆盖变化↑ | 首次p95变化ms↓ | 最大静音变化ms↓ | 音频覆盖变化↑ |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label in ("RL epoch1", "RL epoch2", "RL epoch3"):
            value = by_label[label]["aggregate"]["overall"]
            lines.append(
                f"| {label} | {fmt(float(value['asr_error_rate'])-float(sft['asr_error_rate']))} | "
                f"{fmt(float(value['mt_corpus_bleu'])-float(sft['mt_corpus_bleu']),2)} | "
                f"{fmt(float(value['mt_corpus_chrf'])-float(sft['mt_corpus_chrf']),2)} | "
                f"{fmt(float(value['final_translation_lcs_coverage_mean'])-float(sft['final_translation_lcs_coverage_mean']))} | "
                f"{fmt(float(value['first_audio_source_ms']['p95'])-float(sft['first_audio_source_ms']['p95']),0)} | "
                f"{fmt(float(value['maximum_internal_timeline_silence_ms_mean'])-float(sft['maximum_internal_timeline_silence_ms_mean']),0)} | "
                f"{fmt(float(value['translation_audio_to_source_duration_ratio_mean'])-float(sft['translation_audio_to_source_duration_ratio_mean']))} |"
            )
        lines.extend(
            [
                "",
                "结论：RL 确实改变了真实 ASR/MT/TTS 路由，不再是旧实验中 ASR 输出逐字不变的无效更新。若以本次用户指定的 train-seen 目标优先选择内容完整度和双向折中，推荐 `RL epoch2 / iter_0000082`：它相对 SFT64 取得最低整体 ASR error、最高 LCS 覆盖、最高音频覆盖，并显著恢复英→中 BLEU/chrF。若更强调较短内部静音和中→英质量，则 `RL epoch1 / iter_0000041` 更稳健。`RL epoch3 / iter_0000123` 的 BLEU、chrF 和静音开始回退，不推荐作为部署 checkpoint。",
                "",
                "严格边界：相对原始 Phase A，所有新 arm 都改善了中→英 CER/chrF，但英→中 WER 和 chrF 仍未完全恢复，因此没有通过‘双向质量均不退化’的严格门。当前实验只证明约束式 GRPO 能在这 8 条训练样本上学习并部分修复 SFT64，不证明外部泛化，也不代表已经达到低于 1 秒的同传延迟。",
                "",
                "### 推荐试听与失败样本",
                "",
                "- `episode_000006_cmn_eng`：RL epoch2 是最清楚的中→英正例；相对 Phase A，ASR error、chrF、覆盖和最大静音均改善，但首次发声仍为 14.08 s。",
                "- `episode_000033_eng_cmn`：RL epoch2 相对 SFT64 明显恢复长段英文识别与中文翻译，适合听 RL 的修复作用；但仍弱于原始 Phase A，且内部最大静音达到 38 s。",
                "- `episode_000028_cmn_eng`：三个 RL arm 都较稳定，epoch3 单样本分数最好，但不能据此覆盖其整体过训结论。",
                "- `episode_000004_cmn_eng`：SFT64 已大幅优于 Phase A，继续 RL 后逐 epoch 回落，是过度优化的反例。",
                "- `episode_000035_eng_cmn`：所有 arm 都很差，RL epoch2 仅 5 次 WRITE、音频覆盖 0.148，是当前最明显的失败样本。",
            ]
        )
    sample_ids = [str(row["sample_id"]) for row in arms[0][1]["results"]]
    lines.extend(["", "## 6. 逐样本试听", ""])
    for sample_id in sample_ids:
        source = next(
            row for row in arms[0][1]["results"] if str(row["sample_id"]) == sample_id
        )
        lines.extend(
            [
                f"### {sample_id}（{source['src_lang']}→{source['tgt_lang']}）",
                "",
                f"- 源音频：`{source['source_audio']}`",
            ]
        )
        for label, payload in arms:
            row = next(
                value
                for value in payload["results"]
                if str(value["sample_id"]) == sample_id
            )
            metric = row["reference_metrics"]
            lines.extend(
                [
                    f"- {label}：ASR error={fmt(metric['asr_error_rate'])}，chrF={fmt(metric['mt_sentence_chrf'],2)}，覆盖={fmt(metric['final_translation_lcs_coverage'])}，首次发声={fmt(row['first_audio_source_ms'],0)} ms，最大静音={fmt(row['maximum_internal_timeline_silence_ms'],0)} ms。",
                    f"  - 连续译音：`{row['continuous_audio_path']}`",
                    f"  - 时间轴：`{row['timeline_audio_path']}`",
                    f"  - 左源右译：`{row['stereo_audio_path']}`",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## 7. 判定原则",
            "",
            "只有当 ASR error 不升、双向 chrF 和文本覆盖不下降、pending/TTS failure 为零时，first-WRITE、WRITE gap 或静音改善才计为有效。训练内提升仅证明当前方法能在给定数据上学到目标，不等于外部泛化。",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
