#!/usr/bin/env python3
"""Write the joint train-seen and external long-audio Chinese report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments.uniss_phasea_rl_trainlong_eval_v1.evaluation.score_results import (
    audit_wav,
)


LABELS = {
    "phasea_iter381_runtime_v2": "Phase A iter381 + Runtime v2（修复阶段）",
    "rl_iter15_runtime_v2": "RL iter15 + Runtime v2",
    "rl_iter30_runtime_v2": "RL iter30 + Runtime v2",
    "rl_iter45_runtime_v2": "RL iter45 + Runtime v2",
}

EXTERNAL_RUN_IDS = {
    "phasea_iter381_runtime_v2": "phasea_iter381_runtime_v2",
    "rl_epoch1_runtime_v2": "rl_iter15_runtime_v2",
    "rl_epoch2_runtime_v2": "rl_iter30_runtime_v2",
    "rl_epoch3_runtime_v2": "rl_iter45_runtime_v2",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def avg(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values]
    return statistics.fmean(rows) if rows else None


def fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def external_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(payload["results"])
    audits = [
        {
            "continuous": audit_wav(str(row["continuous_audio_path"]), 1),
            "timeline": audit_wav(str(row["timeline_audio_path"]), 1),
            "stereo": audit_wav(str(row["stereo_audio_path"]), 2),
        }
        for row in rows
    ]
    first = [float(row["first_audio_source_ms"]) for row in rows]
    return {
        "samples": len(rows),
        "first_audio_mean_ms": avg(first),
        "first_audio_min_ms": min(first),
        "first_audio_max_ms": max(first),
        "coverage_mean": avg(
            row["translation_audio_to_source_duration_ratio"] for row in rows
        ),
        "silence_mean_ms": avg(
            row["maximum_internal_timeline_silence_ms"] for row in rows
        ),
        "silence_max_ms": max(
            float(row["maximum_internal_timeline_silence_ms"]) for row in rows
        ),
        "writes": sum(int(row["audio_writes"]) for row in rows),
        "pending": sum(int(row["tts_pending_unspoken_items"]) for row in rows),
        "tts_failures": sum(int(row["tts_failures"]) for row in rows),
        "rtf_mean": avg(row["rtf"] for row in rows),
        "healthy": sum(
            all(bool(value[key]["healthy"]) for key in ("continuous", "timeline", "stereo"))
            for value in audits
        ),
        "rows": rows,
        "audits": audits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--external", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    scored_payloads = [load(path) for path in args.score]
    if len(scored_payloads) != 4:
        raise ValueError("expected Phase A plus three RL scored results")
    scored = {str(value["run_id"]): value for value in scored_payloads}
    if set(scored) != set(LABELS):
        raise ValueError(f"unexpected train-seen arms: {sorted(scored)}")
    external_payloads = [load(path) for path in args.external]
    if len(external_payloads) != 4:
        raise ValueError("expected four external Runtime-v2 result sets")
    unknown_external = sorted(
        str(value["run_id"])
        for value in external_payloads
        if str(value["run_id"]) not in EXTERNAL_RUN_IDS
    )
    if unknown_external:
        raise ValueError(f"unexpected raw external run IDs: {unknown_external}")
    external = {
        EXTERNAL_RUN_IDS[str(value["run_id"])]: external_summary(value)
        for value in external_payloads
    }
    if set(external) != set(LABELS):
        raise ValueError(f"unexpected external arms: {sorted(external)}")

    lines = [
        "# Phase A 修复与 long-episode RL：训练长样本和外部长音频联合评估",
        "",
        "## 1. 评估问题与声明边界",
        "",
        "本报告回答两个不同问题。第一组是正式 RL rollout 真正见过的 8 条长训练 episode，用于判断 Runtime v2 修复以及 RL iter15/30/45 是否学会训练目标；这组结果是 **train-seen/in-domain**，不能用于宣称泛化。第二组是之前反复试听的 4 条 Wikimedia 外部长音频，用于继续观察模型在非训练音频上的结构表现；因为它们没有与当前协议匹配的人工 reference，不能伪造 BLEU、chrF、WER 或 CER。两组结果始终分表讨论。",
        "",
        "固定推理条件：640 ms decision chunk、160 ms 物理声学 block、24 s acoustic ring、相同 Phase A speaker token、相同 Runtime v2。所有样本均输出连续译音、全局时间轴和左源右译立体声。",
        "",
        "8 条训练 episode 是由 6–13 个真实 15-shard 训练短句以约 160 ms 间隔拼接而成，时长约 70.8–79.0 秒。协议同时审计 episode 音频哈希和 component sample ID，确认不与 validation 重叠。",
        "",
        "## 2. Train-seen 长 episode：正式 reference 指标",
        "",
        "### 2.1 ASR 与 MT 内容质量",
        "",
        "| 系统 | 中→英中文 CER↓ | 英→中英文 WER↓ | 中→英 BLEU/chrF↑ | 英→中 BLEU/chrF↑ | LCS 文本覆盖↑ | 4-gram 重复率↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_id, label in LABELS.items():
        value = scored[run_id]["aggregate"]
        cmn = value["by_direction"]["cmn->eng"]
        eng = value["by_direction"]["eng->cmn"]
        overall = value["overall"]
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
            "CER/WER 衡量流式 ASR 编辑错误；BLEU/chrF 衡量最终增量译文与 teacher translation 的匹配；LCS 文本覆盖是 hypothesis 与 reference 的最长公共子序列召回，只用于判断漏译趋势，不等同于语义指标；4-gram 重复率用于暴露循环扩写。",
            "",
            "### 2.2 WRITE、TTS、覆盖与运行效率",
            "",
            "| 系统 | 首次发声 p50/p95 ms↓ | WRITE gap p95/max ms↓ | 最大内部静音 mean/max ms↓ | 译音/源音时长比 | WRITE 总数 | pending/TTS失败 | pre-final率 | WAV健康 | RTF↓ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run_id, label in LABELS.items():
        value = scored[run_id]["aggregate"]["overall"]
        first, gap = value["first_audio_source_ms"], value["write_gap_ms"]
        healthy = min(
            float(value["continuous_wav_health_rate"]),
            float(value["timeline_wav_health_rate"]),
            float(value["stereo_wav_health_rate"]),
        )
        lines.append(
            f"| {label} | {fmt(first['p50'],0)}/{fmt(first['p95'],0)} | "
            f"{fmt(gap['p95'],0)}/{fmt(gap['maximum'],0)} | "
            f"{fmt(value['maximum_internal_timeline_silence_ms_mean'],0)}/"
            f"{fmt(value['maximum_internal_timeline_silence_ms_max'],0)} | "
            f"{fmt(value['translation_audio_to_source_duration_ratio_mean'])} | "
            f"{value['audio_writes_total']} | {value['pending_unspoken_total']}/"
            f"{value['tts_failures_total']} | {fmt(value['prefinal_audio_rate'])} | "
            f"{fmt(healthy)} | {fmt(value['rtf_mean'])} |"
        )

    lines.extend(["", "### 2.3 逐训练样本试听路径", ""])
    baseline_rows = {
        str(row["sample_id"]): row for row in scored["phasea_iter381_runtime_v2"]["results"]
    }
    for sample_id in baseline_rows:
        reference = baseline_rows[sample_id]
        lines.extend(
            [
                f"#### {sample_id}（{reference['src_lang']}→{reference['tgt_lang']}，{float(reference['source_duration_ms'])/1000:.2f}s）",
                "",
                f"- 源音频：`{reference['source_audio']}`",
                f"- reference transcription：{reference['reference_transcription']}",
                f"- reference translation：{reference['reference_translation']}",
            ]
        )
        for run_id, label in LABELS.items():
            row = next(
                value
                for value in scored[run_id]["results"]
                if str(value["sample_id"]) == sample_id
            )
            metric = row["reference_metrics"]
            lines.extend(
                [
                    f"- {label}：CER/WER={fmt(metric['asr_error_rate'])}，chrF={fmt(metric['mt_sentence_chrf'],2)}，LCS覆盖={fmt(metric['final_translation_lcs_coverage'])}，4-gram重复={fmt(metric['translation_4gram_repetition']['repetition_rate'])}，首次发声={fmt(row['first_audio_source_ms'],0)} ms。",
                    f"  - 连续译音：`{row['continuous_audio_path']}`",
                    f"  - 时间轴：`{row['timeline_audio_path']}`",
                    f"  - 左源右译：`{row['stereo_audio_path']}`",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## 3. 之前反复试听的四条外部长音频",
            "",
            "这四条分别是 Helen Keller、Shimon Peres、新加坡—越南关系和张河桥乡。它们不属于训练数据。本节复用已经在相同 checkpoint、相同 Runtime v2 和相同 640 ms 配置下完成的正式结果，并重新独立读取 WAV 检查采样率、声道、finite、RMS、peak 和非静音比例；不重复消耗 GPU 做完全相同的推理。",
            "",
            "| 系统 | 首次发声 mean[min,max] ms↓ | 译音/源音覆盖 | 最大内部静音 mean/max ms↓ | WRITE | pending/TTS失败 | WAV健康 | RTF↓ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run_id, label in LABELS.items():
        value = external[run_id]
        lines.append(
            f"| {label} | {fmt(value['first_audio_mean_ms'],0)}["
            f"{fmt(value['first_audio_min_ms'],0)},{fmt(value['first_audio_max_ms'],0)}] | "
            f"{fmt(value['coverage_mean'])} | {fmt(value['silence_mean_ms'],0)}/"
            f"{fmt(value['silence_max_ms'],0)} | {value['writes']} | "
            f"{value['pending']}/{value['tts_failures']} | {value['healthy']}/4 | "
            f"{fmt(value['rtf_mean'])} |"
        )

    lines.extend(["", "### 3.1 外部长音频试听矩阵", ""])
    external_ids = [str(row["sample_id"]) for row in external["phasea_iter381_runtime_v2"]["rows"]]
    for sample_id in external_ids:
        lines.extend([f"#### {sample_id}", ""])
        for run_id, label in LABELS.items():
            row = next(
                value for value in external[run_id]["rows"] if str(value["sample_id"]) == sample_id
            )
            lines.extend(
                [
                    f"- {label}：首次发声={fmt(row['first_audio_source_ms'],0)} ms，覆盖={fmt(row['translation_audio_to_source_duration_ratio'])}，最大静音={fmt(row['maximum_internal_timeline_silence_ms'],0)} ms，WRITE={row['audio_writes']}，pending/TTS失败={row['tts_pending_unspoken_items']}/{row['tts_failures']}。",
                    f"  - 连续译音：`{row['continuous_audio_path']}`",
                    f"  - 时间轴：`{row['timeline_audio_path']}`",
                    f"  - 左源右译：`{row['stereo_audio_path']}`",
                ]
            )
        lines.append("")

    base_train = scored["phasea_iter381_runtime_v2"]["aggregate"]["overall"]
    lines.extend(
        [
            "## 4. 如何判定 RL 和修复阶段是否有效",
            "",
            "- Runtime v2/Phase A 是修复阶段基线；RL 净收益只能比较 iter15/30/45 与该基线，不能把旧 bounded-window runtime 到 Runtime v2 的收益算给 RL。",
            "- train-seen 上，如果 BLEU/chrF、ASR similarity、LCS 覆盖提升，同时重复、首次发声、WRITE gap、内部静音、pending 和 TTS failure 不恶化，才能说明 RL 在训练目标上形成了正向联合收益。",
            "- 外部长音频没有 reference，因此这里只能判断结构稳定性与试听表现，不能据此证明语义质量提升。既有结果已经表明 iter15 的结构通常最稳定；iter30 偏保守且易少译；iter45 的 validation loss 最优但会出现 pending/TTS failure 和中文 phrase-loop。最终 train-seen reference 结果用于验证这些现象究竟是“没学会训练目标”还是“只在训练域学会、外部不泛化”。",
            f"- Phase A train-seen 基线的首次发声 p50={fmt(base_train['first_audio_source_ms']['p50'],0)} ms。若所有 RL checkpoint 仍基本不变，就必须明确结论为：RL 没有解决首次 WRITE 过晚。",
            "",
            "## 5. 可复现文件",
            "",
            "- 固定 train-seen 协议：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/uniss_phasea_rl_trainlong_eval_v1/evaluation/protocol_train_seen_long8.json`",
            "- 每个 arm 的完整 reference、文本、event、WAV audit 和逐样本指标位于：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/<arm>/SCORED.json`",
            "- 外部长音频旧联合报告：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/reports/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/REPORT.zh-CN.md`",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
