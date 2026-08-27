#!/usr/bin/env python3
"""Write a clearly bounded report for the saved pre-RL group-0 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = json.loads(args.scored.read_text(encoding="utf-8"))
    rows = list(payload["results"])
    overall = payload["aggregate"]["overall"]
    directions = payload["aggregate"]["by_direction"]
    best = max(rows, key=lambda row: float(row["reference_metrics"]["mt_sentence_chrf"]))
    worst = min(rows, key=lambda row: float(row["reference_metrics"]["mt_sentence_chrf"]))
    lines = [
        "# 8 条 train-seen 长 episode：pre-RL Phase A group-0 临时基线",
        "",
        "## 重要边界",
        "",
        "本报告从已经完成的正式 train64×group4 rollout 中提取固定 group-0，不需要重新占用 GPU。它能立即确认训练长 episode、reference scorer 和现有 Phase A 问题，但 **不是** 新生成的 Phase A/iter15/iter30/iter45 同 seed 正式对照，不能代替最终报告。",
        "",
        "全部 8 条样本都来自 RL 正式训练 rollout，属于 train-seen/in-domain；episode 音频和组成 component 均已确认不与 validation 重叠。",
        "",
        "## 临时结果",
        "",
        f"- 中→英流式 ASR CER={fmt(directions['cmn->eng']['asr_error_rate'])}；英→中流式 ASR WER={fmt(directions['eng->cmn']['asr_error_rate'])}。",
        f"- 中→英 MT BLEU/chrF={fmt(directions['cmn->eng']['mt_corpus_bleu'],2)}/{fmt(directions['cmn->eng']['mt_corpus_chrf'],2)}；英→中={fmt(directions['eng->cmn']['mt_corpus_bleu'],2)}/{fmt(directions['eng->cmn']['mt_corpus_chrf'],2)}。",
        f"- 平均 LCS 文本覆盖={fmt(overall['final_translation_lcs_coverage_mean'])}；平均 hypothesis/reference 内容长度比={fmt(overall['translation_length_ratio_mean'])}；平均 4-gram 重复率={fmt(overall['translation_4gram_repetition_rate_mean'])}。",
        f"- 首次发声 p50/p95/max={fmt(overall['first_audio_source_ms']['p50'],0)}/{fmt(overall['first_audio_source_ms']['p95'],0)}/{fmt(overall['first_audio_source_ms']['maximum'],0)} ms。",
        f"- WRITE gap p95/max={fmt(overall['write_gap_ms']['p95'],0)}/{fmt(overall['write_gap_ms']['maximum'],0)} ms；最大内部静音 mean/max={fmt(overall['maximum_internal_timeline_silence_ms_mean'],0)}/{fmt(overall['maximum_internal_timeline_silence_ms_max'],0)} ms。",
        f"- 译音/源音时长比={fmt(overall['translation_audio_to_source_duration_ratio_mean'])}；总 WRITE={overall['audio_writes_total']}；pending/TTS failure={overall['pending_unspoken_total']}/{overall['tts_failures_total']}；RTF={fmt(overall['rtf_mean'])}。",
        f"- continuous/timeline/stereo WAV 健康率={fmt(overall['continuous_wav_health_rate'])}/{fmt(overall['timeline_wav_health_rate'])}/{fmt(overall['stereo_wav_health_rate'])}。",
        "",
        "临时结论：声音文件本身全部健康、TTS 队列也能清空，但 Phase A 在这些约 1 分钟拼接 episode 上的 ASR、翻译内容覆盖、首次 WRITE 和长空白均明显有问题。当前最主要瓶颈不是 WAV 写坏，而是 free-running ASR/增量 MT 的内容错误与过晚/稀疏 WRITE。RL 是否真正修复它，必须等 iter15/30/45 在同一协议上的新结果。",
        "",
        f"当前单条 chrF 最好的是 `{best['sample_id']}`（{fmt(best['reference_metrics']['mt_sentence_chrf'],2)}），最差的是 `{worst['sample_id']}`（{fmt(worst['reference_metrics']['mt_sentence_chrf'],2)}）；不能只挑最好样本试听。",
        "",
        "## 逐样本指标与试听",
        "",
        "| episode | 方向 | 秒 | CER/WER↓ | chrF↑ | LCS覆盖↑ | 首次发声ms↓ | 最大静音ms↓ | WRITE | 译音/源音 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metric = row["reference_metrics"]
        lines.append(
            f"| {row['sample_id']} | {row['src_lang']}→{row['tgt_lang']} | "
            f"{float(row['source_duration_ms'])/1000:.2f} | {fmt(metric['asr_error_rate'])} | "
            f"{fmt(metric['mt_sentence_chrf'],2)} | {fmt(metric['final_translation_lcs_coverage'])} | "
            f"{fmt(row['first_audio_source_ms'],0)} | {fmt(row['maximum_internal_timeline_silence_ms'],0)} | "
            f"{row['audio_writes']} | {fmt(row['translation_audio_to_source_duration_ratio'])} |"
        )
    lines.extend(["", "### 音频路径", ""])
    for row in rows:
        lines.extend(
            [
                f"#### {row['sample_id']}（{row['src_lang']}→{row['tgt_lang']}）",
                "",
                f"- 源音频：`{row['source_audio']}`",
                f"- 连续译音：`{row['continuous_audio_path']}`",
                f"- 全局时间轴：`{row['timeline_audio_path']}`",
                f"- 左源右译立体声：`{row['stereo_audio_path']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 后续正式对照",
            "",
            "GPU device node 恢复后，`run_all_8gpu.sh` 会在相同 8 条 episode、相同 Runtime v2、640 ms/24 s 配置下依次重跑 Phase A、RL iter15、iter30 和 iter45；最终报告还会把之前反复试听的 Helen Keller、Shimon Peres、新加坡—越南关系和张河桥乡四条外部长音频放在独立章节继续比较。",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
