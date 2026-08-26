#!/usr/bin/env python3
"""Write the Chinese A/B/C/D long-episode fault-attribution report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def mean(rows, fn) -> float:
    return statistics.fmean(float(fn(row)) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = json.loads(args.attribution.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError("attribution is incomplete")
    rows = payload["results"]
    offline_asr = mean(rows, lambda row: row["offline_asr_similarity"])
    streaming_asr = mean(
        rows,
        lambda row: row["runtime_v2_group0"]["observation"]["asr_teacher_similarity"],
    )
    gold_mt = mean(rows, lambda row: row["gold_source_mt_chrf"]) / 100.0
    streaming_mt = mean(
        rows,
        lambda row: row["runtime_v2_group0"]["observation"]["mt_teacher_similarity"],
    )
    gold_tts = mean(rows, lambda row: row["gold_target_tts_phrase_success_fraction"])
    runtime_health = mean(
        rows,
        lambda row: row["runtime_v2_group0"]["observation"]["healthy_audio_fraction"],
    )
    runtime_spoken = mean(
        rows,
        lambda row: row["runtime_v2_group0"]["observation"]["spoken_text_fraction"],
    )
    first_write = mean(
        rows,
        lambda row: row["runtime_v2_group0"]["observation"]["first_write_ms"],
    )
    lines = [
        "# Phase A 长 episode A/B/C/D 故障归因",
        "",
        "## 归因协议",
        "",
        "本报告只在带 teacher transcription/translation 的 valid 长 episode 上计算质量，避免给四条外部无人工参考音频伪造 WER/BLEU。A/B/C 使用相同 Phase A `iter_0000381` 与同一 speaker/BiCodec 条件；D 使用真实自由运行 Runtime v2 的固定 group-0 候选。",
        "",
        "- A：完整 episode 一次性 full-context ASR，对 teacher transcription 计算相似度。",
        "- B：输入 gold source text 做 MT，对 teacher translation 计算 chrF。",
        "- C：输入 gold target text 分短语做 semantic TTS + BiCodec，统计健康发音覆盖。",
        "- D：stateful Runtime v2 自由运行 ASR→incremental MT→ACK TTS，所有上游误差会级联。",
        "",
        "## 汇总结论",
        "",
        "| 路由 | 指标 | 结果 |",
        "|---|---|---:|",
        f"| A offline/full-context ASR | teacher similarity | {offline_asr:.4f} |",
        f"| D streaming ASR | teacher similarity | {streaming_asr:.4f} |",
        f"| A→D ASR 退化 | similarity 差值 | {offline_asr-streaming_asr:+.4f} |",
        f"| B gold-source MT | chrF/100 | {gold_mt:.4f} |",
        f"| D free-running MT | chrF/100 | {streaming_mt:.4f} |",
        f"| B→D MT 级联退化 | chrF/100 差值 | {gold_mt-streaming_mt:+.4f} |",
        f"| C gold-target TTS | 健康短语覆盖 | {gold_tts:.4f} |",
        f"| D runtime TTS | 健康音频覆盖 | {runtime_health:.4f} |",
        f"| D runtime | 已发音文本比例 | {runtime_spoken:.4f} |",
        f"| D runtime | mean first WRITE | {first_write:.1f} ms |",
        "",
        "如果 A 明显好于 D 的 ASR，主因是流式声学/长会话 ASR；如果 B 明显好于 D 的 MT，主因包含 ASR 误差传播和 incremental MT；如果 C 接近 1 而 D 发音覆盖低，则 TTS 本体可用但输入片段、END 或队列状态有问题。Runtime v1/v2 的窗口重置差异另由四条外部长音频 Stage-1 报告量化。",
        "",
        "## 逐 episode 结果与试听",
        "",
        "| episode | 方向 | A ASR | B MT chrF | C TTS覆盖 | D ASR | D MT | D首WRITE | D已发音 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda value: str(value["episode_id"])):
        obs = row["runtime_v2_group0"]["observation"]
        lines.append(
            f"| {row['episode_id']} | {row['direction']} | {float(row['offline_asr_similarity']):.4f} | {float(row['gold_source_mt_chrf']):.2f} | {float(row['gold_target_tts_phrase_success_fraction']):.4f} | {float(obs['asr_teacher_similarity']):.4f} | {float(obs['mt_teacher_similarity'])*100:.2f} | {float(obs['first_write_ms']):.0f} ms | {float(obs['spoken_text_fraction']):.4f} |"
        )
    lines.extend(["", "## C 路由 gold-target TTS 试听", ""])
    for row in sorted(rows, key=lambda value: str(value["episode_id"])):
        lines.extend(
            [
                f"### {row['episode_id']} / {row['direction']}",
                "",
                f"- 左源右 gold-target TTS：`{row['gold_target_tts_stereo']}`",
                f"- 连续 gold-target TTS：`{row['gold_target_tts_audio']}`",
                f"- 健康短语覆盖：{float(row['gold_target_tts_phrase_success_fraction']):.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## D 路由自由运行试听",
            "",
            "D 路由的连续、全局时间轴和左源右译文件保存在 formal valid rollout 报告中；本 JSON 直接保留每条 group-0 的绝对路径和完整 observation/reward，便于后续训练前后做完全相同协议对照。",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
