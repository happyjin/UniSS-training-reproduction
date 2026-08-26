#!/usr/bin/env python3
"""Write a Chinese analysis report for a merged free-running rollout."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = json.loads(args.rollout.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError("rollout is incomplete")
    candidates: list[dict[str, object]] = []
    for summary in payload["summaries"]:
        for candidate in summary["candidates"]:
            candidates.append(
                {
                    "episode_id": summary["episode_id"],
                    "direction": summary["direction"],
                    **candidate,
                }
            )
    rewards = [float(row["reward"]["total"]) for row in candidates]
    first = [float(row["observation"]["first_write_ms"]) for row in candidates]
    silence = [
        float(row["observation"]["maximum_internal_silence_ms"]) for row in candidates
    ]
    spoken = [float(row["observation"]["spoken_text_fraction"]) for row in candidates]
    mt = [float(row["observation"]["mt_teacher_similarity"]) for row in candidates]
    asr = [float(row["observation"]["asr_teacher_similarity"]) for row in candidates]
    best = sorted(candidates, key=lambda row: float(row["reward"]["total"]), reverse=True)
    worst = list(reversed(best))
    lines = [
        f"# {args.stage_name}：真实自由运行长 episode rollout 分析",
        "",
        "## 协议",
        "",
        f"共 {payload['episodes']} 条 45–90 秒 episode，{payload['workers']} 个 GPU worker，group size={payload['group_size']}，产生 {payload['candidates']} 条完整自由运行候选。策略在自己的 ASR/MT/TTS 历史上继续生成，不使用 gold prefix。质量门只记录与排序，不会中断后续打包、训练或评估。",
        "",
        "## 汇总",
        "",
        "| 指标 | mean | p50 | p95 |",
        "|---|---:|---:|---:|",
        f"| episode reward | {statistics.fmean(rewards):.4f} | {percentile(rewards,0.50):.4f} | {percentile(rewards,0.95):.4f} |",
        f"| ASR teacher similarity | {statistics.fmean(asr):.4f} | {percentile(asr,0.50):.4f} | {percentile(asr,0.95):.4f} |",
        f"| MT teacher chrF/100 | {statistics.fmean(mt):.4f} | {percentile(mt,0.50):.4f} | {percentile(mt,0.95):.4f} |",
        f"| first WRITE (ms) | {statistics.fmean(first):.1f} | {percentile(first,0.50):.1f} | {percentile(first,0.95):.1f} |",
        f"| 最大内部静音 (ms) | {statistics.fmean(silence):.1f} | {percentile(silence,0.50):.1f} | {percentile(silence,0.95):.1f} |",
        f"| 已发音文本比例 | {statistics.fmean(spoken):.4f} | {percentile(spoken,0.50):.4f} | {percentile(spoken,0.95):.4f} |",
        "",
        "## 最好候选试听",
        "",
    ]
    for row in best[:8]:
        result = row["result"]
        observation = row["observation"]
        lines.extend(
            [
                f"### {row['episode_id']} / group {row['group_index']} / {row['direction']}",
                "",
                f"- reward={float(row['reward']['total']):.4f}；first WRITE={float(observation['first_write_ms']):.0f} ms；MT chrF/100={float(observation['mt_teacher_similarity']):.4f}；已发音比例={float(observation['spoken_text_fraction']):.4f}。",
                f"- 连续译音：`{result['continuous_audio_path']}`",
                f"- 全局时间轴：`{result['timeline_audio_path']}`",
                f"- 左源右译：`{result['stereo_audio_path']}`",
                "",
            ]
        )
    lines.extend(["## 最差候选诊断", ""])
    for row in worst[:8]:
        observation = row["observation"]
        result = row["result"]
        lines.extend(
            [
                f"### {row['episode_id']} / group {row['group_index']} / {row['direction']}",
                "",
                f"- reward={float(row['reward']['total']):.4f}；first WRITE={float(observation['first_write_ms']):.0f} ms；最大内部静音={float(observation['maximum_internal_silence_ms']):.0f} ms；MT chrF/100={float(observation['mt_teacher_similarity']):.4f}；已发音比例={float(observation['spoken_text_fraction']):.4f}。",
                f"- 左源右译：`{result['stereo_audio_path']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 本阶段用途",
            "",
            "这些候选不是最终模型结论，而是带 old-policy log-probability 的训练轨迹。组内优势会偏好质量、完整发音、稳定提交和健康音频，同时保留较弱的首 WRITE/内部静音项；Phase3 replay 与 KL 在下一阶段抑制灾难性遗忘。",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
