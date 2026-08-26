#!/usr/bin/env python3
"""Write a Chinese listening/metric report for one stateful runtime stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _f(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _old_by_id(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["sample_id"]): row for row in payload["results"]}


def _new_by_id(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError("stateful result is incomplete")
    return {str(row["sample_id"]): row for row in payload["results"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--old-results", type=Path, required=True)
    parser.add_argument("--new-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old = _old_by_id(args.old_results)
    new = _new_by_id(args.new_results)
    if set(old) != set(new):
        raise ValueError("old/new sample IDs differ")

    lines = [
        f"# {args.stage_name}：四条长音频试听与问题分析",
        "",
        "## 结论口径",
        "",
        "旧对照是每个 18–30 秒窗口重置全部状态的 bounded-window pseudo-streaming。新结果在完整文件内保留因果 WhisperVQ 前端状态、ASR/MT 已提交文本、TTS ACK 队列和播放时钟；仅 LLM acoustic prompt 使用 24 秒有界 ring 并重算，因此这里不宣称 LLM KV-cache 实时部署。质量门只记录，不阻断后续阶段。",
        "",
        "## 总表",
        "",
        "| 音频 | 方向 | 旧首音频 | 新首音频 | 旧译音覆盖 | 新译音覆盖 | 旧最大内部静音 | 新最大内部静音 | 新 WRITE | 未发音队列 | TTS失败 | RTF旧→新 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sample_id in sorted(new):
        before, after = old[sample_id], new[sample_id]
        old_ratio = float(before["translation_duration_seconds"]) / max(
            1e-9, float(before["source_duration_seconds"])
        )
        new_ratio = float(after["translation_audio_to_source_duration_ratio"])
        lines.append(
            "| {sample} | {src}→{tgt} | {old_first} ms | {new_first} ms | {old_cov} | {new_cov} | {old_sil} ms | {new_sil} ms | {writes} | {pending} | {failures} | {old_rtf}→{new_rtf} |".format(
                sample=sample_id,
                src=after["src_lang"],
                tgt=after["tgt_lang"],
                old_first=_f(before.get("first_audio_global_ms"), 0),
                new_first=_f(after.get("first_audio_source_ms"), 0),
                old_cov=_f(old_ratio, 3),
                new_cov=_f(new_ratio, 3),
                old_sil=_f(before.get("timeline_silence", {}).get("maximum_internal_silence_ms"), 0),
                new_sil=_f(after.get("maximum_internal_timeline_silence_ms"), 0),
                writes=after["audio_writes"],
                pending=after["tts_pending_unspoken_items"],
                failures=after["tts_failures"],
                old_rtf=_f(before.get("rtf"), 2),
                new_rtf=_f(after.get("rtf"), 2),
            )
        )

    lines.extend(["", "## 分音频试听与诊断", ""])
    for sample_id in sorted(new):
        row = new[sample_id]
        events = list(row["events"])
        conflicts = [event for event in events if event.get("mt_acceptance") == "rejected_early_end"]
        writes = [event for event in events if event.get("tts_emissions")]
        lines.extend(
            [
                f"### {sample_id}",
                "",
                f"- 源音频：`{row['source_audio']}`",
                f"- 连续翻译音频：`{row['continuous_audio_path']}`",
                f"- 全局时间轴：`{row['timeline_audio_path']}`",
                f"- 左源右译立体声：`{row['stereo_audio_path']}`",
                f"- 完整 ASR：{row['generated_streaming_transcription']}",
                f"- 完整增量翻译：{row['generated_streaming_translation']}",
                f"- 首次发声 {_f(row['first_audio_source_ms'],0)} ms；共 {row['audio_writes']} 次发声；最大 WRITE 间隔 {_f(row['inter_write_gap_ms']['maximum'],0)} ms。",
                f"- 24 秒 acoustic ring rollover {row['memory_rollovers']} 次，WhisperVQ encoder position reset {row['frontend_encoder_resets']} 次；人工窗口误 final 次数 {row['artificial_boundary_finalizations']}。",
                f"- early-END 拒绝 {len(conflicts)} 次；semantic continuation {row['semantic_continuations']} 次；TTS 失败 {row['tts_failures']} 次；最终未发音队列 {row['tts_pending_unspoken_items']} 条。",
                f"- 有发声事件 {len(writes)} 个决策点；连续音频健康={row['continuous_audio_health']['healthy']}，译音/源音时长比={_f(row['translation_audio_to_source_duration_ratio'],3)}。",
                "",
            ]
        )

    lines.extend(
        [
            "## 本阶段能回答与不能回答的问题",
            "",
            "可以直接判断：窗口状态是否重置、文本是否在 TTS 失败后丢失、320 semantic token 截断是否继续、是否在真实文件结束前发声、WRITE 间隔和全局时间轴空白是否改善。",
            "",
            "不能仅凭这四条无人工参考的外部长音频报告 BLEU/WER，也不能把固定 speaker token 等同于客观音色一致性分数。后续 A/B/C/D 归因会加入离线 teacher/reference 路由；最终 RL 对照仍使用同一 runtime v2，避免把 runtime 修复误算成模型训练收益。",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()

