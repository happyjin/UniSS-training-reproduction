#!/usr/bin/env python3
"""Render the final Chinese report from immutable experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


ARMS = (
    "a1_sft_full_recovery1",
    "a2_g4_full_recovery1",
    "a3_g8_full_recovery1",
    "a4_g8_seed2_full_recovery1",
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise ValueError(f"incomplete report input: {path}")
    return value


def _f(value, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _pct(value, digits: int = 2) -> str:
    return "—" if value is None else f"{100.0*float(value):.{digits}f}%"


def _mt_row(metrics: Mapping[str, object], path: str, direction: str) -> Mapping[str, object]:
    return metrics["e_mt"][path]["directions"][direction]  # type: ignore[index]


def _listening_table(
    roots: Mapping[str, dict[str, object]], title: str
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| arm | chunk | ASR WER/CER | 首音频 p50 | pre-final 发声 | WAV 健康 | RTF |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        payload = roots[arm]
        for chunk, row in sorted(
            payload["chunks_ms"].items(), key=lambda item: int(item[0])  # type: ignore[union-attr]
        ):
            first = row["first_audio_source_ms"]["p50"]  # type: ignore[index]
            lines.append(
                "| {arm} | {chunk} ms | {asr} | {first} ms | {prefinal} | {healthy} | {rtf} |".format(
                    arm=arm,
                    chunk=chunk,
                    asr=_pct(row["weighted_asr_error_rate"]),  # type: ignore[index]
                    first=_f(first, 1),
                    prefinal=_pct(row["prefinal_audio_rate"]),  # type: ignore[index]
                    healthy=_pct(row["healthy_audio_rate"]),  # type: ignore[index]
                    rtf=_f(row["runtime_rtf"], 3),  # type: ignore[index]
                )
            )
    lines.extend(
        [
            "",
            "该表的首音频时延是相对该条源音频起点的 source-availability 时刻；RTF 还包含当前 Python 自回归实现，不能等同于优化后的线上服务吞吐。",
            "",
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--training-audit", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--short-root", type=Path, required=True)
    parser.add_argument("--long-prefix-root", type=Path, required=True)
    parser.add_argument("--best-longform", type=Path, required=True)
    parser.add_argument("--stage-a-longform", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    training = _load(args.training_audit)
    comparison = _load(args.comparison)
    evaluations = {
        arm: _load(args.evaluation_root / arm / "SUMMARY.json") for arm in ARMS
    }
    short = {arm: _load(args.short_root / arm / "SUMMARY.json") for arm in ARMS}
    long_prefix = {
        arm: _load(args.long_prefix_root / arm / "SUMMARY.json") for arm in ARMS
    }
    best_long = _load(args.best_longform / "results.json")
    base_long = _load(args.stage_a_longform / "results.json")
    best = str(comparison["best_arm"])

    lines = [
        "# Stage A 质量优先 SFT / GRPO 四组完整对照实验报告",
        "",
        "## 1. 结论摘要",
        "",
        f"固定质量优先排序选择的最佳实验为 **{best}**。排序先比较结构错误、非静音率和 source EOS 前语义输出，再比较相对 Stage A 的配对质量，最后才比较首语义时延；训练过程中没有使用该排序提前停止。",
        "",
        "本实验回答两个问题：第一，继续 SFT 或 GRPO 是否能相对不可变 Stage A `iter_0000381` 改善 incremental MT / semantic TTS / WAIT-WRITE；第二，GRPO 是否优于同训练预算的 matched continued SFT。所有结论均来自相同 checkpoint 初始化、相同 15-shard 全局 shuffle、相同 2-GPU/arm 预算和相同固定评估样本。",
        "",
        "## 2. 训练设置与完整性",
        "",
        "| arm | 方法 | steps | NaN / skipped | GPU util mean / p95 | power mean / p95 / max | max memory |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        ARMS[0]: "matched continued SFT",
        ARMS[1]: "GRPO G4",
        ARMS[2]: "GRPO G8",
        ARMS[3]: "GRPO G8 seed-2 + stronger anchor",
    }
    for arm in ARMS:
        value = training["arms"][arm]  # type: ignore[index]
        train = value["training"]  # type: ignore[index]
        gpu = value["gpu"]  # type: ignore[index]
        lines.append(
            "| {arm} | {label} | {step}/{total} | {nan}/{skip} | {mean}/{p95} | {pmean}/{pp95}/{pmax} W | {mem} MiB |".format(
                arm=arm,
                label=labels[arm],
                step=train["last_step"],
                total=train["target_steps"],
                nan=train["nan_iterations"],
                skip=train["skipped_iterations"],
                mean=_pct(float(gpu["utility_mean_percent"]) / 100.0),
                p95=_pct(float(gpu["utility_p95_percent"]) / 100.0),
                pmean=_f(gpu["power_mean_w"], 1),
                pp95=_f(gpu["power_p95_w"], 1),
                pmax=_f(gpu["power_max_w"], 1),
                mem=_f(gpu["memory_max_mib"], 0),
            )
        )
    lines.extend(
        [
            "",
            "共同训练几何：Megatron 单机 2 GPU/arm；4 arm 并发使用 8×H200；GBS=16、MBS=1、sequence length=18,000、2,510 updates、40,150 packs 一次严格全局 shuffle coverage。A2–A4 前 256 updates 为共同 SFT bootstrap，之后 GRPO reference 与 group reward 激活。",
            "",
            "功率不以无关 synthetic kernel 填充；上表只报告真实训练监控，因此 H200 未达到 700W 并不代表程序空闲。变长 18k pack、同步、checkpoint 和 GRPO reference forward 会造成 utility/power 波动。",
            "",
            "## 3. 固定 64 条 validation 定量评估",
            "",
            "评估集按方向和 short/medium/long 时长分层冻结为 64 条，其中 16 条运行完整 E2E S2S 与音频解码。ASR route 固定禁用 adapter；MT、semantic TTS 与 control route 启用 adapter。每个 candidate 同时在相同 worker 内计算 Stage A adapter-off 配对基线。",
            "",
            "### 3.1 ASR 与 MT",
            "",
            "| arm | 中文 CER | 英文 WER | gold cmn→eng BLEU/chrF | gold eng→cmn BLEU/chrF | free cmn→eng BLEU/chrF | free eng→cmn BLEU/chrF |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        metrics = evaluations[arm]["candidate"]  # type: ignore[index]
        asr = metrics["e_asr"]  # type: ignore[index]
        values = []
        for path in ("gold_source", "free_running_source"):
            for direction in ("cmn->eng", "eng->cmn"):
                row = _mt_row(metrics, path, direction)
                values.append(f"{_f(row['candidate_bleu'],2)}/{_f(row['candidate_chrf'],2)}")
        lines.append(
            f"| {arm} | {_pct(asr['cmn']['error_rate'])} | {_pct(asr['eng']['error_rate'])} | "
            + " | ".join(values)
            + " |"
        )
    lines.extend(
        [
            "",
            "ASR 理论上应在四组完全一致，因为 adapter 在 ASR route 关闭；若存在仅为浮点/运行噪声。这里的主要可学习差异是 incremental MT、TTS semantic 与外部 control。",
            "",
            "### 3.2 E2E S2S、结构与时延",
            "",
            "| arm | semantic coverage mean/min | pre-EOS text | pre-EOS semantic | non-silent | malformed | first text p50 | first semantic p50 | AL / DAL | RTF |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        metrics = evaluations[arm]["candidate"]  # type: ignore[index]
        s2s = metrics["e_s2s_free"]  # type: ignore[index]
        latency = metrics["latency"]  # type: ignore[index]
        count = max(1, int(s2s.get("samples", 0)))
        lines.append(
            "| {arm} | {mean}/{minimum} | {text} | {semantic} | {audio} | {malformed} | {ft} ms | {fs} ms | {al}/{dal} ms | {rtf} |".format(
                arm=arm,
                mean=_f(s2s.get("semantic_coverage_mean"), 3),
                minimum=_f(s2s.get("semantic_coverage_min"), 3),
                text=_pct(float(s2s.get("target_text_before_source_eos", 0)) / count),
                semantic=_pct(float(s2s.get("target_semantic_before_source_eos", 0)) / count),
                audio=_pct(float(s2s.get("non_silent_pcm", 0)) / count),
                malformed=s2s.get("malformed_segments", 0),
                ft=_f(latency["first_text_write_ms"]["p50"], 1),
                fs=_f(latency["first_semantic_write_ms"]["p50"], 1),
                al=_f(latency["average_lagging_ms"], 1),
                dal=_f(latency["differentiable_average_lagging_ms"], 1),
                rtf=_f(latency["generation_rtf"], 3),
            )
        )
    lines.extend(
        [
            "",
            "### 3.3 相对 Stage A 的配对结论",
            "",
            "| arm | quality retention mean | non-silent | pre-EOS semantic | structure errors | first semantic Δ | 判定 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for arm in ARMS:
        row = comparison["arms"][arm]  # type: ignore[index]
        positive = (
            int(row["structure_errors"]) == 0
            and float(row["non_silent_rate"]) == 1.0
            and float(row.get("quality_retention_vs_stage_a_mean") or 0.0) > 1.0
        )
        lines.append(
            f"| {arm} | {_f(row.get('quality_retention_vs_stage_a_mean'),4)} | {_pct(row['non_silent_rate'])} | {_pct(row['pre_eos_semantic_rate'])} | {row['structure_errors']} | {_f(row.get('first_semantic_p50_delta_ms'),1)} ms | {'相对 Stage A 有效' if positive else '未证明全面有效'} |"
        )
    lines.extend(
        [
            "",
            "GRPO 是否优于 matched SFT 必须直接比较 A2–A4 与 A1，而不是只看各自相对 Stage A。若 GRPO 的 quality retention、结构健康度或首语义时延没有同时优于 A1，则只能说明 GRPO reward 在训练内有效激活，不能说明其外部性能优于 SFT。",
            "",
        ]
    )
    lines.extend(_listening_table(short, "4. 短音频 160/320/640/1280ms 可试听结果"))
    lines.extend(_listening_table(long_prefix, "5. 四条中英文 60 秒严格因果前缀"))
    lines.extend(
        [
            "## 6. 完整 5–7 分钟有界滑窗",
            "",
            "| model | audio | source | windows | first audio | RTF | max internal silence | stereo |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for label, payload in ((best, best_long), ("Stage A", base_long)):
        for row in payload["results"]:  # type: ignore[index]
            lines.append(
                f"| {label} | {row['sample_id']} | {_f(row['source_duration_seconds'],1)}s | {row['planned_windows']} | {_f(row['first_audio_global_ms'],1)}ms | {_f(row['rtf'],3)} | {_f(row['timeline_silence']['maximum_internal_silence_ms'],1)}ms | `{row['stereo_path']}` |"
            )
    lines.extend(
        [
            "",
            "完整长音频模式在每个 18–30 秒窗口内部遵守 160ms PCM 逐块可见性，但窗口间重置模型状态；因此它是 bounded-window pseudo-streaming，不是因果 encoder/KV cache 的严格长时 streaming。60 秒前缀表才用于严格因果长前缀判断。",
            "",
            "## 7. 音频与报告路径",
            "",
            f"- 定量评估根目录：`{args.evaluation_root.resolve()}`",
            f"- 短音频根目录：`{args.short_root.resolve()}`",
            f"- 60 秒严格前缀根目录：`{args.long_prefix_root.resolve()}`",
            f"- 最佳 arm 完整长音频：`{args.best_longform.resolve()}`",
            f"- Stage A 完整长音频：`{args.stage_a_longform.resolve()}`",
            "",
            "每个短/前缀样本目录均包含 `source.wav`、`translation_continuous.wav`、`translation_timeline.wav`、`stereo_left_source_right_translation.wav` 与逐 event JSON；立体声左声道是源语音，右声道是翻译语音。",
            "",
            "## 8. 方法边界与限制",
            "",
            "1. 训练 route mask 使用 gold next-token loss family；自由运行无法提前知道下一 token 的 oracle family，因此评估采用确定性状态机近似：ASR 内 adapter off，MT/TTS/control on。",
            "2. 当前 GRPO 是 utterance-level grouped token/action surrogate，不是逐事件真实音频 rollout 的 on-policy GRPO。reward 有方差、KL 和 policy update 不为零能证明优化路径生效，但最终是否有效只由外部评估决定。",
            "3. 64 条 validation 是冻结的配对对照集，不等同于 CVSS-T 或全量 UniST test；四条外部长音频没有参考译文，因此只报告运行、时延、空白与可试听音频，不报告 BLEU。",
            "4. 独立 TTS segment 使用固定 32-token speaker condition，condition 本身不变化；这减少显式音色漂移来源，但不等价于客观 speaker-similarity 指标。",
            "",
            "## 9. 最终回答",
            "",
            f"本次固定质量优先选择为 **{best}**。是否相对 Stage A 有效，以及 GRPO 是否比 A1 matched SFT 更好，应以上述配对表中的质量、结构、pre-EOS 发声和时延共同判断；不能仅凭训练 reward 上升或 loss 下降下结论。",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
