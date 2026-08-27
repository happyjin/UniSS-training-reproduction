#!/usr/bin/env python3
"""Write the joint train-seen and external long-audio Chinese report."""

from __future__ import annotations

import argparse
import hashlib
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


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_identity_audit(
    scored: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Audit which outputs really change relative to the immutable Phase-A arm."""
    baseline_id = "phasea_iter381_runtime_v2"
    baseline = {
        str(row["sample_id"]): row for row in scored[baseline_id]["results"]
    }
    sample_rows: list[dict[str, Any]] = []
    paired: dict[str, dict[str, int]] = {}
    hash_keys = {
        "continuous_same": "continuous_audio_path",
        "timeline_same": "timeline_audio_path",
        "stereo_same": "stereo_audio_path",
    }
    for run_id in LABELS:
        if run_id == baseline_id:
            continue
        counts = {
            "asr_same": 0,
            "mt_same": 0,
            "continuous_same": 0,
            "timeline_same": 0,
            "stereo_same": 0,
        }
        candidates = {
            str(row["sample_id"]): row for row in scored[run_id]["results"]
        }
        for sample_id, base in baseline.items():
            row = candidates[sample_id]
            counts["asr_same"] += int(
                row["generated_streaming_transcription"]
                == base["generated_streaming_transcription"]
            )
            counts["mt_same"] += int(
                row["generated_streaming_translation"]
                == base["generated_streaming_translation"]
            )
            for output_key, path_key in hash_keys.items():
                counts[output_key] += int(
                    file_sha256(str(row[path_key]))
                    == file_sha256(str(base[path_key]))
                )
        paired[run_id] = counts
    for sample_id, base in baseline.items():
        rows = [
            next(
                row
                for row in scored[run_id]["results"]
                if str(row["sample_id"]) == sample_id
            )
            for run_id in LABELS
        ]
        sample_rows.append(
            {
                "sample_id": sample_id,
                "asr_unique": len(
                    {str(row["generated_streaming_transcription"]) for row in rows}
                ),
                "mt_unique": len(
                    {str(row["generated_streaming_translation"]) for row in rows}
                ),
                "continuous_unique": len(
                    {file_sha256(str(row["continuous_audio_path"])) for row in rows}
                ),
                "timeline_unique": len(
                    {file_sha256(str(row["timeline_audio_path"])) for row in rows}
                ),
                "stereo_unique": len(
                    {file_sha256(str(row["stereo_audio_path"])) for row in rows}
                ),
            }
        )
    return sample_rows, paired


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
    identity_rows, paired_identity = paired_identity_audit(scored)

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
            "## 4. 逐样本变化审计：RL 到底改了什么",
            "",
            "当前严格级联协议对 ASR prompt 强制关闭 RL adapter，只在 MT、semantic TTS 和 control prompt 上启用。因此 ASR 不是 RL 可学习路径。下面使用逐样本精确字符串比较和 WAV 文件 SHA256，而不是仅比较四舍五入后的指标。`唯一数=1` 表示四个 arm 完全相同；大于 1 表示至少一个 RL checkpoint 真正改变了输出。",
            "",
            "| 样本 | ASR文本唯一数 | MT文本唯一数 | 连续WAV唯一数 | 时间轴WAV唯一数 | 立体声WAV唯一数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in identity_rows:
        lines.append(
            f"| {row['sample_id']} | {row['asr_unique']} | {row['mt_unique']} | "
            f"{row['continuous_unique']} | {row['timeline_unique']} | "
            f"{row['stereo_unique']} |"
        )
    lines.extend(
        [
            "",
            "| RL checkpoint 相对 Phase A | ASR文本相同 | MT文本相同 | 连续WAV相同 | 时间轴WAV相同 | 立体声WAV相同 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for run_id in (
        "rl_iter15_runtime_v2",
        "rl_iter30_runtime_v2",
        "rl_iter45_runtime_v2",
    ):
        value = paired_identity[run_id]
        lines.append(
            f"| {LABELS[run_id]} | {value['asr_same']}/8 | {value['mt_same']}/8 | "
            f"{value['continuous_same']}/8 | {value['timeline_same']}/8 | "
            f"{value['stereo_same']}/8 |"
        )
    lines.extend(
        [
            "",
            "审计结论：ASR 在全部 8 条样本、全部 checkpoint 上逐字完全相同，确认 CER/WER 一致不是显示精度导致。iter15/30 各改变 4/8 条最终 MT，iter45 改变 5/8 条；其余样本最终译文未变。所有 WAV 均发生变化，因为 adapter 在 semantic TTS route 上启用，即使最终可见 MT 文本相同，声学 token 仍可改变。固定随机种子后仍观察到这种差异，所以不能把 WAV SHA256 差异误判为结构或内容必然改善。",
            "",
            "## 5. 结果结论与当前问题",
            "",
            "- **train-seen 最均衡 checkpoint 是 iter30，但提升很小。** 相比 Phase A，平均最大内部静音从 26.88 s 降到 23.58 s，WRITE gap p95 从 24.06 s 降到 22.40 s，英→中 chrF 从 17.19 微升到 17.32，4-gram 重复率从 0.004 降到 0.003；同时 LCS 覆盖从 0.196 降到 0.193，中→英 BLEU 也略升但幅度不足以构成稳定联合收益。",
            "- **iter15 在 train-seen 上整体退化。** 英→中 BLEU/chrF、文本覆盖、WRITE gap 与内部静音都比 Phase A 差；它只是在无 reference 的四条外部长音频上显示出较好的结构覆盖和较少失败，因此不能据此宣称语义质量更好。",
            "- **iter45 是混合收益。** 中→英 BLEU/chrF 与音频覆盖最高，但英→中质量下降，外部长音频仍有 pending/TTS failure，不适合作为统一最佳 checkpoint。",
            f"- **首次 WRITE 没有被 RL 改善。** 四个 arm 的 train-seen 首次发声均为 p50={fmt(base_train['first_audio_source_ms']['p50'],0)} ms、p95={fmt(base_train['first_audio_source_ms']['p95'],0)} ms；外部长音频仍约 10.24–75.52 s。当前系统不能声称低延迟实时同传。",
            "- **ASR 是首要瓶颈。** 中→英中文 CER=0.763、英→中英文 WER=0.689，且 RL 路由根本不更新 ASR。上游错误和漏识别直接限制增量 MT，后续 reward 无法补回没有进入文本上下文的源内容。",
            "- **训练信号仍偏弱且域不匹配。** 当前仅 64 条 episode、45 次 update；group-relative rollout 候选之间的差异有限。episode 由多个短句以约 160 ms 间隔拼接，虽然长度约一分钟，但不等价于自然连续讲话的停顿、共发音和话题连续性。",
            "- **RTF 只适合本轮 arm 间近似比较。** 正式运行每个 arm 同时启动 8 个模型进程，四个 arm 顺序执行；表中 RTF 包含并发 GPU contention 与解码开销，不应当解释为单流部署 RTF。",
            "",
            "下一版应优先让训练目标与推理路由一致：若 reward 包含 ASR 项，就必须让可训练参数实际进入 ASR route，或明确把 ASR reward 移出 policy 优化并单独固定强 ASR。随后扩大自然连续长语音 episode，并直接优化 first-WRITE、WRITE-gap、覆盖和双向 MT 质量的联合 Pareto reward。",
            "",
            "## 6. 可复现文件",
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
