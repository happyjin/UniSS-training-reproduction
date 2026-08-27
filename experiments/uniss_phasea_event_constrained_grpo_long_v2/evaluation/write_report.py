#!/usr/bin/env python3
"""Compare complete 64x4 rollouts and write a train-seen Chinese report."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


VALIDATION_RE = re.compile(r"validation loss at iteration\s+(\d+).*?\|")
METRIC_RE = re.compile(
    r"([a-zA-Z0-9_/]+)(?: value)?:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)[Ee][-+]?\d+)"
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def flatten(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in payload["summaries"]:
        for candidate in summary["candidates"]:
            rows.append(
                {
                    "episode_id": str(summary["episode_id"]),
                    "direction": str(summary["direction"]),
                    "source_audio": str(summary["source_audio"]),
                    **candidate,
                }
            )
    return rows


def validate_rollout(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("status") != "complete":
        raise ValueError("rollout is not complete")
    if int(payload.get("episodes", -1)) != 64 or int(payload.get("group_size", -1)) != 4:
        raise ValueError("expected immutable 64-episode group-four geometry")
    rows = flatten(payload)
    if len(rows) != 256:
        raise ValueError(f"expected 256 candidates, got {len(rows)}")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["episode_id"]] = counts.get(row["episode_id"], 0) + 1
    if len(counts) != 64 or set(counts.values()) != {4}:
        raise ValueError("each episode must contribute exactly four candidates")
    return rows


def value(row: dict[str, Any], name: str) -> float:
    if name == "reward":
        reward = row["reward"]
        return float(reward["total"] if isinstance(reward, dict) else reward)
    if name in row["observation"]:
        return float(row["observation"][name])
    result = row["result"]
    mapping = {
        "rtf": "rtf",
        "audio_writes": "audio_writes",
        "pending": "tts_pending_unspoken_items",
        "tts_failures": "tts_failures",
        "coverage_audio": "translation_audio_to_source_duration_ratio",
    }
    return float(result[mapping[name]])


METRICS = (
    "reward",
    "asr_teacher_similarity",
    "mt_teacher_similarity",
    "translation_length_ratio",
    "spoken_text_fraction",
    "first_write_ms",
    "maximum_internal_silence_ms",
    "coverage_audio",
    "audio_writes",
    "pending",
    "tts_failures",
    "rtf",
)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"candidates": len(rows)}
    for metric in METRICS:
        values = [value(row, metric) for row in rows]
        result[metric] = {
            "mean": mean(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
        }
    result["healthy_fraction"] = mean(
        [float(row["observation"]["healthy_audio_fraction"]) for row in rows]
    )
    return result


def best_per_episode(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["episode_id"]
        if key not in selected or value(row, "reward") > value(selected[key], "reward"):
            selected[key] = row
    return [selected[key] for key in sorted(selected)]


def parse_arm(specification: str) -> tuple[str, Path]:
    label, separator, path = specification.partition("=")
    if not separator or not label or not path:
        raise ValueError("arm must use LABEL=ROLLOUT.json")
    return label, Path(path)


def parse_terminal_validation(path: Path) -> dict[str, Any] | None:
    selected = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = VALIDATION_RE.search(line)
        if match is None:
            continue
        metrics = {name: float(number) for name, number in METRIC_RE.findall(line)}
        if "loss/total" in metrics:
            selected = {"iteration": int(match.group(1)), "metrics": metrics}
    return selected


def fmt(number: float, digits: int = 3) -> str:
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--training-log", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    arms = []
    for specification in args.arm:
        label, path = parse_arm(specification)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = validate_rollout(payload)
        arms.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "round_index": payload.get("round_index"),
                "all": summarize(rows),
                "best_of_four": summarize(best_per_episode(rows)),
                "by_direction": {
                    direction: summarize(
                        [row for row in rows if row["direction"] == direction]
                    )
                    for direction in ("cmn->eng", "eng->cmn")
                },
                "best_of_four_by_direction": {
                    direction: summarize(
                        [
                            row
                            for row in best_per_episode(rows)
                            if row["direction"] == direction
                        ]
                    )
                    for direction in ("cmn->eng", "eng->cmn")
                },
                "rows": rows,
            }
        )

    validations = {}
    for specification in args.training_log:
        label, path = parse_arm(specification)
        validations[label] = {
            "path": str(path.resolve()),
            "terminal": parse_terminal_validation(path),
        }

    machine = {
        "schema_version": "uniss_event_constrained_grpo_long_v2_evaluation_v1",
        "status": "passed",
        "claim_scope": "train_seen_only",
        "geometry": {"episodes": 64, "group_size": 4, "candidates_per_arm": 256},
        "arms": [{key: value for key, value in arm.items() if key != "rows"} for arm in arms],
        "training_validations": validations,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "METRICS.json").write_text(
        json.dumps(machine, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Phase A 事件约束长 Episode GRPO：64×4 统一评估",
        "",
        "## 1. 结论边界",
        "",
        "本报告严格复用同一批 64 条双向长 episode（中→英 32、英→中 32），每个 policy 生成 4 个候选，共 256 candidates。结果只说明 train-seen 方法有效性，不证明 validation 或外部泛化。`first WRITE` 是源音频时间轴上的决策时延，不是 wall-clock 服务时延；当前没有 LLM KV cache，且 TTS 同步执行，因此不能据此宣称真实 wall-clock 低于 1 秒。",
        "",
        "`all 256` 衡量随机采样 policy 的总体行为；`best-of-4` 是每条 episode 按同一 reward 选出的试听上界，不能当成单次部署性能。",
        "历史 baseline 使用旧 reward 定义，而 fresh arms 使用当前带质量保留、连续时延和 failure penalty 的 reward；因此历史行的 reward 只作原始记录，不能与 fresh arms 的 reward 数值直接排序。fresh arms 之间使用同一定义，可以横向比较。",
        "",
        "## 2. 全部 256 candidates",
        "",
        "| Policy | reward↑ | ASR相似度↑ | MT相似度↑ | 文本完整度↑ | 首次WRITE p50/p95 ms↓ | 最大静音均值 ms↓ | 译音覆盖↑ | RTF↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        metric = arm["all"]
        lines.append(
            f"| {arm['label']} | {fmt(metric['reward']['mean'])} | "
            f"{fmt(metric['asr_teacher_similarity']['mean'])} | "
            f"{fmt(metric['mt_teacher_similarity']['mean'])} | "
            f"{fmt(metric['translation_length_ratio']['mean'])} | "
            f"{fmt(metric['first_write_ms']['p50'],0)}/{fmt(metric['first_write_ms']['p95'],0)} | "
            f"{fmt(metric['maximum_internal_silence_ms']['mean'],0)} | "
            f"{fmt(metric['coverage_audio']['mean'])} | {fmt(metric['rtf']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## 3. 分方向全部 candidates",
            "",
            "| Policy | 方向 | ASR相似度↑ | MT相似度↑ | 文本完整度↑ | 首次WRITE p50/p95 ms↓ | 最大静音均值 ms↓ | 译音覆盖↑ |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in arms:
        for direction in ("cmn->eng", "eng->cmn"):
            metric = arm["by_direction"][direction]
            lines.append(
                f"| {arm['label']} | {direction} | "
                f"{fmt(metric['asr_teacher_similarity']['mean'])} | "
                f"{fmt(metric['mt_teacher_similarity']['mean'])} | "
                f"{fmt(metric['translation_length_ratio']['mean'])} | "
                f"{fmt(metric['first_write_ms']['p50'],0)}/{fmt(metric['first_write_ms']['p95'],0)} | "
                f"{fmt(metric['maximum_internal_silence_ms']['mean'],0)} | "
                f"{fmt(metric['coverage_audio']['mean'])} |"
            )
    lines.extend(
        [
            "",
            "## 4. 每条 episode 的 best-of-4",
            "",
            "| Policy | reward↑ | ASR相似度↑ | MT相似度↑ | 文本完整度↑ | 首次WRITE p50/p95 ms↓ | 最大静音均值 ms↓ | pending/TTS失败↓ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in arms:
        metric = arm["best_of_four"]
        lines.append(
            f"| {arm['label']} | {fmt(metric['reward']['mean'])} | "
            f"{fmt(metric['asr_teacher_similarity']['mean'])} | "
            f"{fmt(metric['mt_teacher_similarity']['mean'])} | "
            f"{fmt(metric['translation_length_ratio']['mean'])} | "
            f"{fmt(metric['first_write_ms']['p50'],0)}/{fmt(metric['first_write_ms']['p95'],0)} | "
            f"{fmt(metric['maximum_internal_silence_ms']['mean'],0)} | "
            f"{fmt(metric['pending']['mean'],1)}/{fmt(metric['tts_failures']['mean'],1)} |"
        )

    lines.extend(["", "## 5. 每轮双向最佳试听样本", ""])
    for arm in arms:
        lines.extend([f"### {arm['label']}", ""])
        for direction in ("cmn->eng", "eng->cmn"):
            rows = [
                row
                for row in best_per_episode(arm["rows"])
                if row["direction"] == direction
            ]
            for row in sorted(rows, key=lambda item: value(item, "reward"), reverse=True)[:4]:
                result = row["result"]
                lines.extend(
                    [
                        f"- `{row['episode_id']}` / group {row['group_index']} / {direction}：reward={fmt(value(row,'reward'))}，first WRITE={fmt(value(row,'first_write_ms'),0)} ms，MT={fmt(value(row,'mt_teacher_similarity'))}。",
                        f"  - 源音频：`{row['source_audio']}`",
                        f"  - 连续译音：`{result['continuous_audio_path']}`",
                        f"  - 全局时间轴：`{result['timeline_audio_path']}`",
                        f"  - 左源右译：`{result['stereo_audio_path']}`",
                    ]
                )
        lines.append("")

    lines.extend(["## 6. 训练终点 validation", ""])
    for label, record in validations.items():
        terminal = record["terminal"]
        if terminal is None:
            lines.append(f"- {label}：未解析到终点 validation；日志 `{record['path']}`。")
            continue
        metrics = terminal["metrics"]
        lines.append(
            f"- {label} / iter {terminal['iteration']}：total={fmt(metrics.get('loss/total',float('nan')),6)}，policy={fmt(metrics.get('loss/policy',float('nan')),6)}，KL={fmt(metrics.get('loss/reference_kl',float('nan')),6)}，control clipping={fmt(metrics.get('diagnostic/control_ratio_clipped_fraction',float('nan')),6)}；日志 `{record['path']}`。"
        )
    lines.extend(
        [
            "",
            "## 7. 选择原则",
            "",
            "只在 ASR/MT 相似度、文本与译音完整度、音频健康不明显下降时，才把更早 first WRITE、更短内部静音视为有效提升。训练 validation loss 只用于检查优化稳定性；最终试听选择必须同时查看自由运行 64×4 指标，不能仅按 loss 最低选 checkpoint。",
            "",
        ]
    )
    (args.output_dir / "REPORT.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"OUTPUT={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
