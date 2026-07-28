"""Build a CVSS-T Phase3 report aligned with UniSS Table 1."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.io_utils import write_json


PAPER_REFERENCE = Path(__file__).resolve().parents[1] / "reference_data/uniss_paper_cvss_t_table1.json"
METRIC_FILES = {
    "speech_bleu": "metrics/speech_bleu.json",
    "text_bleu": "metrics/text_bleu.json",
    "autopcp": "metrics/autopcp.json",
    "slc": "metrics/slc.json",
    "utmos": "metrics/utmos.json",
}
METRIC_COLUMNS = ("speech_bleu", "text_bleu", "autopcp", "slc_0_2", "slc_0_4", "utmos")
METRIC_LABELS = {
    "speech_bleu": "Speech-BLEU",
    "text_bleu": "Text-BLEU",
    "autopcp": "AutoPCP",
    "slc_0_2": "SLC-0.2",
    "slc_0_4": "SLC-0.4",
    "utmos": "UTMOS",
}
MODE_LABELS = {"quality": "Quality (Q)", "performance": "Performance (P)"}
DIRECTION_LABELS = {"eng->cmn": "EN→ZH", "cmn->eng": "ZH→EN"}


def read_optional(path: Path) -> object | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_run(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "summary": read_optional(path / "summary.json"),
        "generation_summary": read_optional(path / "vllm/generation_summary.json"),
        "integrity": read_optional(path / "metrics/result_integrity.json"),
        "metrics": {name: read_optional(path / relative) for name, relative in METRIC_FILES.items()},
    }


def metric_records(runs: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for run_name, run in runs.items():
        metrics = run["metrics"]
        for metric_file, report in metrics.items():  # type: ignore[union-attr]
            if not isinstance(report, Mapping):
                continue
            for group_name, values in report.get("groups", {}).items():  # type: ignore[union-attr]
                mode, direction = str(group_name).split(":", 1)
                if not isinstance(values, Mapping):
                    continue
                fields: list[tuple[str, object]]
                if metric_file in {"speech_bleu", "text_bleu"}:
                    fields = [(str(metric_file), values.get("score"))]
                elif metric_file == "slc":
                    fields = [("slc_0_2", values.get("slc_0_2")), ("slc_0_4", values.get("slc_0_4"))]
                else:
                    fields = [(str(metric_file), values.get("mean"))]
                for metric, value in fields:
                    if value is None:
                        continue
                    key = (mode, direction, metric)
                    if key in seen:
                        raise ValueError(f"Duplicate CVSS-T metric cell across runs: {key}")
                    seen.add(key)
                    records.append(
                        {
                            "run": run_name,
                            "mode": mode,
                            "direction": direction,
                            "metric": metric,
                            "value": float(value),
                            "sample_count": values.get("sample_count"),
                        }
                    )
    return records


def completeness(
    records: Sequence[Mapping[str, object]],
    *,
    expected_pairs: int,
    formal_pair_count: int = 4897,
) -> dict[str, object]:
    index = {(str(row["mode"]), str(row["direction"]), str(row["metric"])): row for row in records}
    expected = {
        (mode, direction, metric)
        for mode in MODE_LABELS
        for direction in DIRECTION_LABELS
        for metric in METRIC_COLUMNS
    }
    missing = sorted(expected - set(index))
    unexpected = sorted(set(index) - expected)
    wrong_sample_counts = []
    for key, row in sorted(index.items()):
        sample_count = row.get("sample_count")
        if sample_count is not None and int(sample_count) != expected_pairs:
            wrong_sample_counts.append({"cell": list(key), "actual": int(sample_count), "expected": expected_pairs})
    protocol_complete = not missing and not unexpected and not wrong_sample_counts
    return {
        "protocol_complete": protocol_complete,
        "formal_complete": protocol_complete and expected_pairs == formal_pair_count,
        "evaluation_scope": "formal" if expected_pairs == formal_pair_count else "smoke_or_subset",
        "formal_pair_count": formal_pair_count,
        "expected_metric_cells": len(expected),
        "observed_metric_cells": len(index),
        "missing_cells": [list(key) for key in missing],
        "unexpected_cells": [list(key) for key in unexpected],
        "wrong_sample_counts": wrong_sample_counts,
    }


def matching_paper_deltas(
    records: Sequence[Mapping[str, object]], paper: Mapping[str, object]
) -> list[dict[str, object]]:
    paper_models = {str(row["model"]): row for row in paper["models"]}  # type: ignore[index]
    output = []
    for row in records:
        paper_name = "UniSS (Q)" if row["mode"] == "quality" else "UniSS (P)"
        reference = paper_models[paper_name]["metrics"].get(row["metric"], {}).get(row["direction"])
        if reference is None:
            continue
        output.append(
            {
                **row,
                "paper_model": paper_name,
                "paper_value": float(reference),
                "delta_local_minus_paper": float(row["value"]) - float(reference),
            }
        )
    return output


def fmt(value: object) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def paper_pair(model: Mapping[str, object], metric: str) -> str:
    values = model["metrics"].get(metric, {})  # type: ignore[union-attr]
    return f"{fmt(values.get('eng->cmn'))} / {fmt(values.get('cmn->eng'))}"


def leakage_section(leakage: Mapping[str, object] | None) -> list[str]:
    if not leakage:
        return ["泄漏审计文件未提供；正式发布结果前必须补充训练集重合检查。"]
    text_counts = leakage.get("text_match_counts", {})
    dataset_counts = leakage.get("dataset_match_counts", {})
    lines = [
        f"- 训练集扫描：{leakage.get('train_shard_count', '-')} shards，{leakage.get('train_row_count', '-')} rows。",
        f"- CVSS ID 精确命中：{leakage.get('id_match_count', '-')}。",
        f"- 归一化文本命中的训练记录：{leakage.get('matched_train_record_count', '-')}。",
        f"- 文本字段分布：`{json.dumps(text_counts, ensure_ascii=False, sort_keys=True)}`。",
        f"- 训练数据来源分布：`{json.dumps(dataset_counts, ensure_ascii=False, sort_keys=True)}`。",
        f"- 音频 exact-overlap：{leakage.get('audio_exact_overlap_status', 'unknown')}。",
        "",
        "文本命中不等于音频逐字节泄漏，但它说明本地分数不能被解释为完全无重合的 held-out 泛化。"
        "正式结论必须同时报告该审计；训练 parquet 没有原始音频 hash，音频重合仍是当前协议的限制。",
    ]
    return lines


def markdown_report(
    runs: Mapping[str, Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
    status: Mapping[str, object],
    deltas: Sequence[Mapping[str, object]],
    paper: Mapping[str, object],
    leakage: Mapping[str, object] | None,
    *,
    expected_pairs: int,
) -> str:
    index = {(str(row["mode"]), str(row["direction"]), str(row["metric"])): row for row in records}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# UniSS Phase3 full198：CVSS-T zh/en Table 1 评估报告",
        "",
        f"> 生成时间：{now}",
        f"> 论文：[UniSS, arXiv:{paper['paper']['arxiv']}]({paper['paper']['url']})，{paper['paper']['table']}",
        f"> 数据：CVSS-T test，{expected_pairs} pairs，EN→ZH 与 ZH→EN，Quality/Performance 双模式",
        "",
        "## 1. 当前结论与完整性",
        "",
    ]
    if status["formal_complete"]:
        lines.append("全部 24 个正式指标单元均已生成，且每个单元的样本数与 CVSS-T test 一致。")
    elif status["protocol_complete"]:
        lines.extend(
            [
                "双方向 Q/P 的 24 个指标单元均已跑通，但当前是 **smoke/subset 功能验证**，不是正式 Table 1 结果。",
                f"当前每个单元 {expected_pairs} 条；正式 CVSS-T test 要求 {status['formal_pair_count']} 条。",
            ]
        )
    else:
        lines.extend(
            [
                "当前报告仍是 **未完成/非正式状态**，不能作为论文复现最终数值。",
                f"已生成 {status['observed_metric_cells']}/{status['expected_metric_cells']} 个指标单元；"
                f"缺失 {len(status['missing_cells'])} 个，样本数不完整 {len(status['wrong_sample_counts'])} 个。",
            ]
        )
    lines.extend(
        [
            "",
            "本地自动评估不包含论文主观 MOS。MOS 需要六名双语评分者和 webMUSHRA，不能用 UTMOS 替代。",
            "",
            "## 2. 本地 CVSS-T 客观指标",
            "",
            "所有指标均为 higher-is-better。方向顺序与论文一致；EN→ZH 的英文输入是 CVSS-T 合成语音，"
            "ZH→EN 的中文输入是真实 Common Voice v4 语音。",
            "",
            "| Mode | 方向 | Speech-BLEU | Text-BLEU | AutoPCP | SLC-0.2 | SLC-0.4 | UTMOS | N |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode in ("performance", "quality"):
        for direction in ("eng->cmn", "cmn->eng"):
            values = [index.get((mode, direction, metric)) for metric in METRIC_COLUMNS]
            counts = [row.get("sample_count") for row in values if row and row.get("sample_count") is not None]
            count_text = str(min(int(value) for value in counts)) if counts else "-"
            lines.append(
                f"| {MODE_LABELS[mode]} | {DIRECTION_LABELS[direction]} | "
                + " | ".join(fmt(row.get("value") if row else None) for row in values)
                + f" | {count_text} |"
            )

    lines.extend(
        [
            "",
            "## 3. 与原论文 UniSS P/Q 的同协议差值",
            "",
            "仅在同一 CVSS-T test、同方向、同 mode、同指标下计算 Δ(本地−论文)。"
            "解码 seed、checkpoint 大小/训练数据和 metric 软件版本仍会带来差异。",
            "",
            "| Mode | 方向 | 指标 | 本地 | 论文 | Δ | N |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    if not deltas:
        lines.append("| - | - | 尚无可比本地指标 | - | - | - | - |")
    for row in sorted(deltas, key=lambda item: (str(item["mode"]), str(item["direction"]), str(item["metric"]))):
        lines.append(
            f"| {MODE_LABELS.get(str(row['mode']), row['mode'])} | "
            f"{DIRECTION_LABELS.get(str(row['direction']), row['direction'])} | "
            f"{METRIC_LABELS.get(str(row['metric']), row['metric'])} | {fmt(row['value'])} | "
            f"{fmt(row['paper_value'])} | {float(row['delta_local_minus_paper']):+.4f} | "
            f"{row.get('sample_count', '-')} |"
        )

    lines.extend(
        [
            "",
            "## 4. 原论文 Table 1 完整基线",
            "",
            "表中每格为 EN→ZH / ZH→EN。",
            "",
            "| 类别 | 方法 | 参数量 | Speech-BLEU | Text-BLEU | AutoPCP | SLC-0.2 | SLC-0.4 | UTMOS |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in paper["models"]:  # type: ignore[index]
        lines.append(
            f"| {model['category']} | {model['model']} | {model['size']} | "
            f"{paper_pair(model, 'speech_bleu')} | {paper_pair(model, 'text_bleu')} | "
            f"{paper_pair(model, 'autopcp')} | {paper_pair(model, 'slc_0_2')} | "
            f"{paper_pair(model, 'slc_0_4')} | {paper_pair(model, 'utmos')} |"
        )

    lines.extend(["", "## 5. 数据泄漏审计", "", *leakage_section(leakage), ""])
    lines.extend(
        [
            "## 6. 协议与可解释性边界",
            "",
            "- 解码参数：temperature 0.7、top-p 0.8、top-k -1、repetition penalty 1.1；Q/P 分开生成。",
            "- Text-BLEU 使用模型生成翻译文本；Speech-BLEU 使用生成语音经目标语言 ASR 后的文本。",
            "- 英文 ASR 为 Whisper-large-v3；中文 ASR 为 Paraformer-zh；中文统一简体化并按字符 BLEU。",
            "- AutoPCP 比较官方源语音与生成语音的跨语言韵律；SLC 比较生成时长与输入时长；UTMOS 是预测音质。",
            "- source/reference 均保留 canonical 官方 WAV；只对模型 semantic tokens 解码生成 WAV，避免 BiCodec 重建 reference 污染指标。",
            "- SimulS2ST-Omni 的 greedy unified re-score 是另一套协议，不能与本报告的 UniSS 采样协议混成一张主表。",
            "",
            "## 7. 运行完整性与产物路径",
            "",
            "| Run | decoded | failed | generated | no semantic | integrity | path |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for name, run in sorted(runs.items()):
        summary = run.get("summary") if isinstance(run.get("summary"), Mapping) else {}
        generation = run.get("generation_summary") if isinstance(run.get("generation_summary"), Mapping) else {}
        integrity = run.get("integrity") if isinstance(run.get("integrity"), Mapping) else {}
        lines.append(
            f"| {name} | {summary.get('decoded', '-')} | {summary.get('failed', '-')} | "
            f"{generation.get('total_results', generation.get('generated', '-'))} | "
            f"{generation.get('no_semantic_tokens', '-')} | {integrity.get('valid', '-')} | `{run['path']}` |"
        )
    if status["missing_cells"]:
        lines.extend(["", "缺失指标单元：", "", f"`{json.dumps(status['missing_cells'], ensure_ascii=False)}`"])
    if status["wrong_sample_counts"]:
        lines.extend(["", "样本数不完整单元：", "", f"`{json.dumps(status['wrong_sample_counts'], ensure_ascii=False)}`"])
    return "\n".join(lines) + "\n"


def build_report(args: argparse.Namespace) -> dict[str, object]:
    runs = {path.name: collect_run(path) for path in args.run}
    records = metric_records(runs)
    paper = json.loads(args.paper_reference.read_text(encoding="utf-8"))
    leakage = read_optional(args.leakage_audit) if args.leakage_audit else None
    if leakage is not None and not isinstance(leakage, Mapping):
        raise TypeError("leakage audit must be a JSON object")
    status = completeness(records, expected_pairs=args.expected_pairs)
    deltas = matching_paper_deltas(records, paper)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expected_pairs": args.expected_pairs,
        "status": status,
        "runs": runs,
        "metric_records": records,
        "paper_deltas": deltas,
        "leakage_audit": leakage,
        "paper_reference": paper,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "cvss_t_phase3_table1_report.json", payload)
    markdown = markdown_report(
        runs,
        records,
        status,
        deltas,
        paper,
        leakage,
        expected_pairs=args.expected_pairs,
    )
    (args.output_dir / "cvss_t_phase3_table1_report.md").write_text(markdown, encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, default=4897)
    parser.add_argument("--paper-reference", type=Path, default=PAPER_REFERENCE)
    parser.add_argument("--leakage-audit", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    payload = build_report(parse_args(argv))
    print(json.dumps(payload["status"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
