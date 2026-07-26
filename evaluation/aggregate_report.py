"""Build a detailed Phase2/Phase3 evaluation report with guarded paper comparison."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.io_utils import write_json


METRIC_FILES = {
    "text_bleu": "metrics/text_bleu.json",
    "speech_bleu": "metrics/speech_bleu.json",
    "slc": "metrics/slc.json",
    "utmos": "metrics/utmos.json",
    "autopcp": "metrics/autopcp.json",
}
PAPER_REFERENCE = Path(__file__).parent / "reference_data/uniss_paper_cvss_t_table1.json"
METRIC_LABELS = {
    "text_bleu": "Text-BLEU",
    "speech_bleu": "Speech-BLEU",
    "autopcp": "AutoPCP (A.PCP)",
    "slc_0_2": "SLC-0.2",
    "slc_0_4": "SLC-0.4",
    "utmos": "UTMOS",
}
DIRECTION_LABELS = {"eng->cmn": "EN→ZH", "cmn->eng": "ZH→EN"}
MODE_LABELS = {"quality": "Quality (Q)", "performance": "Performance (P)"}


def read_optional(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def classify_run(path: Path, run_config: Mapping[str, object] | None) -> dict[str, str]:
    name = path.name.lower()
    manifest = str((run_config or {}).get("manifest", "")).lower()
    combined = f"{name} {manifest}"
    stage = "phase2" if "phase2" in combined else "phase3" if "phase3" in combined else "unknown"
    dataset = "cvss_t" if "cvss" in combined else "unist" if "unist" in combined else "unknown"
    split = "test" if "test" in combined else "dev" if "dev" in combined else "unknown"
    if "vllm_smoke" in combined:
        scope = "vllm_smoke"
    elif "smoke" in combined:
        scope = "smoke"
    elif "listen" in combined:
        scope = "listen"
    elif "full" in combined:
        scope = "full"
    else:
        scope = "unknown"
    return {"stage": stage, "dataset": dataset, "split": split, "scope": scope}


def collect_run(path: Path) -> dict[str, object]:
    run_config = read_optional(path / "run_config.json") or read_optional(path / "vllm/run_config.json")
    return {
        "path": str(path.resolve()),
        "metadata": classify_run(path, run_config),
        "run_config": run_config,
        "summary": read_optional(path / "summary.json"),
        "generation_summary": read_optional(path / "vllm/generation_summary.json"),
        "metrics": {name: read_optional(path / relative) for name, relative in METRIC_FILES.items()},
    }


def metric_records(name: str, run: Mapping[str, object]) -> list[dict[str, object]]:
    metadata = run["metadata"]
    records: list[dict[str, object]] = []
    for metric_name, report in run["metrics"].items():  # type: ignore[union-attr]
        if not report:
            continue
        for group, values in report.get("groups", {}).items():
            mode, direction = group.split(":", 1)
            fields = []
            if metric_name in {"text_bleu", "speech_bleu"}:
                fields = [(metric_name, values.get("score"))]
            elif metric_name == "slc":
                fields = [("slc_0_2", values.get("slc_0_2")), ("slc_0_4", values.get("slc_0_4"))]
            elif metric_name == "utmos":
                fields = [("utmos", values.get("mean"))]
            elif metric_name == "autopcp":
                fields = [("autopcp", values.get("mean"))]
            for metric, value in fields:
                if value is None:
                    continue
                records.append(
                    {
                        "run": name,
                        **metadata,  # type: ignore[arg-type]
                        "mode": mode,
                        "direction": direction,
                        "metric": metric,
                        "value": float(value),
                        "sample_count": values.get("sample_count"),
                    }
                )
    return records


def build_comparisons(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    primary = [row for row in records if row["scope"] == "full" and row["dataset"] in {"unist", "cvss_t"}]
    index = {
        (row["dataset"], row["split"], row["mode"], row["direction"], row["metric"], row["stage"]): row
        for row in primary
    }
    comparisons = []
    keys = sorted({key[:-1] for key in index})
    for key in keys:
        phase2 = index.get((*key, "phase2"))
        phase3 = index.get((*key, "phase3"))
        if not phase2 or not phase3:
            continue
        comparisons.append(
            {
                "dataset": key[0],
                "split": key[1],
                "mode": key[2],
                "direction": key[3],
                "metric": key[4],
                "phase2": phase2["value"],
                "phase3": phase3["value"],
                "delta_phase3_minus_phase2": float(phase3["value"]) - float(phase2["value"]),
                "phase2_sample_count": phase2["sample_count"],
                "phase3_sample_count": phase3["sample_count"],
            }
        )
    return comparisons


def paper_comparability(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    cvss_rows = [row for row in records if row["dataset"] == "cvss_t" and row["split"] == "test" and row["scope"] == "full"]
    directions = sorted({str(row["direction"]) for row in cvss_rows})
    return {
        "direct_numeric_comparison_allowed": bool(cvss_rows),
        "reason": (
            "Detected full CVSS-T test metrics; compare only matching direction, mode, and metric."
            if cvss_rows
            else "Current full runs use UniST dev/test, while the paper Table 1 uses CVSS-T test; numeric deltas and rankings are invalid."
        ),
        "available_cvss_t_directions": directions,
    }


def build_paper_comparisons(
    records: Sequence[Mapping[str, object]], paper: Mapping[str, object]
) -> list[dict[str, object]]:
    selected = {"3-Stage", "2-Stage", "Seamless-L", "Seamless-Ex", "UniSS (P)", "UniSS (Q)"}
    paper_models = {model["model"]: model for model in paper["models"] if model["model"] in selected}  # type: ignore[index]
    comparisons = []
    for row in records:
        if row["dataset"] != "cvss_t" or row["split"] != "test" or row["scope"] != "full":
            continue
        for model_name, model in paper_models.items():
            if model_name == "UniSS (P)" and row["mode"] != "performance":
                continue
            if model_name == "UniSS (Q)" and row["mode"] != "quality":
                continue
            reference = model["metrics"].get(row["metric"], {}).get(row["direction"])
            if reference is None:
                continue
            comparisons.append(
                {
                    "stage": row["stage"],
                    "mode": row["mode"],
                    "direction": row["direction"],
                    "metric": row["metric"],
                    "local_value": row["value"],
                    "paper_model": model_name,
                    "paper_value": reference,
                    "delta_local_minus_paper": float(row["value"]) - float(reference),
                }
            )
    return comparisons


def fmt(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def paper_pair(model: Mapping[str, object], metric: str) -> str:
    values = model["metrics"].get(metric, {})  # type: ignore[union-attr]
    return f"{fmt(values.get('eng->cmn'))} | {fmt(values.get('cmn->eng'))}"


def integrity_lines(runs: Mapping[str, Mapping[str, object]]) -> list[str]:
    lines = ["| Run | Stage | Dataset/split | Scope | Decoded/total | Failed | No semantic | Dummy tokens |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for name, run in sorted(runs.items()):
        meta = run["metadata"]
        summary = run.get("summary") or {}
        generation = run.get("generation_summary") or {}
        decoded = summary.get("decoded", summary.get("total", "-"))
        failed = summary.get("failed", "-")
        lines.append(
            f"| {name} | {meta['stage']} | {meta['dataset']}/{meta['split']} | {meta['scope']} | "
            f"{decoded} | {failed} | {generation.get('no_semantic_tokens', '-')} | "
            f"{generation.get('dummy_generated_tokens', '-')} |"
        )
    return lines


def markdown_report(
    runs: Mapping[str, Mapping[str, object]],
    comparisons: Sequence[Mapping[str, object]],
    paper: Mapping[str, object],
    comparability: Mapping[str, object],
    paper_comparisons: Sequence[Mapping[str, object]] = (),
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# UniSS full198 Phase2 / Phase3 详细评估报告",
        "",
        f"> 生成时间：{now}",
        f"> 论文参考：[arXiv:{paper['paper']['arxiv']}]({paper['paper']['url']})，{paper['paper']['table']}",
        "",
        "## 1. 结论与比较边界",
        "",
    ]
    if comparability["direct_numeric_comparison_allowed"]:
        lines.append("本报告检测到完整 CVSS-T test 结果；只对相同方向、mode 和指标进行论文数值对比。")
    else:
        lines.extend(
            [
                "当前 Phase2/Phase3 全量结果来自 **UniST dev/test**，论文 Table 1 来自 **CVSS-T test 4,897 对**。",
                "因此本报告只做 Phase2 与 Phase3 的同数据内部比较；论文表格作为参考背景展示，**不计算跨数据集差值、胜负或排名**。",
            ]
        )
    lines.extend(["", "## 2. Phase2 与 Phase3 全量指标对比", ""])
    if not comparisons:
        lines.append("尚未发现同时完成的 Phase2/Phase3 full 指标。")
    else:
        for dataset, split in sorted({(str(row["dataset"]), str(row["split"])) for row in comparisons}):
            lines.extend(
                [
                    f"### {dataset} {split}",
                    "",
                    "| 指标 | Mode | 方向 | Phase2 | Phase3 | Δ(Phase3-Phase2) | N2/N3 |",
                    "| --- | --- | --- | ---: | ---: | ---: | ---: |",
                ]
            )
            subset = [row for row in comparisons if row["dataset"] == dataset and row["split"] == split]
            for row in sorted(subset, key=lambda item: (str(item["metric"]), str(item["mode"]), str(item["direction"]))):
                lines.append(
                    f"| {METRIC_LABELS.get(str(row['metric']), row['metric'])} | "
                    f"{MODE_LABELS.get(str(row['mode']), row['mode'])} | "
                    f"{DIRECTION_LABELS.get(str(row['direction']), row['direction'])} | "
                    f"{fmt(row['phase2'])} | {fmt(row['phase3'])} | "
                    f"{float(row['delta_phase3_minus_phase2']):+.4f} | "
                    f"{row['phase2_sample_count']}/{row['phase3_sample_count']} |"
                )
            positive = sum(float(row["delta_phase3_minus_phase2"]) > 0 for row in subset)
            negative = sum(float(row["delta_phase3_minus_phase2"]) < 0 for row in subset)
            equal = len(subset) - positive - negative
            lines.extend(
                [
                    "",
                    f"在全部 higher-is-better 指标单元中：Phase3 上升 {positive} 项，下降 {negative} 项，持平 {equal} 项。",
                    "该计数用于定位变化，不替代按任务重要性、置信区间和人工试听做模型选择。",
                    "",
                ]
            )

    lines.extend(["## 3. 生成完整性与失败审计", "", *integrity_lines(runs), ""])
    lines.extend(
        [
            "## 4. 原论文 CVSS-T Table 1 基线参考",
            "",
            "下表严格抄录论文的 EN→ZH | ZH→EN 口径。只有本地结果同样来自 CVSS-T test 时才可直接比较。",
            "",
            "| 类别 | 方法 | 参数量 | Speech-BLEU | Text-BLEU | A.PCP | SLC-0.2 | SLC-0.4 | UTMOS |",
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
    if paper_comparisons:
        lines.extend(
            [
                "",
                "### 同测试集本地结果与论文方法的直接差值",
                "",
                "| 本地Stage/Mode | 指标 | 方向 | 本地值 | 论文方法 | 论文值 | Δ(本地-论文) |",
                "| --- | --- | --- | ---: | --- | ---: | ---: |",
            ]
        )
        for row in sorted(
            paper_comparisons,
            key=lambda item: (str(item["stage"]), str(item["mode"]), str(item["metric"]), str(item["direction"]), str(item["paper_model"])),
        ):
            lines.append(
                f"| {row['stage']}/{MODE_LABELS.get(str(row['mode']), row['mode'])} | "
                f"{METRIC_LABELS.get(str(row['metric']), row['metric'])} | "
                f"{DIRECTION_LABELS.get(str(row['direction']), row['direction'])} | "
                f"{fmt(row['local_value'])} | {row['paper_model']} | {fmt(row['paper_value'])} | "
                f"{float(row['delta_local_minus_paper']):+.4f} |"
            )
    small = paper["small_model_efficiency_reference"]  # type: ignore[index]
    lines.extend(
        [
            "",
            "### 论文0.5B模型效率参考",
            "",
            f"论文 Table 3 使用 {small['dataset']}，硬件/推理条件为 {small['hardware']}，与本地批量vLLM结果也不能直接比较速度。",
            "",
            "| 模型 | 参数量 | 平均Speech-BLEU | 时间(s) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for model in small["models"]:
        lines.append(f"| {model['model']} | {model['size']} | {model['average_speech_bleu']:.2f} | {model['time_seconds']:.2f} |")

    lines.extend(
        [
            "",
            "## 5. 分析原则",
            "",
            "- Text-BLEU 与 Speech-BLEU 分开解读：前者评估模型生成的翻译文本，后者还包含语音解码和ASR误差。",
            "- AutoPCP、SLC 和 UTMOS 分别反映韵律、时长一致性和预测音质，任何单项都不能替代试听。",
            "- UniST source/reference WAV 是BiCodec token重建音频，不是原始数据集波形；论文CVSS-T使用真实配对WAV。",
            "- Phase2/Phase3使用相同manifest、seed、Q/P参数和指标模型，内部差值具有可解释性。",
            "- 随机采样生成仍可能有方差；最终checkpoint选择建议结合多seed子集和人工盲听。",
            "",
            "## 6. 结果与试听目录",
            "",
        ]
    )
    for name, run in sorted(runs.items()):
        lines.append(f"- `{name}`: `{run['path']}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-reference", type=Path, default=PAPER_REFERENCE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {path.name: collect_run(path) for path in args.run}
    records = [record for name, run in runs.items() for record in metric_records(name, run)]
    comparisons = build_comparisons(records)
    comparability = paper_comparability(records)
    paper = json.loads(args.paper_reference.read_text(encoding="utf-8"))
    paper_comparisons = build_paper_comparisons(records, paper)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "metric_records": records,
        "phase2_phase3_comparisons": comparisons,
        "paper_comparability": comparability,
        "paper_comparisons": paper_comparisons,
        "paper_reference": paper,
    }
    write_json(args.output_dir / "aggregate_report.json", payload)
    markdown = markdown_report(runs, comparisons, paper, comparability, paper_comparisons)
    (args.output_dir / "aggregate_report.md").write_text(markdown, encoding="utf-8")
    (args.output_dir / "phase2_phase3_detailed_evaluation_report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"runs": list(runs), "comparisons": len(comparisons)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
