"""Build the Reward-v2 dev-selected four-way full-test report and continuity chapter."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

LABELS = {
    "r0_e3_v1_bias": "R0 E3-v1 + WRITE bias",
    "r1_rebalanced_coverage": "R1 rebalanced + coverage",
    "r2_explicit_latency": "R2 explicit latency",
    "r3_bilingual_adaptive": "R3 bilingual + adaptive KL",
}
DESCRIPTIONS = {
    "r0_e3_v1_bias": "Frozen E3-v1 policy with dev-selected WRITE bias +0.50",
    "r1_rebalanced_coverage": "Reward-v2 action costs plus coverage/final-flush terms",
    "r2_explicit_latency": "R1 plus explicit first-WRITE/ATD/LAAL-style latency deltas",
    "r3_bilingual_adaptive": "R2 plus bilingual balance and adaptive KL",
}
DIRECTIONS = ("cmn->eng", "eng->cmn")
START_MARKER = "<!-- REWARD_V2_FULL_TEST_CHAPTER_START -->"
END_MARKER = "<!-- REWARD_V2_FULL_TEST_CHAPTER_END -->"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_by_direction(metric: dict[str, Any], field: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, values in metric.get("groups", {}).items():
        direction = key.rsplit(":", 1)[-1]
        if field in values:
            result[direction] = float(values[field])
    return result


def extract(run_dir: Path) -> dict[str, Any]:
    aggregate = read_json(run_dir / "aggregate_metrics.json")
    latency_path = run_dir / "latency_batch1" / "aggregate_metrics.json"
    latency = read_json(latency_path) if latency_path.is_file() else aggregate
    common = aggregate["common_metrics"]
    means = aggregate["streaming_metrics"]["overall"]["means"]
    latency_means = latency["streaming_metrics"]["overall"]["means"]
    manifest_path = run_dir / "environment" / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    commit_path = run_dir / "environment" / "git_commit.txt"
    git_commit = commit_path.read_text(encoding="utf-8").strip() if commit_path.is_file() else None
    slc = common.get("slc", {})
    return {
        "run_dir": str(run_dir.resolve()),
        "manifest": manifest,
        "git_commit": git_commit,
        "samples": int(aggregate["streaming_metrics"]["overall"]["samples"]),
        "text_bleu": group_by_direction(common["text_bleu"], "score"),
        "speech_bleu": group_by_direction(common["speech_bleu"], "score"),
        "utmos": group_by_direction(common["utmos"], "mean"),
        "autopcp": group_by_direction(common["autopcp"], "mean"),
        "slc_0_2": group_by_direction(slc, "slc_0_2"),
        "slc_0_4": group_by_direction(slc, "slc_0_4"),
        "streaming": {
            name: means.get(name)
            for name in (
                "first_write_ms_proxy",
                "start_offset_nca_ms",
                "start_offset_ca_ms",
                "atd_ms_proxy",
                "al_glm_tokens_proxy",
                "laal_glm_tokens_proxy",
                "dal_glm_tokens_proxy",
                "ap_proxy",
                "premature_write_given_wait",
                "unnecessary_wait_given_write",
                "write_f1",
                "final_flush_success",
                "forced_actions",
                "audio_chunks",
                "structural_recoveries",
                "rtf_source_audio",
                "rtf_generated_audio",
                "nonempty_text",
                "nonempty_semantic",
            )
        },
        "latency_batch1": {
            name: latency_means.get(name)
            for name in (
                "action_ttft_seconds_mean",
                "write_ttft_seconds_mean",
                "rtf_source_audio",
                "rtf_generated_audio",
                "first_write_ms_proxy",
                "start_offset_nca_ms",
                "atd_ms_proxy",
            )
        },
        "gpu": aggregate.get("gpu_monitor", {}),
        "offline_comparisons": aggregate.get("offline_comparisons", []),
    }


def fmt(value: Any, digits: int = 3) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    return "N/A" if value is None else f"{value * 100:+.{digits}f}%"


def delta(value: float | None, reference: float | None, digits: int = 3) -> str:
    if value is None or reference is None:
        return "N/A"
    return f"{float(value) - float(reference):+.{digits}f}"


def reduction(value: float | None, reference: float | None) -> float | None:
    if value is None or reference in (None, 0):
        return None
    return (float(reference) - float(value)) / abs(float(reference))


def mean_metric(values: dict[str, float]) -> float | None:
    available = [float(values[item]) for item in DIRECTIONS if item in values]
    return statistics.fmean(available) if available else None


def pass_text(value: bool) -> str:
    return "PASS" if value else "FAIL"


def table(headers: list[str], rows: list[list[str]], aligns: list[str] | None = None) -> str:
    if aligns is None:
        aligns = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_sections(results: dict[str, dict[str, Any]], dev: dict[str, Any] | None) -> list[tuple[str, str]]:
    comparison_root = Path(results["r0_e3_v1_bias"]["run_dir"]).parents[1]
    design_rows = []
    quality_rows = []
    slc_rows = []
    stream_rows = []
    latency_rows = []
    gpu_rows = []
    for label, title in LABELS.items():
        value = results[label]
        manifest = value["manifest"]
        commit = value["git_commit"]
        design_rows.append(
            [
                title,
                DESCRIPTIONS[label],
                str(manifest.get("best_step", "N/A")),
                fmt(manifest.get("write_logit_bias"), 2),
                str(manifest.get("gpus", "N/A")),
                f"`{commit[:12] if commit else 'N/A'}`",
                f"`{manifest.get('model', 'N/A')}`",
            ]
        )
        quality_rows.append(
            [
                title,
                fmt(value["text_bleu"].get("cmn->eng")),
                fmt(value["text_bleu"].get("eng->cmn")),
                fmt(value["speech_bleu"].get("cmn->eng")),
                fmt(value["speech_bleu"].get("eng->cmn")),
                fmt(value["utmos"].get("cmn->eng")),
                fmt(value["utmos"].get("eng->cmn")),
                fmt(value["autopcp"].get("cmn->eng")),
                fmt(value["autopcp"].get("eng->cmn")),
            ]
        )
        slc_rows.append(
            [
                title,
                fmt(value["slc_0_2"].get("cmn->eng")),
                fmt(value["slc_0_2"].get("eng->cmn")),
                fmt(value["slc_0_4"].get("cmn->eng")),
                fmt(value["slc_0_4"].get("eng->cmn")),
            ]
        )
        stream = value["streaming"]
        stream_rows.append(
            [
                title,
                fmt(stream["first_write_ms_proxy"], 1),
                fmt(stream["start_offset_nca_ms"], 1),
                fmt(stream["atd_ms_proxy"], 1),
                fmt(stream["laal_glm_tokens_proxy"], 2),
                fmt(stream["write_f1"]),
                fmt(stream["premature_write_given_wait"]),
                fmt(stream["unnecessary_wait_given_write"]),
                fmt(stream["final_flush_success"]),
                fmt(stream["forced_actions"]),
                fmt(stream["rtf_source_audio"]),
            ]
        )
        latency = value["latency_batch1"]
        latency_rows.append(
            [
                title,
                fmt(latency["action_ttft_seconds_mean"]),
                fmt(latency["write_ttft_seconds_mean"]),
                fmt(latency["rtf_source_audio"]),
                fmt(latency["first_write_ms_proxy"], 1),
                fmt(latency["atd_ms_proxy"], 1),
            ]
        )
        gpu = value["gpu"]
        gpu_rows.append(
            [
                title,
                f"{fmt(gpu.get('utilization_mean'), 1)}%",
                f"{fmt(gpu.get('utilization_p95'), 1)}%",
                f"{fmt(gpu.get('power_mean_w'), 1)} W",
                f"{fmt(gpu.get('power_p95_w'), 1)} W",
            ]
        )

    r1 = results["r1_rebalanced_coverage"]
    direct_rows = []
    metrics = (
        ("Text BLEU zh→en", "text_bleu", "cmn->eng", "higher"),
        ("Text BLEU en→zh", "text_bleu", "eng->cmn", "higher"),
        ("Speech BLEU zh→en", "speech_bleu", "cmn->eng", "higher"),
        ("Speech BLEU en→zh", "speech_bleu", "eng->cmn", "higher"),
        ("UTMOS mean", "utmos", None, "higher"),
        ("AutoPCP mean", "autopcp", None, "higher"),
        ("First WRITE ms", "streaming", "first_write_ms_proxy", "lower"),
        ("ATD ms", "streaming", "atd_ms_proxy", "lower"),
        ("LAAL proxy", "streaming", "laal_glm_tokens_proxy", "lower"),
        ("Premature WRITE", "streaming", "premature_write_given_wait", "lower"),
        ("Unnecessary WAIT", "streaming", "unnecessary_wait_given_write", "lower"),
        ("Final flush", "streaming", "final_flush_success", "higher"),
        ("Batch-one source RTF", "latency_batch1", "rtf_source_audio", "lower"),
    )
    for title, family, key, preference in metrics:
        reference_family = r1[family]
        reference = mean_metric(reference_family) if key is None else reference_family.get(key)
        row = [title, "↑" if preference == "higher" else "↓"]
        for label in ("r0_e3_v1_bias", "r2_explicit_latency", "r3_bilingual_adaptive"):
            candidate_family = results[label][family]
            candidate = mean_metric(candidate_family) if key is None else candidate_family.get(key)
            row.append(delta(candidate, reference))
        direct_rows.append(row)

    gate_rows = []
    gate_payload: dict[str, dict[str, bool]] = {}
    for label in ("r0_e3_v1_bias", "r2_explicit_latency", "r3_bilingual_adaptive"):
        value = results[label]
        first_reduction = reduction(
            value["streaming"]["first_write_ms_proxy"], r1["streaming"]["first_write_ms_proxy"]
        )
        atd_reduction = reduction(value["streaming"]["atd_ms_proxy"], r1["streaming"]["atd_ms_proxy"])
        laal_reduction = reduction(
            value["streaming"]["laal_glm_tokens_proxy"], r1["streaming"]["laal_glm_tokens_proxy"]
        )
        text_deltas = [
            value["text_bleu"].get(direction, float("-inf"))
            - r1["text_bleu"].get(direction, float("inf"))
            for direction in DIRECTIONS
        ]
        premature_delta = (
            value["streaming"]["premature_write_given_wait"]
            - r1["streaming"]["premature_write_given_wait"]
        )
        gates = {
            "first": first_reduction is not None and first_reduction >= 0.05,
            "atd": atd_reduction is not None and atd_reduction >= 0.05,
            "laal": laal_reduction is not None and laal_reduction >= 0.05,
            "quality": min(text_deltas) >= -0.5,
            "premature": premature_delta <= 0.01,
            "unnecessary": (
                value["streaming"]["unnecessary_wait_given_write"]
                <= min(0.12, r1["streaming"]["unnecessary_wait_given_write"] * 0.8)
            ),
            "final": value["streaming"]["final_flush_success"] >= 0.999,
            "rtf": value["latency_batch1"]["rtf_source_audio"] < 1.0,
        }
        gate_payload[label] = gates
        gate_rows.append(
            [
                LABELS[label],
                f"{pass_text(gates['first'])} ({fmt_pct(first_reduction)})",
                f"{pass_text(gates['atd'])} ({fmt_pct(atd_reduction)})",
                f"{pass_text(gates['laal'])} ({fmt_pct(laal_reduction)})",
                f"{pass_text(gates['quality'])} (worst {min(text_deltas):+.3f})",
                f"{pass_text(gates['premature'])} ({premature_delta:+.3f})",
                pass_text(gates["unnecessary"]),
                pass_text(gates["final"]),
                pass_text(gates["rtf"]),
            ]
        )

    dev_rows: list[list[str]] = []
    if dev:
        dev_results = dev.get("results", {})
        for label, title in LABELS.items():
            if label not in dev_results:
                continue
            dev_value = dev_results[label]
            test_value = results[label]
            dev_rows.append(
                [
                    title,
                    fmt(dev_value.get("text_bleu", {}).get("cmn->eng")),
                    fmt(test_value["text_bleu"].get("cmn->eng")),
                    fmt(dev_value.get("text_bleu", {}).get("eng->cmn")),
                    fmt(test_value["text_bleu"].get("eng->cmn")),
                    fmt(dev_value.get("streaming", {}).get("first_write_ms_proxy"), 1),
                    fmt(test_value["streaming"].get("first_write_ms_proxy"), 1),
                    fmt(dev_value.get("streaming", {}).get("atd_ms_proxy"), 1),
                    fmt(test_value["streaming"].get("atd_ms_proxy"), 1),
                ]
            )

    offline_rows: list[list[str]] = []
    for label, title in LABELS.items():
        for item in results[label]["offline_comparisons"]:
            if item.get("offline_mode") != "quality":
                continue
            if item.get("metric") not in {"text_bleu", "speech_bleu", "utmos", "autopcp"}:
                continue
            offline_rows.append(
                [
                    title,
                    str(item.get("direction")),
                    str(item.get("metric")),
                    fmt(item.get("streaming_value")),
                    fmt(item.get("offline_value")),
                    fmt(item.get("delta_streaming_minus_offline")),
                ]
            )

    successful = [
        LABELS[label]
        for label in ("r2_explicit_latency", "r3_bilingual_adaptive")
        if all(gate_payload[label][key] for key in ("first", "atd", "quality", "premature", "final", "rtf"))
    ]
    conclusions = [
        "- R1 是 Reward-v2 matched control；R2/R3 必须在双向质量保持条件下同时降低 First WRITE 与 ATD，才能证明显式 latency reward 有独立贡献。",
        "- R0 只回答 E3-v1 是否存在可用的 operating-point 偏置，不应与新训练 reward 混为同一种因果解释。",
        "- test operating point、checkpoint 与 WRITE bias 均已由 dev 冻结；本章没有根据 test 重新调参。",
    ]
    if successful:
        conclusions.append("- 自动 pilot gate 通过：" + "、".join(successful) + "。仍需 paired bootstrap 与多随机种子后才能声明统计显著。")
    else:
        conclusions.append("- 自动 pilot gate：当前没有 R2/R3 同时通过质量、安全、First WRITE、ATD、final flush 与 batch-one RTF 门槛。")

    sections = [
        (
            "实验设计与数据边界",
            table(
                ["Experiment", "Role", "Best step", "WRITE bias", "GPUs", "Eval commit", "Exported model"],
                design_rows,
                ["---", "---", "---:", "---:", "---", "---", "---"],
            )
            + "\n\n四组使用同一组23,369条UniST test schedules、greedy generation、BiCodec decode和指标实现。dev只用于选择，test只用于冻结后的确认。",
        ),
        (
            "翻译、语音与韵律质量",
            table(
                ["Experiment", "Text BLEU zh→en", "Text BLEU en→zh", "Speech BLEU zh→en", "Speech BLEU en→zh", "UTMOS zh→en", "UTMOS en→zh", "AutoPCP zh→en", "AutoPCP en→zh"],
                quality_rows,
                ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
            )
            + "\n\n"
            + table(
                ["Experiment", "SLC-0.2 zh→en", "SLC-0.2 en→zh", "SLC-0.4 zh→en", "SLC-0.4 en→zh"],
                slc_rows,
                ["---", "---:", "---:", "---:", "---:"],
            ),
        ),
        (
            "Streaming policy与端到端延迟",
            table(
                ["Experiment", "First WRITE ms", "StartOffset NCA ms", "ATD ms", "LAAL proxy", "WRITE F1", "Premature", "Unnecessary WAIT", "Final flush", "Forced actions", "Source RTF"],
                stream_rows,
                ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
            )
            + "\n\n带`proxy`的指标仍基于当前pseudo alignment/capacity gate；First WRITE、ATD、LAAL和Unnecessary WAIT越低越好。",
        ),
        (
            "Batch-one可部署延迟",
            table(
                ["Experiment", "Action TTFT s", "WRITE TTFT s", "Source RTF", "First WRITE ms", "ATD ms"],
                latency_rows,
                ["---", "---:", "---:", "---:", "---:", "---:"],
            )
            + "\n\nbatch-one固定每组200条，避免用大batch吞吐冒充真实单请求延迟。",
        ),
        (
            "GPU利用率与功率",
            table(
                ["Experiment", "Util mean", "Util p95", "Power mean", "Power p95"],
                gpu_rows,
                ["---", "---:", "---:", "---:", "---:"],
            )
            + "\n\nGPU数据只用于验证真实工作负载吞吐，没有使用dummy computation。",
        ),
        (
            "相对R1 matched control的直接差值",
            table(
                ["Metric", "Better", "R0−R1", "R2−R1", "R3−R1"],
                direct_rows,
                ["---", ":---:", "---:", "---:", "---:"],
            ),
        ),
        (
            "Reward-v2 pilot gate",
            table(
                ["Experiment", "First WRITE", "ATD", "LAAL", "Text BLEU retention", "Premature Δ", "Unnecessary WAIT", "Final flush", "RTF<1"],
                gate_rows,
                ["---", "---", "---", "---", "---", "---", "---", "---", "---"],
            )
            + "\n\n门槛相对R1：First WRITE/ATD/LAAL至少降低5%，双向Text BLEU最差下降不超过0.5，premature增加不超过0.01。",
        ),
    ]
    if dev_rows:
        sections.append(
            (
                "Dev到test的一致性",
                table(
                    ["Experiment", "Dev Text zh→en", "Test Text zh→en", "Dev Text en→zh", "Test Text en→zh", "Dev First ms", "Test First ms", "Dev ATD ms", "Test ATD ms"],
                    dev_rows,
                    ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
                )
                + "\n\n该表用于判断dev上的排序能否迁移到test；test结果不得反向修改本轮operating point。",
            )
        )
    if offline_rows:
        sections.append(
            (
                "与offline Phase3 quality模式的同指标差值",
                table(
                    ["Streaming experiment", "Direction", "Metric", "Streaming", "Offline Phase3", "Streaming−offline"],
                    offline_rows,
                    ["---", "---", "---", "---:", "---:", "---:"],
                )
                + "\n\nOffline Phase3是质量上界参考，不参与Reward-v2训练因果判断。",
            )
        )
    sections.extend(
        [
            ("结论", "\n".join(conclusions)),
            (
                "结论边界与复现路径",
                "- 本轮仍是单训练种子；没有完成3-seed mean±std、fixed wait-k frontier或10,000次paired bootstrap。\n"
                "- action-only GRPO没有更新text/semantic/BiCodec backbone，不能称为full-model GRPO。\n"
                f"- Machine-readable comparison: `{comparison_root / 'comparison.json'}`。\n"
                + "\n".join(f"- {LABELS[label]}: `{value['run_dir']}`" for label, value in results.items()),
            ),
        ]
    )
    return sections


def render_report(results: dict[str, dict[str, Any]], dev: dict[str, Any] | None) -> tuple[str, str]:
    sample_count = next(iter(results.values()))["samples"]
    sections = build_sections(results, dev)
    preface = (
        f"> Scope: {sample_count:,}-sample frozen full-test streaming S2ST; four experiments use two fixed H200 GPUs each.\n"
        "> R0–R3 use operating points selected only on dev and the same protocol as the prior E0–E3 full-test report."
    )
    standalone = ["# Simul-UniSS Stage7A Reward-v2 four-way full-test report", "", preface]
    for index, (title, body) in enumerate(sections, start=1):
        standalone.extend(["", f"## {index}. {title}", "", body])
    standalone.append("")

    chapter = [
        START_MARKER,
        "## 18. 2026-07-28 Reward-v2 dev选择与full-test确认",
        "",
        preface,
    ]
    for index, (title, body) in enumerate(sections, start=1):
        chapter.extend(["", f"### 18.{index} {title}", "", body])
    chapter.extend(["", END_MARKER])
    return "\n".join(standalone), "\n".join(chapter)


def update_continuity_report(path: Path, chapter: str) -> None:
    original = path.read_text(encoding="utf-8")
    has_start = START_MARKER in original
    has_end = END_MARKER in original
    if has_start != has_end:
        raise RuntimeError(f"Incomplete Reward-v2 chapter markers in {path}")
    if has_start:
        prefix, remainder = original.split(START_MARKER, 1)
        _, suffix = remainder.split(END_MARKER, 1)
        updated = prefix.rstrip() + "\n\n" + chapter + suffix
    else:
        updated = original.rstrip() + "\n\n" + chapter + "\n"
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", default="full_test_v1")
    parser.add_argument("--dev-comparison")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--continuity-report", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    results = {label: extract(root / label / args.run_id) for label in LABELS}
    dev_path = Path(args.dev_comparison) if args.dev_comparison else None
    dev = read_json(dev_path) if dev_path and dev_path.is_file() else None
    payload = {
        "schema_version": "simul_uniss_stage7a_reward_v2_full_test_comparison_v1",
        "selection_split": "dev",
        "evaluation_split": "test",
        "dev_comparison": str(dev_path.resolve()) if dev_path and dev_path.is_file() else None,
        "results": results,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report, chapter = render_report(results, dev)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    update_continuity_report(Path(args.continuity_report), chapter)


if __name__ == "__main__":
    main()
