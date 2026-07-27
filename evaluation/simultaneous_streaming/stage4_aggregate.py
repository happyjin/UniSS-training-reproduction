"""Merge Stage4 shards and build streaming/offline comparison reports."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.io_utils import iter_jsonl, write_json
from evaluation.simultaneous_streaming.stage3_aggregate import load_gpu_monitor
from evaluation.simultaneous_streaming.stage4_metrics import aggregate_rows
from evaluation.text_metrics import corpus_bleu


COMMON_METRIC_FILES = {
    "text_bleu": "metrics/text_bleu.json",
    "speech_bleu": "metrics/speech_bleu.json",
    "slc": "metrics/slc.json",
    "utmos": "metrics/utmos.json",
    "autopcp": "metrics/autopcp.json",
}


def merge_rank_jsonl(
    input_dir: Path,
    *,
    pattern: str,
    output: Path,
    expected_records: int,
    expected_ranks: int,
) -> dict[str, object]:
    paths = sorted(input_dir.glob(pattern))
    if len(paths) != expected_ranks:
        raise ValueError(f"found {len(paths)} rank files, expected {expected_ranks}: {pattern}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite merged output: {output}")
    rows = [row for path in paths for row in iter_jsonl(path)]
    indexes = [int(row["index"]) for row in rows]
    if len(rows) != expected_records:
        raise ValueError(f"merged records {len(rows)} != expected {expected_records}")
    if len(set(indexes)) != len(indexes):
        raise ValueError("duplicate Stage4 record indexes across rank outputs")
    expected_indexes = set(range(expected_records))
    if set(indexes) != expected_indexes:
        missing = sorted(expected_indexes - set(indexes))
        extra = sorted(set(indexes) - expected_indexes)
        raise ValueError(f"Stage4 merged index mismatch: missing={missing[:10]} extra={extra[:10]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda value: int(value["index"])):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "output": str(output),
        "records": len(rows),
        "rank_files": [str(path) for path in paths],
    }


def read_optional(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_common_metrics(root: Path) -> dict[str, object]:
    return {
        name: read_optional(root / relative)
        for name, relative in COMMON_METRIC_FILES.items()
    }


def flatten_common(metrics: Mapping[str, object]) -> dict[tuple[str, str, str], float]:
    flattened: dict[tuple[str, str, str], float] = {}
    for metric_name, report in metrics.items():
        if not isinstance(report, Mapping):
            continue
        for group, values in report.get("groups", {}).items():  # type: ignore[union-attr]
            mode, direction = str(group).split(":", 1)
            if metric_name in {"text_bleu", "speech_bleu"}:
                flattened[(metric_name, mode, direction)] = float(values["score"])
            elif metric_name == "slc":
                flattened[("slc_0_2", mode, direction)] = float(values["slc_0_2"])
                flattened[("slc_0_4", mode, direction)] = float(values["slc_0_4"])
            elif metric_name == "utmos":
                flattened[("utmos", mode, direction)] = float(values["mean"])
            elif metric_name == "autopcp":
                flattened[("autopcp", mode, direction)] = float(values["mean"])
    return flattened


def common_comparisons(
    streaming: Mapping[str, object],
    offline: Mapping[str, object],
    *,
    streaming_mode: str = "streaming_stage4",
) -> list[dict[str, object]]:
    stream_values = flatten_common(streaming)
    offline_values = flatten_common(offline)
    rows: list[dict[str, object]] = []
    for (metric, mode, direction), value in sorted(stream_values.items()):
        if mode != streaming_mode:
            continue
        for offline_mode in ("quality", "performance"):
            reference = offline_values.get((metric, offline_mode, direction))
            if reference is None:
                continue
            rows.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "streaming_value": value,
                    "offline_mode": offline_mode,
                    "offline_value": reference,
                    "delta_streaming_minus_offline": value - reference,
                }
            )
    return rows


def prefix_bleu(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    empty_by_direction: dict[str, int] = defaultdict(int)
    total_events = 0
    skipped = 0
    for row in rows:
        direction = f"{row['src_lang']}->{row['tgt_lang']}"
        for event in row["event_trace"]:  # type: ignore[index]
            if not isinstance(event, Mapping):
                continue
            hypothesis = event.get("generated_prefix_text")
            reference = event.get("reference_prefix_text")
            if not isinstance(reference, str) or not reference.strip():
                continue
            total_events += 1
            if not isinstance(hypothesis, str) or not hypothesis.strip():
                skipped += 1
                empty_by_direction[direction] += 1
                hypothesis = ""
            groups[direction].append((hypothesis, reference))
    output = {}
    for direction, pairs in sorted(groups.items()):
        tgt_lang = direction.split("->", 1)[1]
        output[direction] = {
            **corpus_bleu(
            [hypothesis for hypothesis, _ in pairs],
            [reference for _, reference in pairs],
            language=tgt_lang,
            ),
            "empty_hypothesis_count": empty_by_direction[direction],
        }
    return {
        "groups": output,
        "events": total_events,
        "empty_hypothesis_events": skipped,
    }


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(aggregate: Mapping[str, object]) -> str:
    streaming = aggregate["streaming_metrics"]
    assert isinstance(streaming, Mapping)
    overall = streaming["overall"]
    assert isinstance(overall, Mapping)
    means = overall["means"]
    p95 = overall["p95"]
    assert isinstance(means, Mapping) and isinstance(p95, Mapping)
    integrity = aggregate["integrity"]
    assert isinstance(integrity, Mapping)
    split_label = str(aggregate.get("split_label", "dev"))
    stage_label = str(aggregate.get("stage_label", "Stage4"))
    stage_iteration = int(aggregate.get("stage_iteration", 4753))
    stage_description = str(
        aggregate.get("stage_description", "phrase-level interleaved S2ST")
    )
    streaming_mode = str(aggregate.get("streaming_mode", "streaming_stage4"))
    lines = [
        f"# Simul-UniSS full198 {stage_label} end-to-end streaming {split_label} report",
        "",
        f"> Run directory: `{aggregate['run_dir']}`",
        f"> Model: {stage_label} {stage_description} iteration {stage_iteration}",
        "> Scope: free-running Qwen actions/text/semantic + real streaming BiCodec waveform",
        "",
        "## 1. 结论边界",
        "",
        f"本运行使用真实 {stage_label} 自由运行输出和真实 BiCodec waveform，但 source chunk boundary",
        "仍来自 UniST 的 pseudo proportional schedule。NCA latency 是策略时间轴，CA latency",
        "加入 vLLM request 与 BiCodec wall time。批量4-GPU吞吐和 batch=1 latency 必须分开解释。",
        "",
        "## 2. 完整性",
        "",
        "| 项目 | 数值 |",
        "| --- | ---: |",
        f"| Samples | {fmt(integrity['samples'])} |",
        f"| Decode failures | {fmt(integrity['decode_failures'])} |",
        f"| Forced actions | {fmt(integrity['forced_actions'])} |",
        f"| Structural recoveries | {fmt(integrity['structural_recoveries'])} |",
        f"| Training-context exceeded samples | {fmt(integrity['training_context_exceeded'])} |",
        f"| Maximum realized prompt tokens | {fmt(integrity['max_prompt_tokens'])} |",
        "",
        "## 3. Streaming policy 与延迟",
        "",
        "| 指标 | Mean | p95 |",
        "| --- | ---: | ---: |",
    ]
    selected = [
        ("binary_accuracy", "WAIT/WRITE accuracy"),
        ("macro_f1", "Action Macro-F1"),
        ("write_f1", "WRITE F1"),
        ("premature_write_given_wait", "Premature WRITE / WAIT"),
        ("unnecessary_wait_given_write", "Unnecessary WAIT / WRITE"),
        ("first_write_ms_proxy", "First WRITE NCA proxy (ms)"),
        ("al_glm_tokens_proxy", "AL GLM-token proxy"),
        ("ap_proxy", "AP proxy"),
        ("dal_glm_tokens_proxy", "DAL GLM-token proxy"),
        ("laal_glm_tokens_proxy", "LAAL GLM-token proxy"),
        ("atd_ms_proxy", "ATD proxy (ms)"),
        ("start_offset_nca_ms", "StartOffset NCA (ms)"),
        ("start_offset_ca_ms", "StartOffset CA (ms)"),
        ("end_offset_nca_ms", "EndOffset NCA (ms)"),
        ("end_offset_ca_ms", "EndOffset CA (ms)"),
        ("rtf_generated_audio", "RTF / generated audio"),
        ("rtf_source_audio", "RTF / source audio"),
        ("action_ttft_seconds_mean", "Action TTFT mean/sample (s)"),
        ("write_ttft_seconds_mean", "WRITE TTFT mean/sample (s)"),
        ("chunk_act_seconds_mean", "Chunk ACT mean/sample (s)"),
        ("chunk_act_seconds_p95", "Chunk ACT p95/sample (s)"),
    ]
    for key, label in selected:
        lines.append(f"| {label} | {fmt(means.get(key))} | {fmt(p95.get(key))} |")
    lines.extend(
        [
            "",
            "## 4. Streaming audio 连续性",
            "",
            "| 指标 | Mean | p95 |",
            "| --- | ---: | ---: |",
        ]
    )
    continuity = [
        ("num_audio_chunks", "NumChunks"),
        ("playback_gap_count_nca", "Playback gap count NCA"),
        ("playback_gap_sum_nca_ms", "Playback gap sum NCA (ms)"),
        ("playback_gap_count_ca", "Playback gap count CA"),
        ("playback_gap_sum_ca_ms", "Playback gap sum CA (ms)"),
        ("boundary_amplitude_jump_mean", "Boundary amplitude jump"),
        ("boundary_rms_jump_mean", "Boundary RMS jump"),
        ("boundary_spectral_distance_mean", "Boundary spectral distance"),
        ("boundary_click_rate", "Boundary click rate"),
    ]
    for key, label in continuity:
        lines.append(f"| {label} | {fmt(means.get(key))} | {fmt(p95.get(key))} |")
    lines.extend(
        [
            "",
            "## 5. Text / semantic 内容诊断",
            "",
            "| 指标 | Mean | p95 |",
            "| --- | ---: | ---: |",
        ]
    )
    content = [
        ("text_length_ratio", "Text length ratio"),
        ("semantic_length_ratio", "Semantic length ratio"),
        ("semantic_aligned_token_accuracy", "Semantic aligned accuracy"),
        ("semantic_unigram_f1", "Semantic unigram F1"),
        ("semantic_bigram_f1", "Semantic bigram F1"),
        ("semantic_max_identical_run", "Max identical semantic run"),
        ("semantic_adjacent_repeat_rate", "Adjacent repeat rate"),
    ]
    for key, label in content:
        lines.append(f"| {label} | {fmt(means.get(key))} | {fmt(p95.get(key))} |")
    prefix = aggregate["prefix_bleu"]
    assert isinstance(prefix, Mapping)
    lines.extend(
        [
            "",
            "### Prefix translation quality",
            "",
            "| Direction | Prefix events | Empty hypotheses | Prefix BLEU |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for direction, values in prefix.get("groups", {}).items():  # type: ignore[union-attr]
        lines.append(
            f"| {direction} | {fmt(values['sample_count'])} | "
            f"{fmt(values['empty_hypothesis_count'])} | {fmt(values['score'])} |"
        )
    common = aggregate["common_metrics"]
    assert isinstance(common, Mapping)
    flattened = flatten_common(common)
    lines.extend(
        [
            "",
            "## 6. 与 offline 相同的最终质量指标",
            "",
            f"| 指标 | 方向 | {stage_label} streaming |",
            "| --- | --- | ---: |",
        ]
    )
    for (metric, mode, direction), value in sorted(flattened.items()):
        if mode == streaming_mode:
            lines.append(f"| {metric} | {direction} | {fmt(value)} |")
    lines.extend(
        [
            "",
            f"## 7. {stage_label} streaming vs offline Phase3",
            "",
            "| 指标 | 方向 | Streaming | Offline mode | Offline | Δ |",
            "| --- | --- | ---: | --- | ---: | ---: |",
        ]
    )
    for row in aggregate["offline_comparisons"]:  # type: ignore[index]
        lines.append(
            f"| {row['metric']} | {row['direction']} | {fmt(row['streaming_value'])} | "
            f"{row['offline_mode']} | {fmt(row['offline_value'])} | "
            f"{float(row['delta_streaming_minus_offline']):+.4f} |"
        )
    gpu = aggregate.get("gpu_monitor")
    lines.extend(["", "## 8. GPU 监控", ""])
    if isinstance(gpu, Mapping) and gpu.get("available"):
        lines.extend(
            [
                f"- Utilization mean / p95: {fmt(gpu.get('utilization_mean'), 1)}% / {fmt(gpu.get('utilization_p95'), 1)}%",
                f"- Power mean / p95: {fmt(gpu.get('power_mean_w'), 1)} W / {fmt(gpu.get('power_p95_w'), 1)} W",
            ]
        )
    else:
        lines.append("GPU monitor unavailable.")
    lines.extend(
        [
            "",
            "## 9. 产物",
            "",
            f"- Results: `{aggregate['results_path']}`",
            f"- Aggregate JSON: `{aggregate['output_json']}`",
            "",
        ]
    )
    return "\n".join(lines)


def report(args: argparse.Namespace) -> dict[str, object]:
    results_path = Path(args.results)
    rows = list(iter_jsonl(results_path))
    indexes = [int(row["index"]) for row in rows]
    if len(rows) != args.expected_records or len(set(indexes)) != len(indexes):
        raise ValueError("Stage4 report input is incomplete or contains duplicate indexes")
    streaming_metrics = aggregate_rows(rows)
    prefix = prefix_bleu(rows)
    run_dir = Path(args.run_dir)
    common = load_common_metrics(run_dir)
    offline = load_common_metrics(Path(args.offline_phase3_root))
    comparisons = common_comparisons(
        common,
        offline,
        streaming_mode=args.streaming_mode,
    )
    integrity = {
        "samples": len(rows),
        "decode_failures": sum(bool(row.get("error")) for row in rows),
        "forced_actions": sum(int(row.get("forced_action_count", 0)) for row in rows),
        "structural_recoveries": sum(int(row.get("structural_recovery_count", 0)) for row in rows),
        "training_context_exceeded": sum(bool(row.get("training_context_exceeded")) for row in rows),
        "max_prompt_tokens": max(int(row.get("max_prompt_tokens", 0)) for row in rows),
    }
    gpu_ids = {int(value) for value in args.gpu_ids.split(",") if value.strip()}
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU index")
    gpu = load_gpu_monitor(Path(args.gpu_monitor), gpu_ids) if args.gpu_monitor else {"available": False}
    aggregate = {
        "schema_version": "simul_uniss_stage4_end_to_end_aggregate_v1",
        "stage_label": args.stage_label,
        "stage_iteration": args.stage_iteration,
        "stage_description": args.stage_description,
        "streaming_mode": args.streaming_mode,
        "split_label": args.split_label,
        "gpu_ids": sorted(gpu_ids),
        "run_dir": str(run_dir.resolve()),
        "results_path": str(results_path.resolve()),
        "output_json": str(Path(args.output_json).resolve()),
        "integrity": integrity,
        "streaming_metrics": streaming_metrics,
        "prefix_bleu": prefix,
        "common_metrics": common,
        "offline_phase3_metrics": offline,
        "offline_comparisons": comparisons,
        "gpu_monitor": gpu,
    }
    write_json(Path(args.output_json), aggregate)
    Path(args.report).write_text(build_report(aggregate), encoding="utf-8")
    return aggregate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--input-dir", required=True)
    merge.add_argument("--pattern", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--expected-records", type=int, required=True)
    merge.add_argument("--expected-ranks", type=int, default=4)
    make_report = subparsers.add_parser("report")
    make_report.add_argument("--run-dir", required=True)
    make_report.add_argument("--results", required=True)
    make_report.add_argument("--offline-phase3-root", required=True)
    make_report.add_argument("--output-json", required=True)
    make_report.add_argument("--report", required=True)
    make_report.add_argument("--gpu-monitor", default=None)
    make_report.add_argument("--gpu-ids", default="0,1,2,3")
    make_report.add_argument("--split-label", default="dev")
    make_report.add_argument("--stage-label", default="Stage4")
    make_report.add_argument("--stage-iteration", type=int, default=4753)
    make_report.add_argument(
        "--stage-description",
        default="phrase-level interleaved S2ST",
    )
    make_report.add_argument("--streaming-mode", default="streaming_stage4")
    make_report.add_argument("--expected-records", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "merge":
        result = merge_rank_jsonl(
            Path(args.input_dir),
            pattern=args.pattern,
            output=Path(args.output),
            expected_records=args.expected_records,
            expected_ranks=args.expected_ranks,
        )
    else:
        result = report(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
