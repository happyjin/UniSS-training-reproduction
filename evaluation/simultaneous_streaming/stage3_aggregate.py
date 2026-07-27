"""Aggregate Stage3 action evaluation shards and write a detailed report."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


def iter_jsonl(paths: Iterable[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def class_metrics(counts: Counter[tuple[str, str]], name: str) -> dict[str, float]:
    tp = counts[(name, name)]
    fp = sum(value for (reference, prediction), value in counts.items() if prediction == name and reference != name)
    fn = sum(value for (reference, prediction), value in counts.items() if reference == name and prediction != name)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate_events(events: list[dict[str, object]]) -> dict[str, object]:
    confusion: Counter[tuple[str, str]] = Counter()
    global_correct = 0
    binary_correct = 0
    invalid_global = 0
    target_ce: list[float] = []
    write_probabilities: list[float] = []
    for event in events:
        reference = str(event["reference_action"])
        prediction = str(event["binary_prediction"])
        confusion[(reference, prediction)] += 1
        binary_correct += reference == prediction
        global_action = str(event["global_prediction_action"])
        global_correct += global_action == reference
        invalid_global += global_action == "other"
        target_ce.append(float(event["target_ce"]))
        write_probabilities.append(float(event["binary_write_probability"]))
    wait = class_metrics(confusion, "wait")
    write = class_metrics(confusion, "write")
    mean_ce = statistics.fmean(target_ce)
    total = len(events)
    return {
        "events": total,
        "reference_wait": sum(value for (reference, _), value in confusion.items() if reference == "wait"),
        "reference_write": sum(value for (reference, _), value in confusion.items() if reference == "write"),
        "predicted_wait": sum(value for (_, prediction), value in confusion.items() if prediction == "wait"),
        "predicted_write": sum(value for (_, prediction), value in confusion.items() if prediction == "write"),
        "binary_accuracy": safe_div(binary_correct, total),
        "global_action_top1_accuracy": safe_div(global_correct, total),
        "invalid_global_top1_rate": safe_div(invalid_global, total),
        "macro_f1": (wait["f1"] + write["f1"]) / 2,
        "wait": wait,
        "write": write,
        "premature_write_rate": safe_div(confusion[("wait", "write")], total),
        "premature_write_given_wait": safe_div(
            confusion[("wait", "write")],
            confusion[("wait", "wait")] + confusion[("wait", "write")],
        ),
        "unnecessary_wait_rate": safe_div(confusion[("write", "wait")], total),
        "unnecessary_wait_given_write": safe_div(
            confusion[("write", "wait")],
            confusion[("write", "write")] + confusion[("write", "wait")],
        ),
        "confusion": {
            "wait_wait": confusion[("wait", "wait")],
            "wait_write": confusion[("wait", "write")],
            "write_wait": confusion[("write", "wait")],
            "write_write": confusion[("write", "write")],
        },
        "mean_target_ce": mean_ce,
        "target_perplexity": math.exp(min(mean_ce, 50.0)),
        "binary_write_probability_mean": statistics.fmean(write_probabilities),
    }


def aggregate_samples(samples: list[dict[str, object]]) -> dict[str, object]:
    first_write_deltas = [
        float(sample["first_write_delta_ms"])
        for sample in samples
        if sample.get("first_write_delta_ms") is not None
    ]
    first_write_abs = [abs(value) for value in first_write_deltas]
    missing_predicted_first = sum(sample.get("predicted_first_write_ms") is None for sample in samples)
    final_flush_success = sum(bool(sample.get("final_flush_success")) for sample in samples)
    write_count_delta = [
        float(sample["predicted_write_count"]) - float(sample["reference_write_count"])
        for sample in samples
    ]
    return {
        "samples": len(samples),
        "final_flush_success_rate": safe_div(final_flush_success, len(samples)),
        "missing_predicted_first_write_rate": safe_div(missing_predicted_first, len(samples)),
        "first_write_delta_ms_mean": mean(first_write_deltas),
        "first_write_delta_ms_p50": percentile(first_write_deltas, 0.50),
        "first_write_delta_ms_p95": percentile(first_write_deltas, 0.95),
        "first_write_absolute_error_ms_mean": mean(first_write_abs),
        "first_write_absolute_error_ms_p95": percentile(first_write_abs, 0.95),
        "early_first_write_rate": safe_div(sum(value < 0 for value in first_write_deltas), len(samples)),
        "late_first_write_rate": safe_div(sum(value > 0 for value in first_write_deltas), len(samples)),
        "exact_first_write_rate": safe_div(sum(value == 0 for value in first_write_deltas), len(samples)),
        "write_count_delta_mean": mean(write_count_delta),
        "write_count_delta_p95": percentile(write_count_delta, 0.95),
    }


def aggregate_rank_summaries(paths: list[Path]) -> dict[str, object]:
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    required_ranks = set(range(len(summaries)))
    observed_ranks = {int(summary["rank"]) for summary in summaries}
    if observed_ranks != required_ranks:
        raise ValueError(f"rank summaries are incomplete: {observed_ranks} != {required_ranks}")
    total_inference_seconds = max(float(summary["inference_seconds"]) for summary in summaries)
    return {
        "ranks": len(summaries),
        "samples": sum(int(summary["samples"]) for summary in summaries),
        "events": sum(int(summary["action_events"]) for summary in summaries),
        "actual_tokens": sum(int(summary["actual_tokens"]) for summary in summaries),
        "padded_tokens": sum(int(summary["padded_tokens"]) for summary in summaries),
        "padding_efficiency": safe_div(
            sum(int(summary["actual_tokens"]) for summary in summaries),
            sum(int(summary["padded_tokens"]) for summary in summaries),
        ),
        "wall_inference_seconds": total_inference_seconds,
        "aggregate_samples_per_second": safe_div(
            sum(int(summary["samples"]) for summary in summaries), total_inference_seconds
        ),
        "aggregate_actual_tokens_per_second": safe_div(
            sum(int(summary["actual_tokens"]) for summary in summaries), total_inference_seconds
        ),
        "rank_tokens_per_second": [float(summary["tokens_per_second"]) for summary in summaries],
        "rank_peak_memory_gib": [
            float(summary["peak_memory_bytes"]) / (1024**3) for summary in summaries
        ],
        "gpu_names": sorted({str(summary["gpu_name"]) for summary in summaries}),
        "dtype": sorted({str(summary["dtype"]) for summary in summaries}),
        "attention_implementation": sorted(
            {str(summary["attention_implementation"]) for summary in summaries}
        ),
        "max_batch_tokens": sorted({int(summary["max_batch_tokens"]) for summary in summaries}),
        "max_batch_size": sorted({int(summary["max_batch_size"]) for summary in summaries}),
    }


def load_gpu_monitor(path: Path, gpu_indexes: set[int]) -> dict[str, object]:
    if not path.is_file():
        return {"available": False, "path": str(path)}
    by_gpu: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                index = int(row[1].strip())
                memory = float(row[2].strip())
                utilization = float(row[3].strip())
                power = float(row[4].strip())
                power_limit = float(row[5].strip())
            except ValueError:
                continue
            if index not in gpu_indexes or memory < 512:
                continue
            by_gpu[index]["memory_mib"].append(memory)
            by_gpu[index]["utilization"].append(utilization)
            by_gpu[index]["power_w"].append(power)
            by_gpu[index]["power_limit_w"].append(power_limit)
    gpu_results: dict[str, object] = {}
    all_utilization: list[float] = []
    all_power: list[float] = []
    for index in sorted(gpu_indexes):
        values = by_gpu.get(index, {})
        utilization = list(values.get("utilization", []))
        power = list(values.get("power_w", []))
        memory = list(values.get("memory_mib", []))
        limits = list(values.get("power_limit_w", []))
        all_utilization.extend(utilization)
        all_power.extend(power)
        gpu_results[str(index)] = {
            "active_samples": len(utilization),
            "utilization_mean": mean(utilization),
            "utilization_p50": percentile(utilization, 0.50),
            "utilization_p95": percentile(utilization, 0.95),
            "power_mean_w": mean(power),
            "power_p95_w": percentile(power, 0.95),
            "power_limit_w": max(limits) if limits else None,
            "memory_peak_mib": max(memory) if memory else None,
        }
    return {
        "available": bool(all_utilization),
        "path": str(path),
        "gpus": gpu_results,
        "utilization_mean": mean(all_utilization),
        "utilization_p95": percentile(all_utilization, 0.95),
        "power_mean_w": mean(all_power),
        "power_p95_w": percentile(all_power, 0.95),
    }


def aggregate_split(split_dir: Path) -> dict[str, object]:
    event_paths = sorted(split_dir.glob("events.rank*.jsonl"))
    sample_paths = sorted(split_dir.glob("samples.rank*.jsonl"))
    summary_paths = sorted(split_dir.glob("summary.rank*.json"))
    if not event_paths or len(event_paths) != len(sample_paths) or len(event_paths) != len(summary_paths):
        raise ValueError(f"incomplete split outputs under {split_dir}")
    events = list(iter_jsonl(event_paths))
    samples = list(iter_jsonl(sample_paths))
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"duplicate sample ids under {split_dir}")
    event_keys = [(str(event["sample_id"]), int(event["event_index"])) for event in events]
    if len(set(event_keys)) != len(event_keys):
        raise ValueError(f"duplicate event keys under {split_dir}")
    rank_summary = aggregate_rank_summaries(summary_paths)
    if rank_summary["samples"] != len(samples) or rank_summary["events"] != len(events):
        raise ValueError(f"rank/event accounting mismatch under {split_dir}")
    directions: dict[str, dict[str, object]] = {}
    grouped_events: dict[str, list[dict[str, object]]] = defaultdict(list)
    grouped_samples: dict[str, list[dict[str, object]]] = defaultdict(list)
    for event in events:
        grouped_events[f"{event['src_lang']}->{event['tgt_lang']}"].append(event)
    for sample in samples:
        grouped_samples[f"{sample['src_lang']}->{sample['tgt_lang']}"].append(sample)
    for direction in sorted(grouped_events):
        directions[direction] = {
            **aggregate_events(grouped_events[direction]),
            **aggregate_samples(grouped_samples[direction]),
        }
    return {
        "split_dir": str(split_dir),
        "rank_performance": rank_summary,
        "events": aggregate_events(events),
        "samples": aggregate_samples(samples),
        "directions": directions,
    }


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def metric_table(name: str, result: dict[str, object]) -> str:
    events = result["events"]
    samples = result["samples"]
    assert isinstance(events, dict) and isinstance(samples, dict)
    wait = events["wait"]
    write = events["write"]
    assert isinstance(wait, dict) and isinstance(write, dict)
    return "\n".join(
        [
            f"### {name}",
            "",
            "| 指标 | 数值 |",
            "| --- | ---: |",
            f"| 样本数 | {fmt(samples['samples'])} |",
            f"| action events | {fmt(events['events'])} |",
            f"| Binary action accuracy | {fmt(events['binary_accuracy'])} |",
            f"| Macro-F1 | {fmt(events['macro_f1'])} |",
            f"| WAIT precision / recall / F1 | {fmt(wait['precision'])} / {fmt(wait['recall'])} / {fmt(wait['f1'])} |",
            f"| WRITE precision / recall / F1 | {fmt(write['precision'])} / {fmt(write['recall'])} / {fmt(write['f1'])} |",
            f"| Premature WRITE / given WAIT | {fmt(events['premature_write_rate'])} / {fmt(events['premature_write_given_wait'])} |",
            f"| Unnecessary WAIT / given WRITE | {fmt(events['unnecessary_wait_rate'])} / {fmt(events['unnecessary_wait_given_write'])} |",
            f"| Global-vocab top1 action accuracy | {fmt(events['global_action_top1_accuracy'])} |",
            f"| Global-vocab invalid top1 rate | {fmt(events['invalid_global_top1_rate'])} |",
            f"| Action CE / PPL | {fmt(events['mean_target_ce'])} / {fmt(events['target_perplexity'])} |",
            f"| Final flush success | {fmt(samples['final_flush_success_rate'])} |",
            f"| First-WRITE exact rate | {fmt(samples['exact_first_write_rate'])} |",
            f"| First-WRITE MAE | {fmt(samples['first_write_absolute_error_ms_mean'], 2)} ms |",
            f"| First-WRITE MAE p95 | {fmt(samples['first_write_absolute_error_ms_p95'], 2)} ms |",
            "",
        ]
    )


def gpu_table(gpu: dict[str, object]) -> str:
    if not gpu.get("available"):
        return "GPU monitor data unavailable.\n"
    rows = [
        "| GPU | Active samples | Util mean | Util p95 | Power mean | Power p95 | Peak memory |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    gpus = gpu["gpus"]
    assert isinstance(gpus, dict)
    for index, values in sorted(gpus.items(), key=lambda item: int(item[0])):
        assert isinstance(values, dict)
        rows.append(
            f"| {index} | {fmt(values['active_samples'])} | {fmt(values['utilization_mean'], 1)}% | "
            f"{fmt(values['utilization_p95'], 1)}% | {fmt(values['power_mean_w'], 1)} W | "
            f"{fmt(values['power_p95_w'], 1)} W | {fmt(values['memory_peak_mib'], 0)} MiB |"
        )
    return "\n".join(rows) + "\n"


def build_report(
    run_dir: Path,
    dev: dict[str, object],
    evaluation: dict[str, object],
    gpu: dict[str, object],
    aggregate_path: Path,
) -> str:
    dev_perf = dev["rank_performance"]
    eval_perf = evaluation["rank_performance"]
    assert isinstance(dev_perf, dict) and isinstance(eval_perf, dict)
    sections = [
        "# Simul-UniSS full198 Stage3 action evaluation report",
        "",
        f"> Run directory: `{run_dir}`  ",
        "> Scope: teacher-forced Stage3 WAIT/WRITE logits on pseudo-proportional 640 ms schedules  ",
        "> GPU allocation: UniST dev on GPU 0–3; UniST test/eval on GPU 4–7",
        "",
        "## 1. 结论边界",
        "",
        "本报告严格评估 Stage3 action-only checkpoint 在每个真实 action label 位置的模型 logits。",
        "它能够回答 WAIT/WRITE 分类、过早 WRITE、无必要 WAIT、first-WRITE 和 final flush 是否正确；",
        "但当前输入仍是 `pseudo_proportional_token_alignment`，并且采用 teacher forcing，因此本报告",
        "不把 proxy first-WRITE 结果称为论文中的真实 AL/LAAL/ATD，也不报告 ASR-BLEU。",
        "真实 free-running Simul-S2TT/S2ST 需要后续 Qwen streaming adapter 和 waveform 时间戳。",
        "",
        "## 2. 数据与运行设置",
        "",
        "| 项目 | Dev | Eval/Test |",
        "| --- | ---: | ---: |",
        f"| GPU ranks | {fmt(dev_perf['ranks'])} | {fmt(eval_perf['ranks'])} |",
        f"| Samples | {fmt(dev_perf['samples'])} | {fmt(eval_perf['samples'])} |",
        f"| Action events | {fmt(dev_perf['events'])} | {fmt(eval_perf['events'])} |",
        f"| Actual tokens | {fmt(dev_perf['actual_tokens'])} | {fmt(eval_perf['actual_tokens'])} |",
        f"| Padding efficiency | {fmt(dev_perf['padding_efficiency'])} | {fmt(eval_perf['padding_efficiency'])} |",
        f"| 4-GPU wall inference | {fmt(dev_perf['wall_inference_seconds'], 2)} s | {fmt(eval_perf['wall_inference_seconds'], 2)} s |",
        f"| Aggregate tokens/s | {fmt(dev_perf['aggregate_actual_tokens_per_second'], 1)} | {fmt(eval_perf['aggregate_actual_tokens_per_second'], 1)} |",
        "",
        "Batching does not truncate samples and does not pack independent samples under a shared causal mask.",
        "每条样本有独立 attention mask；bf16/FlashAttention 只改变执行方式，不改变 action 标签或评估范围。",
        "",
        "## 3. Stage3 action 指标",
        "",
        metric_table("UniST dev", dev),
        metric_table("UniST test/eval", evaluation),
        "## 4. 按方向结果",
        "",
    ]
    for split_name, result in (("Dev", dev), ("Eval/Test", evaluation)):
        directions = result["directions"]
        assert isinstance(directions, dict)
        sections.extend([f"### {split_name}", ""])
        sections.extend(
            [
                "| Direction | Samples | Events | Accuracy | Macro-F1 | WRITE F1 | Premature WRITE | Final flush |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for direction, values in directions.items():
            assert isinstance(values, dict)
            write = values["write"]
            assert isinstance(write, dict)
            sections.append(
                f"| {direction} | {fmt(values['samples'])} | {fmt(values['events'])} | "
                f"{fmt(values['binary_accuracy'])} | {fmt(values['macro_f1'])} | {fmt(write['f1'])} | "
                f"{fmt(values['premature_write_rate'])} | {fmt(values['final_flush_success_rate'])} |"
            )
        sections.append("")
    sections.extend(
        [
            "## 5. GPU 利用率与功率",
            "",
            gpu_table(gpu),
            "GPU利用率与功率是吞吐实现诊断，不进入模型质量分数。模型只有0.5B，",
            "因此即使采用大token-batch，也不保证达到长序列训练时的700W功率；正确性优先于通过",
            "重复计算或无效padding人为抬高功率。本报告同时给出padding efficiency用于审计。",
            "",
            "## 6. 指标解释与论文对应",
            "",
            "- `WRITE F1`、`premature WRITE` 和 `unnecessary WAIT` 用于评估 Stage3 策略，",
            "  对应 StreamSpeech 的 READ/WRITE eligibility 和 Hibiki-Zero 对过早输出的控制思想。",
            "- `global-vocab top1 action accuracy` 要求 WAIT/WRITE 在完整180,480词表中成为top1；",
            "  `binary accuracy` 则只在 WAIT/WRITE 两者中比较。两者差值可定位模型是否偏向其他token。",
            "- `first-WRITE MAE` 当前以640ms pseudo schedule为参考，只是policy proxy，",
            "  不能直接与 SimulS2S-LLM 的 ATD、Hibiki 的 LAAL 或 StreamSpeech 的 AL 比较。",
            "- Stage3不训练完整phrase/semantic输出，因此 ASR-BLEU、BLASER、RTF和Discontinuity",
            "  应在 Stage4/6自由运行S2ST阶段评估。",
            "",
            "## 7. 产物",
            "",
            f"- Aggregate JSON: `{aggregate_path}`",
            f"- Dev shards: `{run_dir / 'dev'}`",
            f"- Eval shards: `{run_dir / 'eval'}`",
            f"- GPU monitor: `{run_dir / 'gpu_monitor.csv'}`",
            "",
        ]
    )
    return "\n".join(sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--gpu-monitor", default=None)
    parser.add_argument("--expected-dev-samples", type=int, default=None)
    parser.add_argument("--expected-eval-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    dev = aggregate_split(run_dir / "dev")
    evaluation = aggregate_split(run_dir / "eval")
    if args.expected_dev_samples is not None and dev["samples"]["samples"] != args.expected_dev_samples:
        raise ValueError("dev sample count does not match expectation")
    if args.expected_eval_samples is not None and evaluation["samples"]["samples"] != args.expected_eval_samples:
        raise ValueError("eval sample count does not match expectation")
    gpu_path = Path(args.gpu_monitor) if args.gpu_monitor else run_dir / "gpu_monitor.csv"
    gpu = load_gpu_monitor(gpu_path, set(range(8)))
    aggregate = {
        "schema_version": "simul_uniss_stage3_action_aggregate_v1",
        "run_dir": str(run_dir),
        "dev": dev,
        "eval": evaluation,
        "gpu_monitor": gpu,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(run_dir, dev, evaluation, gpu, output), encoding="utf-8"
    )
    (run_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
