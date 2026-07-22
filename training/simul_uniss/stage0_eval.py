"""Evaluate pseudo-streaming schedules and write TensorBoard diagnostics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterator


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def evaluate(path: Path, limit_records: int | None = None) -> dict[str, float]:
    first_write_ms: list[float] = []
    wait_counts: list[float] = []
    write_counts: list[float] = []
    semantic_chunk_lengths: list[float] = []
    text_chunk_lengths: list[float] = []
    proxy_lag_ms: list[float] = []
    total_records = 0
    final_flush_failures = 0

    for record in iter_jsonl(path):
        total_records += 1
        events = record["events"]
        writes = [event for event in events if event["action"] == "write"]
        waits = [event for event in events if event["action"] == "wait"]
        if not writes:
            final_flush_failures += 1
        else:
            first_write_ms.append(float(writes[0]["source_end_ms"]))
            if not bool(events[-1]["source_is_final"]):
                final_flush_failures += 1
        wait_counts.append(float(len(waits)))
        write_counts.append(float(len(writes)))
        source_length = max(1, int(record["source_glm_length"]))
        target_length = max(1, int(record["target_text_length"]))
        for event in writes:
            text_ids = event["target_text_ids"]
            semantic = event["target_semantic"]
            text_chunk_lengths.append(float(len(text_ids)))
            semantic_chunk_lengths.append(float(len(semantic)))
            target_end = max(phrase["text_end"] for phrase in event["target_phrases"])
            ideal_source_tokens = source_length * target_end / target_length
            actual_source_tokens = int(event["source_glm_end"])
            proxy_lag_ms.append(max(0.0, actual_source_tokens - ideal_source_tokens) * 80.0)
        if limit_records is not None and total_records >= limit_records:
            break

    if total_records == 0:
        raise ValueError(f"{path} contains no schedules")

    def mean(values: list[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]

    return {
        "records": float(total_records),
        "first_write_ms_mean": mean(first_write_ms),
        "first_write_ms_p50": percentile(first_write_ms, 0.50),
        "first_write_ms_p95": percentile(first_write_ms, 0.95),
        "wait_events_mean": mean(wait_counts),
        "write_events_mean": mean(write_counts),
        "text_tokens_per_write_mean": mean(text_chunk_lengths),
        "semantic_tokens_per_write_mean": mean(semantic_chunk_lengths),
        "proxy_lag_ms_mean": mean(proxy_lag_ms),
        "final_flush_failure_rate": final_flush_failures / total_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensorboard-dir", default=None)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--limit-records", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(Path(args.input), limit_records=args.limit_records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter

        with SummaryWriter(args.tensorboard_dir) as writer:
            for name, value in metrics.items():
                writer.add_scalar(f"stage0/{name}", value, args.step)
            writer.flush()
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
