"""Token-level simultaneous translation latency metrics for schedules."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterator


def target_emission_source_positions(schedule: dict[str, object]) -> list[int]:
    positions: list[int] = []
    for event in schedule["events"]:
        if event["action"] == "write":
            positions.extend([int(event["source_glm_end"])] * len(event["target_text_ids"]))
    return positions


def schedule_latency_metrics(schedule: dict[str, object]) -> dict[str, float]:
    source_length = max(1, int(schedule["source_glm_length"]))
    target_length = max(1, int(schedule["target_text_length"]))
    positions = target_emission_source_positions(schedule)
    if len(positions) != target_length:
        raise ValueError(
            f"target emission count {len(positions)} does not match target length {target_length}"
        )
    ratio = target_length / source_length
    tau = next((index + 1 for index, value in enumerate(positions) if value >= source_length), target_length)
    lags = [positions[index] - index / ratio for index in range(tau)]
    al = sum(lags) / max(1, tau)
    ap = sum(positions) / (source_length * target_length)
    ideal = [(index + 1) * source_length / target_length for index in range(target_length)]
    atd_ms = sum(max(0.0, actual - expected) * 80.0 for actual, expected in zip(positions, ideal)) / target_length
    first_write = next(
        (float(event["source_end_ms"]) for event in schedule["events"] if event["action"] == "write"),
        0.0,
    )
    return {
        "al_glm_tokens": al,
        "laal_glm_tokens": al,
        "ap": ap,
        "atd_ms_proxy": atd_ms,
        "first_write_ms": first_write,
        "num_chunks": float(len(schedule["events"])),
        "num_wait": float(sum(event["action"] == "wait" for event in schedule["events"])),
        "num_write": float(sum(event["action"] == "write" for event in schedule["events"])),
    }


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def aggregate(path: Path, limit_records: int | None = None) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    count = 0
    for schedule in iter_jsonl(path):
        metrics = schedule_latency_metrics(schedule)
        for name, value in metrics.items():
            values.setdefault(name, []).append(value)
        count += 1
        if limit_records is not None and count >= limit_records:
            break
    if count == 0:
        raise ValueError(f"{path} contains no schedules")
    result = {name: statistics.fmean(metric_values) for name, metric_values in values.items()}
    result["records"] = float(count)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensorboard-dir", default=None)
    parser.add_argument("--limit-records", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = aggregate(Path(args.input), args.limit_records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter

        with SummaryWriter(args.tensorboard_dir) as writer:
            for name, value in metrics.items():
                writer.add_scalar(f"latency/{name}", value, 0)
            writer.flush()
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
