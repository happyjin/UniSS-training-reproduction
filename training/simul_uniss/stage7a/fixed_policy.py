"""Evaluate deterministic fixed wait-k policies on Simul-UniSS schedules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.io_utils import iter_jsonl, write_json


def fixed_wait_k_actions(events: list[dict[str, object]], wait_k: int) -> list[str]:
    if wait_k < 1:
        raise ValueError("wait_k must be positive")
    committed_proxy = 0
    actions: list[str] = []
    for index, event in enumerate(events):
        supported = int(event.get("target_ctc_count_proxy", 0))
        is_final = bool(event.get("source_is_final", False))
        can_write = index + 1 >= wait_k and supported > committed_proxy
        action = "write" if is_final or can_write else "wait"
        actions.append(action)
        if action == "write":
            committed_proxy = max(committed_proxy, supported)
    return actions


def policy_counts(reference: list[str], predicted: list[str]) -> dict[str, float]:
    if len(reference) != len(predicted):
        raise ValueError("reference and predicted actions must have equal length")
    wait_tp = sum(p == "wait" and r == "wait" for p, r in zip(predicted, reference))
    wait_fp = sum(p == "wait" and r == "write" for p, r in zip(predicted, reference))
    wait_fn = sum(p == "write" and r == "wait" for p, r in zip(predicted, reference))
    write_tp = sum(p == "write" and r == "write" for p, r in zip(predicted, reference))
    write_fp = sum(p == "write" and r == "wait" for p, r in zip(predicted, reference))
    write_fn = sum(p == "wait" and r == "write" for p, r in zip(predicted, reference))
    return {
        "events": float(len(reference)),
        "correct": float(sum(p == r for p, r in zip(predicted, reference))),
        "wait_tp": float(wait_tp),
        "wait_fp": float(wait_fp),
        "wait_fn": float(wait_fn),
        "write_tp": float(write_tp),
        "write_fp": float(write_fp),
        "write_fn": float(write_fn),
        "premature": float(write_fp),
        "reference_wait": float(sum(value == "wait" for value in reference)),
        "unnecessary": float(write_fn),
        "reference_write": float(sum(value == "write" for value in reference)),
    }


def _f1(tp: float, fp: float, fn: float) -> float:
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def evaluate_fixed_wait_k(
    schedules: Path,
    wait_k_values: list[int],
    *,
    limit_records: int = 0,
) -> dict[str, object]:
    aggregates = {
        wait_k: {
            name: 0.0
            for name in (
                "events",
                "correct",
                "wait_tp",
                "wait_fp",
                "wait_fn",
                "write_tp",
                "write_fp",
                "write_fn",
                "premature",
                "reference_wait",
                "unnecessary",
                "reference_write",
                "samples",
                "first_write_abs_ms_sum",
                "first_write_pairs",
                "predicted_writes",
                "reference_writes",
                "final_flush_success",
            )
        }
        for wait_k in wait_k_values
    }
    for record_index, schedule in enumerate(iter_jsonl(schedules)):
        if limit_records > 0 and record_index >= limit_records:
            break
        events = [dict(event) for event in schedule["events"]]
        reference = [str(event["action"]) for event in events]
        source_end_ms = [float(event["source_end_ms"]) for event in events]
        for wait_k in wait_k_values:
            predicted = fixed_wait_k_actions(events, wait_k)
            counts = policy_counts(reference, predicted)
            target = aggregates[wait_k]
            for name, value in counts.items():
                target[name] += value
            target["samples"] += 1.0
            target["predicted_writes"] += sum(value == "write" for value in predicted)
            target["reference_writes"] += sum(value == "write" for value in reference)
            target["final_flush_success"] += predicted[-1] == "write"
            reference_first = next(
                (
                    source_end_ms[index]
                    for index, value in enumerate(reference)
                    if value == "write"
                ),
                None,
            )
            predicted_first = next(
                (
                    source_end_ms[index]
                    for index, value in enumerate(predicted)
                    if value == "write"
                ),
                None,
            )
            if reference_first is not None and predicted_first is not None:
                target["first_write_abs_ms_sum"] += abs(
                    predicted_first - reference_first
                )
                target["first_write_pairs"] += 1.0

    results: dict[str, object] = {}
    for wait_k, values in aggregates.items():
        events = max(1.0, values["events"])
        samples = max(1.0, values["samples"])
        results[str(wait_k)] = {
            "wait_k": wait_k,
            "samples": values["samples"],
            "events": values["events"],
            "accuracy": values["correct"] / events,
            "wait_f1": _f1(values["wait_tp"], values["wait_fp"], values["wait_fn"]),
            "write_f1": _f1(values["write_tp"], values["write_fp"], values["write_fn"]),
            "premature_write_given_wait": values["premature"]
            / max(1.0, values["reference_wait"]),
            "unnecessary_wait_given_write": values["unnecessary"]
            / max(1.0, values["reference_write"]),
            "first_write_mae_ms": values["first_write_abs_ms_sum"]
            / max(1.0, values["first_write_pairs"]),
            "predicted_writes_per_sample": values["predicted_writes"] / samples,
            "reference_writes_per_sample": values["reference_writes"] / samples,
            "final_flush_success": values["final_flush_success"] / samples,
        }
    return {
        "schema_version": "simul_uniss_stage7a_fixed_wait_k_v1",
        "schedules": str(schedules.resolve()),
        "limit_records": limit_records,
        "policies": results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait-k", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--limit-records", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = evaluate_fixed_wait_k(
        Path(args.schedules), args.wait_k, limit_records=args.limit_records
    )
    write_json(Path(args.output), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
