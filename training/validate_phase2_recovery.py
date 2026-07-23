"""Validate the Phase2 recovery pilot before a full recovery run starts."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def max_consecutive(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def validate(
    tensorboard_dir: Path,
    log_path: Path,
    required_step: int,
    max_valid_loss: float,
    max_last_valid_loss: float,
    grad_spike_threshold: float,
    max_consecutive_grad_spikes: int,
) -> dict[str, object]:
    events = EventAccumulator(str(tensorboard_dir))
    events.Reload()
    scalar_tags = events.Tags().get("scalars", [])
    required_tags = {"lm loss", "lm loss validation", "grad-norm", "learning-rate"}
    missing = sorted(required_tags - set(scalar_tags))
    if missing:
        raise ValueError(f"missing TensorBoard scalars: {missing}")

    all_scalars = [value for tag in scalar_tags for value in events.Scalars(tag)]
    nonfinite = [(value.step, value.value) for value in all_scalars if not math.isfinite(value.value)]
    if nonfinite:
        raise ValueError(f"non-finite TensorBoard values: {nonfinite[:5]}")

    validation = events.Scalars("lm loss validation")
    if not validation or validation[-1].step < required_step:
        raise ValueError(f"pilot validation only reached step {validation[-1].step if validation else -1}")
    high_validation_run = max_consecutive([value.value > max_valid_loss for value in validation])
    if high_validation_run >= 2:
        raise ValueError(f"validation exceeded {max_valid_loss} for {high_validation_run} consecutive evaluations")
    if validation[-1].value > max_last_valid_loss:
        raise ValueError(f"last validation {validation[-1].value} exceeds {max_last_valid_loss}")

    gradients = [value for value in events.Scalars("grad-norm") if value.step > 0]
    grad_spike_run = max_consecutive([value.value > grad_spike_threshold for value in gradients])
    if grad_spike_run > max_consecutive_grad_spikes:
        raise ValueError(
            f"grad norm exceeded {grad_spike_threshold} for {grad_spike_run} consecutive log points"
        )

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    fatal_patterns = (
        r"Traceback \(most recent call last\)",
        r"CUDA out of memory",
        r"number of skipped iterations:\s+[1-9]",
        r"number of nan iterations:\s+[1-9]",
    )
    for pattern in fatal_patterns:
        if re.search(pattern, log_text):
            raise ValueError(f"pilot log matched fatal pattern: {pattern}")

    return {
        "status": "pass",
        "required_step": required_step,
        "last_validation_step": validation[-1].step,
        "last_validation_loss": validation[-1].value,
        "best_validation_loss": min(value.value for value in validation),
        "max_consecutive_high_validation": high_validation_run,
        "max_grad_norm": max(value.value for value in gradients),
        "max_consecutive_grad_spikes": grad_spike_run,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorboard-dir", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--required-step", required=True, type=int)
    parser.add_argument("--max-valid-loss", required=True, type=float)
    parser.add_argument("--max-last-valid-loss", required=True, type=float)
    parser.add_argument("--grad-spike-threshold", required=True, type=float)
    parser.add_argument("--max-consecutive-grad-spikes", required=True, type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate(
        args.tensorboard_dir,
        args.log,
        args.required_step,
        args.max_valid_loss,
        args.max_last_valid_loss,
        args.grad_spike_threshold,
        args.max_consecutive_grad_spikes,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
