"""Create action-only curriculum samples from full interleaved samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training import constants_uniss as c
from training.simul_uniss import SAMPLE_SCHEMA_VERSION
from training.simul_uniss.sample_builders import ACTION_WEIGHT


def mask_action_sample(sample: dict[str, object]) -> dict[str, object]:
    if sample.get("schema_version") != SAMPLE_SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={SAMPLE_SCHEMA_VERSION}")
    input_ids = sample["input_ids"]
    if not isinstance(input_ids, list) or not all(isinstance(token, int) for token in input_ids):
        raise TypeError("input_ids must be a list of ints")
    action_ids = {c.TOKEN_WAIT_READ, c.TOKEN_WRITE_GENERATE}
    weights = [ACTION_WEIGHT if token in action_ids else 0.0 for token in input_ids]
    if not any(weights):
        raise ValueError(f"sample {sample.get('id')} contains no WAIT/WRITE action")
    result = dict(sample)
    result["task"] = "simul_action"
    result["token_weights"] = weights
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {input_path}:{line_number}") from exc
            masked = mask_action_sample(sample)
            destination.write(json.dumps(masked, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    print(json.dumps({"output": str(output_path), "samples": count}, sort_keys=True))


if __name__ == "__main__":
    main()
