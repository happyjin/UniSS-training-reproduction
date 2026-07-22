"""Pack weighted Simul-UniSS samples without changing legacy packers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from training import constants_uniss as c
from training.simul_uniss import PACKED_SCHEMA_VERSION, SAMPLE_SCHEMA_VERSION


@dataclass(frozen=True)
class ShiftedWeightedSample:
    tokens: list[int]
    labels: list[int]
    loss_mask: list[float]
    position_ids: list[int]
    task: str
    source_id: str

    @property
    def length(self) -> int:
        return len(self.tokens)


def make_shifted_sample(sample: Mapping[str, object]) -> ShiftedWeightedSample:
    if sample.get("schema_version") != SAMPLE_SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={SAMPLE_SCHEMA_VERSION}")
    input_ids = sample["input_ids"]
    token_weights = sample["token_weights"]
    if not isinstance(input_ids, list) or not all(isinstance(value, int) for value in input_ids):
        raise TypeError("input_ids must be a list of ints")
    if not isinstance(token_weights, list) or not all(isinstance(value, (int, float)) for value in token_weights):
        raise TypeError("token_weights must be a list of numbers")
    if len(input_ids) != len(token_weights):
        raise ValueError("input_ids and token_weights lengths differ")
    if len(input_ids) < 2:
        raise ValueError("sample is too short")
    for token_id in input_ids:
        c.validate_token_id(token_id)
    return ShiftedWeightedSample(
        tokens=input_ids[:-1],
        labels=input_ids[1:],
        loss_mask=[float(value) for value in token_weights[1:]],
        position_ids=list(range(len(input_ids) - 1)),
        task=str(sample.get("task", "simul_s2st")),
        source_id=str(sample.get("id", "")),
    )


def _padded(values: list, length: int, fill):
    return [*values, *([fill] * (length - len(values)))]


def pack_samples(
    samples: Iterable[ShiftedWeightedSample],
    seq_length: int,
    *,
    drop_overlong: bool = False,
) -> Iterator[dict[str, object]]:
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")
    current: list[ShiftedWeightedSample] = []
    current_length = 0

    def emit() -> dict[str, object] | None:
        if not current:
            return None
        tokens: list[int] = []
        labels: list[int] = []
        loss_mask: list[float] = []
        position_ids: list[int] = []
        boundaries: list[list[int]] = []
        tasks: list[str] = []
        source_ids: list[str] = []
        for sample in current:
            start = len(tokens)
            tokens.extend(sample.tokens)
            labels.extend(sample.labels)
            loss_mask.extend(sample.loss_mask)
            position_ids.extend(sample.position_ids)
            boundaries.append([start, len(tokens)])
            tasks.append(sample.task)
            source_ids.append(sample.source_id)
        return {
            "schema_version": PACKED_SCHEMA_VERSION,
            "tokens": _padded(tokens, seq_length, c.TOKEN_PAD),
            "labels": _padded(labels, seq_length, c.TOKEN_PAD),
            "loss_mask": _padded(loss_mask, seq_length, 0.0),
            "position_ids": _padded(position_ids, seq_length, 0),
            "sample_boundaries": boundaries,
            "tasks": tasks,
            "source_ids": source_ids,
        }

    for sample in samples:
        if sample.length > seq_length:
            if drop_overlong:
                continue
            raise ValueError(f"sample length {sample.length} exceeds seq_length {seq_length}")
        if current and current_length + sample.length > seq_length:
            packed = emit()
            if packed is not None:
                yield packed
            current = []
            current_length = 0
        current.append(sample)
        current_length += sample.length
    packed = emit()
    if packed is not None:
        yield packed


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--drop-overlong", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shifted = (make_shifted_sample(sample) for sample in iter_jsonl(Path(args.input)))
    packed = pack_samples(shifted, args.seq_length, drop_overlong=args.drop_overlong)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for item in packed:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    print(json.dumps({"output": str(output), "packed_sequences": count}, sort_keys=True))


if __name__ == "__main__":
    main()
