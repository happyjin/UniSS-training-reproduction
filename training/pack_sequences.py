"""Pack UniSS prompt/target samples into fixed-length language-model sequences.

The packer keeps sample boundaries explicit so a Megatron-LM dataset can reset
position IDs and block attention across packed samples. It also aligns loss masks
with next-token labels, so the first target token is learned from the last prompt
token.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training import constants_uniss as c


@dataclass(frozen=True)
class ShiftedSample:
    tokens: list[int]
    labels: list[int]
    loss_mask: list[int]
    position_ids: list[int]
    task: str
    source_id: str | None

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class PackedSequence:
    tokens: list[int]
    labels: list[int]
    loss_mask: list[int]
    position_ids: list[int]
    sample_boundaries: list[tuple[int, int]]
    tasks: list[str]
    source_ids: list[str | None]

    @property
    def seq_length(self) -> int:
        return len(self.tokens)


def load_jsonl_samples(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def _coerce_ids(value: object, name: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list, got {type(value).__name__}")
    ids: list[int] = []
    for token_id in value:
        if not isinstance(token_id, int):
            raise TypeError(f"{name} item must be int, got {type(token_id).__name__}")
        c.validate_token_id(token_id)
        ids.append(token_id)
    return ids


def make_shifted_sample(sample: Mapping[str, object]) -> ShiftedSample:
    prompt_ids = _coerce_ids(sample["prompt_ids"], "prompt_ids")
    target_ids = _coerce_ids(sample["target_ids"], "target_ids")
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    if not target_ids:
        raise ValueError("target_ids must be non-empty")

    input_ids = [*prompt_ids, *target_ids]
    tokens = input_ids[:-1]
    labels = input_ids[1:]

    prompt_len = len(prompt_ids)
    # labels[i] is predicted from tokens[i]. Target labels start when i+1
    # reaches prompt_len, so the first target loss is at i == prompt_len - 1.
    loss_mask = [1 if index >= prompt_len - 1 else 0 for index in range(len(labels))]
    position_ids = list(range(len(tokens)))

    return ShiftedSample(
        tokens=tokens,
        labels=labels,
        loss_mask=loss_mask,
        position_ids=position_ids,
        task=str(sample.get("task", "unknown")),
        source_id=str(sample["id"]) if sample.get("id") is not None else None,
    )


def pad_packed_sequence(
    packed: PackedSequence,
    seq_length: int,
    pad_token_id: int = c.TOKEN_PAD,
) -> PackedSequence:
    pad_len = seq_length - packed.seq_length
    if pad_len < 0:
        raise ValueError(f"packed sequence length {packed.seq_length} exceeds {seq_length}")
    if pad_len == 0:
        return packed
    return PackedSequence(
        tokens=[*packed.tokens, *([pad_token_id] * pad_len)],
        labels=[*packed.labels, *([pad_token_id] * pad_len)],
        loss_mask=[*packed.loss_mask, *([0] * pad_len)],
        position_ids=[*packed.position_ids, *([0] * pad_len)],
        sample_boundaries=packed.sample_boundaries,
        tasks=packed.tasks,
        source_ids=packed.source_ids,
    )


def pack_shifted_samples(
    samples: Iterable[ShiftedSample],
    seq_length: int,
    drop_overlong: bool = False,
) -> Iterator[PackedSequence]:
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")

    tokens: list[int] = []
    labels: list[int] = []
    loss_mask: list[int] = []
    position_ids: list[int] = []
    boundaries: list[tuple[int, int]] = []
    tasks: list[str] = []
    source_ids: list[str | None] = []

    def emit_current() -> PackedSequence | None:
        if not tokens:
            return None
        packed = PackedSequence(
            tokens=list(tokens),
            labels=list(labels),
            loss_mask=list(loss_mask),
            position_ids=list(position_ids),
            sample_boundaries=list(boundaries),
            tasks=list(tasks),
            source_ids=list(source_ids),
        )
        return pad_packed_sequence(packed, seq_length)

    def clear_current() -> None:
        tokens.clear()
        labels.clear()
        loss_mask.clear()
        position_ids.clear()
        boundaries.clear()
        tasks.clear()
        source_ids.clear()

    for sample in samples:
        if sample.length > seq_length:
            if drop_overlong:
                continue
            raise ValueError(f"sample length {sample.length} exceeds seq_length {seq_length}")

        if tokens and len(tokens) + sample.length > seq_length:
            packed = emit_current()
            if packed is not None:
                yield packed
            clear_current()

        start = len(tokens)
        end = start + sample.length
        tokens.extend(sample.tokens)
        labels.extend(sample.labels)
        loss_mask.extend(sample.loss_mask)
        position_ids.extend(sample.position_ids)
        boundaries.append((start, end))
        tasks.append(sample.task)
        source_ids.append(sample.source_id)

    packed = emit_current()
    if packed is not None:
        yield packed


def build_dense_attention_mask(boundaries: Sequence[tuple[int, int]], seq_length: int) -> list[list[int]]:
    """Build a small block-diagonal causal mask for tests/debugging.

    This is intentionally dense and should not be used for 18k-token production
    batches. Megatron integration should use boundaries to construct efficient
    packed-sequence metadata or reset masks.
    """

    mask = [[0 for _ in range(seq_length)] for _ in range(seq_length)]
    for start, end in boundaries:
        for row in range(start, end):
            for col in range(start, row + 1):
                mask[row][col] = 1
    return mask


def packed_to_json(packed: PackedSequence) -> dict[str, object]:
    return {
        "tokens": packed.tokens,
        "labels": packed.labels,
        "loss_mask": packed.loss_mask,
        "position_ids": packed.position_ids,
        "sample_boundaries": packed.sample_boundaries,
        "tasks": packed.tasks,
        "source_ids": packed.source_ids,
    }


def write_packed_jsonl(packed_sequences: Iterable[PackedSequence], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for packed in packed_sequences:
            handle.write(json.dumps(packed_to_json(packed), separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Sample JSONL files")
    parser.add_argument("--output", required=True, help="Packed JSONL output path")
    parser.add_argument("--seq-length", type=int, default=18_000)
    parser.add_argument("--drop-overlong", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_samples = (
        sample for path in args.input for sample in load_jsonl_samples(Path(path))
    )
    shifted = (make_shifted_sample(sample) for sample in raw_samples)
    packed = pack_shifted_samples(
        shifted, seq_length=args.seq_length, drop_overlong=args.drop_overlong
    )
    count = write_packed_jsonl(packed, Path(args.output))
    print(json.dumps({"output": args.output, "packed_sequences": count}, sort_keys=True))


if __name__ == "__main__":
    main()
