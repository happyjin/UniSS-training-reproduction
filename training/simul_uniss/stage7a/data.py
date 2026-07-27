"""Streaming, shuffled Stage7A action samples and dynamic token batches."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from training import constants_uniss as c
from training.simul_uniss import SAMPLE_SCHEMA_VERSION
from training.simul_uniss.shuffle import buffered_shuffle

ACTION_TO_BINARY = {
    c.TOKEN_WAIT_READ: 0,
    c.TOKEN_WRITE_GENERATE: 1,
}


@dataclass(frozen=True)
class ActionSample:
    sample_id: str
    input_ids: list[int]
    prediction_positions: list[int]
    labels: list[int]
    event_fractions: list[float]
    final_flags: list[bool]

    @property
    def length(self) -> int:
        return len(self.input_ids)

    @property
    def events(self) -> int:
        return len(self.labels)


@dataclass
class ActionBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    selected_rows: torch.Tensor
    selected_positions: torch.Tensor
    labels: torch.Tensor
    event_sample_ids: torch.Tensor
    event_fractions: torch.Tensor
    final_flags: torch.Tensor
    sample_ids: list[str]
    sample_event_counts: list[int]
    actual_tokens: int

    @property
    def samples(self) -> int:
        return len(self.sample_ids)

    @property
    def events(self) -> int:
        return int(self.labels.numel())

    @property
    def padded_tokens(self) -> int:
        return int(self.input_ids.numel())

    def to(self, device: torch.device) -> ActionBatch:
        for name in (
            "input_ids",
            "attention_mask",
            "selected_rows",
            "selected_positions",
            "labels",
            "event_sample_ids",
            "event_fractions",
            "final_flags",
        ):
            setattr(self, name, getattr(self, name).to(device, non_blocking=True))
        return self


def parse_action_sample(
    item: dict[str, object], *, max_sequence_length: int
) -> ActionSample:
    if item.get("schema_version") != SAMPLE_SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={SAMPLE_SCHEMA_VERSION}")
    if item.get("task") not in {"simul_action", "simul_s2st"}:
        raise ValueError(
            "expected task=simul_action or simul_s2st, "
            f"got {item.get('task')!r}"
        )
    values = item.get("input_ids")
    if not isinstance(values, list) or not all(
        isinstance(value, int) for value in values
    ):
        raise TypeError("input_ids must be a list of ints")
    input_ids = [int(value) for value in values]
    if len(input_ids) > max_sequence_length:
        raise OverflowError(
            f"sample {item.get('id')} length {len(input_ids)} exceeds {max_sequence_length}"
        )
    action_positions = [
        index
        for index, token_id in enumerate(input_ids)
        if token_id in ACTION_TO_BINARY
    ]
    if not action_positions or action_positions[0] == 0:
        raise ValueError(f"sample {item.get('id')} contains no valid action positions")
    labels = [ACTION_TO_BINARY[input_ids[index]] for index in action_positions]
    event_count = len(labels)
    return ActionSample(
        sample_id=str(item.get("id", "")),
        input_ids=input_ids,
        prediction_positions=[index - 1 for index in action_positions],
        labels=labels,
        event_fractions=[(index + 1) / event_count for index in range(event_count)],
        final_flags=[index == event_count - 1 for index in range(event_count)],
    )


def _rank_items(
    path: Path,
    *,
    rank: int,
    world_size: int,
    max_sequence_length: int,
    skip_overlong: bool,
) -> Iterator[ActionSample]:
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line_index % world_size != rank or not line.strip():
                continue
            item = json.loads(line)
            try:
                yield parse_action_sample(item, max_sequence_length=max_sequence_length)
            except OverflowError:
                if not skip_overlong:
                    raise


def iter_action_samples(
    path: str | Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    max_sequence_length: int = 18_000,
    shuffle_buffer_size: int = 8192,
    seed: int = 20260727,
    skip_overlong: bool = True,
) -> Iterator[ActionSample]:
    """Yield deterministic per-rank samples, changing shuffle order every epoch."""

    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    epoch = 0
    while True:
        ordered = _rank_items(
            data_path,
            rank=rank,
            world_size=world_size,
            max_sequence_length=max_sequence_length,
            skip_overlong=skip_overlong,
        )
        yield from buffered_shuffle(ordered, shuffle_buffer_size, seed + epoch)
        epoch += 1


def iter_action_samples_once(
    path: str | Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    max_sequence_length: int = 18_000,
    skip_overlong: bool = True,
) -> Iterator[ActionSample]:
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    yield from _rank_items(
        data_path,
        rank=rank,
        world_size=world_size,
        max_sequence_length=max_sequence_length,
        skip_overlong=skip_overlong,
    )


def collate_action_samples(samples: list[ActionSample]) -> ActionBatch:
    if not samples:
        raise ValueError("cannot collate an empty action batch")
    max_length = max(sample.length for sample in samples)
    input_ids = torch.full((len(samples), max_length), c.TOKEN_PAD, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    selected_rows: list[int] = []
    selected_positions: list[int] = []
    labels: list[int] = []
    event_sample_ids: list[int] = []
    event_fractions: list[float] = []
    final_flags: list[bool] = []
    actual_tokens = 0
    for row, sample in enumerate(samples):
        length = sample.length
        input_ids[row, :length] = torch.tensor(sample.input_ids, dtype=torch.long)
        attention_mask[row, :length] = 1
        actual_tokens += length
        selected_rows.extend([row] * sample.events)
        selected_positions.extend(sample.prediction_positions)
        labels.extend(sample.labels)
        event_sample_ids.extend([row] * sample.events)
        event_fractions.extend(sample.event_fractions)
        final_flags.extend(sample.final_flags)
    return ActionBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        selected_rows=torch.tensor(selected_rows, dtype=torch.long),
        selected_positions=torch.tensor(selected_positions, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        event_sample_ids=torch.tensor(event_sample_ids, dtype=torch.long),
        event_fractions=torch.tensor(event_fractions, dtype=torch.float32),
        final_flags=torch.tensor(final_flags, dtype=torch.bool),
        sample_ids=[sample.sample_id for sample in samples],
        sample_event_counts=[sample.events for sample in samples],
        actual_tokens=actual_tokens,
    )


def batch_action_samples(
    samples: Iterable[ActionSample],
    *,
    max_batch_tokens: int,
    max_batch_size: int,
) -> Iterator[ActionBatch]:
    if max_batch_tokens < 1 or max_batch_size < 1:
        raise ValueError("batch limits must be positive")
    pending: list[ActionSample] = []
    max_length = 0
    for sample in samples:
        next_max = max(max_length, sample.length)
        next_size = len(pending) + 1
        if pending and (
            next_size > max_batch_size or next_max * next_size > max_batch_tokens
        ):
            yield collate_action_samples(pending)
            pending = []
            max_length = 0
            next_max = sample.length
        if sample.length > max_batch_tokens:
            raise ValueError(
                f"sample {sample.sample_id} length {sample.length} exceeds max_batch_tokens"
            )
        pending.append(sample)
        max_length = next_max
    if pending:
        yield collate_action_samples(pending)
