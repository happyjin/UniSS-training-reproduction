"""PyTorch dataset adapter for UniSS packed samples.

The returned item schema mirrors the batch keys consumed by Megatron-LM's
``pretrain_gpt.py``: tokens, labels, loss_mask, position_ids, cu_seqlens, and
max_seqlen. Attention masks are intentionally omitted so Megatron can construct
PackedSeqParams from cu_seqlens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import Dataset


class UniSSPackedJsonlDataset(Dataset):
    def __init__(self, path: str | Path, seq_length: int) -> None:
        self.path = Path(path)
        self.seq_length = seq_length
        if seq_length <= 0:
            raise ValueError("seq_length must be positive")
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._byte_offsets = self._index_file()

    def _index_file(self) -> list[int]:
        offsets: list[int] = []
        offset = 0
        with self.path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    offsets.append(offset)
                offset += len(line)
        if not offsets:
            raise ValueError(f"{self.path} contains no packed samples")
        return offsets

    def __len__(self) -> int:
        return len(self._byte_offsets)

    def _read_item(self, index: int) -> dict[str, object]:
        offset = self._byte_offsets[index]
        with self.path.open("rb") as handle:
            handle.seek(offset)
            return json.loads(handle.readline().decode("utf-8"))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self._read_item(index)
        return packed_json_to_megatron_item(item, seq_length=self.seq_length)


class RepeatToLengthDataset(Dataset):
    """Repeat a finite map-style dataset until it reaches Megatron's target size."""

    def __init__(self, dataset: Dataset, length: int) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        if len(dataset) == 0:
            raise ValueError("dataset must be non-empty")
        self.dataset = dataset
        self.length = length
        self.split = getattr(dataset, "split", None)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return self.dataset[index % len(self.dataset)]


def _tensor_from_int_list(item: Mapping[str, object], key: str, length: int, dtype: torch.dtype) -> torch.Tensor:
    value = item[key]
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list, got {type(value).__name__}")
    if len(value) != length:
        raise ValueError(f"{key} length {len(value)} does not match seq_length {length}")
    if not all(isinstance(token, int) for token in value):
        raise TypeError(f"{key} must contain ints")
    return torch.tensor(value, dtype=dtype)


def _validate_boundaries(boundaries: Sequence[Sequence[int]], seq_length: int) -> list[tuple[int, int]]:
    if not boundaries:
        raise ValueError("sample_boundaries must be non-empty")
    parsed: list[tuple[int, int]] = []
    previous_end = 0
    for boundary in boundaries:
        if len(boundary) != 2:
            raise ValueError(f"invalid boundary {boundary!r}")
        start, end = int(boundary[0]), int(boundary[1])
        if start != previous_end:
            raise ValueError(f"boundaries must be contiguous: expected start {previous_end}, got {start}")
        if end <= start:
            raise ValueError(f"boundary end must exceed start: {(start, end)}")
        if end > seq_length:
            raise ValueError(f"boundary {boundary!r} exceeds seq_length {seq_length}")
        parsed.append((start, end))
        previous_end = end
    return parsed


def boundaries_to_padded_cu_seqlens(boundaries: Sequence[Sequence[int]], seq_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    parsed = _validate_boundaries(boundaries, seq_length)
    cu_values = [0]
    cu_values.extend(end for _, end in parsed)
    cu_seqlens = torch.full((seq_length + 1,), seq_length, dtype=torch.int32)
    cu_seqlens[: len(cu_values)] = torch.tensor(cu_values, dtype=torch.int32)
    lengths = [end - start for start, end in parsed]
    padding_tail_length = seq_length - parsed[-1][1]
    if padding_tail_length:
        lengths.append(padding_tail_length)
    max_seqlen = torch.tensor(max(lengths), dtype=torch.int32)
    return cu_seqlens, max_seqlen


def packed_json_to_megatron_item(item: Mapping[str, object], seq_length: int) -> dict[str, torch.Tensor]:
    tokens = _tensor_from_int_list(item, "tokens", seq_length, torch.int64)
    labels = _tensor_from_int_list(item, "labels", seq_length, torch.int64)
    loss_mask = _tensor_from_int_list(item, "loss_mask", seq_length, torch.float32)
    position_ids = _tensor_from_int_list(item, "position_ids", seq_length, torch.int64)

    boundaries = item.get("sample_boundaries")
    if not isinstance(boundaries, list):
        raise TypeError("sample_boundaries must be a list")
    cu_seqlens, max_seqlen = boundaries_to_padded_cu_seqlens(boundaries, seq_length)

    return {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
    }
