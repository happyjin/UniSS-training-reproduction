"""Megatron-compatible dataset for weighted Simul-UniSS packed JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import Dataset

from training.simul_uniss import PACKED_SCHEMA_VERSION


def _tensor(item: Mapping[str, object], key: str, length: int, dtype: torch.dtype) -> torch.Tensor:
    values = item[key]
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{key} must be a list of length {length}")
    return torch.tensor(values, dtype=dtype)


def boundaries_to_cu_seqlens(
    boundaries: Sequence[Sequence[int]], seq_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not boundaries:
        raise ValueError("sample_boundaries must not be empty")
    ends = [0]
    previous = 0
    lengths: list[int] = []
    for boundary in boundaries:
        if len(boundary) != 2:
            raise ValueError(f"invalid boundary: {boundary}")
        start, end = int(boundary[0]), int(boundary[1])
        if start != previous or end <= start or end > seq_length:
            raise ValueError(f"invalid contiguous boundary: {(start, end)}")
        ends.append(end)
        lengths.append(end - start)
        previous = end
    if previous < seq_length:
        lengths.append(seq_length - previous)
    padded = torch.full((seq_length + 1,), seq_length, dtype=torch.int32)
    padded[: len(ends)] = torch.tensor(ends, dtype=torch.int32)
    return padded, torch.tensor(max(lengths), dtype=torch.int32)


def packed_json_to_item(item: Mapping[str, object], seq_length: int) -> dict[str, torch.Tensor]:
    if item.get("schema_version") != PACKED_SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={PACKED_SCHEMA_VERSION}")
    boundaries = item["sample_boundaries"]
    if not isinstance(boundaries, list):
        raise TypeError("sample_boundaries must be a list")
    cu_seqlens, max_seqlen = boundaries_to_cu_seqlens(boundaries, seq_length)
    return {
        "tokens": _tensor(item, "tokens", seq_length, torch.int64),
        "labels": _tensor(item, "labels", seq_length, torch.int64),
        "loss_mask": _tensor(item, "loss_mask", seq_length, torch.float32),
        "position_ids": _tensor(item, "position_ids", seq_length, torch.int64),
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
    }


class SimulPackedJsonlDataset(Dataset):
    def __init__(self, path: str | Path, seq_length: int) -> None:
        self.path = Path(path)
        self.seq_length = seq_length
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.offsets: list[int] = []
        offset = 0
        with self.path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line)
        if not self.offsets:
            raise ValueError(f"{self.path} contains no samples")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            item = json.loads(handle.readline().decode("utf-8"))
        return packed_json_to_item(item, self.seq_length)
