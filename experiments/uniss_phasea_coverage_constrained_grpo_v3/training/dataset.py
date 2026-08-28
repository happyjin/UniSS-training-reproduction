"""Memory-mapped action-warmup and event-GRPO packed datasets."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
from torch.utils.data import Dataset

from training.megatron_uniss_dataset import boundaries_to_padded_cu_seqlens


INT_FIELDS = ("tokens", "labels", "position_ids", "family_ids")
FLOAT_FIELDS = ("loss_mask", "response_mask", "action_mask", "replay_mask")


class EventPolicyPackedDataset(Dataset[dict[str, object]]):
    def __init__(self, path: str | Path, seq_length: int):
        self.path = Path(path)
        self.seq_length = int(seq_length)
        index = self.path.with_suffix(self.path.suffix + ".offsets.bin")
        if not self.path.is_file() or not index.is_file():
            raise FileNotFoundError(self.path if not self.path.is_file() else index)
        raw = index.read_bytes()
        if len(raw) % 8:
            raise ValueError("dataset offset index is malformed")
        self.offsets = list(struct.unpack(f"<{len(raw)//8}Q", raw))
        if not self.offsets:
            raise ValueError("event policy dataset is empty")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, object]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[int(index)])
            row = json.loads(handle.readline())
        output: dict[str, object] = {}
        for name in INT_FIELDS:
            if len(row[name]) != self.seq_length:
                raise ValueError(f"{name} length differs from seq_length")
            output[name] = torch.tensor(row[name], dtype=torch.long)
        for name in FLOAT_FIELDS:
            if len(row[name]) != self.seq_length:
                raise ValueError(f"{name} length differs from seq_length")
            output[name] = torch.tensor(row[name], dtype=torch.float32)
        boundaries = [tuple(map(int, value)) for value in row["sample_boundaries"]]
        output["sample_boundaries"] = boundaries
        output["cu_seqlens"], output["max_seqlen"] = boundaries_to_padded_cu_seqlens(
            boundaries, self.seq_length
        )
        return output


def collate_event_policy(batch: list[dict[str, object]]) -> dict[str, object]:
    if not batch:
        raise ValueError("cannot collate an empty event policy batch")
    output = {
        name: torch.stack([row[name] for row in batch])
        for name in (*INT_FIELDS, *FLOAT_FIELDS, "cu_seqlens", "max_seqlen")
    }
    output["sample_boundaries"] = [row["sample_boundaries"] for row in batch]
    return output


__all__ = ["EventPolicyPackedDataset", "collate_event_policy"]

