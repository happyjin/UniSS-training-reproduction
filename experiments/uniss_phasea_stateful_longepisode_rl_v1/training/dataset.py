"""Memory-mapped JSONL dataset for free-running episode GRPO packs."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
from torch.utils.data import Dataset

from training.megatron_uniss_dataset import boundaries_to_padded_cu_seqlens


INT_FIELDS = ("tokens", "labels", "position_ids", "family_ids")
FLOAT_FIELDS = (
    "loss_mask",
    "response_mask",
    "old_log_probs",
    "advantages",
    "replay_mask",
)


class EpisodeGRPOPackedDataset(Dataset[dict[str, object]]):
    def __init__(self, path: str | Path, seq_length: int, *, target_length: int | None = None):
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
            raise ValueError("episode GRPO dataset is empty")
        self.target_length = int(target_length) if target_length is not None else len(self.offsets)
        if self.target_length <= 0:
            raise ValueError("target length must be positive")

    def __len__(self) -> int:
        return self.target_length

    def __getitem__(self, index: int) -> dict[str, object]:
        offset = self.offsets[int(index) % len(self.offsets)]
        with self.path.open("rb") as handle:
            handle.seek(offset)
            row = json.loads(handle.readline())
        output: dict[str, object] = {}
        for name in INT_FIELDS:
            values = row[name]
            if len(values) != self.seq_length:
                raise ValueError(f"{name} length differs from seq_length")
            output[name] = torch.tensor(values, dtype=torch.long)
        for name in FLOAT_FIELDS:
            values = row[name]
            if len(values) != self.seq_length:
                raise ValueError(f"{name} length differs from seq_length")
            output[name] = torch.tensor(values, dtype=torch.float32)
        boundaries = row["sample_boundaries"]
        output["sample_boundaries"] = [tuple(map(int, value)) for value in boundaries]
        output["cu_seqlens"], output["max_seqlen"] = boundaries_to_padded_cu_seqlens(
            boundaries, self.seq_length
        )
        return output


def collate_episode_grpo(batch: list[dict[str, object]]) -> dict[str, object]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    output = {
        name: torch.stack([row[name] for row in batch])
        for name in (*INT_FIELDS, *FLOAT_FIELDS, "cu_seqlens", "max_seqlen")
    }
    output["sample_boundaries"] = [row["sample_boundaries"] for row in batch]
    return output


__all__ = ["EpisodeGRPOPackedDataset", "collate_episode_grpo"]

