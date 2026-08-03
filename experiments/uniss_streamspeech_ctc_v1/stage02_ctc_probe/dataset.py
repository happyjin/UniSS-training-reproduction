"""Mmap-backed probe dataset preserving latent-shard locality."""

from __future__ import annotations

import bisect
import json
import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


DIRECTION_TO_ID = {"eng->cmn": 0, "cmn->eng": 1}


class CTCProbeDataset(Dataset[dict[str, object]]):
    def __init__(self, dataset_index: str | Path, split: str) -> None:
        index = json.loads(Path(dataset_index).read_text(encoding="utf-8"))
        self.parts = []
        self.cumulative = []
        total = 0
        for entry in index["parts"][split]:
            records = int(entry["records"])
            if records == 0:
                continue
            offsets = np.memmap(entry["offsets"], mode="r", dtype=np.uint64)
            if len(offsets) != records:
                raise ValueError(f"offset count mismatch for {entry['manifest']}")
            self.parts.append((Path(entry["manifest"]), offsets, records))
            total += records
            self.cumulative.append(total)

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    @lru_cache(maxsize=16)
    def _load_shard(self, path: str) -> dict[str, object]:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        part_index = bisect.bisect_right(self.cumulative, index)
        before = self.cumulative[part_index - 1] if part_index else 0
        local_index = index - before
        path, offsets, _ = self.parts[part_index]
        with path.open("rb") as handle:
            handle.seek(int(offsets[local_index]))
            row = json.loads(handle.readline())
        shard = self._load_shard(str(row["shard_path"]))
        start, end = int(row["hidden_start"]), int(row["hidden_end"])
        hidden = shard["pre_vq_hidden"][start:end]  # type: ignore[index]
        if len(hidden) != int(row["hidden_frames"]):
            raise ValueError(f"hidden length mismatch for {row['id']}")
        return {
            "id": row["id"],
            "direction_id": DIRECTION_TO_ID[row["direction"]],
            "hidden": hidden,
            "source_token_ids": torch.tensor(row["source_token_ids"], dtype=torch.long),
            "target_token_ids": torch.tensor(row["target_token_ids"], dtype=torch.long),
        }


def collate_probe(batch: list[dict[str, object]]) -> dict[str, torch.Tensor | list[str]]:
    lengths = torch.tensor([len(row["hidden"]) for row in batch], dtype=torch.long)  # type: ignore[arg-type]
    hidden_size = int(batch[0]["hidden"].shape[-1])  # type: ignore[union-attr]
    hidden = torch.zeros(len(batch), int(lengths.max()), hidden_size, dtype=torch.bfloat16)
    source_lengths = torch.tensor(
        [len(row["source_token_ids"]) for row in batch], dtype=torch.long  # type: ignore[arg-type]
    )
    target_lengths = torch.tensor(
        [len(row["target_token_ids"]) for row in batch], dtype=torch.long  # type: ignore[arg-type]
    )
    for row_index, row in enumerate(batch):
        value = row["hidden"]
        hidden[row_index, : len(value)] = value  # type: ignore[arg-type]
    return {
        "ids": [str(row["id"]) for row in batch],
        "direction_ids": torch.tensor(
            [int(row["direction_id"]) for row in batch], dtype=torch.long
        ),
        "hidden": hidden,
        "hidden_lengths": lengths,
        "source_targets": torch.cat(
            [row["source_token_ids"] for row in batch]  # type: ignore[list-item]
        ),
        "source_lengths": source_lengths,
        "target_targets": torch.cat(
            [row["target_token_ids"] for row in batch]  # type: ignore[list-item]
        ),
        "target_lengths": target_lengths,
    }


class DistributedContiguousBatchSampler(Sampler[list[int]]):
    """Shuffle contiguous batches, retaining the source shard's I/O locality."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        rank: int,
        world_size: int,
        *,
        shuffle: bool,
        seed: int = 2026,
        drop_last: bool = True,
    ) -> None:
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _global_batches(self) -> list[list[int]]:
        stop = self.dataset_size
        batches = [
            list(range(start, min(start + self.batch_size, stop)))
            for start in range(0, stop, self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        usable = len(batches) - len(batches) % self.world_size
        return batches[:usable]

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._global_batches()
        yield from batches[self.rank :: self.world_size]

    def __len__(self) -> int:
        count = self.dataset_size // self.batch_size
        if not self.drop_last:
            count = math.ceil(self.dataset_size / self.batch_size)
        return (count - count % self.world_size) // self.world_size

