"""Exact one-coverage global shuffle for the immutable interleaved task pool."""

from __future__ import annotations

import hashlib
import math

import torch
from torch.utils.data import Dataset


def _seed(seed: int, cycle: int) -> int:
    value = hashlib.blake2b(f"{seed}:{cycle}".encode(), digest_size=8).digest()
    return int.from_bytes(value, "little") & ((1 << 63) - 1)


class OneFamilyCoverageSchedule(Dataset):
    """Globally shuffled records padded only to a complete optimizer batch."""

    def __init__(
        self,
        source: Dataset,
        *,
        total_samples: int,
        global_batch_size: int,
        data_parallel_group_size: int,
        shuffle_seed: int,
        split: str,
        require_full_coverage: bool = True,
    ) -> None:
        if (
            len(source) <= 0
            or total_samples <= 0
            or global_batch_size <= 0
            or data_parallel_group_size <= 0
        ):
            raise ValueError("invalid one-family schedule geometry")
        if total_samples % global_batch_size:
            raise ValueError("schedule must end on a global-batch boundary")
        minimum = len(source) if split == "train" and require_full_coverage else 1
        if total_samples < minimum:
            raise ValueError("schedule truncates the requested source coverage")
        self.source = source
        self.total_samples = int(total_samples)
        self.global_batch_size = int(global_batch_size)
        self.data_parallel_group_size = int(data_parallel_group_size)
        if self.global_batch_size % self.data_parallel_group_size:
            raise ValueError("global batch must be divisible by the DP group")
        self.shuffle_seed = int(shuffle_seed)
        self.total_blocks = self.total_samples // self.global_batch_size
        self.synchronize_task_family = True
        from megatron.core.datasets.utils import Split

        self.split = Split.train if split == "train" else Split.valid
        cycles = math.ceil(self.total_samples / len(self.source))
        self.permutations = [
            torch.randperm(
                len(self.source),
                generator=torch.Generator().manual_seed(_seed(self.shuffle_seed, cycle)),
            )
            for cycle in range(cycles)
        ]

    def __len__(self) -> int:
        return self.total_samples

    def source_index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        cycle, position = divmod(index, len(self.source))
        return int(self.permutations[cycle][position])

    def __getitem__(self, index: int):
        return self.source[self.source_index(index)]


__all__ = ["OneFamilyCoverageSchedule"]
