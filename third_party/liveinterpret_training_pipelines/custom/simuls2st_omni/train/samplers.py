"""Batch samplers for the training stages.

`QwenOmniTalkerTrainer.get_train_dataloader` passes `batch_sampler=` rather than
`sampler=`, which has three consequences:

1. Without a sampler the loader degrades to batch size 1.
2. DDP sharding must happen here; there is no `DistributedSampler` wrapper.
3. A batch carries a single `task_type`, and label validity is judged per batch,
   so task mixing can only happen *between* batches.

The trainer also discards any batch exceeding `max_seq_len`,
`max_talker_seq_len` or `max_talker_tokens_per_batch`, counting it as
`skipped_long_seq` (`modeling_omni_talker.py:776-809`). An over-long batch is
wasted outright rather than truncated, so batches are packed against a token
budget instead of a fixed batch size.
"""

from __future__ import annotations

import math
from typing import Iterator, Optional, Sequence

import numpy as np


class TokenBudgetBatchSampler:
    """Length-bucketed batches capped by a talker-token budget.

    Batch size is derived as `budget // longest_in_batch`, so short utterances
    batch wide and long ones batch narrow, keeping peak activation roughly
    constant and staying clear of the trainer's skip thresholds.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        token_budget: int,
        max_batch_size: int = 64,
        min_batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
    ):
        assert token_budget > 0, "token_budget must be positive"
        assert world_size >= 1 and 0 <= rank < world_size
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.token_budget = int(token_budget)
        self.max_batch_size = int(max_batch_size)
        self.min_batch_size = int(min_batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._cached_len: Optional[int] = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_batches(self) -> list[list[int]]:
        order = np.argsort(self.lengths, kind="stable")

        batches: list[list[int]] = []
        current: list[int] = []
        current_max = 0
        for idx in order:
            length = int(self.lengths[idx])
            candidate_max = max(current_max, length)
            # +3 mirrors the prefix/pad/bos overhead the collator adds.
            if current and (
                (len(current) + 1) * (candidate_max + 3) > self.token_budget
                or len(current) >= self.max_batch_size
            ):
                batches.append(current)
                current, current_max = [int(idx)], length
            else:
                current.append(int(idx))
                current_max = candidate_max
        if current:
            batches.append(current)

        batches = [b for b in batches if len(b) >= self.min_batch_size]

        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(batches)

        if self.world_size > 1:
            # Trim to a multiple of world_size so every rank runs the same number
            # of steps; unequal step counts deadlock DDP at the next all-reduce.
            usable = (len(batches) // self.world_size) * self.world_size
            batches = batches[self.rank : usable : self.world_size]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._build_batches()
        self._cached_len = len(batches)
        yield from batches

    def __len__(self) -> int:
        # HF Trainer needs this up front to compute total steps.
        if self._cached_len is None:
            self._cached_len = len(self._build_batches())
        return self._cached_len


class ConcatBatchSampler:
    """Chain per-task batch samplers whose batches index a local list.

    Each part is (global_indices, sampler); the sampler's positions are mapped
    through `global_indices`. Every part already trims to a multiple of
    world_size, so all ranks see the same number of batches.
    """

    def __init__(self, parts: Sequence[tuple[Sequence[int], "TokenBudgetBatchSampler"]]):
        self.parts = [(list(indices), sampler) for indices, sampler in parts]

    def set_epoch(self, epoch: int) -> None:
        for _, sampler in self.parts:
            sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        for indices, sampler in self.parts:
            for batch in sampler:
                yield [indices[i] for i in batch]

    def __len__(self) -> int:
        return sum(len(sampler) for _, sampler in self.parts)


class RepeatSampler:
    """Yield one fixed batch repeatedly, for overfit diagnostics.

    Used to answer a single question: with the inferred supervision window, can
    the talker memorise a handful of utterances? If the loss will not fall, the
    label placement is wrong and no amount of training will fix it.
    """

    def __init__(self, indices: Sequence[int], batch_size: int, steps: int):
        assert len(indices) > 0 and batch_size > 0 and steps > 0
        self.indices = list(indices)
        self.batch_size = int(batch_size)
        self.steps = int(steps)

    def set_epoch(self, epoch: int) -> None:
        del epoch

    def __iter__(self) -> Iterator[list[int]]:
        n_batches = math.ceil(len(self.indices) / self.batch_size)
        for step in range(self.steps):
            start = (step % n_batches) * self.batch_size
            batch = self.indices[start : start + self.batch_size]
            if batch:
                yield batch

    def __len__(self) -> int:
        return self.steps
