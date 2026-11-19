"""Task-mixing batch sampler for Stage 2.

The paper mixes ASR : S2TT : MT : TTS : S2ST = 0.2 : 1 : 0.5 : 1 : 1.5. Because a
batch can only carry one task (see `collators/multitask.py`), the ratio is
realised as a per-step categorical draw over tasks.

DDP requirement: every rank must pick the *same* task at the same step. If ranks
diverge, one computes a text-head loss while another computes an audio-head loss,
so the two produce gradients for disjoint parameter sets and the all-reduce
mismatches. The task sequence is therefore drawn from a seed shared by all ranks,
and only the within-task sample assignment is rank-dependent.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class TaskMixBatchSampler:
    """Emit batches drawn task-by-task according to fixed mixing weights.

    Each task supplies its own index list and per-sample cost (talker tokens for
    code tasks, thinker tokens otherwise); batches are packed against a token
    budget exactly as in `TokenBudgetBatchSampler`.
    """

    def __init__(
        self,
        task_indices: dict[str, Sequence[int]],
        task_lengths: dict[str, Sequence[int]],
        task_weights: dict[str, float],
        token_budget: int,
        max_batch_size: int = 64,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        steps_per_epoch: Optional[int] = None,
    ):
        assert task_indices.keys() == task_lengths.keys() == task_weights.keys(), (
            "task_indices, task_lengths and task_weights must cover the same tasks"
        )
        assert token_budget > 0
        assert world_size >= 1 and 0 <= rank < world_size

        self.tasks = sorted(t for t, w in task_weights.items() if w > 0)
        assert self.tasks, "no task has positive weight"
        for task in self.tasks:
            assert len(task_indices[task]) == len(task_lengths[task]), (
                f"{task}: index/length mismatch"
            )
            assert len(task_indices[task]) > 0, f"{task}: no samples"

        self.task_indices = {t: np.asarray(task_indices[t]) for t in self.tasks}
        self.task_lengths = {t: np.asarray(task_lengths[t]) for t in self.tasks}
        weights = np.array([task_weights[t] for t in self.tasks], dtype=np.float64)
        self.task_probs = weights / weights.sum()

        self.token_budget = int(token_budget)
        self.max_batch_size = int(max_batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

        # Per-epoch step count: one pass over the task with the largest
        # weight-normalised size, so no task is starved and none is oversampled
        # by more than its weight implies.
        if steps_per_epoch is None:
            batches_per_task = {
                t: self._estimate_batches(t) for t in self.tasks
            }
            scaled = [
                batches_per_task[t] / p for t, p in zip(self.tasks, self.task_probs)
            ]
            steps_per_epoch = int(min(scaled))
        self.steps_per_epoch = max(1, int(steps_per_epoch))

    def _estimate_batches(self, task: str) -> int:
        """How many budget-packed batches one pass over `task` yields."""
        lengths = np.sort(self.task_lengths[task])
        total = 0
        i = 0
        n = len(lengths)
        while i < n:
            # Batch size is limited by the longest member, which after sorting is
            # the last one taken.
            take = min(self.max_batch_size, n - i)
            while take > 1 and take * (int(lengths[i + take - 1]) + 3) > self.token_budget:
                take -= 1
            i += take
            total += 1
        return total

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _task_batches(self, task: str, rng: np.random.Generator) -> list[list[int]]:
        """Shuffled, budget-packed batches for one task, sharded across ranks."""
        indices = self.task_indices[task]
        lengths = self.task_lengths[task]
        order = rng.permutation(len(indices))
        # Sort within large chunks so batches are length-homogeneous without
        # becoming deterministic across epochs.
        chunk = max(self.max_batch_size * 32, 1024)
        packed: list[list[int]] = []
        for start in range(0, len(order), chunk):
            block = order[start : start + chunk]
            block = block[np.argsort(lengths[block], kind="stable")]
            current: list[int] = []
            current_max = 0
            for pos in block:
                length = int(lengths[pos])
                cand_max = max(current_max, length)
                if current and (
                    (len(current) + 1) * (cand_max + 3) > self.token_budget
                    or len(current) >= self.max_batch_size
                ):
                    packed.append(current)
                    current, current_max = [int(indices[pos])], length
                else:
                    current.append(int(indices[pos]))
                    current_max = cand_max
            if current:
                packed.append(current)
        rng.shuffle(packed)
        if self.world_size > 1:
            usable = (len(packed) // self.world_size) * self.world_size
            packed = packed[self.rank : usable : self.world_size]
        return packed

    def __iter__(self) -> Iterator[list[int]]:
        # Shared seed -> identical task sequence on every rank.
        task_rng = np.random.default_rng(self.seed + self.epoch)
        # Rank-specific seed -> different samples per rank within each task.
        data_rng = np.random.default_rng(
            self.seed + self.epoch * 1000 + self.rank + 1
        )

        logger.info(
            "packing task batches epoch=%s sizes=%s",
            self.epoch,
            {task: int(self.task_indices[task].shape[0]) for task in self.tasks},
        )
        pools = {t: self._task_batches(t, data_rng) for t in self.tasks}
        logger.info(
            "packed batches epoch=%s counts=%s",
            self.epoch,
            {task: len(pools[task]) for task in self.tasks},
        )
        cursors = {t: 0 for t in self.tasks}

        for _ in range(self.steps_per_epoch):
            task = self.tasks[int(task_rng.choice(len(self.tasks), p=self.task_probs))]
            pool = pools[task]
            if cursors[task] >= len(pool):
                # Exhausted this task for the epoch; reshuffle and continue so the
                # step count stays identical across ranks.
                pools[task] = self._task_batches(task, data_rng)
                pool = pools[task]
                cursors[task] = 0
            yield pool[cursors[task]]
            cursors[task] += 1

    def __len__(self) -> int:
        return self.steps_per_epoch
