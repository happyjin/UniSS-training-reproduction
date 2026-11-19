"""Concatenate per-source datasets of one task, with integer upsampling.

A task pool that mixes languages (e.g. Emilia zh/en plus ja/ko TTS) is built as
one dataset per source so each source keeps its own root, filters and hour cap.
`repeat` lists a source's rows that many times, which is how a small language is
upsampled relative to a large one inside a single task.

Global index layout: source p occupies `[offset_p, offset_p + repeat_p * len_p)`,
copy k of local row i sitting at `offset_p + k * len_p + i`.
"""

from __future__ import annotations

import bisect
from typing import Sequence

from torch.utils.data import Dataset


class ConcatSourcesDataset(Dataset):
    def __init__(self, parts: Sequence[Dataset], repeats: Sequence[int] | None = None, names=None):
        assert parts, "no source datasets"
        self.parts = list(parts)
        self.repeats = [int(r) for r in (repeats or [1] * len(parts))]
        assert len(self.repeats) == len(self.parts) and all(r >= 1 for r in self.repeats)
        self.names = list(names or [str(i) for i in range(len(parts))])
        self.sizes = [len(p) for p in self.parts]
        self.offsets = [0]
        for size, rep in zip(self.sizes, self.repeats):
            self.offsets.append(self.offsets[-1] + size * rep)

    def __len__(self) -> int:
        return self.offsets[-1]

    def _locate(self, index: int) -> tuple[int, int]:
        p = bisect.bisect_right(self.offsets, index) - 1
        return p, (index - self.offsets[p]) % self.sizes[p]

    def __getitem__(self, index: int):
        p, local = self._locate(index)
        return self.parts[p][local]

    def code_length(self, index: int) -> int:
        p, local = self._locate(index)
        return self.parts[p].code_length(local)

    def token_length(self, index: int) -> int:
        p, local = self._locate(index)
        part = self.parts[p]
        getter = getattr(part, "token_length", None) or part.code_length
        return getter(local)

    def train_eval_split(self, eval_n: int) -> tuple[list[int], list[int]]:
        """Hold out the tail rows of every source, proportional to its size.

        Every repeated copy of a held-out row is dropped from training, so the
        eval slice stays disjoint from what the model trains on.
        """
        total = sum(self.sizes)
        held = [min(size, max(1, round(eval_n * size / total))) if eval_n else 0 for size in self.sizes]
        train, evals = [], []
        for p, (size, rep, h) in enumerate(zip(self.sizes, self.repeats, held)):
            base = self.offsets[p]
            evals.extend(base + i for i in range(size - h, size))
            for k in range(rep):
                start = base + k * size
                train.extend(range(start, start + size - h))
        return train, evals

    def describe(self) -> dict:
        return {name: {"rows": size, "repeat": rep}
                for name, size, rep in zip(self.names, self.sizes, self.repeats)}
