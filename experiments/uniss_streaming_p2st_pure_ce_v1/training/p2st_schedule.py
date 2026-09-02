"""Restart-exact global-step schedule over the three prefix-to-prefix families.

Differences from the five-family schedule this replaces
-------------------------------------------------------
``FiveFamilyGlobalSchedule`` is wired to ``TASK_FAMILIES``, to
``FAMILY_INTERLEAVED`` as the coverage anchor, and to a three-phase weight ramp
that exists to bring the incremental-MT family in gradually.  None of that
transfers: there are three families here, none of them is primary, and there is
nothing to ramp because no family depends on another being learned first.

So the weights are uniform and constant -- one third of the optimizer steps
each -- which is the plain reading of "pure shuffle" across the three tasks.
Note what that allocates: a family's share is measured in *steps*, not tokens.
The three differ by an order of magnitude in supervised tokens (on the valid
pool, 4.92M for TTS against 0.71M for ASR and 0.41M for MT), and weighting by
tokens instead would hand TTS most of the run.  Each loss is a mean over its
own denominator inside a step, so equal steps is equal say.

Coverage is defined over every family rather than an anchor: one coverage epoch
means each family's packed rows have each been visited at least once, so the
block count is set by whichever family needs the most.

The retained property is restart exactness.  Family order and per-family sample
order both come from ``shuffle_seed`` alone, so a run resumed at step N sees the
same batch it would have seen without the interruption.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch.utils.data import Dataset

from experiments.uniss_streaming_p2st_pure_ce_v1.training.task_samples_p2st import (
    FAMILY_PHASE3_PERFORMANCE,
    FAMILY_PHASE3_QUALITY,
    FAMILY_P2ST_ASR,
    FAMILY_P2ST_MT,
    FAMILY_P2ST_TTS,
    POOL_FAMILIES,
    P2ST_FAMILIES,
)

UNIFORM_WEIGHTS = {family: 1.0 / len(P2ST_FAMILIES) for family in P2ST_FAMILIES}

# With replay mixed in, the three task families stay uniform among themselves
# and the replay share is taken from the lineage's own steady state rather than
# invented: the interleaved schedule's STEADY_WEIGHTS give phase3 replay
# 0.15 + 0.10 = 0.25, and B''s realised geometry spent 321 of 1132 blocks on it.
# Matching that number means the anti-forgetting pressure here is the same
# strength this血脉 has always used, so a difference in forgetting is
# attributable to the pure-CE change and not to a new mixing ratio.
POOL_WEIGHTS = {
    FAMILY_P2ST_ASR: 0.25,
    FAMILY_P2ST_MT: 0.25,
    FAMILY_P2ST_TTS: 0.25,
    FAMILY_PHASE3_QUALITY: 0.15,
    FAMILY_PHASE3_PERFORMANCE: 0.10,
}
assert abs(sum(POOL_WEIGHTS.values()) - 1.0) < 1e-12


def scheduled_families(weights: Mapping[str, float]) -> tuple[str, ...]:
    """The families a weight map actually schedules, in canonical order.

    Deriving this from the weights rather than from a module constant is what
    lets ``UNIFORM_WEIGHTS`` keep producing exactly the old three-family
    schedule -- bit-identically, since the order is unchanged -- while
    ``POOL_WEIGHTS`` produces five.
    """
    return tuple(
        family for family in POOL_FAMILIES if float(weights.get(family, 0.0)) > 0.0
    )


def _stable_seed(seed: int, *values: object) -> int:
    digest = hashlib.blake2b(
        ":".join([str(seed), *(str(value) for value in values)]).encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def largest_remainder(
    total: int,
    weights: Mapping[str, float],
    families: Sequence[str] | None = None,
) -> dict[str, int]:
    """Split ``total`` by ``weights`` with no block lost to rounding.

    ``families`` defaults to whichever families ``weights`` actually schedules,
    so the same call works for the three-family task-only map and the
    five-family map that includes phase3 replay.
    """
    if families is None:
        families = scheduled_families(weights)
    if total < 0 or set(weights) != set(families):
        raise ValueError("invalid p2st family allocation geometry")
    if not math.isclose(sum(float(v) for v in weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("p2st family weights do not sum to one")
    exact = {name: total * float(weights[name]) for name in families}
    output = {name: math.floor(exact[name]) for name in families}
    remaining = total - sum(output.values())
    order = sorted(
        families,
        key=lambda name: (exact[name] - output[name], -families.index(name)),
        reverse=True,
    )
    for name in order[:remaining]:
        output[name] += 1
    return output


def family_blocks(
    total_blocks: int,
    *,
    seed: int,
    weights: Mapping[str, float] = UNIFORM_WEIGHTS,
) -> tuple[str, ...]:
    """One family per global block, shuffled once over the whole run."""
    if total_blocks <= 0:
        raise ValueError("p2st schedule must contain global blocks")
    counts = largest_remainder(total_blocks, weights)
    families = scheduled_families(weights)
    blocks = [family for family in families for _ in range(counts[family])]
    generator = torch.Generator().manual_seed(_stable_seed(seed, "p2st_blocks"))
    order = torch.randperm(len(blocks), generator=generator).tolist()
    return tuple(blocks[index] for index in order)


def required_total_blocks(
    family_rows: Mapping[str, int],
    *,
    global_batch_size: int,
    coverage_epochs: int,
    seed: int,
    weights: Mapping[str, float] = UNIFORM_WEIGHTS,
) -> int:
    """Blocks needed for every family to be covered ``coverage_epochs`` times."""
    if global_batch_size <= 0 or coverage_epochs <= 0:
        raise ValueError("invalid p2st coverage geometry")
    families = scheduled_families(weights)
    if set(family_rows) != set(families) or any(
        int(value) <= 0 for value in family_rows.values()
    ):
        raise ValueError("p2st coverage needs a positive row count per family")
    needed = {
        family: math.ceil(coverage_epochs * int(rows) / global_batch_size)
        for family, rows in family_rows.items()
    }
    total = max(
        1,
        max(
            math.ceil(needed[family] / float(weights[family]))
            for family in families
        ),
    )
    # Rounding can leave a family one block short of its requirement, so close
    # the gap by measurement rather than by adding a fudge factor.
    while True:
        blocks = family_blocks(total, seed=seed, weights=weights)
        if all(blocks.count(family) >= needed[family] for family in families):
            return total
        total += 1


@dataclass(frozen=True)
class FamilyScheduledIndex:
    global_block: int
    family: str
    source_index: int
    source_cycle: int


class _BlockSchedule(Dataset):
    """Shared indexing for the train and validation schedules."""

    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        *,
        blocks: Sequence[str],
        global_batch_size: int,
        data_parallel_group_size: int,
        shuffle_seed: int,
        shuffle_samples: bool,
    ) -> None:
        # Whichever families this schedule's blocks name -- three for a
        # task-only run, five once phase3 replay is mixed in -- must all be
        # present and non-empty.  Deriving the set from the blocks keeps the
        # three-family smoke schedules working unchanged.
        families = tuple(dict.fromkeys(blocks))
        if not families:
            raise ValueError("p2st schedule must contain global blocks")
        if set(datasets) != set(families) or any(
            len(datasets[name]) <= 0 for name in families
        ):
            raise ValueError(
                "p2st schedule requires a non-empty dataset for each of "
                f"{sorted(families)}"
            )
        if global_batch_size <= 0 or data_parallel_group_size <= 0:
            raise ValueError("p2st schedule batch geometry must be positive")
        if global_batch_size % data_parallel_group_size:
            raise ValueError("global batch must be divisible by the DP group")
        self.datasets = dict(datasets)
        self.blocks = tuple(blocks)
        self.total_blocks = len(self.blocks)
        self.global_batch_size = int(global_batch_size)
        self.data_parallel_group_size = int(data_parallel_group_size)
        self.shuffle_seed = int(shuffle_seed)
        self.total_samples = self.total_blocks * self.global_batch_size
        # Megatron reads both of these off the dataset object.
        self.synchronize_task_family = True
        from megatron.core.datasets.utils import Split

        self.split = Split.train
        self.families = tuple(
            family for family in POOL_FAMILIES if family in set(families)
        )
        self.family_block_counts = {
            family: self.blocks.count(family) for family in self.families
        }
        running = {family: 0 for family in self.families}
        self._block_family_ordinals: list[int] = []
        for family in self.blocks:
            self._block_family_ordinals.append(running[family])
            running[family] += 1
        self._permutations: dict[tuple[str, int], torch.Tensor] = {}
        if shuffle_samples:
            for family in self.families:
                samples = self.family_block_counts[family] * self.global_batch_size
                cycles = math.ceil(samples / len(self.datasets[family]))
                for cycle in range(cycles):
                    self._permutations[(family, cycle)] = torch.randperm(
                        len(self.datasets[family]),
                        generator=torch.Generator().manual_seed(
                            _stable_seed(self.shuffle_seed, family, cycle)
                        ),
                    )

    def __len__(self) -> int:
        return self.total_samples

    def scheduled_index(self, index: int) -> FamilyScheduledIndex:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        global_block, lane = divmod(index, self.global_batch_size)
        family = self.blocks[global_block]
        family_flat = (
            self._block_family_ordinals[global_block] * self.global_batch_size + lane
        )
        dataset_length = len(self.datasets[family])
        cycle, position = divmod(family_flat, dataset_length)
        permutation = self._permutations.get((family, cycle))
        source = int(permutation[position]) if permutation is not None else position
        return FamilyScheduledIndex(global_block, family, source, cycle)

    def __getitem__(self, index: int):
        scheduled = self.scheduled_index(index)
        value = dict(self.datasets[scheduled.family][scheduled.source_index])
        if value.get("family") != scheduled.family:
            raise ValueError("scheduled p2st family disagrees with source dataset")
        return value


class ThreeFamilyGlobalSchedule(_BlockSchedule):
    """All samples in one optimizer global batch come from one task family."""

    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        *,
        coverage_epochs: int,
        global_batch_size: int,
        data_parallel_group_size: int,
        shuffle_seed: int,
        weights: Mapping[str, float] = UNIFORM_WEIGHTS,
    ) -> None:
        rows = {family: len(datasets[family]) for family in scheduled_families(weights)}
        total_blocks = required_total_blocks(
            rows,
            global_batch_size=int(global_batch_size),
            coverage_epochs=int(coverage_epochs),
            seed=int(shuffle_seed),
            weights=weights,
        )
        super().__init__(
            datasets,
            blocks=family_blocks(
                total_blocks, seed=int(shuffle_seed), weights=weights
            ),
            global_batch_size=global_batch_size,
            data_parallel_group_size=data_parallel_group_size,
            shuffle_seed=shuffle_seed,
            shuffle_samples=True,
        )
        self.coverage_epochs = int(coverage_epochs)
        self.weights = dict(weights)


class ThreeFamilyValidationSchedule(_BlockSchedule):
    """Deterministic validation blocks that never mix task families."""

    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        *,
        total_samples: int,
        global_batch_size: int,
        data_parallel_group_size: int,
        shuffle_seed: int,
        weights: Mapping[str, float] = UNIFORM_WEIGHTS,
    ) -> None:
        if total_samples <= 0:
            raise ValueError("p2st validation needs a positive sample count")
        total_blocks = max(1, math.ceil(int(total_samples) / int(global_batch_size)))
        families = scheduled_families(weights)
        # Round-robin rather than shuffled: a validation curve is only
        # comparable across steps if every evaluation sees the same batches in
        # the same order.
        blocks = tuple(
            families[index % len(families)] for index in range(total_blocks)
        )
        super().__init__(
            datasets,
            blocks=blocks,
            global_batch_size=global_batch_size,
            data_parallel_group_size=data_parallel_group_size,
            shuffle_seed=shuffle_seed,
            shuffle_samples=False,
        )
        from megatron.core.datasets.utils import Split

        self.split = Split.valid


__all__ = [
    "POOL_WEIGHTS",
    "UNIFORM_WEIGHTS",
    "scheduled_families",
    "ThreeFamilySchedulePrefix",
    "ThreeFamilySingleBlock",
    "FamilyScheduledIndex",
    "ThreeFamilyGlobalSchedule",
    "ThreeFamilyValidationSchedule",
    "family_blocks",
    "largest_remainder",
    "required_total_blocks",
]


class ThreeFamilySchedulePrefix(Dataset):
    """Global-batch-aligned prefix used only by bounded smoke runs.

    Takes the first N blocks of a full schedule rather than a fresh short
    schedule, so a smoke sees exactly the batches a formal run would see
    first.  The base experiment's equivalent hard-codes the five families in
    its block tally, which is the only reason this exists.
    """

    def __init__(self, dataset: Dataset, total_samples: int) -> None:
        global_batch_size = int(getattr(dataset, "global_batch_size", 0))
        if (
            total_samples <= 0
            or total_samples > len(dataset)
            or global_batch_size <= 0
            or total_samples % global_batch_size
        ):
            raise ValueError("invalid p2st smoke schedule prefix")
        self.dataset = dataset
        self.total_samples = int(total_samples)
        self.global_batch_size = global_batch_size
        self.data_parallel_group_size = int(dataset.data_parallel_group_size)
        self.total_blocks = self.total_samples // self.global_batch_size
        self.blocks = tuple(getattr(dataset, "blocks"))[: self.total_blocks]
        self.families = tuple(getattr(dataset, "families", POOL_FAMILIES))
        self.family_block_counts = {
            family: self.blocks.count(family) for family in self.families
        }
        self.synchronize_task_family = True
        self.split = getattr(dataset, "split")

    def __len__(self) -> int:
        return self.total_samples

    def scheduled_index(self, index: int) -> FamilyScheduledIndex:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.dataset.scheduled_index(index)

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.dataset[index]


class ThreeFamilySingleBlock(Dataset):
    """One explicit family block for an isolated one-update GPU canary.

    A smoke run is capped at two optimizer updates and one family owns a whole
    global batch, so a mixed smoke can never touch all three families.  This
    makes the family explicit instead, which is also the stronger check: it is
    what proves each family's own losses fire rather than that the run survived.
    """

    def __init__(self, dataset: Dataset, family: str) -> None:
        if family not in POOL_FAMILIES:
            raise ValueError(f"unknown p2st canary family {family!r}")
        blocks = tuple(getattr(dataset, "blocks", ()))
        if family not in blocks:
            raise ValueError("p2st canary family has no scheduled block")
        self.dataset = dataset
        self.family = family
        self.global_batch_size = int(getattr(dataset, "global_batch_size", 0))
        self.data_parallel_group_size = int(
            getattr(dataset, "data_parallel_group_size", 0)
        )
        if self.global_batch_size <= 0 or self.data_parallel_group_size <= 0:
            raise ValueError("invalid p2st canary schedule geometry")
        self.source_block = blocks.index(family)
        self.total_blocks = 1
        self.blocks = (family,)
        self.families = tuple(getattr(dataset, "families", POOL_FAMILIES))
        self.family_block_counts = {
            name: int(name == family) for name in self.families
        }
        self.synchronize_task_family = True
        self.split = getattr(dataset, "split", None)

    def __len__(self) -> int:
        return self.global_batch_size

    def scheduled_index(self, index: int) -> FamilyScheduledIndex:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.dataset.scheduled_index(
            self.source_block * self.global_batch_size + index
        )

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        value = self.dataset[self.source_block * self.global_batch_size + index]
        if value.get("family") != self.family:
            raise ValueError("p2st canary block escaped its requested family")
        return value
