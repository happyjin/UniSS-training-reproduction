"""Deterministic rank-local schedules for paired S4 context-view experiments."""

from __future__ import annotations

import hashlib
import operator
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP
from typing import Iterable, Iterator, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


ViewKind = Literal["full", "thinker_crop", "talker_crop"]
SCHEDULE_SCHEMA_VERSION = 1
_MASK64 = (1 << 64) - 1
_MAX_SIGNED_63 = (1 << 63) - 1


@dataclass(frozen=True)
class S4ViewIndex:
    index: int
    designated_kind: ViewKind
    applied_kind: ViewKind
    draw_seed: int


def _splitmix64(value: int) -> int:
    """Stable integer mixer; deliberately independent of Python's randomized hash."""
    z = (value + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def _generator_seed(seed: int, epoch: int, rank: int, world_size: int) -> int:
    mixed = _splitmix64(SCHEDULE_SCHEMA_VERSION)
    for value in (seed, epoch, rank, world_size):
        mixed = _splitmix64(mixed ^ (int(value) & _MASK64))
    return mixed & _MAX_SIGNED_63


def context_view_replacement_seed(draw_seed: int, designated_kind: ViewKind) -> int:
    """Derive a stable per-draw lazy-admission seed without touching global RNG."""
    try:
        draw_seed = operator.index(draw_seed)
    except TypeError as exc:
        raise ValueError("draw_seed must be an integer") from exc
    kind_code = {"full": 0, "thinker_crop": 1, "talker_crop": 2}.get(
        designated_kind)
    if kind_code is None:
        raise ValueError(f"unsupported designated view kind: {designated_kind!r}")
    mixed = _splitmix64(draw_seed)
    return _splitmix64(mixed ^ kind_code) & _MAX_SIGNED_63


@dataclass
class ViewScheduleCounters:
    scheduled_full: int = 0
    scheduled_thinker_crop: int = 0
    applied_full: int = 0
    applied_thinker_crop: int = 0
    replacements: int = 0
    failures: Counter | None = None

    def __post_init__(self):
        if self.failures is None:
            self.failures = Counter()

    def update(self, batch: dict) -> None:
        designated = batch.get("designated_view_kinds")
        applied = batch.get("view_kinds")
        if designated is None and applied is None:
            return
        if not isinstance(designated, list) or not isinstance(applied, list):
            raise ValueError("view counter batch metadata must be lists")
        if len(designated) != len(applied):
            raise ValueError("designated/applied view metadata lengths differ")
        self.scheduled_full += designated.count("full")
        self.scheduled_thinker_crop += designated.count("thinker_crop")
        self.applied_full += applied.count("full")
        self.applied_thinker_crop += applied.count("thinker_crop")
        replacement_counts = batch.get("replacement_counts", [])
        if torch.is_tensor(replacement_counts):
            self.replacements += int(replacement_counts.sum().item())
        else:
            self.replacements += sum(int(value) for value in replacement_counts)
        for reasons in batch.get("replacement_reasons", []):
            self.failures.update(str(reason) for reason in reasons)


def assert_view_schedule_counters(counters: ViewScheduleCounters, apply_crop: bool) -> None:
    """Fail if realized views violate the scheduled experiment/control contract."""
    if apply_crop:
        if counters.applied_thinker_crop != counters.scheduled_thinker_crop:
            raise AssertionError(
                "experiment applied Thinker crop count does not match scheduled count: "
                f"{counters.applied_thinker_crop} != {counters.scheduled_thinker_crop}"
            )
    elif counters.applied_thinker_crop != 0:
        raise AssertionError(
            f"paired control applied {counters.applied_thinker_crop} Thinker crops"
        )


def _round_half_up_nonnegative(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _local_hamilton_quota(
    num_samples: int, world_size: int, rank: int, ratio: float
) -> int:
    ratio_decimal = Decimal(str(ratio))
    ideal = Decimal(num_samples) * ratio_decimal
    base = int(ideal.to_integral_value(rounding=ROUND_FLOOR))
    global_quota = _round_half_up_nonnegative(
        Decimal(num_samples * world_size) * ratio_decimal
    )
    extras = global_quota - base * world_size
    if not 0 <= extras <= world_size:
        raise ValueError("Hamilton crop quota is inconsistent with the requested ratio")
    return base + int(rank < extras)


def _eligible_indices(values: Iterable[int] | Sequence[bool], size: int, name: str) -> list[int]:
    if isinstance(values, torch.Tensor):
        raw = values.detach().cpu().numpy()
    elif isinstance(values, np.ndarray):
        raw = values
    else:
        raw = list(values)

    if isinstance(raw, np.ndarray) and raw.dtype == np.bool_:
        if raw.ndim != 1 or raw.size != size:
            raise ValueError(f"{name} boolean mask must have length {size}")
        return np.flatnonzero(raw).astype(np.int64).tolist()
    if isinstance(raw, list) and raw and all(isinstance(x, (bool, np.bool_)) for x in raw):
        if len(raw) != size:
            raise ValueError(f"{name} boolean mask must have length {size}")
        return [i for i, selected in enumerate(raw) if selected]

    try:
        indices = [operator.index(x) for x in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer indices or be a boolean mask") from exc
    if len(indices) != len(set(indices)):
        raise ValueError(f"{name} indices must be unique")
    if any(i < 0 or i >= size for i in indices):
        raise ValueError(f"{name} indices must be in range [0, {size})")
    return sorted(indices)


class MixedContextViewSampler(Sampler[S4ViewIndex]):
    """Build an exact-ratio deterministic schedule for one DDP rank.

    ``num_samples`` is the equal per-rank draw horizon. The global crop quota is
    round-half-up over all ranks and Hamilton-allocated with ascending-rank tie breaks.
    Iteration never advances ``draw_offset``; callers set that offset only after examples
    have been committed.
    """

    def __init__(
        self,
        weights,
        thinker_eligible,
        talker_eligible,
        thinker_ratio: float,
        talker_ratio: float,
        num_samples: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        apply_crop: bool = True,
    ) -> None:
        try:
            weight_tensor = torch.as_tensor(weights, dtype=torch.double, device="cpu")
        except (TypeError, ValueError) as exc:
            raise ValueError("weights must be a numeric 1-D vector") from exc
        if weight_tensor.ndim != 1 or weight_tensor.numel() == 0:
            raise ValueError("weights must be a nonempty 1-D vector")
        if not bool(torch.isfinite(weight_tensor).all()):
            raise ValueError("weights must contain only finite values")
        if bool((weight_tensor < 0).any()):
            raise ValueError("weights must be nonnegative")
        if float(weight_tensor.sum()) <= 0:
            raise ValueError("weights must have positive sum")

        self.thinker_ratio = self._validate_ratio(thinker_ratio, "thinker_ratio")
        self.talker_ratio = self._validate_ratio(talker_ratio, "talker_ratio")
        if self.thinker_ratio > 0 and self.talker_ratio > 0:
            raise ValueError("only one crop experiment may be positive")
        if not isinstance(world_size, int) or world_size < 1:
            raise ValueError("world_size must be an integer >= 1")
        if not isinstance(rank, int) or not 0 <= rank < world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        if not isinstance(num_samples, int) or num_samples < 0:
            raise ValueError("num_samples must be an integer >= 0")
        try:
            normalized_seed = operator.index(seed)
        except TypeError as exc:
            raise ValueError("seed must be an integer") from exc

        self.weights = weight_tensor.contiguous()
        self.thinker_eligible = _eligible_indices(
            thinker_eligible, self.weights.numel(), "thinker_eligible")
        self.talker_eligible = _eligible_indices(
            talker_eligible, self.weights.numel(), "talker_eligible")
        self.num_samples = num_samples
        self.seed = normalized_seed
        self.rank = rank
        self.world_size = world_size
        self.apply_crop = bool(apply_crop)
        self.epoch = 0
        self.draw_offset = 0
        self._schedule: list[S4ViewIndex] | None = None

        if self.thinker_ratio > 0:
            self._validate_positive_pool(self.thinker_eligible, "thinker")
        if self.talker_ratio > 0:
            self._validate_positive_pool(self.talker_eligible, "talker")
        self._pool_digest = self._compute_sampling_pool_digest()

    @staticmethod
    def _validate_ratio(value: float, name: str) -> float:
        try:
            ratio = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number in [0, 1]") from exc
        if not np.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
        return ratio

    def _validate_positive_pool(self, eligible: list[int], label: str) -> None:
        if not eligible or float(self.weights[eligible].sum()) <= 0:
            raise ValueError(f"{label} eligible pool must have positive weight")

    def _compute_sampling_pool_digest(self) -> str:
        """Compactly bind resume state to the otherwise multi-million-row pools."""
        digest = hashlib.sha256()
        weights = self.weights.numpy().astype('<f8', copy=False)
        digest.update(b'weights\0')
        digest.update(np.asarray([weights.size], dtype='<u8').tobytes())
        digest.update(weights.tobytes())
        for name, indices in (
            ('thinker', self.thinker_eligible), ('talker', self.talker_eligible)
        ):
            values = np.asarray(indices, dtype='<i8')
            digest.update(name.encode('ascii') + b'\0')
            digest.update(np.asarray([values.size], dtype='<u8').tobytes())
            digest.update(values.tobytes())
        return digest.hexdigest()

    @property
    def _crop_contract(self) -> tuple[ViewKind, float, list[int]]:
        if self.thinker_ratio > 0:
            return "thinker_crop", self.thinker_ratio, self.thinker_eligible
        if self.talker_ratio > 0:
            return "talker_crop", self.talker_ratio, self.talker_eligible
        return "full", 0.0, []

    def _build_schedule(self) -> list[S4ViewIndex]:
        crop_kind, ratio, eligible = self._crop_contract
        quota = _local_hamilton_quota(
            self.num_samples, self.world_size, self.rank, ratio)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_generator_seed(
            self.seed, self.epoch, self.rank, self.world_size))

        designated_positions = torch.randperm(
            self.num_samples, generator=generator, device="cpu")[:quota]
        designated_positions = designated_positions.sort().values
        designated_mask = torch.zeros(self.num_samples, dtype=torch.bool)
        designated_mask[designated_positions] = True
        full_positions = torch.arange(self.num_samples)[~designated_mask]

        indices = torch.empty(self.num_samples, dtype=torch.int64)
        if full_positions.numel():
            indices[full_positions] = torch.multinomial(
                self.weights, int(full_positions.numel()), replacement=True,
                generator=generator)
        if designated_positions.numel():
            pool = torch.as_tensor(eligible, dtype=torch.int64)
            selected = torch.multinomial(
                self.weights[pool], int(designated_positions.numel()), replacement=True,
                generator=generator)
            indices[designated_positions] = pool[selected]

        draw_seeds = torch.randint(
            0, _MAX_SIGNED_63, (self.num_samples,), dtype=torch.int64,
            generator=generator, device="cpu")
        result = []
        for position in range(self.num_samples):
            designated: ViewKind = crop_kind if designated_mask[position] else "full"
            applied: ViewKind = designated if self.apply_crop else "full"
            result.append(S4ViewIndex(
                index=int(indices[position]),
                designated_kind=designated,
                applied_kind=applied,
                draw_seed=int(draw_seeds[position]),
            ))
        return result

    def _current_schedule(self) -> list[S4ViewIndex]:
        if self._schedule is None:
            self._schedule = self._build_schedule()
        return self._schedule

    def __iter__(self) -> Iterator[S4ViewIndex]:
        return iter(self._current_schedule()[self.draw_offset:])

    def __len__(self) -> int:
        return self.num_samples - self.draw_offset

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be an integer >= 0")
        self.epoch = epoch
        self.draw_offset = 0
        self._schedule = None

    def set_draw_offset(self, offset: int) -> None:
        if not isinstance(offset, int) or not 0 <= offset <= self.num_samples:
            raise ValueError(
                f"draw_offset must satisfy 0 <= draw_offset <= {self.num_samples}")
        self.draw_offset = offset

    def state_dict(self) -> dict:
        return {
            "schema_version": SCHEDULE_SCHEMA_VERSION,
            "seed": self.seed,
            "epoch": self.epoch,
            "rank": self.rank,
            "world_size": self.world_size,
            "num_samples": self.num_samples,
            "thinker_ratio": self.thinker_ratio,
            "talker_ratio": self.talker_ratio,
            "sampling_pool_digest": self._pool_digest,
            "draw_offset": self.draw_offset,
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise ValueError("sampler state must be a dictionary")
        expected = self.state_dict()
        for field in (
            "schema_version", "seed", "rank", "world_size", "num_samples",
            "thinker_ratio", "talker_ratio", "sampling_pool_digest",
        ):
            if field not in state:
                raise ValueError(f"sampler state is missing {field}")
            if state[field] != expected[field]:
                if field == "sampling_pool_digest":
                    raise ValueError("sampler state sampling pools do not match runtime")
                raise ValueError(f"sampler state {field}={state[field]!r} does not match "
                                 f"runtime {expected[field]!r}")
        if "epoch" not in state:
            raise ValueError("sampler state is missing epoch")
        if "draw_offset" not in state:
            raise ValueError("sampler state is missing draw_offset")
        self.set_epoch(state["epoch"])
        self.set_draw_offset(state["draw_offset"])
