"""Deterministic single-run curriculum for the full198 experiment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


TASKS = ("replay", "prefix", "semantic", "commit")


@dataclass(frozen=True)
class CurriculumPoint:
    replay: float
    prefix: float
    semantic: float
    commit: float
    prefix_ratios: tuple[float, ...]

    def __post_init__(self) -> None:
        values = (self.replay, self.prefix, self.semantic, self.commit)
        if any(value < 0.0 for value in values):
            raise ValueError("task probabilities must be non-negative")
        if abs(sum(values) - 1.0) > 1e-8:
            raise ValueError(f"task probabilities must sum to one, got {sum(values)}")
        if not self.prefix_ratios or any(not 0.0 < value <= 1.0 for value in self.prefix_ratios):
            raise ValueError("prefix ratios must be in (0, 1]")

    @property
    def probabilities(self) -> tuple[float, ...]:
        return (self.replay, self.prefix, self.semantic, self.commit)


def current_training_iteration(args: object) -> int:
    """Return Megatron's live loop iteration, with checkpoint compatibility.

    Megatron keeps ``args.iteration`` as the checkpoint/start iteration and
    advances ``args.curr_iteration`` inside the training loop.  Reading only
    ``args.iteration`` therefore pins a curriculum to its initial stage for the
    whole run.  The fallback preserves compatibility with startup, tests, and
    older Megatron versions that do not expose ``curr_iteration`` yet.
    """

    current = getattr(args, "curr_iteration", None)
    if current is not None:
        return int(current)
    return int(getattr(args, "iteration", 0) or 0)


def point_for_iteration(iteration: int) -> CurriculumPoint:
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    if iteration <= 1500:
        return CurriculumPoint(0.40, 0.50, 0.10, 0.00, (0.70, 0.85, 1.00))
    if iteration <= 4000:
        return CurriculumPoint(0.30, 0.50, 0.15, 0.05, (0.55, 0.70, 0.85, 1.00))
    if iteration <= 7000:
        return CurriculumPoint(0.30, 0.30, 0.30, 0.10, (0.40, 0.55, 0.70, 0.85, 1.00))
    if iteration <= 10000:
        return CurriculumPoint(0.30, 0.25, 0.25, 0.20, (0.25, 0.40, 0.55, 0.70, 0.85, 1.00))
    return CurriculumPoint(0.35, 0.20, 0.20, 0.25, (0.25, 0.40, 0.55, 0.70, 0.85, 1.00))


def stable_uniform(*values: object) -> float:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    integer = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return integer / float(1 << 64)


def choose_task(point: CurriculumPoint, *, sample_id: str, iteration: int, salt: int = 0) -> str:
    value = stable_uniform(sample_id, iteration, salt, "task")
    cumulative = 0.0
    for task, probability in zip(TASKS, point.probabilities):
        cumulative += probability
        if value < cumulative:
            return task
    return TASKS[-1]


def choose_prefix_pair(
    point: CurriculumPoint, *, sample_id: str, iteration: int, salt: int = 0
) -> tuple[float, float]:
    ratios = point.prefix_ratios
    short_index = min(
        int(stable_uniform(sample_id, iteration, salt, "prefix") * len(ratios)),
        len(ratios) - 1,
    )
    short = ratios[short_index]
    long = ratios[min(short_index + 1, len(ratios) - 1)]
    return short, long


def choose_semantic_geometry(
    *, sample_id: str, iteration: int, semantic_length: int, salt: int = 0
) -> tuple[float, int, int]:
    if semantic_length < 2:
        raise ValueError("semantic sequence must contain at least two tokens")
    progress = 0.10 + 0.80 * stable_uniform(sample_id, iteration, salt, "semantic-progress")
    cut = max(1, min(semantic_length - 1, int(round(progress * semantic_length))))
    block = 25 if stable_uniform(sample_id, iteration, salt, "semantic-block") < 0.5 else 50
    block = min(block, semantic_length - cut)
    jitter = 0.30 * (stable_uniform(sample_id, iteration, salt, "text-jitter") - 0.5)
    text_ratio = min(1.0, max(0.10, progress + jitter))
    return text_ratio, cut, block
