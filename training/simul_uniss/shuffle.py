"""Deterministic bounded-memory shuffling for streaming Simul-UniSS data."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def buffered_shuffle(items: Iterable[T], buffer_size: int, seed: int) -> Iterator[T]:
    """Shuffle an iterable without loading the complete dataset into memory."""

    if buffer_size < 1:
        raise ValueError("shuffle buffer size must be positive")
    if buffer_size == 1:
        yield from items
        return

    rng = random.Random(seed)
    buffer: list[T] = []
    for item in items:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = item
    rng.shuffle(buffer)
    yield from buffer


def distributed_stride(items: Iterable[T], rank: int, world_size: int) -> Iterator[T]:
    """Partition one deterministic iterable across ranks without overlap."""

    if world_size < 1:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    for index, item in enumerate(items):
        if index % world_size == rank:
            yield item
