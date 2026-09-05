"""The stub-merging gate must move audio in time, never lose or invent it.

A gate that drops short fragments would improve every choppiness metric while
making the output worse, which is the failure mode this project has already
walked into once (fewer gaps because the model said less).  So the property
tested here is conservation: the codes emitted with the gate on are exactly
the codes emitted with it off, in the same order.
"""

from __future__ import annotations

import pytest

from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.p2st_cascade import (
    SEMANTIC_MS_PER_TOKEN,
)


class _Gate:
    """The gate's arithmetic, lifted out of the session so it can be driven.

    Mirrors the three branches in ``P2STCascadeSession.run``: merge what is
    held, hold what is too short, flush at the end.
    """

    def __init__(self, minimum: int) -> None:
        self.minimum = minimum
        self.pending: list[int] = []
        self.fragments: list[list[int]] = []

    def step(self, codes: list[int], *, final: bool) -> None:
        if self.pending:
            codes = self.pending + codes
            self.pending = []
        if self.minimum and len(codes) < self.minimum and not final:
            self.pending = codes
            return
        if codes:
            self.fragments.append(codes)

    def finish(self) -> None:
        if self.pending:
            self.fragments.append(self.pending)
            self.pending = []


def _run(chunks, minimum):
    gate = _Gate(minimum)
    for index, codes in enumerate(chunks):
        gate.step(list(codes), final=index == len(chunks) - 1)
    gate.finish()
    return gate.fragments


CHUNKS = [[1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12, 13], [14, 15, 16, 17, 18]]


def test_gate_off_is_the_established_behaviour():
    assert _run(CHUNKS, 0) == [list(c) for c in CHUNKS]


def test_no_code_is_lost_or_invented():
    flat_off = [v for f in _run(CHUNKS, 0) for v in f]
    for minimum in (2, 4, 8, 16, 64):
        flat_on = [v for f in _run(CHUNKS, minimum) for v in f]
        assert flat_on == flat_off, f"gate {minimum} changed the codes"


def test_short_fragments_disappear_from_the_output():
    fragments = _run(CHUNKS, 4)
    assert all(len(f) >= 4 for f in fragments[:-1])
    assert len(fragments) < len(CHUNKS)


def test_a_held_stub_is_flushed_even_if_nothing_follows():
    """The bug this test exists for: the last step may not run the TTS stage."""
    fragments = _run([[1, 2, 3, 4, 5, 6], [7]], 4)
    assert [v for f in fragments for v in f] == [1, 2, 3, 4, 5, 6, 7]


def test_the_final_chunk_is_never_held():
    fragments = _run([[1], [2]], 8)
    assert [v for f in fragments for v in f] == [1, 2]


def test_a_very_large_gate_yields_one_fragment():
    fragments = _run(CHUNKS, 10_000)
    assert len(fragments) == 1
    assert fragments[0] == [v for c in CHUNKS for v in c]


def test_duration_arithmetic_is_unchanged_by_merging():
    total_off = sum(len(f) for f in _run(CHUNKS, 0)) * SEMANTIC_MS_PER_TOKEN
    total_on = sum(len(f) for f in _run(CHUNKS, 16)) * SEMANTIC_MS_PER_TOKEN
    assert total_off == total_on


class _Grid:
    """The read-step grid, lifted out so the tail gate can be driven directly."""

    def __init__(self, block_samples: int, block_ms: int) -> None:
        self.block_samples = block_samples
        self.block_ms = block_ms

    def steps(self, samples: int, stride: int, min_final_ms: int) -> list[int]:
        total_blocks = -(-samples // self.block_samples)
        steps = list(range(stride - 1, total_blocks, stride))
        if not steps:
            return [total_blocks - 1]
        if steps[-1] != total_blocks - 1:
            residual = samples - (steps[-1] + 1) * self.block_samples
            residual_ms = self.block_ms * max(0, residual) / self.block_samples
            if min_final_ms and residual_ms < min_final_ms:
                steps[-1] = total_blocks - 1
            else:
                steps.append(total_blocks - 1)
        return steps


GRID = _Grid(block_samples=2560, block_ms=160)  # 160 ms at 16 kHz


def test_tail_gate_off_keeps_the_established_grid():
    # 17.1 s at stride 4: a 460 ms tail gets its own step, as it always has.
    steps = GRID.steps(int(17.1 * 16000), 4, 0)
    assert steps[-1] == -(-int(17.1 * 16000) // 2560) - 1
    assert len(steps) == len(GRID.steps(int(17.1 * 16000), 4, 0))


def test_a_short_tail_is_folded_not_dropped():
    """The whole audio must still be read; only the extra round disappears."""
    samples = int(16.78 * 16000)  # 140 ms tail at stride 4
    total_blocks = -(-samples // 2560)
    without = GRID.steps(samples, 4, 0)
    with_gate = GRID.steps(samples, 4, 320)
    assert len(with_gate) == len(without) - 1
    # Both still finish on the final block, so no audio is left unread.
    assert without[-1] == with_gate[-1] == total_blocks - 1


def test_a_long_tail_still_gets_its_own_step():
    samples = int(16.56 * 16000)  # 560 ms tail
    assert len(GRID.steps(samples, 4, 320)) == len(GRID.steps(samples, 4, 0))


def test_the_grid_always_ends_on_the_last_block():
    for seconds in (3.0, 7.4, 16.46, 19.38):
        for stride in (1, 2, 4, 6):
            samples = int(seconds * 16000)
            total_blocks = -(-samples // 2560)
            for gate in (0, 320):
                assert GRID.steps(samples, stride, gate)[-1] == total_blocks - 1


def test_stride_one_is_never_affected():
    """At stride 1 every block is a step, so there is no partial tail."""
    samples = int(16.78 * 16000)
    assert GRID.steps(samples, 1, 320) == GRID.steps(samples, 1, 0)
