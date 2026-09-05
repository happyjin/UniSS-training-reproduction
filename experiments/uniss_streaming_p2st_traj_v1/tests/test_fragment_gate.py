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
