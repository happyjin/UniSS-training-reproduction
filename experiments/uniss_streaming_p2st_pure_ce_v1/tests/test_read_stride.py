"""The read stride must change only the read step, never stride 1's behaviour."""
from __future__ import annotations

import pytest

from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.p2st_cascade import (
    BLOCK_MS,
    BLOCK_SAMPLES,
)


def step_indices(total_blocks: int, stride: int) -> list[int]:
    """The selection rule from P2STCascadeSession.run, kept in one place."""
    steps = list(range(stride - 1, total_blocks, stride))
    if not steps or steps[-1] != total_blocks - 1:
        steps.append(total_blocks - 1)
    return steps


def test_stride_one_is_every_block():
    """This is the regression lock: stride 1 must be the old loop exactly."""
    for total in (1, 2, 7, 40, 121):
        assert step_indices(total, 1) == list(range(total))


def test_the_last_block_is_always_consumed():
    """Otherwise the tail of the utterance is silently dropped."""
    for total in (1, 5, 20, 121, 1875):
        for stride in (1, 3, 6, 12, 18, 24):
            steps = step_indices(total, stride)
            assert steps[-1] == total - 1, (total, stride)
            assert steps == sorted(set(steps)), (total, stride)


def test_read_step_is_the_stride_times_the_block():
    """k=6 is a 960 ms read step, which is what makes it comparable to m=1."""
    assert BLOCK_MS == 160
    assert BLOCK_SAMPLES == 2560
    for stride, expected_ms in ((3, 480), (6, 960), (12, 1920), (18, 2880), (24, 3840)):
        steps = step_indices(1000, stride)
        # Every step but the forced last one ends on a stride boundary.
        consumed = [(index + 1) * BLOCK_SAMPLES * 1000 // 16000 for index in steps[:-1]]
        assert consumed[0] == expected_ms, (stride, consumed[0])
        assert all(
            later - earlier == expected_ms
            for earlier, later in zip(consumed, consumed[1:])
        ), stride


def test_a_stride_longer_than_the_utterance_still_reads_once():
    """A 2 s clip at stride 24 must not produce zero read steps."""
    total = 13  # 13 blocks = 2080 ms
    steps = step_indices(total, 24)
    assert steps == [12]


def test_stride_must_be_at_least_one():
    from experiments.uniss_streaming_p2st_pure_ce_v1.runtime import p2st_cascade

    with pytest.raises(ValueError):
        p2st_cascade.P2STCascadeSession(
            model=None,
            tokenizer=None,
            objective=None,
            frontend=None,
            src_lang="eng",
            tgt_lang="cmn",
            speaker_global=tuple(range(32)),
            read_stride=0,
        )
