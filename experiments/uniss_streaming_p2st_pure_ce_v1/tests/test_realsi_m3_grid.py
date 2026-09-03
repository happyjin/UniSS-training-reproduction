"""The synthesized read grid must satisfy what ``_append_source`` demands.

``PersistentInterleavedSession._append_source`` raises unless
``0 <= start <= stop <= len(embeddings)``, and ``schema.py:225`` treats a gap
between one event's ``source_glm_end`` and the next event's
``source_glm_start`` as a trajectory fault.  A grid that violated either would
fail deep inside a GPU rollout, so it is checked here instead.
"""
from __future__ import annotations

import math

import pytest

from experiments.uniss_streaming_p2st_pure_ce_v1.evaluation.realsi_m3_rollout import (
    BLOCK_SAMPLES,
    GLM_FRAMES_PER_BLOCK,
    build_read_grid,
)


def glm_length_for(samples: int) -> int:
    return math.ceil(samples / 1280)


@pytest.mark.parametrize("seconds", [0.4, 1.0, 3.26, 6.94, 10.62, 30.0])
@pytest.mark.parametrize("stride", [1, 6, 25])
def test_grid_is_contiguous_monotone_and_clamped(seconds: float, stride: int) -> None:
    samples = int(round(16_000 * seconds))
    glm_length = glm_length_for(samples)
    events = build_read_grid(
        source_samples=samples, glm_length=glm_length, read_stride=stride
    )
    assert events, "every utterance must get at least one read step"
    assert events[0].source_glm_start == 0
    assert events[-1].source_glm_end == glm_length, "the last step must read to the end"
    previous = 0
    for index, event in enumerate(events):
        assert event.event_index == index
        assert event.source_glm_start == previous, "a gap would be a trajectory fault"
        assert event.source_glm_start <= event.source_glm_end <= glm_length
        assert 0 < event.source_end_ms <= math.ceil(1000 * seconds)
        previous = event.source_glm_end
    assert [e.source_final for e in events] == [False] * (len(events) - 1) + [True], (
        "EOS is legal only on the final step, so exactly one event may be final"
    )
    ends = [e.source_end_ms for e in events]
    assert ends == sorted(ends) and len(set(ends)) == len(ends)


def test_stride_one_is_one_event_per_160ms_block() -> None:
    samples = int(round(16_000 * 6.94))
    total_blocks = math.ceil(samples / BLOCK_SAMPLES)
    events = build_read_grid(
        source_samples=samples, glm_length=glm_length_for(samples), read_stride=1
    )
    assert len(events) == total_blocks
    # 160 ms of audio is exactly two 80 ms GLM frames.
    assert GLM_FRAMES_PER_BLOCK == 2
    spans = [e.source_glm_end - e.source_glm_start for e in events[:-1]]
    assert set(spans) == {2}
    assert [e.source_end_ms for e in events[:3]] == [160, 320, 480]


def test_larger_stride_lands_on_multiples_of_the_read_step() -> None:
    samples = int(round(16_000 * 20.0))
    events = build_read_grid(
        source_samples=samples, glm_length=glm_length_for(samples), read_stride=6
    )
    # 6 x 160 ms = 960 ms.  Every step but the clamped last is a multiple.
    assert all(e.source_end_ms % 960 == 0 for e in events[:-1])
    assert events[-1].source_end_ms == 20_000


def test_stride_must_be_positive() -> None:
    with pytest.raises(ValueError):
        build_read_grid(source_samples=16_000, glm_length=13, read_stride=0)
