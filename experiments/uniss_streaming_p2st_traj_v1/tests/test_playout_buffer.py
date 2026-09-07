"""The buffer may only ever delay, never advance.

That invariant is the whole reason this is an acceptable knob: relaxing it --
letting a fragment be heard before the source that justified it -- would make
every latency number better without the system knowing any more than it did.
So it is the first thing tested, at every buffer size.
"""

from __future__ import annotations

from experiments.uniss_streaming_p2st_traj_v1.evaluation.playout_buffer import (
    schedule,
    silence_ratio_from_schedule,
)

# A source boundary every 640 ms, each fragment speaking for 300 ms: the
# shape that produces our gaps.
DELAYS = [640.0, 1280.0, 1920.0, 2560.0, 3200.0]
DURATIONS = [300.0, 300.0, 300.0, 300.0, 300.0]


def test_no_fragment_is_ever_scheduled_before_its_source():
    for buffer_ms in (0.0, 100.0, 500.0, 1000.0, 5000.0):
        starts = schedule(DELAYS, DURATIONS, buffer_ms)
        assert all(s >= d for s, d in zip(starts, DELAYS)), buffer_ms


def test_zero_buffer_reproduces_the_streaming_placement():
    starts = schedule(DELAYS, DURATIONS, 0.0)
    expected = []
    cursor = DELAYS[0]
    for d, dur in zip(DELAYS, DURATIONS):
        s = max(cursor, d)
        expected.append(s)
        cursor = s + dur
    assert starts == expected


def test_fragments_never_overlap():
    for buffer_ms in (0.0, 500.0, 2000.0):
        starts = schedule(DELAYS, DURATIONS, buffer_ms)
        for i in range(len(starts) - 1):
            assert starts[i] + DURATIONS[i] <= starts[i + 1] + 1e-9


def test_a_larger_buffer_never_increases_silence():
    previous = None
    for buffer_ms in (0.0, 200.0, 400.0, 800.0, 1600.0, 3200.0):
        ratio = silence_ratio_from_schedule(
            schedule(DELAYS, DURATIONS, buffer_ms), DURATIONS
        )
        if previous is not None:
            assert ratio <= previous + 1e-9, buffer_ms
        previous = ratio


def test_a_large_enough_buffer_removes_silence_entirely():
    starts = schedule(DELAYS, DURATIONS, 10_000.0)
    assert silence_ratio_from_schedule(starts, DURATIONS) == 0.0


def test_the_buffer_is_paid_for_in_lag():
    """Continuity is bought with delay, and the cost must be visible."""
    zero = schedule(DELAYS, DURATIONS, 0.0)
    buffered = schedule(DELAYS, DURATIONS, 1000.0)
    lag_zero = sum(s - d for s, d in zip(zero, DELAYS)) / len(DELAYS)
    lag_buf = sum(s - d for s, d in zip(buffered, DELAYS)) / len(DELAYS)
    assert lag_buf > lag_zero


def test_silences_under_100ms_are_not_counted():
    delays = [0.0, 1000.0]
    durations = [950.0, 500.0]  # a 50 ms hole
    assert silence_ratio_from_schedule(schedule(delays, durations, 0.0), durations) == 0.0


def test_a_single_fragment_has_no_silence():
    assert silence_ratio_from_schedule(schedule([500.0], [400.0], 0.0), [400.0]) == 0.0


def test_empty_input_is_handled():
    assert schedule([], [], 500.0) == []
    assert silence_ratio_from_schedule([], []) is None
