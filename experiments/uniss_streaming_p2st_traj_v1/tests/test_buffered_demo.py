"""Slicing the packed translation back into fragments, and the buffer's invariant."""
import numpy as np

from experiments.uniss_streaming_p2st_traj_v1.evaluation.buffered_demo import (
    slice_fragments,
)
from experiments.uniss_streaming_p2st_traj_v1.evaluation.playout_buffer import (
    SAMPLE_RATE,
    render,
    schedule,
)


def test_slices_add_back_up_to_the_packed_audio():
    durations = [620.0, 480.0, 340.0]
    audio = np.arange(int(sum(durations) * SAMPLE_RATE / 1000), dtype=np.float32)
    parts = slice_fragments(audio, durations)
    assert [len(p) for p in parts] == [int(d * SAMPLE_RATE / 1000) for d in durations]
    assert np.array_equal(np.concatenate(parts), audio)


def test_a_short_final_fragment_is_clipped_not_padded():
    """The packed file can be a few samples short of the rounded durations."""
    durations = [100.0, 100.0]
    audio = np.ones(int(150 * SAMPLE_RATE / 1000), dtype=np.float32)
    parts = slice_fragments(audio, durations)
    assert len(parts[0]) == int(100 * SAMPLE_RATE / 1000)
    assert len(parts[1]) == len(audio) - len(parts[0])


def test_the_buffer_never_plays_a_fragment_early():
    delays = [2560.0, 3200.0, 9600.0]
    durations = [620.0, 480.0, 2180.0]
    for buffer_ms in (0.0, 500.0, 1000.0, 3000.0):
        starts = schedule(delays, durations, buffer_ms)
        for start, delay in zip(starts, delays):
            assert start >= delay, "a fragment may only ever be pushed later"


def test_the_buffer_closes_gaps_it_can_reach():
    # a 1860 ms gap before the third fragment, and a buffer wide enough to span it
    delays = [0.0, 640.0, 2500.0]
    durations = [640.0, 640.0, 640.0]
    plain = schedule(delays, durations, 0.0)
    assert plain[2] - (plain[1] + durations[1]) > 1000
    buffered = schedule(delays, durations, 1300.0)
    assert buffered[2] == buffered[1] + durations[1], "the gap should be gone"


def test_render_places_audio_at_the_scheduled_offsets():
    frag = np.ones(int(100 * SAMPLE_RATE / 1000), dtype=np.float32)
    out = render([frag, frag], [0.0, 500.0], 600.0)
    assert out[0] == 1.0
    assert out[int(200 * SAMPLE_RATE / 1000)] == 0.0
    assert out[int(500 * SAMPLE_RATE / 1000)] == 1.0
