"""Snapping the split to a quiet frame, without losing a sample or cheating."""
import numpy as np

from experiments.uniss_streaming_p2st_traj_v1.evaluation.snap_cuts import (
    SAMPLE_RATE,
    snap,
)


def _tone(ms, level=0.5):
    return np.full(int(ms * SAMPLE_RATE / 1000), level, dtype=np.float32)


def test_total_duration_is_conserved():
    durations = [200.0, 200.0, 200.0]
    audio = np.concatenate([_tone(200), _tone(200), _tone(200)])
    new, _ = snap(durations, audio)
    assert abs(sum(new) - sum(durations)) < 1e-6, "audio may move, never vanish"


def test_the_split_moves_back_to_a_quiet_frame():
    # fragment 1 is loud, goes quiet for 40 ms, then loud again before its end
    a = np.concatenate([_tone(160), _tone(40, 0.001), _tone(100)])
    b = _tone(200)
    new, shifts = snap([300.0, 200.0], np.concatenate([a, b]), window_ms=120.0)
    assert shifts[0] > 0, "a quiet frame inside the window should be chosen"
    assert 90.0 <= new[0] <= 210.0, f"split landed at {new[0]} ms"
    assert abs(new[1] - (200.0 + shifts[0])) < 1e-6, "the tail joins the next fragment"


def test_it_never_moves_a_split_later():
    durations = [200.0, 200.0]
    audio = np.concatenate([_tone(200, 0.001), _tone(200)])
    new, shifts = snap(durations, audio)
    assert all(s >= 0 for s in shifts)
    assert new[0] <= durations[0] + 1e-9, "moving later would play audio early"


def test_a_fragment_is_never_shortened_past_the_floor():
    durations = [100.0, 200.0]
    # everything before the edge is quiet, so the search would run to the start
    audio = np.concatenate([_tone(100, 0.001), _tone(200)])
    new, _ = snap(durations, audio, window_ms=500.0, min_keep_ms=60.0)
    assert new[0] >= 60.0 - 1e-6


def test_a_single_fragment_is_left_alone():
    new, shifts = snap([200.0], _tone(200))
    assert new == [200.0] and shifts == []
