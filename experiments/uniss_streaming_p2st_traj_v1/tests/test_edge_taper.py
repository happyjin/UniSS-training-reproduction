"""The taper only touches edges that have a real gap beside them."""
import numpy as np

from experiments.uniss_streaming_p2st_traj_v1.evaluation.edge_taper import (
    SAMPLE_RATE,
    raised_cosine,
    taper_edges,
)


def test_raised_cosine_spans_zero_to_one_monotonically():
    up = raised_cosine(64, rising=True)
    assert up[0] == 0.0 and abs(up[-1] - 1.0) < 1e-6
    assert np.all(np.diff(up) >= -1e-7)
    down = raised_cosine(64, rising=False)
    assert abs(down[0] - 1.0) < 1e-6 and down[-1] == 0.0


def test_a_contiguous_join_is_left_untouched():
    """Where the next fragment follows immediately the phone is continuous."""
    audio = np.ones(int(0.4 * SAMPLE_RATE), dtype=np.float32)
    out, touched = taper_edges(audio, [0.0, 200.0], [200.0, 200.0])
    assert touched == 0
    assert np.array_equal(out, audio), "no gap means nothing to taper"


def test_an_edge_before_a_gap_is_faded_out():
    audio = np.ones(int(1.0 * SAMPLE_RATE), dtype=np.float32)
    out, touched = taper_edges(audio, [0.0, 500.0], [200.0, 200.0], taper_ms=40.0)
    assert touched == 2, "the end before the gap and the start after it"
    stop = int(0.200 * SAMPLE_RATE)
    assert out[stop - 1] < 0.05, "the fragment should end near silence"
    assert out[stop - int(0.040 * SAMPLE_RATE)] > 0.95, "and be untouched before the taper"
    begin = int(0.500 * SAMPLE_RATE)
    assert out[begin] < 0.05, "the next fragment should fade in"


def test_the_taper_only_reduces_amplitude():
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(int(1.0 * SAMPLE_RATE)).astype(np.float32)
    out, _ = taper_edges(audio, [0.0, 500.0], [200.0, 200.0])
    assert np.all(np.abs(out) <= np.abs(audio) + 1e-6), "a taper can never add signal"


def test_the_gap_threshold_is_respected():
    """A 200 ms gap: tapered at a 120 ms threshold, ignored at a 250 ms one."""
    audio = np.ones(int(1.0 * SAMPLE_RATE), dtype=np.float32)
    _, tapered = taper_edges(audio, [0.0, 400.0], [200.0, 200.0], gap_ms=120.0)
    _, ignored = taper_edges(audio, [0.0, 400.0], [200.0, 200.0], gap_ms=250.0)
    assert tapered == 2 and ignored == 0
