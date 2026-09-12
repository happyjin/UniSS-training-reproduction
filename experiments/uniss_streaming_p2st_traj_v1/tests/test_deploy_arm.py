"""The deployment stack applied to one sample."""

from __future__ import annotations

import numpy as np

from experiments.uniss_streaming_p2st_traj_v1.evaluation.deploy_arm import render_sample

RATE = 16_000


def _packed(durations):
    """Loud fragments with a quiet dip just before each nominal boundary."""
    total = int(sum(durations) * RATE / 1000.0)
    rng = np.random.default_rng(0)
    audio = 0.2 * rng.standard_normal(total).astype("float32")
    edge = 0.0
    for duration in durations[:-1]:
        edge += duration
        cut = int(edge * RATE / 1000.0)
        audio[max(0, cut - 480) : cut] *= 0.001
    return audio


def test_the_buffer_is_absorbed_by_a_gap_that_already_exists():
    """It costs its own second of onset and nothing else.

    Each fragment keeps its own source boundary as a hard lower bound, so when
    there is already a gap ahead of the next fragment the buffer is swallowed
    by it: the listener waits one second longer to hear the first word and the
    utterance ends at the same moment.  That is the whole reason a one second
    buffer was affordable here.
    """
    durations = [600.0, 600.0]
    delays = [1000.0, 3000.0]
    packed = _packed(durations)
    early, first = render_sample(packed, delays, durations, window_ms=200.0,
                                 buffer_ms=0.0, taper_ms=40.0, gap_ms=120.0, tone=False)
    late, second = render_sample(packed, delays, durations, window_ms=200.0,
                                 buffer_ms=1000.0, taper_ms=40.0, gap_ms=120.0, tone=False)
    assert second["onset_ms"] - first["onset_ms"] == 1000.0
    assert len(late) == len(early), "the gap absorbed it, so the tail did not move"


def test_the_buffer_shifts_everything_when_there_is_no_gap_to_absorb_it():
    durations = [600.0, 600.0]
    delays = [0.0, 600.0]
    packed = _packed(durations)
    early, _ = render_sample(packed, delays, durations, window_ms=200.0,
                             buffer_ms=0.0, taper_ms=40.0, gap_ms=120.0, tone=False)
    late, _ = render_sample(packed, delays, durations, window_ms=200.0,
                            buffer_ms=1000.0, taper_ms=40.0, gap_ms=120.0, tone=False)
    grew = (len(late) - len(early)) * 1000.0 / RATE
    assert 900.0 <= grew <= 1000.0, grew


def test_the_cuts_are_snapped_into_the_quiet_dip():
    durations = [600.0, 600.0]
    packed = _packed(durations)
    _, stats = render_sample(packed, [0.0, 700.0], durations, window_ms=200.0,
                             buffer_ms=0.0, taper_ms=40.0, gap_ms=120.0, tone=False)
    assert stats["cuts"] == 1
    assert stats["cuts_moved"] == 1, "the dip sits 30 ms before the nominal cut"


def test_a_single_fragment_has_no_cut_to_snap():
    packed = _packed([500.0])
    _, stats = render_sample(packed, [0.0], [500.0], window_ms=200.0,
                             buffer_ms=0.0, taper_ms=40.0, gap_ms=120.0, tone=False)
    assert stats["cuts"] == 0 and stats["cuts_moved"] == 0
