"""The paired difference and its interval."""

from __future__ import annotations

from experiments.uniss_streaming_p2st_traj_v1.evaluation.flow_screen import (
    onset_ms,
    paired,
)


def test_the_difference_is_taken_per_sample_before_averaging():
    """Pairing is the whole point: it cancels per-sample variation.

    These two arms differ by exactly -1 on every sample while the samples
    themselves range over 100, so an unpaired comparison would be swamped.
    """
    a = {f"s{i}": float(i) for i in range(100)}
    b = {f"s{i}": float(i) + 1.0 for i in range(100)}
    ids = sorted(a)
    mean, half = paired(a, b, ids)
    assert mean == -1.0
    assert half == 0.0, "no spread in the differences, so no interval"


def test_the_interval_widens_as_the_differences_scatter():
    ids = [f"s{i}" for i in range(100)]
    tight = {i: 0.0 for i in ids}
    loose = {i: (1.0 if n % 2 else -1.0) for n, i in enumerate(ids)}
    _, narrow = paired(tight, {i: 0.5 for i in ids}, ids)
    _, wide = paired(loose, {i: 0.0 for i in ids}, ids)
    assert narrow == 0.0 and wide > 0.1


def test_a_single_sample_has_no_interval():
    mean, half = paired({"s": 2.0}, {"s": 1.0}, ["s"])
    assert mean == 1.0 and half == 0.0


def test_onset_reads_the_first_delay_and_survives_an_empty_one():
    assert onset_ms({"delays": [320.0, 900.0]}) == 320.0
    assert onset_ms({"starts_ms": [640.0]}) == 640.0
    assert onset_ms({"delays": []}) == 0.0
    assert onset_ms({}) == 0.0
