import numpy as np

from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.materialize_episodes import (
    SAMPLE_RATE,
)


def test_gap_geometry_is_exactly_160_ms():
    gap = np.zeros(int(round(160 * SAMPLE_RATE / 1000.0)), dtype=np.float32)
    assert len(gap) == 2560

