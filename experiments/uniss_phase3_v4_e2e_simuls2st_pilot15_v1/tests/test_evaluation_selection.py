from __future__ import annotations

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.selection import (
    _round_robin_strata,
    duration_band,
    stable_rank,
)


def test_duration_bands_have_fixed_boundaries() -> None:
    assert duration_band(4_999) == "short"
    assert duration_band(5_000) == "medium"
    assert duration_band(8_999) == "medium"
    assert duration_band(9_000) == "long"


def test_selection_rank_is_seeded_and_stable() -> None:
    assert stable_rank(7, "sample") == stable_rank(7, "sample")
    assert stable_rank(7, "sample") != stable_rank(8, "sample")


def test_direction_selection_round_robins_duration_strata() -> None:
    pools = {
        ("eng->cmn", "short"): [{"sample_id": "s0"}, {"sample_id": "s1"}],
        ("eng->cmn", "medium"): [{"sample_id": "m0"}],
        ("eng->cmn", "long"): [{"sample_id": "l0"}],
    }
    selected = _round_robin_strata(pools, "eng->cmn", 4)
    assert [value["sample_id"] for value in selected] == ["s0", "m0", "l0", "s1"]

