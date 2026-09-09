"""HPO's hierarchical combination: separate standardisation, and a gate that
stops latency being bought with a bad translation."""
from experiments.uniss_streaming_p2st_traj_v1.evaluation.rerank_candidates import (
    choose,
    standardise,
)


def _rows(pairs):
    return [{"quality": q, "latency": l} for q, l in pairs]


def test_standardise_centres_and_scales():
    z = standardise([1.0, 2.0, 3.0])
    assert abs(sum(z)) < 1e-9
    assert z[0] < z[1] < z[2]


def test_a_constant_group_standardises_to_zero():
    assert standardise([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_with_equal_quality_the_quietest_wins():
    best, _ = choose(_rows([(10.0, 0.30), (10.0, 0.10), (10.0, 0.20)]),
                     lam=0.5, quality_quantile=0.25)
    assert best == 1


def test_with_equal_silence_the_best_translation_wins():
    best, _ = choose(_rows([(5.0, 0.2), (9.0, 0.2), (7.0, 0.2)]),
                     lam=0.5, quality_quantile=0.25)
    assert best == 1


def test_the_gate_refuses_to_buy_silence_with_a_bad_translation():
    """The quietest candidate is also the worst; the gate must reject it."""
    rows = _rows([(1.0, 0.01), (9.0, 0.25), (9.5, 0.26), (10.0, 0.27)])
    best, _ = choose(rows, lam=0.5, quality_quantile=0.25)
    assert best != 0, "a candidate below the quality gate cannot win on latency"


def test_a_larger_lambda_moves_the_choice_towards_silence():
    rows = _rows([(10.0, 0.30), (9.0, 0.05), (9.5, 0.20)])
    quiet_weighted, _ = choose(rows, lam=2.0, quality_quantile=0.0)
    quality_weighted, _ = choose(rows, lam=0.0, quality_quantile=0.0)
    assert quiet_weighted == 1 and quality_weighted == 0
