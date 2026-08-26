from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.rollout import units


def test_language_units_follow_evaluation_convention():
    assert units("你 好 世界", "cmn") == 4
    assert units("hello   world", "eng") == 2

