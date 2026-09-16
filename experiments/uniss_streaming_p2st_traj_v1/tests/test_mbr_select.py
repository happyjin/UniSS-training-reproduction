"""The Bayes-risk selection rule."""

from __future__ import annotations

from experiments.uniss_streaming_p2st_traj_v1.evaluation.mbr_select import (
    mbr_scores,
    select,
    utility,
)

CONSENSUS = ["我今天去了公园", "我今天去了公园玩", "我今天去公园"]


def test_an_outlier_scores_lowest():
    """MBR's whole premise: agreement with the others is the evidence.

    The odd candidate shares nothing with the rest, so it earns no utility,
    while the three that agree carry each other.
    """
    texts = CONSENSUS + ["完全不同的一句话"]
    scores = mbr_scores(texts, language="cmn", kind="chrf")
    assert scores[-1] == min(scores)
    assert max(range(len(scores)), key=lambda i: scores[i]) != len(texts) - 1


def test_a_duplicated_candidate_is_pulled_up():
    """Two identical samples vouch for each other, which is the model's vote."""
    alone = mbr_scores(["A B C D", "E F G H", "I J K L"], language="eng", kind="chrf")
    twinned = mbr_scores(["A B C D", "A B C D", "I J K L"], language="eng", kind="chrf")
    assert twinned[0] > alone[0]


def test_the_flow_weight_moves_the_choice_towards_the_quieter_candidate():
    texts = list(CONSENSUS)
    silences = [0.60, 0.05, 0.55]
    quiet_off, _ = select(texts, silences, language="cmn", kind="chrf", flow_weight=0.0)
    quiet_on, _ = select(texts, silences, language="cmn", kind="chrf", flow_weight=3.0)
    assert silences[quiet_on] <= silences[quiet_off]
    assert silences[quiet_on] == min(silences)


def test_normalisation_makes_punctuation_free_of_charge():
    """ASR output has no punctuation, so the utility must not charge for it."""
    assert utility("我今天去了公园。", "我今天去了公园", language="cmn", kind="chrf") == 100.0


def test_an_empty_candidate_earns_nothing():
    assert utility("", "我今天去了公园", language="cmn", kind="chrf") == 0.0
    scores = mbr_scores(["", "我今天去了公园", "我今天去公园"], language="cmn", kind="chrf")
    assert scores[0] == 0.0


def test_a_single_candidate_has_no_evidence():
    assert mbr_scores(["只有一个"], language="cmn", kind="chrf") == [0.0]


def test_pool_agreement_is_the_mean_pairwise_utility():
    """The number that says how much variation MBR has to work with.

    A pool of identical candidates agrees perfectly and offers nothing; a pool
    of unrelated ones agrees not at all.
    """
    identical = mbr_scores(["A B C", "A B C", "A B C"], language="eng", kind="chrf")
    assert sum(identical) / 3 == 100.0
    varied = mbr_scores(["A B C", "D E F", "G H I"], language="eng", kind="chrf")
    assert sum(varied) / 3 < 50.0
