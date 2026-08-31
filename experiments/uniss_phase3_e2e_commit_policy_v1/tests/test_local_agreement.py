"""CPU unit tests for the local-agreement MT commit policy."""

from __future__ import annotations

from typing import Sequence

from experiments.uniss_phase3_e2e_commit_policy_v1.runtime import (
    local_agreement as la,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.gate import (
    incremental_text_metrics,
    text_units,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.runtime import (
    append_only_commit,
)


# The observed hypothesis sequence of emilia_zh_0005985930 (cmn->eng), which the
# established policy collapses to its first event.
OBSERVED = [
    "That's",
    "Such",
    "Such a",
    "Such a",
    "Such a feeling of",
    "Such a feeling that everything",
    "Such a person who thinks everything is possible",
    "Such a person feels that everything is possible and then",
    "Such a person feels that everything is possible and then he",
    "Such a person feels that everything is possible and then everything in the future is full of hope",
]
REFERENCE = (
    "Such a self one who feels that anything is possible "
    "and that the future is full of hope"
)


def _rollout(raws: Sequence[str], language: str, holdback: int = la.DEFAULT_HOLDBACK):
    """Drive the policy with a scripted generator instead of a model."""

    calls = iter(raws)
    original = la.generate_mt_prefix
    la.generate_mt_prefix = lambda *a, **k: (next(calls), (), True)
    try:
        return la.local_agreement_mt_rollout(
            None,
            None,
            [f"src {index}" for index in range(len(raws))],
            language,
            max_tokens=8,
            holdback=holdback,
        )
    finally:
        la.generate_mt_prefix = original


def test_display_units_matches_text_units_length() -> None:
    for language in ("eng", "cmn"):
        for value in ("Such A Feeling", "他是主席啊", "  spaced   out  "):
            assert len(la.display_units(value, language)) == len(
                text_units(value, language)
            )


def test_display_units_preserves_casing() -> None:
    assert la.display_units("Such A Feeling", "eng") == ["Such", "A", "Feeling"]
    assert text_units("Such A Feeling", "eng") == ["such", "a", "feeling"]


def test_first_event_is_not_committed_without_agreement() -> None:
    result = _rollout(OBSERVED, "eng")
    assert result["hypotheses"][0] == ""


def test_the_established_policy_freezes_on_this_sequence() -> None:
    """Guard the premise: the baseline really does collapse to the first event."""

    committed = ""
    conflicts = 0
    for raw in OBSERVED:
        committed, conflict = append_only_commit(committed, raw, "eng")
        conflicts += int(conflict)
    assert committed == "That's"
    assert conflicts == len(OBSERVED) - 1


def test_local_agreement_recovers_the_translation() -> None:
    result = _rollout(OBSERVED, "eng")
    final = str(result["hypotheses"][-1])
    assert final.startswith("Such a person feels that everything is possible")
    baseline = incremental_text_metrics(["That's"] * len(OBSERVED), REFERENCE, "eng")
    improved = incremental_text_metrics(result["hypotheses"], REFERENCE, "eng")
    assert baseline["coverage"] == 0.0
    assert improved["coverage"] > 0.6


def test_zero_holdback_still_freezes_on_a_real_revision() -> None:
    """Why DEFAULT_HOLDBACK is 1: committing the full agreement is too eager.

    Event 5 agrees on "Such a feeling", event 6 revises to "Such a person ...",
    and an append-only policy can never recover from that.
    """

    eager = _rollout(OBSERVED, "eng", holdback=0)
    assert str(eager["hypotheses"][-1]) == "Such a feeling"
    assert incremental_text_metrics(eager["hypotheses"], REFERENCE, "eng")[
        "coverage"
    ] < incremental_text_metrics(
        _rollout(OBSERVED, "eng")["hypotheses"], REFERENCE, "eng"
    )["coverage"]


def test_committed_prefix_is_monotonic_and_never_rolls_back() -> None:
    result = _rollout(OBSERVED, "eng")
    previous: list[str] = []
    for value in result["hypotheses"]:
        current = text_units(value, "eng")
        assert current[: len(previous)] == previous
        previous = current
    metrics = incremental_text_metrics(result["hypotheses"], REFERENCE, "eng")
    assert metrics["rollback_events"] == 0


def test_conflicts_drop_sharply_versus_the_established_policy() -> None:
    result = _rollout(OBSERVED, "eng")
    assert int(result["commit_conflicts"]) < len(OBSERVED) - 1


def test_final_event_flushes_the_pending_agreement() -> None:
    result = _rollout(["alpha", "alpha beta", "alpha beta gamma"], "eng")
    assert result["hypotheses"][-1] == "alpha beta gamma"


def test_holdback_delays_the_commit_by_that_many_units() -> None:
    raws = ["alpha beta", "alpha beta", "alpha beta gamma"]
    eager = _rollout(raws, "eng", holdback=0)
    withheld = _rollout(raws, "eng", holdback=1)
    # Index 1 is the first event that can commit anything at all, because the
    # policy needs two hypotheses before it has an agreement to measure.
    assert eager["hypotheses"][1] == "alpha beta"
    assert withheld["hypotheses"][1] == "alpha"
    # The final event flushes regardless of holdback.
    assert eager["hypotheses"][-1] == withheld["hypotheses"][-1] == "alpha beta gamma"


def test_chinese_units_are_characters_and_join_without_spaces() -> None:
    result = _rollout(["他是", "他是主席", "他是主席啊"], "cmn")
    assert result["hypotheses"][-1] == "他是主席啊"
    assert " " not in result["hypotheses"][-1]


def test_empty_source_prefixes_do_not_commit_but_final_flushes() -> None:
    result = _rollout(["alpha", "alpha beta"], "eng")
    assert result["hypotheses"][-1] == "alpha beta"
    assert result["raw_hypotheses"][0] == "alpha"


def test_return_shape_matches_the_established_rollout() -> None:
    result = _rollout(OBSERVED, "eng")
    assert set(result) == {
        "hypotheses",
        "raw_hypotheses",
        "commit_conflicts",
        "unterminated_generations",
    }
    assert len(result["hypotheses"]) == len(OBSERVED)
    assert len(result["raw_hypotheses"]) == len(OBSERVED)
