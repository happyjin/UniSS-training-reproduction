"""The seeded committer drops the forced empty first commit, nothing else."""
from __future__ import annotations

from experiments.uniss_phasea_stateful_longepisode_rl_v1.runtime.commit import (
    StablePrefixCommitter,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.seeded_commit import (
    SeededPrefixCommitter,
)


def test_base_rule_commits_nothing_on_the_first_call():
    base = StablePrefixCommitter(holdback=1)
    assert base.update([10, 11, 12]) == []


def test_seeded_rule_commits_the_trimmed_first_hypothesis():
    seeded = SeededPrefixCommitter(holdback=1)
    assert seeded.update([10, 11, 12]) == [10, 11]


def test_the_holdback_is_still_applied():
    """This is the difference from holdback=0, which the sweep rejected."""
    assert SeededPrefixCommitter(holdback=2).update([10, 11, 12]) == [10]
    assert SeededPrefixCommitter(holdback=0).update([10, 11, 12]) == [10, 11, 12]


def test_later_calls_match_the_base_rule():
    seeded = SeededPrefixCommitter(holdback=1)
    base = StablePrefixCommitter(holdback=1)
    seeded.update([10, 11, 12])
    base.update([10, 11, 12])
    for hypothesis in ([10, 11, 12, 13], [10, 11, 12, 13, 14]):
        got = seeded.update(list(hypothesis))
        base.update(list(hypothesis))
        assert seeded.committed[: len(base.committed)] == base.committed
        assert isinstance(got, list)


def test_a_final_call_is_untouched():
    seeded = SeededPrefixCommitter(holdback=1)
    assert seeded.update([10, 11, 12], final=True) == [10, 11, 12]


def test_it_never_rewrites_what_it_already_committed():
    seeded = SeededPrefixCommitter(holdback=1)
    seeded.update([10, 11, 12])
    before = list(seeded.committed)
    seeded.update([99, 98])
    assert seeded.committed == before
    assert seeded.revision_conflicts == 1
