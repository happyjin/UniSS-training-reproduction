"""The deadline fires only when the stable prefix has stalled."""
from experiments.uniss_phasea_stateful_longepisode_rl_v1.runtime.commit import (
    StablePrefixCommitter,
)
from experiments.uniss_streaming_p2st_traj_v1.runtime.backlog_commit import (
    BacklogCappedCommitter,
)


def test_disabled_reproduces_the_stable_prefix_committer():
    a = StablePrefixCommitter(holdback=0)
    b = BacklogCappedCommitter(holdback=0, max_backlog=0)
    for hypothesis in ([1, 2, 3], [1, 2, 3, 4], [1, 2, 9], [1, 2, 9, 5], [1, 2, 9, 5, 6]):
        assert a.update(hypothesis) == b.update(hypothesis)
    assert a.committed == b.committed
    assert b.forced_tokens == 0


def test_a_growing_stable_prefix_is_never_forced():
    c = BacklogCappedCommitter(holdback=0, max_backlog=3, keep_tail=1)
    c.update([1, 2, 3])             # first hypothesis: no deadline yet
    c.update([1, 2, 3, 4])          # LCP 3 -> commits 1,2,3 normally
    assert c.committed == [1, 2, 3]
    assert c.forced_events == 0


def test_the_first_hypothesis_never_triggers_the_deadline():
    """Nothing has stalled until there is a previous hypothesis to compare."""
    c = BacklogCappedCommitter(holdback=0, max_backlog=2, keep_tail=0)
    assert c.update([1, 2, 3, 4, 5]) == []
    assert c.forced_events == 0


def test_a_stalled_prefix_past_the_deadline_is_released():
    """The hypothesis keeps changing its tail, so nothing would ever commit."""
    c = BacklogCappedCommitter(holdback=0, max_backlog=4, keep_tail=1)
    c.update([1, 2])
    c.update([1, 2, 7])             # LCP 2 -> commits 1,2
    assert c.committed == [1, 2]
    c.update([1, 2, 8, 9, 10, 11])  # backlog 4 >= 4, keep 1 -> release 8,9,10
    assert c.committed == [1, 2, 8, 9, 10]
    assert c.forced_events == 1 and c.forced_tokens == 3


def test_keep_tail_leaves_the_volatile_end_uncommitted():
    c = BacklogCappedCommitter(holdback=0, max_backlog=3, keep_tail=2)
    c.update([1])
    c.update([1, 5])                # commits 1
    c.update([1, 6, 7, 8, 9])       # backlog 4 >= 3, keep 2 -> release 6,7
    assert c.committed == [1, 6, 7]


def test_final_still_commits_everything():
    c = BacklogCappedCommitter(holdback=0, max_backlog=3, keep_tail=2)
    c.update([1, 2, 3])
    c.update([1, 2, 3, 4, 5], final=True)
    assert c.committed == [1, 2, 3, 4, 5]


def test_a_revision_of_the_committed_prefix_is_still_refused():
    c = BacklogCappedCommitter(holdback=0, max_backlog=2, keep_tail=0)
    c.update([1, 2]); c.update([1, 2, 3])
    before = list(c.committed)
    assert c.update([9, 9, 9]) == [], "a conflicting prefix commits nothing"
    assert c.committed == before and c.revision_conflicts == 1
