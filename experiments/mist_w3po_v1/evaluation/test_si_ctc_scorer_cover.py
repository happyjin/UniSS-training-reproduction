"""Unit tests for si_ctc_scorer's opt-in coverage (lcs_matched, block_cover); CPU only."""
from experiments.mist_w3po_v1.evaluation.si_ctc_scorer import block_cover, lcs_matched


def test_lcs_matched():
    assert lcs_matched("A B C D".split(), "A C D".split()) == [True, False, True, True]
    assert lcs_matched([], ["X"]) == []
    assert lcs_matched(["X"], []) == [False]
    assert lcs_matched("A B A".split(), "B A".split()).count(True) == 2


def test_block_cover():
    # block words all present after the prefix
    assert block_cover("Hello there,", "you're welcome.", "HELLO THERE YOU'RE WELCOME") == 1.0
    # one of two block words missing
    assert block_cover("Hello there,", "you're welcome.", "HELLO THERE WELCOME") == 0.5
    # a prefix word reused cannot cover the block (order)
    assert block_cover("", "go now", "NOW GO") == 0.5
    assert block_cover("x", "...", "ANY") is None
    assert block_cover("", "Kind of guides", "") == 0.0


if __name__ == "__main__":
    test_lcs_matched()
    test_block_cover()
    print("ok")
