"""Test of evaluation/talker_cer_zh.py (CPU, toy strings).

    PYTHONPATH=. python -m experiments.mist_w3po_v1.evaluation.test_talker_cer_zh
"""
from experiments.mist_w3po_v1.evaluation.talker_cer_zh import align, stream_stats


def main() -> None:
    assert align(list("我们开会"), list("我们开会")) == (0, 0, 0, [True] * 4)
    S, D, I, hit = align(list("我们开会"), list("我门开"))
    assert (S, D, I) == (1, 1, 0) and hit == [True, False, True, False]
    assert align(list("好"), list("好的")) [:3] == (0, 0, 1)
    st = stream_stats("我们，今天开个会吧。", "我们今天")
    assert st["n"] == 8 and st["D"] == 4 and st["trunc"] is True
    st = stream_stats("我们用 PPT 吧", "我们用PPT吧")
    assert st["n"] == 5 and st["S"] + st["D"] + st["I"] == 0 and st["trunc"] is False
    print("ok")


if __name__ == "__main__":
    main()
