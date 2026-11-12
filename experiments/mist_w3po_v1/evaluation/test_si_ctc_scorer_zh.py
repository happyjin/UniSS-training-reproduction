"""Unit test of si_ctc_scorer --lang zh (CPU, no model): the Chinese target reduction and the character coverage; the
English helpers are unchanged.

    PYTHONPATH=. python -m experiments.mist_w3po_v1.evaluation.test_si_ctc_scorer_zh
"""
from experiments.mist_w3po_v1.evaluation.si_ctc_scorer import block_cover, block_cover_zh, ctc_text, ctc_text_zh

VOCAB = {"<pad>", "<s>", "</s>", "<unk>", "|", "A", "B", "P", "T", "我", "们", "用", "吧", "好", "的", "开", "会"}


def main() -> None:
    # Chinese: punctuation / spaces / out-of-vocabulary characters dropped, Latin letters upper-cased
    assert ctc_text_zh("", "我们用 ppt 吧。", VOCAB) == "我们用PPT吧"
    assert ctc_text_zh("好的，", "我们开会", VOCAB) == "好的我们开会"
    assert ctc_text_zh("", "，。！？ 123", VOCAB) == ""
    assert ctc_text_zh("", "我们去", VOCAB) == "我们"          # 去 is not in the vocabulary
    # coverage over characters, in order, prefix excluded from the share
    assert block_cover_zh("好的", "我们开会", "好的我们开会", VOCAB) == 1.0
    assert block_cover_zh("好的", "我们开会", "好的我们", VOCAB) == 0.5
    assert block_cover_zh("", "。", "我们", VOCAB) is None
    assert block_cover_zh("", "我们开会", "会开们我", VOCAB) == 0.25
    # English unchanged (the zh2en path)
    assert ctc_text("Hello,", " it's 3 o'clock.") == "HELLO IT'S O'CLOCK"
    assert block_cover("hello", "big world", "HELLO BIG WORD") == 0.5
    print("ok")


if __name__ == "__main__":
    main()
