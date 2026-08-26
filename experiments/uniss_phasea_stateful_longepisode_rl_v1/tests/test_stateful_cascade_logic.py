from experiments.uniss_phasea_stateful_longepisode_rl_v1.runtime.stateful_cascade import (
    accept_mt_candidate,
    split_ready_prefixes,
)
from training import constants_uniss as c


class CharacterTokenizer:
    def decode(self, values, skip_special_tokens=True):
        return "".join(chr(value) for value in values)


def test_nonfinal_empty_early_end_is_rejected():
    accepted, reason = accept_mt_candidate(
        [c.TOKEN_END_CONTENT], [], source_final=False
    )
    assert not accepted
    assert reason == "rejected_early_end"


def test_true_final_end_is_accepted():
    accepted, reason = accept_mt_candidate(
        [c.TOKEN_END_CONTENT], [], source_final=True
    )
    assert accepted
    assert reason == "true_source_final"


def test_chinese_committed_text_is_split_into_bounded_speakable_phrases():
    tokenizer = CharacterTokenizer()
    ids = [ord(value) for value in "这是第一句。这是第二个足够长的短句。末尾"]
    ready, pending = split_ready_prefixes(ids, tokenizer, "cmn", final=False)
    assert ready
    assert "".join(text for _, text in ready).startswith("这是第一句。")
    assert "".join(chr(value) for value in pending).endswith("末尾")


def test_true_final_flushes_short_tail():
    tokenizer = CharacterTokenizer()
    ready, pending = split_ready_prefixes(
        [ord(value) for value in "末尾"], tokenizer, "cmn", final=True
    )
    assert [text for _, text in ready] == ["末尾"]
    assert pending == []

