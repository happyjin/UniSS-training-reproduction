"""The rebuilt prompt must equal the one the cascade used, exactly."""

from __future__ import annotations

import pytest

from experiments.uniss_streaming_p2st_traj_v1.training.mt_prompt import (
    mt_completion_ids,
    mt_prompt_ids,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.training.task_samples_p2st import (
    FAMILY_P2ST_MT,
    TASK_TOKENS,
)
from training import constants_uniss as c


class _Tokenizer:
    """Character ids, offset clear of the special-token range."""

    def encode(self, text, add_special_tokens=False):
        return [1000 + ord(ch) % 500 for ch in text]


def test_the_prompt_matches_the_cascade_layout():
    tok = _Tokenizer()
    ids = mt_prompt_ids(
        tok, source_prefix="hello", target_prefix="你好", tgt_lang="cmn"
    )
    assert ids == [
        TASK_TOKENS[FAMILY_P2ST_MT],
        c.TOKEN_STREAMING_MODE,
        c.language_token_id("cmn"),
        c.TOKEN_START_CONTENT,
        *tok.encode("hello"),
        c.TOKEN_END_CONTENT,
        c.TOKEN_WRITE_GENERATE,
        c.language_token_id("cmn"),
        c.TOKEN_START_CONTENT,
        *tok.encode("你好"),
    ]


def test_a_longer_source_prefix_only_grows_the_source_block():
    tok = _Tokenizer()
    short = mt_prompt_ids(tok, source_prefix="ab", target_prefix="x", tgt_lang="cmn")
    long = mt_prompt_ids(tok, source_prefix="abcd", target_prefix="x", tgt_lang="cmn")
    assert len(long) - len(short) == 2
    assert long[-1] == short[-1]


def test_the_stop_token_is_part_of_the_completion():
    """Without it the objective cannot prefer speaking on to falling silent."""
    assert mt_completion_ids([7, 8], ended=True) == [7, 8, c.TOKEN_END_CONTENT]
    assert mt_completion_ids([7, 8], ended=False) == [7, 8]


def test_an_empty_step_is_still_a_decision():
    """A read step that committed nothing is a stop token and nothing else."""
    assert mt_completion_ids([], ended=True) == [c.TOKEN_END_CONTENT]
