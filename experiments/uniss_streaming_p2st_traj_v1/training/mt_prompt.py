"""Rebuild the MT prompt the streaming cascade actually used.

DPO scores the policy's own decisions, so the prompt it is scored under has to
be the prompt it decided under -- token for token.  Rather than describing the
format in prose and hoping, this reconstructs it from the same constants the
cascade imports, and ``tests/test_mt_prompt.py`` asserts the result is
identical to ``P2STCascade._mt_prompt`` on a real tokenizer.
"""

from __future__ import annotations

from typing import Sequence

from experiments.uniss_streaming_p2st_pure_ce_v1.training.task_samples_p2st import (
    FAMILY_P2ST_MT,
    TASK_TOKENS,
)
from training import constants_uniss as c


def mt_prompt_ids(
    tokenizer, *, source_prefix: str, target_prefix: str, tgt_lang: str
) -> list[int]:
    """The prompt token ids for one MT read step."""
    return [
        TASK_TOKENS[FAMILY_P2ST_MT],
        c.TOKEN_STREAMING_MODE,
        c.language_token_id(tgt_lang),
        c.TOKEN_START_CONTENT,
        *tokenizer.encode(source_prefix, add_special_tokens=False),
        c.TOKEN_END_CONTENT,
        c.TOKEN_WRITE_GENERATE,
        c.language_token_id(tgt_lang),
        c.TOKEN_START_CONTENT,
        *tokenizer.encode(target_prefix, add_special_tokens=False),
    ]


def mt_completion_ids(produced: Sequence[int], *, ended: bool) -> list[int]:
    """What the model emitted, terminator included.

    ``_generate`` returns the tokens *before* the terminator, so appending it
    back is not bookkeeping: stopping is the decision that leaves the listener
    in silence until the next read step, and a preference objective that never
    sees the stop token cannot express a preference about it.
    """
    tokens = [int(t) for t in produced]
    if ended:
        tokens.append(c.TOKEN_END_CONTENT)
    return tokens
