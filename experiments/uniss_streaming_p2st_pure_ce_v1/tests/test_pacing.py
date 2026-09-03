"""The speech budget may never outpace the source it has consumed."""
from __future__ import annotations

import pytest

from experiments.uniss_phase3_e2e_commit_policy_v1.runtime.semantic_pacing import (
    allowed_event_tokens,
)


def test_budget_is_the_consumed_source_plus_margin():
    """(consumed + margin) / 20 ms, minus what has already been spoken."""
    assert allowed_event_tokens(
        consumed_source_ms=1920.0,
        already_emitted=0,
        source_final=False,
        margin_ms=1200.0,
        tail_ms=2000.0,
    ) == 156


def test_budget_is_cumulative_not_per_event():
    """A quiet stretch may be caught up, but the total stays bounded."""
    first = allowed_event_tokens(
        consumed_source_ms=1920.0, already_emitted=0, source_final=False,
        margin_ms=1200.0, tail_ms=2000.0,
    )
    second = allowed_event_tokens(
        consumed_source_ms=3520.0, already_emitted=first, source_final=False,
        margin_ms=1200.0, tail_ms=2000.0,
    )
    assert first + second == (3520 + 1200) // 20


def test_the_runaway_sample_is_bounded():
    """emilia_zh_0003980703 emitted 3027 codes on a 19,380 ms source.

    Unpaced that is 60.5 s of speech for 19.4 s of audio and the last fragment
    queued 51.4 s behind.  The budget over the whole utterance is what the
    source plus margin plus tail can carry, and nothing more.
    """
    total = allowed_event_tokens(
        consumed_source_ms=19380.0, already_emitted=0, source_final=True,
        margin_ms=1200.0, tail_ms=2000.0,
    )
    assert total == (19380 + 1200 + 2000) // 20 == 1129
    assert total < 3027


def test_floor_is_two_tokens_so_a_fragment_is_never_malformed():
    """END is illegal before one content token, so a zero budget is worse."""
    assert allowed_event_tokens(
        consumed_source_ms=0.0, already_emitted=10_000, source_final=False,
        margin_ms=0.0, tail_ms=0.0,
    ) == 2


def test_token_duration_must_be_positive():
    with pytest.raises(ValueError):
        allowed_event_tokens(
            consumed_source_ms=100.0, already_emitted=0, source_final=False,
            token_ms=0.0,
        )
