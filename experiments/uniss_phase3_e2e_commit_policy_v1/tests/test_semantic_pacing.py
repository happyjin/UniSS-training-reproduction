"""CPU unit tests for the source-paced semantic budget."""

from __future__ import annotations

import pytest

from experiments.uniss_phase3_e2e_commit_policy_v1.runtime import (
    semantic_pacing as sp,
)


def test_the_measured_token_rate_is_fifty_per_second() -> None:
    """20 ms per token, measured as exactly 50.0 tok/s on all seven samples."""

    assert sp.SEMANTIC_TOKEN_MS == 20.0
    assert 1000.0 / sp.SEMANTIC_TOKEN_MS == 50.0


def test_budget_tracks_consumed_source_time() -> None:
    # One second of source consumed allows fifty tokens of output audio.
    assert (
        sp.allowed_event_tokens(
            consumed_source_ms=1000.0, already_emitted=0, source_final=False
        )
        == 50
    )
    # Half of it already spent leaves the other half.
    assert (
        sp.allowed_event_tokens(
            consumed_source_ms=1000.0, already_emitted=25, source_final=False
        )
        == 25
    )


def test_an_over_spent_stream_is_floored_not_starved() -> None:
    """A zero budget would guarantee a malformed fragment, so the floor is two."""

    value = sp.allowed_event_tokens(
        consumed_source_ms=1000.0, already_emitted=500, source_final=False
    )
    assert value == sp.MINIMUM_FRAGMENT_TOKENS == 2


def test_the_final_event_gets_a_tail_allowance() -> None:
    without = sp.allowed_event_tokens(
        consumed_source_ms=1000.0, already_emitted=0, source_final=False
    )
    with_tail = sp.allowed_event_tokens(
        consumed_source_ms=1000.0, already_emitted=0, source_final=True
    )
    assert with_tail - without == int(sp.DEFAULT_TAIL_MS / sp.SEMANTIC_TOKEN_MS)


def test_margin_widens_the_budget_by_its_own_duration() -> None:
    value = sp.allowed_event_tokens(
        consumed_source_ms=1000.0,
        already_emitted=0,
        source_final=False,
        margin_ms=400.0,
    )
    assert value == 70


def test_zero_token_duration_is_rejected() -> None:
    with pytest.raises(ValueError):
        sp.allowed_event_tokens(
            consumed_source_ms=1000.0,
            already_emitted=0,
            source_final=False,
            token_ms=0.0,
        )


def test_the_observed_runaway_sample_would_have_been_bounded() -> None:
    """emilia_zh_0006199435: 6.06 s source, 685 tokens emitted, ratio 2.26.

    Under the pace budget the non-final events can only reach the 1:1 total,
    and the two-token floor over seventeen events bounds the overshoot to 11%.
    """

    source_ms = 6060.0
    one_to_one = int(source_ms / sp.SEMANTIC_TOKEN_MS)
    assert one_to_one == 303
    assert 685 / one_to_one > 2.2
    events = 17
    worst_case = one_to_one + events * sp.MINIMUM_FRAGMENT_TOKENS
    assert worst_case / one_to_one < 1.12


def test_the_good_sample_is_not_constrained() -> None:
    """emilia_zh_0004122419 already fits: 4.12 s source, 227 tokens, 1.10x.

    Its per-event budget must never bite before the tail, otherwise the fix
    would degrade the one sample that already works.
    """

    source_ms = 4120.0
    budget_with_tail = sp.allowed_event_tokens(
        consumed_source_ms=source_ms, already_emitted=0, source_final=True
    )
    assert budget_with_tail >= 227


def _bare_session(**overrides):
    """A PacedInterleavedSession with only the pacing state initialised.

    ``__new__`` avoids the real constructor, which needs a model, a tokenizer
    and speech embeddings.
    """

    session = sp.PacedInterleavedSession.__new__(sp.PacedInterleavedSession)
    session.semantic = []
    session.pace_margin_ms = 0.0
    session.pace_tail_ms = 0.0
    session.minimum_fragment_tokens = 2
    session.pace_budgets = []
    for name, value in overrides.items():
        setattr(session, name, value)
    return session


class _Event:
    def __init__(self, source_end_ms: int, *, final: bool = False, index: int = 0) -> None:
        self.event_index = index
        self.source_end_ms = source_end_ms
        self.source_final = final


def test_paced_session_only_narrows_the_semantic_cap(monkeypatch) -> None:
    """The subclass must clamp, never widen, and never touch other limits."""

    recorded: dict[str, object] = {}
    monkeypatch.setattr(
        sp.PersistentInterleavedSession,
        "run_event",
        lambda self, event, **kwargs: (recorded.update(kwargs), "delegated")[1],
        raising=True,
    )
    session = _bare_session()
    assert (
        session.run_event(
            _Event(640, index=3),
            max_fragments=4,
            max_text_tokens=48,
            max_semantic_tokens=384,
        )
        == "delegated"
    )
    # 640 ms consumed, nothing emitted yet -> 32 tokens, well under the 384 cap.
    assert recorded["max_semantic_tokens"] == 32
    assert recorded["max_fragments"] == 4
    assert recorded["max_text_tokens"] == 48
    assert session.pace_budgets[0]["budget"] == 32.0
    assert session.pace_budgets[0]["effective"] == 32.0


def test_paced_session_never_exceeds_the_established_cap(monkeypatch) -> None:
    monkeypatch.setattr(
        sp.PersistentInterleavedSession,
        "run_event",
        lambda self, event, **kwargs: kwargs["max_semantic_tokens"],
        raising=True,
    )
    session = _bare_session(pace_tail_ms=100_000.0)
    value = session.run_event(
        _Event(160, final=True),
        max_fragments=4,
        max_text_tokens=48,
        max_semantic_tokens=16,
    )
    assert value == 16


def test_pace_budget_shrinks_as_the_stream_over_spends(monkeypatch) -> None:
    """Successive events see the cumulative spend, not a fresh allowance."""

    monkeypatch.setattr(
        sp.PersistentInterleavedSession,
        "run_event",
        lambda self, event, **kwargs: kwargs["max_semantic_tokens"],
        raising=True,
    )
    session = _bare_session()
    first = session.run_event(
        _Event(1000, index=0),
        max_fragments=4,
        max_text_tokens=48,
        max_semantic_tokens=384,
    )
    session.semantic.extend(range(first))
    second = session.run_event(
        _Event(1200, index=1),
        max_fragments=4,
        max_text_tokens=48,
        max_semantic_tokens=384,
    )
    assert first == 50
    # Another 200 ms of source buys ten more tokens, not another fifty.
    assert second == 10
