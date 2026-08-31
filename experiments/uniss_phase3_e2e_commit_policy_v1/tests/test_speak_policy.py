"""CPU unit tests for the content-gated speak policy."""

from __future__ import annotations

import pytest

from experiments.uniss_phase3_e2e_commit_policy_v1.runtime import speak_policy as sp
from training import constants_uniss as c


CONTINUATION = [c.TOKEN_WRITE_GENERATE, c.TOKEN_WAIT_READ, c.TOKEN_START_GLM]
FAMILIES = [c.TOKEN_TASK_ASR, c.TOKEN_TASK_S2T_TRANSLATION, c.TOKEN_TASK_TTS]


def test_family_and_continuation_decisions_are_distinguishable() -> None:
    """The gate has to tell the two ``_choice`` call sites apart."""

    assert sp.is_family_decision(FAMILIES)
    assert sp.is_family_decision([c.TOKEN_TASK_TTS])
    assert not sp.is_family_decision(CONTINUATION)
    assert not sp.is_family_decision([])


class _Trajectory:
    src_lang = "cmn"
    tgt_lang = "eng"


def _session(monkeypatch, *, fallback=None):
    """A ContentGatedSpeakSession with only the state the gate touches."""

    session = sp.ContentGatedSpeakSession.__new__(sp.ContentGatedSpeakSession)
    session.trajectory = _Trajectory()
    session.source_text = ""
    session.semantic = []
    session.force_family_order = True
    session._event_asr_done = False
    session._event_start_source_units = 0
    session.gate_opened = 0
    session.gate_withheld = 0
    session.pace_budgets = []
    # The base class's sampled choice; the gate must delegate to it when it
    # decides not to force anything.
    calls: list[list[int]] = []
    monkeypatch.setattr(
        sp.PacedInterleavedSession,
        "_choice",
        lambda self, values: (calls.append(list(values)), fallback or list(values)[-1])[1],
        raising=True,
    )
    return session, calls


def test_first_decision_of_an_event_always_lets_asr_run(monkeypatch) -> None:
    session, calls = _session(monkeypatch)
    assert session._choice(CONTINUATION) == c.TOKEN_WRITE_GENERATE
    assert calls == []  # forced, never delegated


def test_family_order_is_forced_and_asr_is_recorded(monkeypatch) -> None:
    session, _ = _session(monkeypatch)
    assert session._choice(FAMILIES) == c.TOKEN_TASK_ASR
    assert session._event_asr_done is True
    # Later in the same event only MT and TTS remain.
    assert session._choice(FAMILIES[1:]) == c.TOKEN_TASK_S2T_TRANSLATION


def test_gate_opens_when_asr_added_source_units(monkeypatch) -> None:
    session, calls = _session(monkeypatch)
    session._choice(FAMILIES)  # ASR selected
    session.source_text = "he is no longer"
    assert session._choice(CONTINUATION) == c.TOKEN_WRITE_GENERATE
    assert session.gate_opened == 1
    assert session.gate_withheld == 0
    assert calls == []


def test_gate_withholds_when_asr_added_nothing(monkeypatch) -> None:
    session, calls = _session(monkeypatch, fallback=c.TOKEN_WAIT_READ)
    session._choice(FAMILIES)  # ASR selected, produced no text
    assert session._choice(CONTINUATION) == c.TOKEN_WAIT_READ
    assert session.gate_withheld == 1
    assert session.gate_opened == 0
    assert calls == [CONTINUATION]  # delegated to the model


def test_units_carried_from_earlier_events_do_not_open_the_gate(monkeypatch) -> None:
    """The comparison is per event, not against an empty transcript."""

    session, _ = _session(monkeypatch, fallback=c.TOKEN_WAIT_READ)
    session.source_text = "committed by earlier events"
    session._event_start_source_units = len(
        sp.display_units(session.source_text, "cmn")
    )
    session._choice(FAMILIES)
    assert session._choice(CONTINUATION) == c.TOKEN_WAIT_READ
    assert session.gate_withheld == 1


def test_run_event_resets_the_per_event_snapshot(monkeypatch) -> None:
    session, _ = _session(monkeypatch)
    # src_lang is cmn here, so units are characters: four of them.
    session.source_text = "他是主席"
    session._event_asr_done = True
    monkeypatch.setattr(
        sp.PacedInterleavedSession,
        "run_event",
        lambda self, event, **kwargs: "delegated",
        raising=True,
    )
    assert session.run_event(object()) == "delegated"
    assert session._event_asr_done is False
    assert session._event_start_source_units == 4


def test_english_source_is_counted_by_whitespace_token(monkeypatch) -> None:
    """The unit definition follows the source language, not the target."""

    session, _ = _session(monkeypatch)
    session.trajectory.src_lang = "eng"
    session.source_text = "he is no longer"
    assert session._source_units() == 4


def test_chinese_source_is_counted_by_character(monkeypatch) -> None:
    session, _ = _session(monkeypatch)
    session._choice(FAMILIES)
    session.source_text = "他是"
    assert session._choice(CONTINUATION) == c.TOKEN_WRITE_GENERATE
    assert session._source_units() == 2


def test_the_gate_never_invents_a_token_outside_the_candidates(monkeypatch) -> None:
    session, _ = _session(monkeypatch, fallback=c.TOKEN_WAIT_READ)
    session._choice(FAMILIES)
    for candidates in ([c.TOKEN_WAIT_READ, c.TOKEN_START_GLM], CONTINUATION):
        assert session._choice(candidates) in candidates


def test_eager_and_content_gated_are_separate_classes() -> None:
    assert issubclass(sp.ContentGatedSpeakSession, sp.PacedInterleavedSession)
    assert issubclass(sp.EagerSpeakSession, sp.PacedInterleavedSession)
    assert not issubclass(sp.ContentGatedSpeakSession, sp.EagerSpeakSession)
