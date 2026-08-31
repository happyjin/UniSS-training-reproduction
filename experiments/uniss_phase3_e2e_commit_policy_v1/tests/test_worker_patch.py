"""The worker wrapper must rebind exactly one symbol and nothing else."""

from __future__ import annotations

import functools

import pytest

import experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.run_worker as worker
from experiments.uniss_phase3_e2e_commit_policy_v1.evaluation import (
    run_worker_local_agreement as wrapper,
)
from experiments.uniss_phase3_e2e_commit_policy_v1.runtime.local_agreement import (
    DEFAULT_HOLDBACK,
    local_agreement_mt_rollout,
)


def test_holdback_defaults_to_the_evidence_based_value(monkeypatch) -> None:
    monkeypatch.delenv(wrapper.ENV_HOLDBACK, raising=False)
    assert wrapper.resolve_holdback() == DEFAULT_HOLDBACK == 1


def test_holdback_is_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv(wrapper.ENV_HOLDBACK, "2")
    assert wrapper.resolve_holdback() == 2
    monkeypatch.setenv(wrapper.ENV_HOLDBACK, "  0 ")
    assert wrapper.resolve_holdback() == 0


def test_negative_holdback_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv(wrapper.ENV_HOLDBACK, "-1")
    with pytest.raises(ValueError):
        wrapper.resolve_holdback()


def test_main_rebinds_only_the_commit_policy(monkeypatch) -> None:
    original = worker.incremental_mt_rollout
    calls: list[str] = []
    monkeypatch.setattr(worker, "main", lambda: calls.append("delegated"))
    monkeypatch.delenv(wrapper.ENV_HOLDBACK, raising=False)
    try:
        wrapper.main()
        assert calls == ["delegated"]
        bound = worker.incremental_mt_rollout
        assert isinstance(bound, functools.partial)
        assert bound.func is local_agreement_mt_rollout
        assert bound.keywords == {"holdback": DEFAULT_HOLDBACK}
    finally:
        worker.incremental_mt_rollout = original


def test_the_established_worker_still_reads_the_symbol_it_binds() -> None:
    """Guard against the patch seam moving under us."""

    assert hasattr(worker, "incremental_mt_rollout")
    assert hasattr(worker, "PersistentInterleavedSession")


def test_pacing_is_off_unless_explicitly_enabled(monkeypatch) -> None:
    monkeypatch.delenv(wrapper.ENV_PACE, raising=False)
    assert wrapper.resolve_pacing() is None
    monkeypatch.setenv(wrapper.ENV_PACE, "0")
    assert wrapper.resolve_pacing() is None


def test_pacing_defaults_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv(wrapper.ENV_PACE, "1")
    for name in (
        wrapper.ENV_PACE_MARGIN_MS,
        wrapper.ENV_PACE_TAIL_MS,
        wrapper.ENV_PACE_MINIMUM,
    ):
        monkeypatch.delenv(name, raising=False)
    assert wrapper.resolve_pacing() == {
        "pace_margin_ms": 0.0,
        "pace_tail_ms": 2000.0,
        "minimum_fragment_tokens": 2,
    }


def test_a_one_token_floor_is_refused(monkeypatch) -> None:
    """One token cannot be followed by END, so it always reports malformed."""

    monkeypatch.setenv(wrapper.ENV_PACE, "1")
    monkeypatch.setenv(wrapper.ENV_PACE_MINIMUM, "1")
    with pytest.raises(ValueError):
        wrapper.resolve_pacing()


def test_main_rebinds_the_session_only_when_pacing_is_on(monkeypatch) -> None:
    original_session = worker.PersistentInterleavedSession
    original_rollout = worker.incremental_mt_rollout
    monkeypatch.setattr(worker, "main", lambda: None)
    monkeypatch.delenv(wrapper.ENV_HOLDBACK, raising=False)
    try:
        monkeypatch.delenv(wrapper.ENV_PACE, raising=False)
        wrapper.main()
        assert worker.PersistentInterleavedSession is original_session
        monkeypatch.setenv(wrapper.ENV_PACE, "1")
        wrapper.main()
        assert worker.PersistentInterleavedSession is not original_session
    finally:
        worker.PersistentInterleavedSession = original_session
        worker.incremental_mt_rollout = original_rollout
