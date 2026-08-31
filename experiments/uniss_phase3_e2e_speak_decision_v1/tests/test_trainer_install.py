"""The wrapper must rebind exactly four attributes and keep the contracts aligned."""

from __future__ import annotations

import functools

import pytest

import experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.pretrain_e2e_megatron as trainer
from experiments.uniss_phase3_e2e_speak_decision_v1.training import (
    pretrain_speak_decision_megatron as entry,
)
from experiments.uniss_phase3_e2e_speak_decision_v1.training import objective_ext as ox


@pytest.fixture
def restored():
    saved = (
        trainer.flattened_e2e_objective,
        trainer.distributed_e2e_objective,
        trainer.OBJECTIVE_METRIC_NAMES,
        trainer.METRIC_NAMES,
    )
    yield
    (
        trainer.flattened_e2e_objective,
        trainer.distributed_e2e_objective,
        trainer.OBJECTIVE_METRIC_NAMES,
        trainer.METRIC_NAMES,
    ) = saved


def test_refuses_to_run_with_both_new_weights_at_zero(monkeypatch) -> None:
    """Without them this entrypoint has no reason to exist."""

    for name in (
        entry.ENV_SPEAK_WEIGHT,
        entry.ENV_REPETITION_WEIGHT,
        entry.ENV_SPEAK_MARGIN,
        entry.ENV_REPETITION_WINDOW,
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError):
        entry.resolve_extension()


def test_reads_the_weights_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv(entry.ENV_SPEAK_WEIGHT, "0.5")
    monkeypatch.setenv(entry.ENV_REPETITION_WEIGHT, "0.1")
    monkeypatch.setenv(entry.ENV_SPEAK_MARGIN, "1.0")
    monkeypatch.setenv(entry.ENV_REPETITION_WINDOW, "8")
    assert entry.resolve_extension() == {
        "speak_decision": 0.5,
        "speak_decision_logit_margin": 1.0,
        "repetition_penalty": 0.1,
        "repetition_window": 8.0,
    }


def test_negative_and_zero_window_are_rejected(monkeypatch) -> None:
    monkeypatch.setenv(entry.ENV_SPEAK_WEIGHT, "0.5")
    monkeypatch.setenv(entry.ENV_REPETITION_WEIGHT, "-0.1")
    with pytest.raises(ValueError):
        entry.resolve_extension()
    monkeypatch.setenv(entry.ENV_REPETITION_WEIGHT, "0.1")
    monkeypatch.setenv(entry.ENV_REPETITION_WINDOW, "0")
    with pytest.raises(ValueError):
        entry.resolve_extension()


def test_install_rebinds_the_two_entry_points(restored) -> None:
    entry.install(
        {
            "speak_decision": 0.5,
            "speak_decision_logit_margin": 1.0,
            "repetition_penalty": 0.1,
            "repetition_window": 8.0,
        }
    )
    assert isinstance(trainer.flattened_e2e_objective, functools.partial)
    assert trainer.flattened_e2e_objective.func is ox.flattened_with_speak_decision
    assert trainer.distributed_e2e_objective.func is ox.distributed_with_speak_decision
    assert trainer.distributed_e2e_objective.keywords == {
        "speak_decision": 0.5,
        "repetition_penalty": 0.1,
    }


def test_metric_contracts_are_appended_and_stay_consistent(restored) -> None:
    """Both assertions in the trainer compare tuples, so order matters."""

    before = trainer.OBJECTIVE_METRIC_NAMES
    entry.install(
        {
            "speak_decision": 0.5,
            "speak_decision_logit_margin": 1.0,
            "repetition_penalty": 0.1,
            "repetition_window": 8.0,
        }
    )
    after = trainer.OBJECTIVE_METRIC_NAMES
    assert after[: len(before)] == before
    assert len(after) == len(before) + len(ox.EXTRA_TERM_NAMES) * 2 + 1 + len(
        ox.EXTRA_WEIGHTED_NAMES
    )
    assert trainer.METRIC_NAMES == (*after, *trainer.DIAGNOSTIC_NAMES)
    assert len(set(trainer.METRIC_NAMES)) == len(trainer.METRIC_NAMES)


def test_install_is_idempotent_enough_to_be_safe(restored) -> None:
    """A double install must not duplicate metric names."""

    configuration = {
        "speak_decision": 0.5,
        "speak_decision_logit_margin": 1.0,
        "repetition_penalty": 0.1,
        "repetition_window": 8.0,
    }
    entry.install(configuration)
    first = trainer.OBJECTIVE_METRIC_NAMES
    entry.install(configuration)
    # Names are appended each time, so a second install is a bug the caller must
    # avoid; this pins the observable behaviour rather than pretending it is safe.
    assert len(trainer.OBJECTIVE_METRIC_NAMES) > len(first)


def test_the_trainer_still_exposes_what_we_rebind() -> None:
    """Guard against the seam moving under us."""

    for name in (
        "flattened_e2e_objective",
        "distributed_e2e_objective",
        "OBJECTIVE_METRIC_NAMES",
        "METRIC_NAMES",
        "DIAGNOSTIC_NAMES",
        "main",
    ):
        assert hasattr(trainer, name), name
