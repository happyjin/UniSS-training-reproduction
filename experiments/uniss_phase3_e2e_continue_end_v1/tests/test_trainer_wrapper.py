"""The wrapper must satisfy the trainer's two metric-order assertions."""
from __future__ import annotations

import os
from unittest import mock

import pytest
import torch

from experiments.uniss_phase3_e2e_continue_end_v1.training import objective_ext as ext
from experiments.uniss_phase3_e2e_continue_end_v1.training import (
    pretrain_continue_end_megatron as wrapper,
)


def test_it_refuses_to_run_with_every_weight_at_zero() -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="entrypoint exists to add"):
            wrapper.resolve_extension()


def test_a_zero_margin_is_rejected_because_it_supervises_nothing() -> None:
    env = {
        wrapper.ENV_CONTINUE_WEIGHT: "0.5",
        wrapper.ENV_CONTINUE_MARGIN: "0",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        with pytest.raises(ValueError, match="must be positive"):
            wrapper.resolve_extension()


def test_the_defaults_match_the_measured_requirement() -> None:
    env = {wrapper.ENV_CONTINUE_WEIGHT: "0.5"}
    with mock.patch.dict(os.environ, env, clear=True):
        resolved = wrapper.resolve_extension()
    # The probe measured the continue decision at a median -2.88.
    assert resolved["continue_after_fragment_logit_margin"] == 1.0
    assert resolved["content_end_logit_margin"] == 2.0
    assert resolved["repetition_window"] == 8.0


def test_distributed_metric_order_matches_the_declared_contract() -> None:
    """The trainer asserts this twice; a mismatch aborts the run at iteration 1."""
    import experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.objective as base

    established = tuple(base.E2E_TERM_NAMES)
    terms = {
        name: base.LossTerm(torch.tensor(1.0), torch.tensor(2.0))
        for name in established
    }
    for name in ext.EXTRA_TERM_NAMES:
        terms[name] = base.LossTerm(torch.tensor(1.0), torch.tensor(2.0))
    _, metrics = ext.distributed_with_continue_end(
        terms,
        continue_after_fragment=0.5,
        content_end_margin=0.25,
        repetition_penalty=0.3,
    )
    _, baseline_metrics = base.distributed_e2e_objective(
        {name: terms[name] for name in established}
    )
    declared = ext.extended_objective_metric_names(tuple(baseline_metrics))
    assert tuple(metrics) == declared


def test_every_new_term_is_weighted_into_the_total() -> None:
    import experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.objective as base

    terms = {
        name: base.LossTerm(torch.tensor(0.0), torch.tensor(1.0))
        for name in base.E2E_TERM_NAMES
    }
    for name in ext.EXTRA_TERM_NAMES:
        terms[name] = base.LossTerm(torch.tensor(4.0), torch.tensor(2.0))
    total, metrics = ext.distributed_with_continue_end(
        terms,
        continue_after_fragment=0.5,
        content_end_margin=0.25,
        repetition_penalty=0.3,
    )
    # each term contributes weight * (numerator / denominator) = weight * 2
    assert float(total) == pytest.approx(2 * (0.5 + 0.25 + 0.3), abs=1e-5)
    for name, weight in (
        ("continue_after_fragment", 0.5),
        ("content_end_margin", 0.25),
        ("repetition_penalty", 0.3),
    ):
        assert float(metrics[f"weighted/{name}"]) == pytest.approx(2 * weight, abs=1e-5)


def test_the_wrapper_rebinds_all_four_trainer_attributes() -> None:
    import experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.pretrain_e2e_megatron as trainer

    originals = {
        name: getattr(trainer, name)
        for name in (
            "flattened_e2e_objective",
            "distributed_e2e_objective",
            "OBJECTIVE_METRIC_NAMES",
            "METRIC_NAMES",
        )
    }
    try:
        wrapper.install(
            {
                "continue_after_fragment": 0.5,
                "continue_after_fragment_logit_margin": 1.0,
                "content_end_margin": 0.25,
                "content_end_logit_margin": 2.0,
                "repetition_penalty": 0.3,
                "repetition_window": 8.0,
            }
        )
        for name, original in originals.items():
            assert getattr(trainer, name) is not original, f"{name} was not rebound"
        assert "loss/continue_after_fragment" in trainer.METRIC_NAMES
        assert "loss/content_end_margin" in trainer.METRIC_NAMES
    finally:
        for name, original in originals.items():
            setattr(trainer, name, original)
