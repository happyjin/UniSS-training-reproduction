from __future__ import annotations

import pytest
import torch

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v8.stage_a_causal_whisper_asr.training.objective import (
    DEFAULT_WEIGHTS,
    ctc_seed_strength,
    decision_margin_penalty,
)


def test_v8_seed_retains_long_hold_floor() -> None:
    assert ctc_seed_strength(0.0) == 1.0
    assert ctc_seed_strength(0.4) == pytest.approx(0.10)
    assert ctc_seed_strength(1.0) == pytest.approx(0.10)


def test_v8_decision_margin_penalizes_excess_blank_winners() -> None:
    collapsed = torch.zeros(1, 10, 3)
    collapsed[..., 0] = 2.0
    penalty = decision_margin_penalty(collapsed, torch.tensor([10]), 0)
    assert float(penalty[0]) > 0.0

    healthy = torch.zeros(1, 10, 3)
    healthy[:, :8, 1] = 2.0
    healthy[:, 8:, 0] = 2.0
    penalty = decision_margin_penalty(healthy, torch.tensor([10]), 0)
    assert float(penalty[0]) == pytest.approx(0.0)


def test_v8_strengthens_only_failed_geometry_constraints() -> None:
    assert DEFAULT_WEIGHTS["codebook_commitment"] == pytest.approx(0.30)
    assert DEFAULT_WEIGHTS["codebook_identity_ce"] == pytest.approx(0.50)
    assert DEFAULT_WEIGHTS["code_adapter_residual"] == pytest.approx(0.05)
