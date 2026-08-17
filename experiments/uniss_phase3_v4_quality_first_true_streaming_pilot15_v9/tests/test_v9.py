from __future__ import annotations

import pytest

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v9.stage_a_causal_whisper_asr.check_canary import (
    evaluate,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v9.stage_a_causal_whisper_asr.training.objective import (
    ALLOWED_BLANK_FRACTION,
    DECISION_MARGIN_SCALE,
    DEFAULT_WEIGHTS,
)


def test_v9_strengthens_only_close_v8_failures() -> None:
    assert ALLOWED_BLANK_FRACTION == pytest.approx(0.15)
    assert DECISION_MARGIN_SCALE == pytest.approx(0.20)
    assert DEFAULT_WEIGHTS["codebook_commitment"] == pytest.approx(0.40)
    assert DEFAULT_WEIGHTS["code_adapter_residual"] == pytest.approx(0.10)


def test_v9_gate_has_isolated_schema() -> None:
    result = evaluate("")
    assert result["schema_version"] == "uniss_stage_a_v9_long_hold_canary_gate_v1"
    assert not result["passed"]
    assert result["stage_b_authorized"] is False
