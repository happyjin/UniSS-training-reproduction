from __future__ import annotations

import pytest

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v8.stage_a_causal_whisper_asr.training import (
    objective as v8_objective,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v9.stage_a_causal_whisper_asr.check_canary import (
    evaluate,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v9.stage_a_causal_whisper_asr.training.objective import (
    ALLOWED_BLANK_FRACTION,
    DECISION_MARGIN_SCALE,
    DEFAULT_WEIGHTS,
)


def test_v9_changes_only_blank_decision_margin_in_objective() -> None:
    assert ALLOWED_BLANK_FRACTION == pytest.approx(
        v8_objective.ALLOWED_BLANK_FRACTION
    )
    assert DECISION_MARGIN_SCALE == pytest.approx(0.20)
    assert DECISION_MARGIN_SCALE > v8_objective.DECISION_MARGIN_SCALE
    assert DEFAULT_WEIGHTS == v8_objective.DEFAULT_WEIGHTS
    assert v8_objective.DECISION_MARGIN_SCALE == pytest.approx(0.05)


def test_v9_gate_has_isolated_schema() -> None:
    result = evaluate("")
    assert (
        result["schema_version"]
        == "uniss_stage_a_v9_bridge_freeze_canary_gate_v1"
    )
    assert not result["passed"]
    assert result["formal_v9_authorized"] is False
    assert result["stage_b_authorized"] is False
