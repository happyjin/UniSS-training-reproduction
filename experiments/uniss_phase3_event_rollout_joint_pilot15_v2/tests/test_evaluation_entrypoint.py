from __future__ import annotations

from pathlib import Path

from experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation import (
    evaluate_checkpoint,
)


def test_v2_evaluator_uses_repaired_provenance() -> None:
    source = Path(evaluate_checkpoint.__file__).read_text(encoding="utf-8")
    assert '"version": "uniss_phase3_event_rollout_joint_pilot15_v2"' in source
    assert '"repair": "trainable_causal_frontend"' in source
    assert '"forced_write": False' in source


def test_v2_shell_wrapper_is_non_overwriting_and_configurable() -> None:
    wrapper = Path(evaluate_checkpoint.__file__).with_suffix(".sh")
    source = wrapper.read_text(encoding="utf-8")
    assert 'RUN_NAME="${RUN_NAME:-' in source
    assert '[[ ! -e "${OUTPUT}" ]]' in source
    assert "uniss_phase3_event_rollout_joint_pilot15_v2.evaluation" in source
