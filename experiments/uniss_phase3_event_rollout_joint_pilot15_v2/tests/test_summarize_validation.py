from __future__ import annotations

from pathlib import Path

import pytest

from experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.summarize_validation import (
    markdown,
    parse_log,
)


def test_parse_validation_requires_runtime_selection(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "iter_0000050"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_text("metadata", encoding="utf-8")
    log = tmp_path / "train.log"
    log.write_text(
        "iteration 50/ 717 | number of skipped iterations: 0 | "
        "number of nan iterations: 0 |\n"
        "validation loss at iteration 50 | interleaved_trajectory value: 7.0 | "
        "interleaved_trajectory PPL: 1096 | natural_write_fraction value: 2.5E-01 | "
        "deadline_forced_fraction value: 0.0 | frontend_residual_rms value: 1.0E-02 |\n",
        encoding="utf-8",
    )
    summary = parse_log(log, tmp_path / "checkpoints")
    assert summary["selection_status"] == "exact_runtime_evaluation_required"
    assert summary["maximum_nan_iterations"] == 0
    assert summary["maximum_skipped_iterations"] == 0
    row = summary["checkpoints"][0]
    assert row["checkpoint_exists"] is True
    assert row["metrics"]["natural_write_fraction"] == pytest.approx(0.25)
    assert "not proof of useful audio" in markdown(summary)


def test_parse_validation_rejects_non_finite_metrics(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "validation loss at iteration 50 | interleaved_trajectory value: nan |\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite"):
        parse_log(log, tmp_path / "checkpoints")
