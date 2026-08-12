from __future__ import annotations

from experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.shortlist_checkpoints import (
    shortlist,
)


def _row(iteration: int, trajectory: float, text: float, eos: float):
    return {
        "iteration": iteration,
        "checkpoint_exists": True,
        "metrics": {
            "interleaved_trajectory": trajectory,
            "runtime_text_token_accuracy": text,
            "runtime_action_accuracy": text + 0.3,
            "runtime_eos_recall": eos,
            "safe_commit_f1": 0.6,
            "natural_write_fraction": 0.25,
            "deadline_forced_fraction": 0.0,
            "frontend_residual_rms": 0.02,
        },
    }


def test_shortlist_is_candidate_only_and_not_last_by_definition() -> None:
    report = shortlist(
        {"checkpoints": [_row(50, 8.0, 0.1, 0.0), _row(100, 7.0, 0.2, 0.4)]},
        maximum_candidates=1,
    )
    assert report["candidates"][0]["iteration"] == 100
    assert report["final_selection_status"] == "not_selected"
    assert "exact-runtime" in report["final_selection_rule"]


def test_shortlist_rejects_forced_write_checkpoint() -> None:
    bad = _row(100, 7.0, 0.2, 0.4)
    bad["metrics"]["deadline_forced_fraction"] = 0.1
    report = shortlist(
        {"checkpoints": [_row(50, 8.0, 0.1, 0.0), bad]},
        maximum_candidates=2,
    )
    assert [row["iteration"] for row in report["candidates"]] == [50]
    assert report["rejected"][0]["reasons"] == ["forced_write_nonzero"]
