from __future__ import annotations

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v7.stage_a_causal_whisper_asr.check_formal import (
    evaluate,
)


def trace(
    blank_ratio: float,
    *,
    iteration: int = 381,
    progress: float = 1.0,
    cosine: float = 0.9,
) -> str:
    values = {
        "ar_asr": 1.0,
        "source_ctc": 5.0,
        "offline_teacher_kl": 2.0,
        "ctc_blank_ratio": blank_ratio,
        "causal_glm_agreement": 0.05,
        "ctc_blank_posterior": 0.20,
        "ctc_blank_budget_target": 0.80,
        "codebook_commitment": 0.1,
        "codebook_identity_ce": 1.0,
        "teacher_code_cosine": cosine,
        "teacher_code_margin": -0.01,
        "code_adapter_residual": 0.01,
        "code_adapter_rms": 0.1,
        "curriculum_progress": progress,
        "curriculum_chunk_ms": 160.0,
    }
    fields = " | ".join(f"{key} value: {value:.6E}" for key, value in values.items())
    return "\n".join(
        (
            "stage_a_prefix_schedule ........................ False",
            "> Stage A datasets: source_packs=16195 coverage_epochs=3 "
            "epoch_samples=16256 total_samples=48768 "
            "global_shuffle_seed=20260816 valid_source=167 valid_effective=168",
            f"iteration {iteration}/ 381 | consumed samples: "
            f"{iteration * 128} | number of skipped iterations: 0 | "
            "number of nan iterations: 0 |",
            f"successfully saved checkpoint from iteration {iteration} to /tmp/run",
            f"validation loss at iteration {iteration} on validation set | {fields} |",
        )
    )


def test_v7_formal_gate_authorizes_stage_b_only_after_complete_healthy_run() -> None:
    result = evaluate(trace(0.1))
    assert result["passed"]
    assert result["lr_floor_hold_updates"] == 254
    assert result["stage_b_authorized"] is True
    assert result["blocked_next_stage"] is None


def test_v7_formal_gate_blocks_incomplete_or_collapsed_run() -> None:
    incomplete = evaluate(trace(0.1, iteration=380))
    assert not incomplete["passed"]
    assert not incomplete["stage_b_authorized"]
    assert not incomplete["checks"]["full_three_epoch_schedule_completed"]

    blank = evaluate(trace(0.3))
    assert not blank["checks"]["strict_sustained_ctc_not_blank"]

    geometry = evaluate(trace(0.1, cosine=0.84))
    assert not geometry["checks"]["teacher_geometry_retained"]
