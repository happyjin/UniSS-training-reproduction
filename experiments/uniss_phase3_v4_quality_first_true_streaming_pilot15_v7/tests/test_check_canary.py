from __future__ import annotations

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v7.stage_a_causal_whisper_asr.check_canary import (
    evaluate,
)


def trace(blank_ratio: float, *, progress: float = 1.0, iteration: int = 191) -> str:
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
        "teacher_code_cosine": 0.9,
        "teacher_code_margin": -0.01,
        "code_adapter_residual": 0.01,
        "code_adapter_rms": 0.1,
        "curriculum_progress": progress,
        "curriculum_chunk_ms": 160.0,
    }
    fields = " | ".join(f"{key} value: {value:.6E}" for key, value in values.items())
    return "\n".join(
        (
            f"iteration {iteration}/ 191 | number of skipped iterations: 0 | "
            "number of nan iterations: 0 |",
            f"successfully saved checkpoint from iteration {iteration} to /tmp/run",
            f"validation loss at iteration {iteration} on validation set | {fields} |",
        )
    )


def test_v7_gate_requires_post_decay_hold_and_strict_health() -> None:
    result = evaluate(trace(0.1))
    assert result["passed"]
    assert result["post_decay_hold_updates"] == 64
    assert result["stage_b_authorized"] is False
    assert not evaluate(trace(0.3))["checks"]["strict_sustained_ctc_not_blank"]
    assert not evaluate(trace(0.1, progress=0.99))["checks"][
        "effective_curriculum_saturated"
    ]
    assert not evaluate(trace(0.1, iteration=190))["checks"][
        "post_decay_hold_completed"
    ]
