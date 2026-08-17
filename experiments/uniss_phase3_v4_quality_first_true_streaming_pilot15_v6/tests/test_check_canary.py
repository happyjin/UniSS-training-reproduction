from __future__ import annotations

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v6.stage_a_causal_whisper_asr.check_canary import (
    evaluate,
)


def trace(blank_ratio: float, progress: float = 1.0) -> str:
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
            "iteration 127/ 127 | number of skipped iterations: 0 | "
            "number of nan iterations: 0 |",
            "successfully saved checkpoint from iteration 127 to /tmp/run",
            f"validation loss at iteration 127 on validation set | {fields} |",
        )
    )


def test_v6_gate_requires_strict_blank_health_and_saturated_progress() -> None:
    assert evaluate(trace(0.1))["passed"]
    assert not evaluate(trace(0.3))["checks"]["strict_sustained_ctc_not_blank"]
    assert not evaluate(trace(0.1, 0.99))["checks"][
        "effective_curriculum_saturated"
    ]
