from __future__ import annotations

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v4.stage_a_causal_whisper_asr.check_canary import (
    evaluate,
)


def final_validation(agreement: float, chunk_ms: float = 160.0) -> str:
    values = {
        "ar_asr": 1.0,
        "source_ctc": 5.0,
        "offline_teacher_kl": 2.0,
        "ctc_blank_ratio": 0.1,
        "causal_glm_agreement": agreement,
        "ctc_blank_posterior": 0.75,
        "ctc_blank_budget_target": 0.80,
        "codebook_commitment": 0.1,
        "codebook_identity_ce": 1.0,
        "teacher_code_cosine": 0.9,
        "teacher_code_margin": -0.01,
        "curriculum_chunk_ms": chunk_ms,
    }
    fields = " | ".join(f"{key} value: {value:.6E}" for key, value in values.items())
    return f"validation loss at iteration 127 on validation set | {fields} |"


def trace(agreement: float, chunk_ms: float = 160.0) -> str:
    return "\n".join(
        (
            "iteration 127/ 127 | number of skipped iterations: 0 | "
            "number of nan iterations: 0 |",
            "successfully saved checkpoint from iteration 127 to /tmp/run",
            final_validation(agreement, chunk_ms),
        )
    )


def test_v4_gate_requires_final_160ms_identity() -> None:
    passed = evaluate(trace(0.03))
    assert passed["passed"]
    assert passed["validation_iteration"] == 127
    assert not passed["stage_b_authorized"]

    failed_identity = evaluate(trace(0.009))
    assert not failed_identity["passed"]
    assert not failed_identity["checks"]["causal_code_identity_retained"]

    failed_chunk = evaluate(trace(0.03, 320.0))
    assert not failed_chunk["passed"]
    assert not failed_chunk["checks"]["final_validation_is_160ms"]

