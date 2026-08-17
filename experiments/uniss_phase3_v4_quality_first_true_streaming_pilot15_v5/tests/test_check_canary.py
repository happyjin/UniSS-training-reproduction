from __future__ import annotations

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v5.stage_a_causal_whisper_asr.check_canary import (
    evaluate,
)


def trace(agreement: float, adapter_rms: float = 0.1) -> str:
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
        "code_adapter_residual": adapter_rms**2,
        "code_adapter_rms": adapter_rms,
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


def test_v5_gate_requires_identity_and_bounded_adapter() -> None:
    assert evaluate(trace(0.03))["passed"]
    failed_identity = evaluate(trace(0.01))
    assert not failed_identity["checks"]["causal_code_identity_retained"]
    failed_adapter = evaluate(trace(0.03, 0.6))
    assert not failed_adapter["checks"]["adapter_is_bounded"]

