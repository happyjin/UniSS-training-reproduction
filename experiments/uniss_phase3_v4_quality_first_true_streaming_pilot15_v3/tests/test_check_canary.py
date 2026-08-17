from __future__ import annotations

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v3.stage_a_causal_whisper_asr.check_canary import (
    evaluate,
)


def validation_line(
    blank: float,
    agreement: float,
    *,
    iteration: int = 96,
    final: bool = False,
) -> str:
    values = {
        "ar_asr": 1.0,
        "source_ctc": 5.0,
        "offline_teacher_kl": 2.0,
        "ctc_blank_ratio": blank,
        "causal_glm_agreement": agreement,
        "ctc_blank_posterior": 0.75,
        "ctc_blank_budget_target": 0.80,
        "codebook_commitment": 0.1,
        "teacher_code_cosine": 0.8,
    }
    fields = " | ".join(f"{key} value: {value:.6E}" for key, value in values.items())
    suffix = " on validation set" if final else ""
    return f"validation loss at iteration {iteration}{suffix} | {fields} |"


def test_canary_gate_rejects_all_blank_and_accepts_healthy_trace() -> None:
    iteration = (
        "iteration 96/ 127 | number of skipped iterations: 0 | "
        "number of nan iterations: 0 |"
    )
    failed = evaluate(iteration + "\n" + validation_line(1.0, 0.10))
    assert not failed["passed"]
    assert not failed["checks"]["ctc_not_all_blank"]

    passed = evaluate(iteration + "\n" + validation_line(0.80, 0.10))
    assert passed["passed"]
    assert not passed["stage_b_authorized"]


def test_canary_gate_uses_final_validation_instead_of_iteration_96() -> None:
    trace = "\n".join(
        (
            "iteration 127/ 127 | number of skipped iterations: 0 | "
            "number of nan iterations: 0 |",
            validation_line(0.10, 0.10),
            validation_line(0.10, 0.009, iteration=127, final=True),
        )
    )
    result = evaluate(trace)
    assert result["validation_iteration"] == 127
    assert result["metrics"]["causal_glm_agreement"] == 0.009
    assert not result["passed"]
    assert not result["checks"]["causal_code_identity_retained"]
