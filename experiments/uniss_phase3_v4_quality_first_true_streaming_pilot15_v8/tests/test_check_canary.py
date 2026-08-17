from __future__ import annotations

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v8.stage_a_causal_whisper_asr.check_canary import (
    evaluate,
)


def trace(*, blank: float = 0.10, cosine: float = 0.90, iteration: int = 255) -> str:
    values = {
        "ar_asr": 1.0,
        "source_ctc": 5.0,
        "ctc_blank_ratio": blank,
        "causal_glm_agreement": 0.05,
        "ctc_blank_posterior": 0.20,
        "ctc_blank_budget_target": 0.20,
        "teacher_code_cosine": cosine,
        "code_adapter_rms": 0.1,
        "curriculum_progress": 1.0,
        "curriculum_chunk_ms": 160.0,
    }
    fields = " | ".join(f"{key} value: {value:.6E}" for key, value in values.items())
    return "\n".join(
        (
            "stage_a_prefix_schedule ........................ True",
            "> Stage A v7 prefix datasets: source_packs=16195 coverage_epochs=3 "
            "complete_samples=48768 prefix_samples=32640 "
            "global_shuffle_seed=20260816 valid_source=167 valid_effective=168",
            f"iteration {iteration}/ 255 | consumed samples: {iteration * 128} | "
            "number of skipped iterations: 0 | number of nan iterations: 0 |",
            f"successfully saved checkpoint from iteration {iteration} to /tmp/run",
            f"validation loss at iteration {iteration} on validation set | {fields} |",
        )
    )


def test_v8_long_hold_canary_passes_but_never_authorizes_stage_b() -> None:
    result = evaluate(trace())
    assert result["passed"]
    assert result["formal_v8_authorized"]
    assert result["stage_b_authorized"] is False
    assert result["lr_floor_hold_updates"] == 128


def test_v8_long_hold_canary_blocks_known_v7_failures() -> None:
    assert not evaluate(trace(blank=0.30))["passed"]
    assert not evaluate(trace(cosine=0.84))["passed"]
    assert not evaluate(trace(iteration=254))["passed"]
