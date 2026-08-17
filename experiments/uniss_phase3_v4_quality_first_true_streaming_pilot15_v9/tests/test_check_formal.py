from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v9.stage_a_causal_whisper_asr.check_formal import (
    evaluate,
)


def trace(*, blank: float = 0.1, cosine: float = 0.9, iteration: int = 381) -> str:
    values = {
        "ar_asr": 1.0,
        "source_ctc": 5.0,
        "offline_teacher_kl": 2.0,
        "ctc_blank_ratio": blank,
        "causal_glm_agreement": 0.05,
        "ctc_blank_posterior": 0.20,
        "ctc_blank_budget_target": 0.20,
        "codebook_commitment": 0.1,
        "codebook_identity_ce": 1.0,
        "teacher_code_cosine": cosine,
        "teacher_code_margin": -0.01,
        "code_adapter_residual": 0.01,
        "code_adapter_rms": 0.1,
        "curriculum_progress": 1.0,
        "curriculum_chunk_ms": 160.0,
    }
    fields = " | ".join(f"{key} value: {value:.6E}" for key, value in values.items())
    return "\n".join(
        (
            "stage_a_prefix_schedule ........................ False",
            "> Stage A datasets: source_packs=16195 coverage_epochs=3 "
            "epoch_samples=16256 total_samples=48768 "
            "global_shuffle_seed=20260816 valid_source=167 valid_effective=168",
            f"iteration {iteration}/ 381 | consumed samples: {iteration * 128} | "
            "number of skipped iterations: 0 | number of nan iterations: 0 |",
            f"successfully saved checkpoint from iteration {iteration} to /tmp/run",
            f"validation loss at iteration {iteration} on validation set | {fields} |",
        )
    )


def test_v9_formal_gate_can_authorize_stage_b() -> None:
    result = evaluate(trace())
    assert result["passed"]
    assert result["formal_v9_completed"]
    assert result["stage_b_authorized"]
    assert result["blocked_next_stage"] is None
    assert (
        result["schema_version"]
        == "uniss_stage_a_v9_bridge_freeze_formal_gate_v1"
    )


def test_v9_formal_gate_blocks_incomplete_or_failed_geometry() -> None:
    assert not evaluate(trace(iteration=380))["passed"]
    assert not evaluate(trace(blank=0.3))["passed"]
    assert not evaluate(trace(cosine=0.84))["passed"]
