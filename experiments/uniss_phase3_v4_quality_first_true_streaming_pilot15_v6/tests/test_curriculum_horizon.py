from __future__ import annotations

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v5.stage_a_causal_whisper_asr.training.objective import (
    chunk_pair_for_progress,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v6.stage_a_causal_whisper_asr.training.pretrain_stage_a_megatron import (
    effective_curriculum_progress,
)


def test_formal_curriculum_finishes_in_first_coverage_epoch() -> None:
    gbs = 128
    assert effective_curriculum_progress(0, gbs, 127) == 0.0
    assert effective_curriculum_progress(126 * gbs, gbs, 127) < 1.0
    assert effective_curriculum_progress(127 * gbs, gbs, 127) == 1.0
    assert effective_curriculum_progress(381 * gbs, gbs, 127) == 1.0


def test_saturated_curriculum_retains_only_target_short_chunks() -> None:
    assert {chunk_pair_for_progress(1.0, update)[0] for update in range(20)} == {
        160,
        320,
    }
    assert all(
        chunk_pair_for_progress(1.0, update)[1] == 320 for update in range(20)
    )


def test_hold_canary_saturates_after_42_updates() -> None:
    gbs = 128
    assert effective_curriculum_progress(42 * gbs, gbs, 42) == 1.0
    assert effective_curriculum_progress(127 * gbs, gbs, 42) == 1.0
