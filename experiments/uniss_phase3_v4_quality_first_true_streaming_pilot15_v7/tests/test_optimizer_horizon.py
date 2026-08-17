from __future__ import annotations

from types import SimpleNamespace

import pytest

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v5.stage_a_causal_whisper_asr.training.objective import (
    chunk_pair_for_progress,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v7.stage_a_causal_whisper_asr.training import (
    pretrain_stage_a_megatron as v7,
)


def args(**overrides):
    values = {
        "train_iters": 381,
        "stage_a_curriculum_iters": 127,
        "stage_a_optimizer_iters": 127,
        "stage_a_optimizer_warmup_iters": 6,
        "lr_decay_iters": 381,
        "lr_warmup_iters": 19,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_formal_installs_canary_optimizer_clock() -> None:
    value = args()
    assert v7.configure_training_horizons(value) == 1.0
    assert value.lr_decay_iters == 127
    assert value.lr_warmup_iters == 6


def test_post_decay_canary_uses_same_clock_and_longer_total_run() -> None:
    value = args(train_iters=191)
    assert v7.configure_training_horizons(value) == 1.0
    assert value.lr_decay_iters == 127
    assert value.train_iters - value.lr_decay_iters == 64


@pytest.mark.parametrize(
    "overrides",
    (
        {"stage_a_curriculum_iters": 128},
        {"stage_a_optimizer_iters": 382},
        {"stage_a_optimizer_warmup_iters": 127},
        {"stage_a_optimizer_warmup_iters": -1},
    ),
)
def test_invalid_horizon_order_is_rejected(overrides) -> None:
    with pytest.raises(ValueError):
        v7.configure_training_horizons(args(**overrides))


def test_curriculum_progress_saturates_but_training_continues() -> None:
    gbs = 128
    assert v7.effective_curriculum_progress(126 * gbs, gbs, 127) < 1.0
    assert v7.effective_curriculum_progress(127 * gbs, gbs, 127) == 1.0
    assert v7.effective_curriculum_progress(381 * gbs, gbs, 127) == 1.0


def test_formal_gate_progress_uses_optimizer_over_curriculum_ratio() -> None:
    value = args(stage_a_curriculum_iters=42, stage_a_optimizer_iters=127)
    scale = v7.configure_training_horizons(value)
    assert scale == pytest.approx(127 / 42)
    qwen = {"uniss_stage_a_qwen": True}
    assert v7.curriculum_group_multiplier(qwen, 0.01) == 0.0
    assert v7.curriculum_group_multiplier(qwen, 0.02) == 1.0


def test_post_horizon_curriculum_retains_target_short_chunks() -> None:
    assert {chunk_pair_for_progress(1.0, update)[0] for update in range(128, 192)} == {
        160,
        320,
    }
    assert chunk_pair_for_progress(1.0, 191) == (160, 320)
