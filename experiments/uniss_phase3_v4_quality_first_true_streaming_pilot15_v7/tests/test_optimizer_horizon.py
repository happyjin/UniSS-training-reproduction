from __future__ import annotations

from pathlib import Path
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


def test_post_decay_wrapper_keeps_required_strict_coverage_metadata() -> None:
    wrapper = Path(__file__).parents[1] / "scripts/run_stage_a_post_decay_canary_8gpu.sh"
    text = wrapper.read_text(encoding="utf-8")
    assert "export RUN_COVERAGE_EPOCHS=3" in text
    assert "export RUN_PREFIX_SCHEDULE=1" in text
    assert "export RUN_TRAIN_ITERS=191" in text


class ToySchedule:
    data_parallel_group_size = 8
    global_batch_size = 128
    coverage_epochs = 3
    epoch_samples = 128
    shuffle_seed = 17
    collate_fn = list

    def __len__(self) -> int:
        return 384

    def __getitem__(self, index: int) -> int:
        return index * 7

    def source_index(self, index: int) -> tuple[int, int]:
        return divmod(index, 128)


def test_prefix_schedule_preserves_order_and_megatron_geometry() -> None:
    prefix = v7.PrefixStageASchedule(ToySchedule(), 256)
    assert len(prefix) == 256
    assert prefix[0] == 0
    assert prefix[-1] == 255 * 7
    assert prefix.source_index(255) == (1, 127)
    assert prefix.data_parallel_group_size == 8
    assert prefix.collate_fn is list


@pytest.mark.parametrize("length", (0, 192, 193, 392))
def test_prefix_schedule_rejects_unsafe_boundaries(length: int) -> None:
    with pytest.raises(ValueError):
        v7.PrefixStageASchedule(ToySchedule(), length)
