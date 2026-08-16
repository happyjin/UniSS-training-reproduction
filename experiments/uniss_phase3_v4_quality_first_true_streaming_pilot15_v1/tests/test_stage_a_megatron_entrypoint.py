from __future__ import annotations

from types import SimpleNamespace

import pytest

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.pretrain_stage_a_megatron import (
    curriculum_group_multiplier,
    lr_group_values,
    validate_phase3_handoff_key_sets,
)


def test_stage_a_lr_groups_match_registered_plan() -> None:
    args = SimpleNamespace(
        lr=1e-4,
        stage_a_lr_new_head=1e-4,
        stage_a_lr_bridge=5e-5,
        stage_a_lr_whisper_top=1e-6,
        stage_a_lr_whisper_bottom=2e-7,
        stage_a_lr_whisper_conv=1e-7,
        stage_a_lr_qwen=2e-6,
        stage_a_lr_qwen_io=5e-7,
    )
    groups = lr_group_values(args)
    assert groups["uniss_stage_a_new_head"]["max_lr"] == 1e-4
    assert groups["uniss_stage_a_qwen"]["max_lr"] == 2e-6
    assert groups["uniss_stage_a_whisper_conv"]["max_lr"] == 1e-7


def test_stage_a_unfreeze_curriculum_keeps_new_heads_active() -> None:
    assert curriculum_group_multiplier({"uniss_stage_a_new_head": True}, 0.0) == 1.0
    assert curriculum_group_multiplier({"uniss_stage_a_qwen": True}, 0.049) == 0.0
    assert curriculum_group_multiplier({"uniss_stage_a_qwen": True}, 0.05) == 1.0
    assert curriculum_group_multiplier({"uniss_stage_a_whisper_bottom": True}, 0.29) == 0.0
    assert curriculum_group_multiplier({"uniss_stage_a_whisper_bottom": True}, 0.30) == 1.0


def test_phase3_handoff_allows_only_isolated_stage_a_modules() -> None:
    checkpoint = {"embedding.weight", "decoder.layers.0.weight", "output_layer.weight"}
    current = checkpoint | {
        "stage_a_objective.ctc_head.weight",
        "stage_a_objective.frontend.encoder.conv1.weight",
    }
    audit = validate_phase3_handoff_key_sets(checkpoint, current)
    assert audit["allowed_new_keys"] == 2
    with pytest.raises(RuntimeError):
        validate_phase3_handoff_key_sets(checkpoint, current | {"foreign.weight"})
