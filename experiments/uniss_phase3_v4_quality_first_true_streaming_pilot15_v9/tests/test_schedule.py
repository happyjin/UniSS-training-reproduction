from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v9.stage_a_causal_whisper_asr.training.pretrain_stage_a_megatron import (
    V9_DECISION_MARGIN_SCALE,
    curriculum_group_multiplier,
    install_v9_overrides,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v9.stage_a_causal_whisper_asr.training import (
    objective as v9_objective,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.training import (
    pretrain_stage_a_megatron as v2_entrypoint,
)


def test_v9_strengthens_only_blank_decision_margin() -> None:
    assert V9_DECISION_MARGIN_SCALE == 0.20


def test_v9_freezes_bridge_only_at_optimizer_horizon() -> None:
    bridge = {"uniss_stage_a_bridge": True}
    head = {"uniss_stage_a_new_head": True}
    qwen = {"uniss_stage_a_qwen": True}
    qwen_io = {"uniss_stage_a_qwen_io": True}
    assert curriculum_group_multiplier(bridge, 0.99) == 1.0
    assert curriculum_group_multiplier(bridge, 1.0) == 0.0
    assert curriculum_group_multiplier(head, 1.0) == 1.0
    assert curriculum_group_multiplier(qwen, 1.0) == 1.0
    assert curriculum_group_multiplier(qwen_io, 1.0) == 1.0


def test_v9_runtime_installs_both_objective_and_bridge_schedule() -> None:
    install_v9_overrides()
    native = v2_entrypoint.implementation
    assert native.StageAObjective is v9_objective.StageAObjective
    assert native.curriculum_group_multiplier is curriculum_group_multiplier
