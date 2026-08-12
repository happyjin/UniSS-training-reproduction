import torch

from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize15_action_eos_calibration.inference import (
    continuation_vocab_logits,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize15_action_eos_calibration.pretrain_generalize15 import (
    V15_WEIGHTS,
    calibration_prefix_schedule,
    continuation_supervision_mask,
    is_generalize15_trainable_parameter,
    runtime_equivalent_action_targets,
)
from training import constants_uniss as c


def test_policy_only_weights_preserve_content() -> None:
    assert V15_WEIGHTS["runtime_action"] > 0
    assert V15_WEIGHTS["runtime_continuation"] > 0
    assert V15_WEIGHTS["deadline_survival"] > 0
    assert V15_WEIGHTS["runtime_text_content"] == 0
    assert V15_WEIGHTS["microblock_semantic_content"] == 0


def test_prefix_rollin_never_reaches_destructive_v14_probability() -> None:
    assert calibration_prefix_schedule(0.05).rounds == 0
    final = calibration_prefix_schedule(1.0)
    assert final.rounds == 1
    assert final.probability == 0.10


def test_freeze_policy_selects_only_runtime_policy_heads() -> None:
    assert is_generalize15_trainable_parameter(
        "true_subsecond_objective.action_head.weight"
    )
    assert is_generalize15_trainable_parameter(
        "true_subsecond_objective.continuation_head.weight"
    )
    assert not is_generalize15_trainable_parameter("true_subsecond_lora.0.weight")
    assert not is_generalize15_trainable_parameter(
        "true_subsecond_objective.semantic_microblock_head.content.weight"
    )


def test_continuation_logits_keep_only_legal_choices() -> None:
    reference = torch.zeros(1, 1, c.VOCAB_SIZE, dtype=torch.float32)
    pair = torch.tensor([[2.0, -3.0]])
    value = continuation_vocab_logits(reference, pair)
    assert value[0, 0, c.TOKEN_START_GLM].item() == 2.0
    assert value[0, 0, c.TOKEN_EOS].item() == -3.0
    assert value[0, 0, c.TOKEN_WAIT_READ] < -1e20


def test_continuation_supervision_does_not_depend_on_legacy_loss_mask() -> None:
    labels = torch.tensor(
        [c.TOKEN_PAD, c.TOKEN_START_GLM, c.TOKEN_WAIT_READ, c.TOKEN_EOS]
    )
    assert continuation_supervision_mask(labels).tolist() == [
        False,
        True,
        False,
        True,
    ]


def test_semantic_only_writes_are_not_new_runtime_write_actions() -> None:
    batch = {
        "natural_action": torch.tensor([0, 1, 1, 1]),
        "previous_committed_length": torch.tensor([0, 0, 2, 2]),
        "stable_target_length": torch.tensor([0, 2, 2, 3]),
    }
    assert runtime_equivalent_action_targets(batch).tolist() == [0, 1, 0, 1]
