from __future__ import annotations

import torch
from torch import nn

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v5.stage_a_causal_whisper_asr.training.objective import (
    StageAObjective,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v5.stage_a_causal_whisper_asr.training.pretrain_stage_a_megatron import (
    curriculum_group_multiplier,
)


class TinyFrontend(nn.Module):
    hidden_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.register_buffer("codes", torch.eye(4))

    @property
    def codebook(self) -> torch.Tensor:
        return self.codes


def make_objective() -> StageAObjective:
    return StageAObjective(
        TinyFrontend(),
        qwen_hidden_size=6,
        ctc_output_size=5,
        ctc_blank_id=4,
        glm_semantic_offset=0,
        code_adapter_rank=2,
    )


def test_adapter_is_exactly_zero_at_initialization_and_receives_identity_gradient() -> None:
    objective = make_objective()
    hidden = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]]
    )
    adapted, residual = objective.code_adapter(hidden)
    assert torch.equal(adapted, hidden)
    assert torch.count_nonzero(residual).item() == 0

    batch = {
        "glm_lengths": torch.tensor([2]),
        "glm_ids": torch.tensor([[0, 1]]),
        "waveform_lengths": torch.tensor([2560]),
    }
    identity, _ = objective._codebook_identity(adapted, torch.tensor([2]), batch)
    identity.mean.backward()
    assert objective.code_adapter.up.weight.grad is not None
    assert torch.isfinite(objective.code_adapter.up.weight.grad).all()
    assert objective.code_adapter.up.weight.grad.abs().sum().item() > 0


def test_v5_freezes_all_whisper_groups() -> None:
    for key in (
        "uniss_stage_a_whisper_top",
        "uniss_stage_a_whisper_bottom",
        "uniss_stage_a_whisper_conv",
    ):
        assert curriculum_group_multiplier({key: True}, 1.0) == 0.0
    assert curriculum_group_multiplier({"uniss_stage_a_bridge": True}, 0.0) == 1.0

