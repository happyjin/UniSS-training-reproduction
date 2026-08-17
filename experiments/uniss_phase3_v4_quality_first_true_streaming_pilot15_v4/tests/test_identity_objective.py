from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v4.stage_a_causal_whisper_asr.training.objective import (
    StageAObjective,
    chunk_pair_for_progress,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v4.stage_a_causal_whisper_asr.training.pretrain_stage_a_megatron import (
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
    )


def test_identity_cross_entropy_prefers_the_teacher_code_and_has_gradient() -> None:
    objective = make_objective()
    batch = {
        "glm_lengths": torch.tensor([2]),
        "glm_ids": torch.tensor([[0, 1]]),
        "waveform_lengths": torch.tensor([2560]),
    }
    exact = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    exact_term, exact_margin = objective._codebook_identity(
        exact, torch.tensor([2]), batch
    )
    wrong = exact.detach().flip(-1).requires_grad_(True)
    wrong_term, wrong_margin = objective._codebook_identity(
        wrong, torch.tensor([2]), batch
    )
    assert exact_term.mean.item() < wrong_term.mean.item()
    assert exact_margin.item() > 0
    assert wrong_margin.item() < 0
    wrong_term.mean.backward()
    assert wrong.grad is not None
    assert torch.isfinite(wrong.grad).all()
    assert wrong.grad.abs().sum().item() > 0


def test_v4_curriculum_exposes_final_160ms_and_freezes_bottom() -> None:
    seen = {
        chunk_pair_for_progress(0.70, update)[0]
        for update in range(12)
    }
    assert seen == {160, 320, 640}
    assert chunk_pair_for_progress(1.0, 127) == (160, 320)
    assert curriculum_group_multiplier({"uniss_stage_a_whisper_top": True}, 0.09) == 0.0
    assert curriculum_group_multiplier({"uniss_stage_a_whisper_top": True}, 0.10) == 1.0
    assert curriculum_group_multiplier({"uniss_stage_a_whisper_bottom": True}, 1.0) == 0.0
    assert curriculum_group_multiplier({"uniss_stage_a_whisper_conv": True}, 1.0) == 0.0


def test_identity_temperature_must_be_positive() -> None:
    with pytest.raises(ValueError, match="temperature"):
        StageAObjective(
            TinyFrontend(),
            qwen_hidden_size=6,
            ctc_output_size=5,
            ctc_blank_id=4,
            glm_semantic_offset=0,
            codebook_identity_temperature=0.0,
        )

