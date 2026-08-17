from __future__ import annotations

import torch
import pytest
from torch import nn

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    CausalWhisperOutput,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v3.stage_a_causal_whisper_asr.training.objective import (
    StageAObjective,
    ctc_seed_strength,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v3.stage_a_causal_whisper_asr.training.pretrain_stage_a_megatron import (
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


def test_ctc_repair_initialization_and_curricula() -> None:
    objective = StageAObjective(
        TinyFrontend(),
        qwen_hidden_size=6,
        ctc_output_size=5,
        ctc_blank_id=4,
        glm_semantic_offset=0,
    )
    assert objective.ctc_head.bias[4].item() == -2.0
    assert ctc_seed_strength(0.0) == 1.0
    assert ctc_seed_strength(0.25) == pytest.approx(0.5)
    assert ctc_seed_strength(0.40) == 0.0
    assert curriculum_group_multiplier({"uniss_stage_a_whisper_top": True}, 0.29) == 0.0
    assert curriculum_group_multiplier({"uniss_stage_a_whisper_top": True}, 0.30) == 1.0
    assert curriculum_group_multiplier({"uniss_stage_a_whisper_conv": True}, 0.59) == 0.0
    assert curriculum_group_multiplier({"uniss_stage_a_whisper_conv": True}, 0.60) == 1.0


def test_all_blank_logits_activate_seed_and_budget_gradients() -> None:
    objective = StageAObjective(
        TinyFrontend(),
        qwen_hidden_size=6,
        ctc_output_size=5,
        ctc_blank_id=4,
        glm_semantic_offset=0,
    )
    with torch.no_grad():
        objective.ctc_head.weight.zero_()
        objective.ctc_head.bias.zero_()
        objective.ctc_head.bias[4] = 10.0
    frame_hidden = torch.randn(1, 8, 4, requires_grad=True)
    output = CausalWhisperOutput(
        frame_hidden=frame_hidden,
        frame_lengths=torch.tensor([8]),
        pooled_hidden=torch.zeros(1, 2, 4),
        pooled_lengths=torch.tensor([2]),
    )
    batch = {
        "ctc_ids": torch.tensor([[0, 1]]),
        "ctc_lengths": torch.tensor([2]),
        "training_progress": torch.tensor(0.0),
    }
    (
        _,
        seed,
        budget,
        blank_ratio,
        blank_posterior,
        budget_target,
        strength,
        _,
    ) = objective._ctc_terms(output, batch)
    assert blank_ratio.item() == 1.0
    assert blank_posterior.item() > budget_target.item()
    assert strength.item() == 1.0
    assert seed.mean.item() > 0
    assert budget.mean.item() > 0
    (seed.mean + 20.0 * budget.mean).backward()
    assert objective.ctc_head.bias.grad is not None
    assert objective.ctc_head.bias.grad[4].item() > 0


def test_teacher_code_commitment_is_zero_only_at_teacher_code() -> None:
    objective = StageAObjective(
        TinyFrontend(),
        qwen_hidden_size=6,
        ctc_output_size=5,
        ctc_blank_id=4,
        glm_semantic_offset=0,
    )
    batch = {
        "glm_lengths": torch.tensor([2]),
        "glm_ids": torch.tensor([[0, 1]]),
        "waveform_lengths": torch.tensor([2560]),
    }
    exact = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    term, cosine = objective._codebook_commitment(exact, torch.tensor([2]), batch)
    assert term.mean.item() == 0.0
    assert cosine.item() == 1.0
    wrong = exact.flip(-1)
    wrong_term, wrong_cosine = objective._codebook_commitment(
        wrong, torch.tensor([2]), batch
    )
    assert wrong_term.mean.item() > 0
    assert wrong_cosine.item() < 1.0
