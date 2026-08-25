import torch

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    LOSS_BOUNDARY,
    LOSS_MT,
    LOSS_NONE,
    LOSS_SEMANTIC,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.training.reward import (
    candidate_topk,
    group_relative_objective,
)
from training import constants_uniss as c


def test_group_objective_is_finite_and_backpropagates():
    vocab = c.VOCAB_SIZE
    logits = torch.randn(12, vocab, requires_grad=True)
    labels = torch.zeros(12, dtype=torch.long)
    kinds = torch.full((12,), LOSS_NONE, dtype=torch.long)
    labels[1] = c.TOKEN_WAIT_READ
    labels[3] = 19
    labels[5] = c.BICODEC_SEMANTIC_OFFSET + 3
    labels[7] = c.TOKEN_WRITE_GENERATE
    labels[9] = c.TOKEN_END_SEMANTIC
    kinds[1] = LOSS_BOUNDARY
    kinds[3] = LOSS_MT
    kinds[5] = LOSS_SEMANTIC
    kinds[7] = LOSS_BOUNDARY
    kinds[9] = LOSS_BOUNDARY
    positions, indices, reference = candidate_topk(logits, labels, kinds, width=4)
    assert positions.numel() == 5
    result = group_relative_objective(
        logits,
        labels,
        kinds,
        [[(0, 12)]],
        indices,
        reference.detach(),
        sequence_length=12,
        group_size=4,
        progress=0.8,
        clip_epsilon=0.2,
    )
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
