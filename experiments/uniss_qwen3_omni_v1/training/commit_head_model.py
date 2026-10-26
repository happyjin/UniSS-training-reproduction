"""CRISP's commit head under Megatron: evaluation/si_commit_head.Head, trained on prepared rows.

The head reads one greedy draft's per-token features (si_commit_head.features) and gives the
chance the token's unit is inside the vote's consensus (label cons) or agrees with the finished
translation (label final). Training is binary cross-entropy over rows; the rows arrive as
chunks of a per-pass permutation (training/pretrain_commit_head.py). The normalisation (train
rows' mean and std) is fixed at construction and travels with the exported head, not the save.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from megatron.core.models.huggingface import HuggingFaceModule

from experiments.uniss_qwen3_omni_v1.evaluation.si_commit_head import Head

METRIC_NAMES = ("loss", "acc", "pos_rate", "rows")


class CommitHeadModel(HuggingFaceModule):
    def __init__(self, config, *, mean, std, pg_collection=None):
        super().__init__(config)
        # Megatron's train loop reads pg_collection off the innermost module; in bf16 runs Float16Module carries
        # it, but the head trains in fp32, unwrapped
        self.pg_collection = pg_collection
        self.head = Head(torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x, y):
        logits = self.head(x.float())
        y = y.float()
        loss = F.binary_cross_entropy_with_logits(logits, y)
        with torch.no_grad():
            acc = ((logits > 0).float() == y).float().mean()
        return loss, acc, y.mean(), torch.tensor(float(y.numel()), device=y.device)
