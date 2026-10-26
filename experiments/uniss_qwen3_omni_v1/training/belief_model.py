"""The commit belief under Megatron: evaluation/si_belief.Belief (prior g, log likelihood ratio h), trained on
unit trajectories from si_belief.py prep.

A sample is a chunk of trajectories (training/pretrain_belief.py); the loss is binary cross-entropy of the
belief at every read of every trajectory against Z (does the draft survive the future), so h is learned as a
log likelihood ratio by the additive recursion. The normalisation travels with the export, not the save.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from megatron.core.models.huggingface import HuggingFaceModule

from experiments.uniss_qwen3_omni_v1.evaluation.si_belief import Belief

METRIC_NAMES = ("loss", "loss_born", "loss_repeat", "acc", "stable_rate", "steps")


class BeliefModel(HuggingFaceModule):
    def __init__(self, config, *, mx, sx, mo, so, pg_collection=None):
        super().__init__(config)
        self.pg_collection = pg_collection   # fp32, unwrapped: Megatron reads it off the innermost module
        self.belief = Belief(mx, sx, mo, so)

    def forward(self, x0, obs, z, m):
        logits = self.belief.trajectory_logits(x0.float(), obs.float(), m.float())
        per = F.binary_cross_entropy_with_logits(logits, z.float(), reduction="none") * m
        steps = m.sum().clamp_min(1)
        loss = per.sum() / steps
        with torch.no_grad():
            born = m.clone()
            born[:, 1:] = 0
            rep = m - born
            loss_born = per.mul(born).sum() / born.sum().clamp_min(1)
            loss_rep = per.mul(rep).sum() / rep.sum().clamp_min(1)
            acc = (((logits > 0).float() == z.float()).float() * m).sum() / steps
            rate = (z.float() * m).sum() / steps
        return loss, loss_born, loss_rep, acc, rate, steps
