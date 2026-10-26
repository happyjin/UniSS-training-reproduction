"""Plan 2's CTC head under Megatron: evaluation/si_ctc_head.CTCHead, trained on (semantic codes, text) rows.

CTC over characters, mean over the batch of each utterance's loss divided by its text length.
A sample is one length-bucketed batch (training/pretrain_ctc_head.py), so a micro-batch is one bucket.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from megatron.core.models.huggingface import HuggingFaceModule

from experiments.uniss_qwen3_omni_v1.evaluation.si_ctc_head import CTCHead

METRIC_NAMES = ("loss", "utts", "frames")


class CTCHeadModel(HuggingFaceModule):
    def __init__(self, config, *, n_chars: int, pg_collection=None, **dims):
        super().__init__(config)
        self.pg_collection = pg_collection   # read off the innermost module by Megatron's train loop
        self.ctc = CTCHead(n_chars, **dims)

    def forward(self, codes, pad, targets, code_len, text_len):
        logp = self.ctc(codes, pad)                                   # [B, T, V], fp32
        loss = F.ctc_loss(logp.transpose(0, 1), targets, code_len, text_len, blank=0,
                          reduction="mean", zero_infinity=True)
        return (loss, torch.tensor(float(codes.shape[0]), device=codes.device),
                code_len.float().sum())
