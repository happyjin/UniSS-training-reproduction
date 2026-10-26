"""Plan 2's amortised self-monitoring: the talker learns to say, in one sample, what the Thinker's listening chose.

Rows come from si_infer --select-k --dump-seq on training items: per block, the talker's whole input at that round and
the continuation (codes + closing token) that, of k samples, the Thinker found likeliest to say the block's text. The
talker (trainable, from T1c) is fit to those continuations -- the posterior's choice distilled into the prior -- with a
KL to the frozen talker it started from on the code positions, so it moves toward what is heard right without
drifting from how it spoke.

loss = CE(chosen codes, closing x close_weight) + w_ref * KL(ref || talker) over codes and the end token
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from megatron.core.models.huggingface import HuggingFaceModule

from experiments.uniss_qwen3_omni_v1.training.si_model import SIModel, _frozen

METRIC_NAMES = ("loss", "ce", "kl_ref", "close_acc", "labels")


class SelfMonModel(HuggingFaceModule):
    def __init__(self, config, *, uniss_path: str, ref_path: str, w_ref: float, close_weight: float,
                 recompute: bool = True, pg_collection=None):
        super().__init__(config)
        from transformers import AutoModelForCausalLM

        self.pg_collection = pg_collection
        self.talker = AutoModelForCausalLM.from_pretrained(uniss_path, torch_dtype=torch.bfloat16)
        self.talker.requires_grad_(True)
        if recompute:
            self.talker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.w_ref, self.close_weight = float(w_ref), float(close_weight)
        self.ref = _frozen(ref_path) if self.w_ref > 0 else None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """The talker only; the frozen reference comes back from its path on every start."""
        full = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        frozen = {prefix + name for name, param in self.named_parameters() if not param.requires_grad}
        kept = {k: v for k, v in full.items() if k not in frozen}
        if not kept:
            raise SystemExit("every key was filtered out; the predicate is wrong")
        return kept

    def train(self, mode: bool = True):
        super().train(mode)
        if self.ref is not None:
            self.ref.eval()
        return self

    def forward(self, ids, mask, b, p, labels, is_close):
        own = SIModel._logits(self.talker, ids, mask, b, p)            # p: the state that predicts each label
        ce = F.cross_entropy(own, labels, reduction="none")
        w = 1.0 + (self.close_weight - 1.0) * is_close.float()
        ce_mean = (ce * w).sum() / w.sum()
        kl = torch.zeros((), device=own.device)
        codes = ~is_close
        if self.ref is not None and codes.any():
            with torch.no_grad():
                ref = SIModel._logits(self.ref, ids, mask, b[codes], p[codes])
            kl = SIModel._kl(ref, own[codes]).mean()
        loss = ce_mean + self.w_ref * kl
        with torch.no_grad():
            close_acc = (own[is_close].argmax(-1) == labels[is_close]).float().mean() if is_close.any() else torch.zeros((), device=own.device)
        return loss, ce_mean.detach(), kl.detach(), close_acc, torch.tensor(float(labels.numel()), device=own.device)
