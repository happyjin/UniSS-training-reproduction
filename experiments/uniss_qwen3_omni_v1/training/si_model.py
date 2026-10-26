"""T1: UniSS as a streaming Talker, from Plan 1's model (R2).

  talker   trained; starts as R2 (UniSS after Plan 1, source GLM as inference sees it).
  teacher  frozen R2: reads each si row's whole source and whole translation
           (the offline sequence); at every code the streaming model is pulled
           towards the teacher's distribution -- what the offline model would
           say there knowing everything (future-guided distillation).
  anchor   frozen original UniSS, on offline replay rows only, as in Plan 1;
           rows into Chinese weighted anchor_cmn_mult, since that is the
           direction Plan 1 lost 0.2 BLEU in.

loss = CE(codes and closing tokens) + w_future * KL(teacher || talker)   on si steps
       CE(codes and end token)      + w_anchor * KL(anchor || talker)    on offline steps
Both KLs over what a code position may hold: the 8,192 semantic codes and the end token.

init_wait_from_end: <|wait_read|> has never been trained -- its row (tied: input
and output) gave it probability 0.000 at every closing position, and 100 steps
at 1e-5 did not move it. Started as a copy of the end token's row it is "stop"
from the first step, and the model only has to learn which stop a block's
header asks for. close_weight scales the closing positions' CE: a row has two
or three of them against some four hundred codes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from megatron.core.models.huggingface import HuggingFaceModule

from experiments.uniss_qwen3_omni_v1.training.e2_model import END_SEMANTIC, SEMANTIC_0, SEMANTIC_LAST
from experiments.uniss_qwen3_omni_v1.training.si_batch import LOOK_CLOSE, LOOK_OPEN, WAIT_READ

# one row per step, averaged over the log window: a pool's own mean is its sum over its flag
METRIC_NAMES = ("loss", "si", "si_code_ce", "si_close_acc", "si_kl_future", "off", "off_code_ce", "off_kl_anchor",
                "supervised_tokens")


def _frozen(path: str):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    m.requires_grad_(False)
    m.eval()
    return m


class SIModel(HuggingFaceModule):
    def __init__(self, config, *, uniss_path: str, teacher_path: str, anchor_path: str, w_future: float,
                 w_anchor: float, recompute: bool = True, init_wait_from_end: bool = False, close_weight: float = 1.0,
                 init_look_from_content: bool = False):
        super().__init__(config)
        from transformers import AutoModelForCausalLM

        self.talker = AutoModelForCausalLM.from_pretrained(uniss_path, torch_dtype=torch.bfloat16)
        if init_wait_from_end:   # a resume loads the trained row over this
            with torch.no_grad():
                for table in {id(t): t for t in (self.talker.get_input_embeddings(), self.talker.get_output_embeddings())}.values():
                    table.weight[WAIT_READ] = table.weight[END_SEMANTIC]
        if init_look_from_content:   # the lookahead brackets start as copies of the content brackets
            from experiments.uniss_qwen3_omni_v1.training.e2_batch import END_CONTENT, START_CONTENT

            with torch.no_grad():
                for table in {id(t): t for t in (self.talker.get_input_embeddings(), self.talker.get_output_embeddings())}.values():
                    table.weight[LOOK_OPEN] = table.weight[START_CONTENT]
                    table.weight[LOOK_CLOSE] = table.weight[END_CONTENT]
        self.close_weight = float(close_weight)
        self.talker.requires_grad_(True)
        if recompute:
            self.talker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.w_future, self.w_anchor = float(w_future), float(w_anchor)
        self.teacher = _frozen(teacher_path) if self.w_future > 0 else None
        self.anchor = _frozen(anchor_path) if self.w_anchor > 0 else None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """The talker only; the frozen models come back from their paths on every start."""
        full = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        frozen = {prefix + name for name, param in self.named_parameters() if not param.requires_grad}
        kept = {k: v for k, v in full.items() if k not in frozen}
        if not kept:
            raise SystemExit("every key was filtered out; the predicate is wrong")
        return kept

    def train(self, mode: bool = True):
        super().train(mode)
        for frozen in (self.teacher, self.anchor):
            if frozen is not None:
                frozen.eval()
        return self

    def trainable_report(self) -> dict:
        return {"talker_m": round(sum(p.numel() for p in self.talker.parameters() if p.requires_grad) / 1e6, 1),
                "teacher": self.teacher is not None, "anchor": self.anchor is not None,
                "w_future": self.w_future, "w_anchor": self.w_anchor}

    @staticmethod
    def _logits(model, ids, mask, b, p):
        states = model.model(inputs_embeds=model.get_input_embeddings()(ids), attention_mask=mask,
                             return_dict=True).last_hidden_state
        return model.lm_head(states[b, p]).float()

    @staticmethod
    def _kl(ref_logits, own_logits):
        keep = torch.cat([torch.arange(SEMANTIC_0, SEMANTIC_LAST + 1, device=own_logits.device),
                          torch.tensor([END_SEMANTIC], device=own_logits.device)])
        ref = F.log_softmax(ref_logits[:, keep], dim=-1)
        own = F.log_softmax(own_logits[:, keep], dim=-1)
        return (ref.exp() * (ref - own)).sum(-1)

    def forward(self, stream, uniss_ids, uniss_mask, label_b, label_p, labels, label_w=None, closing=None,
                teacher_ids=None, teacher_mask=None, kl_b=None, kl_s=None, kl_t=None):
        logits = self._logits(self.talker, uniss_ids, uniss_mask, label_b, label_p)
        ce_each = F.cross_entropy(logits, labels, reduction="none")
        ce = ce_each.mean()
        zero = torch.zeros((), device=ce.device)
        n = torch.tensor(float(labels.numel()), device=ce.device)
        if int(stream):
            codes = ~closing
            if self.close_weight != 1.0:
                w = torch.where(closing, self.close_weight, 1.0)
                ce = (ce_each * w).sum() / w.sum()
            code_ce = ce_each[codes].mean() if codes.any() else zero
            close_acc = ((logits[closing].argmax(-1) == labels[closing]).float().mean() if closing.any() else zero)
            kl = zero
            if self.teacher is not None and kl_s.numel():
                with torch.no_grad():
                    ref = self._logits(self.teacher, teacher_ids, teacher_mask, kl_b, kl_t)
                kl = self._kl(ref, logits[kl_s]).mean()
            loss = ce + self.w_future * kl
            return torch.stack([loss, torch.ones_like(ce), code_ce.detach(), close_acc.detach(), kl.detach(),
                                zero, zero, zero, n])
        kl = zero
        if self.anchor is not None:
            with torch.no_grad():
                ref = self._logits(self.anchor, uniss_ids, uniss_mask, label_b, label_p)
            kl = (self._kl(ref, logits) * label_w).sum() / label_w.sum()
        loss = ce + self.w_anchor * kl
        return torch.stack([loss, zero, zero, zero, zero, torch.ones_like(ce), ce.detach(), kl.detach(), n])
