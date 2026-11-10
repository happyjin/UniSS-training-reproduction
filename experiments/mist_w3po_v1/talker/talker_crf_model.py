"""The trainable talker of the two schemes (pretrain_talker_crf.py): T1c's export, full parameters, a frozen copy as
the KL reference, and (align mode) the alignment head.

plem   loss = sum_l w_l * cw_l * CE_l / sum_l w_l * cw_l  +  w_speed * mean_blocks(-sum_v q_v log softmax_35(speed))
              + w_ref * sum_codes w * KL(ref || talker) / sum_codes w
align  loss = CE(chosen rows; closing x close_weight)  +  w_ref * KL(ref || talker) on the chosen rows' codes
              + w_align * mean_rows(alignment loss)  +  w_dur * mean_label1(duration PR)
KL over the 8,192 semantic codes and the end token, as SIModel._kl.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from megatron.core.models.huggingface import HuggingFaceModule

from experiments.mist_w3po_v1.talker import align_crf
from experiments.uniss_qwen3_omni_v1.training.si_model import SIModel, _frozen

SPEED_0, N_SPEED = 180337, 35
ALIGN_DIM = 256


class TalkerCRFModel(HuggingFaceModule):
    def __init__(self, config, *, mode: str, uniss_path: str, ref_path: str, w_ref: float, w_speed: float, w_align: float,
                 w_dur: float, close_weight: float, recompute: bool = True):
        super().__init__(config)
        from transformers import AutoModelForCausalLM

        self.mode = mode
        self.talker = AutoModelForCausalLM.from_pretrained(uniss_path, torch_dtype=torch.bfloat16)
        self.talker.requires_grad_(True)
        if recompute:
            self.talker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.w_ref, self.w_speed, self.w_align, self.w_dur = float(w_ref), float(w_speed), float(w_align), float(w_dur)
        self.close_weight = float(close_weight)
        self.ref = _frozen(ref_path) if self.w_ref > 0 else None
        if mode == "align":
            h = self.talker.config.hidden_size
            self.align_q = torch.nn.Linear(h, ALIGN_DIM, bias=False, dtype=torch.float32)
            self.align_k = torch.nn.Linear(h, ALIGN_DIM, bias=False, dtype=torch.float32)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """trainable tensors only (talker, alignment head); the frozen reference comes back from its path."""
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

    @staticmethod
    def states(model, ids, mask):
        return model.model(inputs_embeds=model.get_input_embeddings()(ids), attention_mask=mask,
                           return_dict=True).last_hidden_state

    def forward(self, t: dict):
        z = torch.zeros((), device=t["ids"].device)
        if self.mode == "plem":
            return self._plem(t, z)
        return self._align(t, z)

    def _ce(self, logits, labels, is_close, w=None):
        ce = F.cross_entropy(logits, labels, reduction="none")
        cw = 1.0 + (self.close_weight - 1.0) * is_close.float()
        ww = cw if w is None else cw * w
        acc = (logits[is_close].argmax(-1) == labels[is_close]).float().mean() if is_close.any() else torch.zeros((), device=logits.device)
        return (ce * ww).sum() / ww.sum().clamp(min=1e-8), acc.detach()

    def _plem(self, t, z):
        h = self.states(self.talker, t["ids"], t["mask"])
        logits = self.talker.lm_head(h[t["b"], t["p"]]).float()
        is_close = t["is_close"].bool()
        ce, acc = self._ce(logits, t["labels"], is_close, t["w"])
        spd = z
        if t["sb"].numel():
            sl = self.talker.lm_head(h[t["sb"], t["sp"]]).float()[:, SPEED_0: SPEED_0 + N_SPEED]
            spd = -(t["sq"] * F.log_softmax(sl, dim=-1)).sum(-1).mean()
        kl = z
        codes = ~is_close
        if self.ref is not None and codes.any():
            with torch.no_grad():
                ref = SIModel._logits(self.ref, t["rids"], t["rmask"], t["b"][codes], t["pr"][codes])
            k = SIModel._kl(ref, logits[codes])
            kl = (k * t["w"][codes]).sum() / t["w"][codes].sum().clamp(min=1e-8)
        loss = ce + self.w_speed * spd + self.w_ref * kl
        n = torch.tensor(float(t["labels"].numel()), device=ce.device)
        return torch.stack([loss, ce.detach(), kl.detach(), spd.detach(), z, z, z, z, acc, n])

    def _align(self, t, z):
        h = self.states(self.talker, t["ids"], t["mask"])
        ce, acc, kl = z, z, z
        if t["b"].numel():
            logits = self.talker.lm_head(h[t["b"], t["p"]]).float()
            is_close = t["is_close"].bool()
            ce, acc = self._ce(logits, t["labels"], is_close)
            codes = ~is_close
            if self.ref is not None and codes.any():
                with torch.no_grad():
                    ref = SIModel._logits(self.ref, t["ids"], t["mask"], t["b"][codes], t["p"][codes])
                kl = SIModel._kl(ref, logits[codes]).mean()
        # alignment rows: text-unit states and code-frame states of each row
        R = t["a_b"].numel()
        Tm, Mm = int(t["a_Tm"]), int(t["a_Mm"])
        dev = h.device
        ar_t = torch.arange(Tm, device=dev)
        ar_m = torch.arange(Mm, device=dev)
        fpos = (t["a_c0"][:, None] + ar_t[None, :]).clamp(max=h.shape[1] - 1)          # code token positions
        upos = (t["a_t0"][:, None] + ar_m[None, :]).clamp(max=h.shape[1] - 1)          # text token positions
        hf = h[t["a_b"][:, None], fpos].float()                                          # R x Tm x H
        hu = h[t["a_b"][:, None], upos].float()                                          # R x Mm x H
        wq, wk = self.align_q.weight, self.align_k.weight   # Megatron's bf16 wrapper may have cast the head
        e = torch.einsum("rtd,rmd->rtm", F.linear(hf.to(wq.dtype), wq).float(), F.linear(hu.to(wk.dtype), wk).float()) / 16.0
        al, prl, p = align_crf.losses(e, t["a_T"], t["a_M"], t["a_lens"], t["a_y"], pr=self.w_dur > 0)
        lab = t["a_y"]
        n_lab = (lab >= 0).sum().clamp(min=1)
        align = al.sum() / n_lab
        pos = lab == 1
        dur = prl.sum() / pos.sum().clamp(min=1)
        loss = ce + self.w_ref * kl + self.w_align * align + self.w_dur * dur
        p_pos = p[pos].mean() if pos.any() else z
        p_neg = p[lab == 0].mean() if (lab == 0).any() else z
        n = torch.tensor(float(t["labels"].numel() + R), device=dev)
        return torch.stack([loss, ce.detach(), kl.detach(), z, align.detach(), dur.detach(), p_pos.detach(),
                            p_neg.detach(), acc, n])
