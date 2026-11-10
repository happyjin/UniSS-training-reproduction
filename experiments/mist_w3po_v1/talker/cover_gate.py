"""Scheme 2's inference rule (reports/mist_w3po_v1/talker_crf_prereg_v1.zh-CN.md §2.4): the alignment posterior's
coverage of a spoken block. P_cover = Z_cover / Z_all of align_crf on the talker's own hidden states (the exported
talker read by transformers, no grad) and the trained alignment head (added.pt: align_q, align_k). si_infer
--cover-gate resamples a block whose P_cover < 0.5 (at most --cover-gate-tries times): rejection sampling of the event
"every text unit is covered before the closing token", block by block.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.mist_w3po_v1.talker import align_crf

START_CONTENT, SEMANTIC_0, SEMANTIC_LAST = 180386, 155761, 163952


class CoverGate:
    def __init__(self, model, wq: torch.Tensor, wk: torch.Tensor, kappa: float | None, device, batch: int = 8):
        self.model, self.wq, self.wk, self.kappa, self.device, self.batch = model, wq, wk, kappa, device, batch

    @classmethod
    def from_paths(cls, model_dir: str, added_pt: str, kappa: float | None, device):
        from transformers import AutoModelForCausalLM

        m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to(device).eval()
        add = torch.load(added_pt, map_location="cpu")["added"]
        return cls(m, add["align_q.weight"].float().to(device), add["align_k.weight"].float().to(device), kappa, device)

    @torch.no_grad()
    def p_cover(self, prompts: list[list[int]], outs: list[list[int]], lens: list[list[int]]) -> list[float]:
        """prompts: the talker input of each block (ends at START_SEMANTIC; the block's text follows its last
        START_CONTENT); outs: the generated ids (codes [+ closing]); lens: the text units' character lengths."""
        res = []
        for s in range(0, len(prompts), self.batch):
            P, O, Ls = prompts[s: s + self.batch], outs[s: s + self.batch], lens[s: s + self.batch]
            seqs = [p + o for p, o in zip(P, O)]
            L = max(len(x) for x in seqs)
            ids = torch.zeros(len(seqs), L, dtype=torch.long, device=self.device)
            mask = torch.zeros_like(ids)
            for j, x in enumerate(seqs):
                ids[j, : len(x)] = torch.tensor(x, device=self.device)
                mask[j, : len(x)] = 1
            h = self.model.model(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
            t0 = [len(p) - p[::-1].index(START_CONTENT) for p in P]          # first text token of the block
            M = [len(x) for x in Ls]
            T = [sum(1 for t in o if SEMANTIC_0 <= t <= SEMANTIC_LAST) for o in O]
            c0 = [len(p) for p in P]
            Tm, Mm = max(max(T), 1), max(max(M), 1)
            ar_t, ar_m = torch.arange(Tm, device=self.device), torch.arange(Mm, device=self.device)
            b = torch.arange(len(seqs), device=self.device)[:, None]
            fpos = (torch.tensor(c0, device=self.device)[:, None] + ar_t[None, :]).clamp(max=L - 1)
            upos = (torch.tensor(t0, device=self.device)[:, None] + ar_m[None, :]).clamp(max=L - 1)
            hf, hu = h[b, fpos].float(), h[b, upos].float()
            e = torch.einsum("rtd,rmd->rtm", F.linear(hf, self.wq), F.linear(hu, self.wk)) / 16.0
            lt = torch.ones(len(seqs), Mm, dtype=torch.long, device=self.device)
            for j, x in enumerate(Ls):
                lt[j, : len(x)] = torch.tensor(x, device=self.device)
            zc, za = align_crf.forward(e, torch.tensor(T, device=self.device), torch.tensor(M, device=self.device), lt,
                                       kappa=self.kappa)
            ok = (zc > align_crf.NEG / 2) & (za > align_crf.NEG / 2)
            res += torch.where(ok, (zc - za).clamp(max=0).exp(), torch.zeros_like(zc)).tolist()
        return res
