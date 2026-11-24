"""sst_hsmm_v1 P10 (prereg change 18): write head on the FROZEN translator's next-token distribution.
Features of one position (log-softmax over the vocabulary, the wait token removed and renormalised):
  [top1 / 5, (top1 - top2) / 5, entropy / 5, max(log P(<|im_end|>), -30) / 10, max(log P(punctuation), -30) / 10]
(no target token: at inference the token to be written is the greedy one). WriteHead: 5 -> 32 -> 1 tanh MLP + linear skip,
last layer and skip zero-initialised (bias 0). The lattice uses l(c, i) = head(features) in place of log(1 - P_wait) - log P_wait
(passed as lp_wait = logsigmoid(-head)); inference (e2e_si_thinker, SST_HEAD) writes iff sigmoid(head + gate) > 0.5.
Shared by train_head.py and e2e_si_thinker.py so both compute identical features.
"""
from __future__ import annotations

import torch
import torch.nn as nn

WAIT, IM_END = 151662, 151645
NF = 5


def head_feats(lsm: torch.Tensor, punct_idx: torch.Tensor) -> torch.Tensor:
    """lsm (..., V) log-softmax (fp32) -> (..., 5)"""
    l = lsm.float().clone()
    l[..., WAIT] = -float("inf")
    l = l - torch.logsumexp(l, dim=-1, keepdim=True)
    t2 = torch.topk(l, 2, dim=-1).values
    p = torch.exp(l)
    ent = -torch.where(p > 0, p * l, torch.zeros_like(l)).sum(-1)
    eos = l[..., IM_END].clamp_min(-30.0)
    pun = torch.logsumexp(l[..., punct_idx.to(l.device)], dim=-1).clamp_min(-30.0)
    return torch.stack([t2[..., 0] / 5.0, (t2[..., 0] - t2[..., 1]) / 5.0, ent / 5.0, eos / 10.0, pun / 10.0], -1)


class WriteHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NF, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)
        self.skip = nn.Linear(NF, 1, bias=False)
        nn.init.zeros_(self.skip.weight)

    def forward(self, x):
        return (self.net(x) + self.skip(x))[..., 0]


# ---- P11 (prereg change 19): consistency + multiplier features ------------------------------------------------------------
NF2 = 10


def prev_feats(lsm: torch.Tensor) -> torch.Tensor:
    """training: lsm (C, n, V) log-softmax of chunks c = 0..C-1 at the same teacher-forced prefixes -> (C, n, 3)
    [agree = 1(greedy at c == greedy at c-1), prev_top1 = log P_{c-1}(greedy at c-1) / 5, known = 1(c > 0)]
    (WAIT removed and renormalised, as head_feats); chunk 0: [0, 0, 0]."""
    l = lsm.float().clone()
    l[..., WAIT] = -float("inf")
    l = l - torch.logsumexp(l, dim=-1, keepdim=True)
    top, g = l.max(-1)
    out = torch.zeros(*g.shape, 3, device=lsm.device)
    if g.shape[0] > 1:
        out[1:, :, 0] = (g[1:] == g[:-1]).float()
        out[1:, :, 1] = top[:-1].clamp_min(-30.0) / 5.0
        out[1:, :, 2] = 1.0
    return out


def prev_feats_online(greedy_tok: int, k: int, out_ids, prev, prevlp, has_prev: bool):
    """inference: the same 3 features from the previous read's greedy draft (tokens + processed log-probs, aligned to the
    committed prefix). Known iff a previous draft covers position k and this read's tokens so far equal its first k tokens
    (then the draft token at k IS the previous chunk's greedy token at this prefix); unknown -> [0, 0, 0] (= chunk 0)."""
    if has_prev and k < len(prev) and list(out_ids) == list(prev[:k]):
        return [float(greedy_tok == prev[k]), max(float(prevlp[k]), -30.0) / 5.0, 1.0]
    return [0.0, 0.0, 0.0]


class WriteHead2(WriteHead):
    """WriteHead on 10 inputs: head_feats (5) + prev feats (3) + (z_L, z_P) (2); same init (output 0)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NF2, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)
        self.skip = nn.Linear(NF2, 1, bias=False)
        nn.init.zeros_(self.skip.weight)
