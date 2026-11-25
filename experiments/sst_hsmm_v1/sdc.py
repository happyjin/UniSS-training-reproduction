"""sst_hsmm_v1 prereg change 27: Support-Determinacy Commit (SDC) -- features and survival head, shared by the rollout /
inference script (sdc_si_thinker.py) and the trainer (train_sdc.py), so both compute identical features.

Per non-final read and every not-yet-committed unit i (n <= i < len(d0)) of N's raw-view draft d0, the features use only
what is known online: N's enhanced-view draft d1, the SAP verifier's draft ds, the previous drafts of this item, and the
per-unit minimum token log-probabilities of d0 (N) and ds (SAP). The head predicts S_i = P(d0[:i+1] == final[:i+1]).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

THINKER_EOS = 151645
PUNCT_END = set("，。,.!?！？；;：:、…")
TAUS = (0.5, 0.8, 0.95)
CLIP = 10.0
NF = 23


def lcp(a, b) -> int:
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k


def unit_lp(tok, units, join, base_units, cont, out, zh):
    """per-unit minimum log-prob of the chosen tokens over the whole draft units(join(base) + sep + cont); committed units 0.0.
    Token j is assigned to the last unit of the decoded prefix ids[:j+1] (clipped to the new units)."""
    sep_ = "" if zh or not base_units or cont[:1].isspace() or cont[:1] in ",.;:!?)" else " "
    head = join(base_units, zh) + sep_
    full = units(head + cont, zh)
    n = len(base_units)
    lp = [0.0] * len(full)
    for u in range(n, len(full)):
        lp[u] = 0.0
    ids = [t for t in out.token_ids if t != THINKER_EOS]
    seen = [False] * len(full)
    cover = n - 1                                   # last unit index reached by an earlier token
    for j, t in enumerate(ids):
        pre = units(head + tok.decode(ids[: j + 1]).replace("�", ""), zh)
        u = min(len(pre) - 1, len(full) - 1)
        if u < n:
            continue
        v = float(out.logprobs[j][t].logprob)
        for x in range(min(cover + 1, u), u + 1):     # every unit this token (co-)produced
            if x >= n:
                lp[x] = v if not seen[x] else min(lp[x], v)
                seen[x] = True
        cover = max(cover, u)
    return full, lp


def _first_below(lp, start, tau):
    lt = math.log(tau)
    for i in range(start, len(lp)):
        if lp[i] <= lt:
            return i
    return len(lp)


def feats(d0, lp0, d1, ds, lps, n, prev_drafts, t_ms, is_zh):
    """-> (m, NF) float32 for units i = n .. len(d0)-1 (m may be 0). prev_drafts: earlier drafts of this item, oldest first."""
    m = len(d0) - n
    if m <= 0:
        return np.zeros((0, NF), np.float32)
    agreed = lcp(d0, prev_drafts[-1]) if prev_drafts else 0
    cv = lcp(d0, d1)
    sv = lcp(d0, ds)
    cN = [_first_below(lp0, n, tau) for tau in TAUS]
    cS = [min(_first_below(lps, n, tau), sv) for tau in TAUS]      # SAP confident AND equal to d0
    Ls = [lcp(p, d0) for p in reversed(prev_drafts)]               # most recent first
    lastp = -1
    for i in range(n):
        if d0[i][-1:] in PUNCT_END:
            lastp = i
    rows = np.zeros((m, NF), np.float32)
    c = lambda x: max(-CLIP, min(CLIP, float(x))) / CLIP
    for r, i in enumerate(range(n, len(d0))):
        age = 0
        for L in Ls:
            if L > i:
                age += 1
            else:
                break
        lpS_i = lps[i] if i < sv else -10.0
        f = [c(i - n), c(i - agreed), c(i - cv), c(i - sv),
             *[c(i - x) for x in cN], *[c(i - x) for x in cS],
             max(lp0[i], -10.0) / 10.0, max(lpS_i, -10.0) / 10.0,
             min(age, 10) / 10.0, float(age == 0), float(bool(prev_drafts)),
             min(t_ms / 1000.0, 30.0) / 30.0, c(i - lastp), float(len(ds) <= i), c(len(d0) - i),
             float(is_zh), float(i < agreed), float(i < cv), float(i < sv)]
        if d0[i][-1:] in PUNCT_END:
            lastp = i
        rows[r] = f
    assert rows.shape[1] == NF
    return rows


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NF, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)[..., 0]


def load_head(path):
    h = Head()
    h.load_state_dict(torch.load(path, map_location="cpu"))
    h.eval()
    return h


@torch.no_grad()
def survival(head, X):
    if len(X) == 0:
        return np.zeros(0, np.float32)
    return torch.sigmoid(head(torch.from_numpy(X))).numpy()


def commit_len(S, theta) -> int:
    """number of leading units with S >= theta (monotone prefix)"""
    k = 0
    while k < len(S) and S[k] >= theta:
        k += 1
    return k
