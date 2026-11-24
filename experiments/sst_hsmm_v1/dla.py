"""sst_hsmm_v1 prereg change 21: differentiable local agreement (DLA) lattice. LA2 commits token i at read c iff the drafts of
reads c and c-1 agree on it (given the committed prefix); the soft surrogate a(c, i) = sum_v p_c(v | y_<i) p_{c-1}(v | y_<i)
(probability that two independent draws agree) is the write probability of hsmm's speech-timeline lattice (OB bins, block
bit), so E[commit time] on the speech timeline and E[#agreed-but-wrong commits] are exact expectations under the surrogate and
differentiable in the Thinker's distributions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.sst_hsmm_v1.hsmm import K0, NEG, PAUSE_K, S, _wt_batch, s_of_bins


def lattice_risk_dla(lp_wait, lcost, g, dbin, punct, X_ms, ob: int, eos=True):
    """prereg change 21 (DLA): = hsmm.lattice_risk_block with gate None, beta 1, and a DIFFERENTIABLE per-write quality cost:
    lcost (C, n) = log cost of committing token i at chunk c (log P(agreed token != y_i | agreement)), added to channel 1 for
    real tokens at non-final chunks only (the final read pays nothing); lp_wait = log(1 - agreement probability).
    Was: prereg change 8 (P5): the policy pi_beta of lattice_risk on the same speech-timeline lattice, with the block bit b
    (b = 1 after a write in the current chunk; every chunk boundary resets b = 0) and the Talker per-block overhead ob bins:
    a block's first token (b = 0) starts its speech at max(s, ob) -> s' = max(s, ob) + d_i and pauses iff started, f = 0,
    s <= -0.3 + ob; a continuing token (b = 1): s' = max(s, 0) + d_i (no pause possible). Three expected costs by the
    expectation semiring (channel 0: A = log P; channels 1..3: log P*M_k):
      q: sum over real tokens of dis[c-1, i] (1 if the prefix model's greedy non-wait token at chunk c != y_i, detached)
      L: sum over real tokens of min(t_c + s', X)      P: number of pauses (real tokens only; the end token pays nothing)
    Returns tensor (E[q], E[L], E[P]) (gradients to lp_wait and the gate only). lp_tok: shape only.
    Checked against brute-force path enumeration (max abs error < 1e-6)."""
    C, n = lp_wait.shape
    dev = lp_wait.device
    X = X_ms / 1000.0
    n_real = n - 1 if eos else n
    s_pos = s_of_bins(dev).clamp_min(0.0)
    E = torch.full((4, S, 2, 2), NEG, device=dev)                  # (channel, s, f, b)
    T = [E] * (n + 1)
    t0 = E.clone(); t0[0, K0, 0, 0] = 0.0
    T[0] = t0
    kk = torch.arange(S, device=dev)
    pm = torch.zeros(S, 2, 2, dtype=torch.bool, device=dev)        # pause on writing (f = 0 only)
    pm[:, 0, 0] = kk <= PAUSE_K + ob
    pm[:, 0, 1] = kk <= PAUSE_K
    msk = torch.zeros(2, 1, S, 2, 2, dtype=torch.bool, device=dev)
    msk[0, :, :, 0, 1] = True
    msk[1, :, :, 1, 1] = True
    ch = torch.arange(4, device=dev)[:, None, None, None]
    done = []
    for c in range(1, C + 1):
        last = c == C
        t_c = X if last else float(c)
        loglat = torch.log(torch.clamp(t_c + s_pos, max=X).clamp_min(1e-6))
        gt = None
        for i in range(n):
            if not last and c < g[i]:
                continue
            if T[i] is E:
                continue
            cur = T[i]
            real = i < n_real
            if last:
                s_ = cur
            else:
                l1 = torch.log1p(-torch.exp(lp_wait[c - 1, i]).clamp(max=1 - 1e-6))
                z = (l1 - lp_wait[c - 1, i]) + (gt[:, :, 1 if i > 0 else 0] if gt is not None else 0.0)
                if not torch.is_tensor(z) or z.dim() == 0:
                    z = torch.as_tensor(z, device=dev).expand(S, 2)
                lw, l1w = F.logsigmoid(z), F.logsigmoid(-z)
                s_ = cur + lw[None, :, :, None]
            if real and not last:
                s_ = torch.where(ch == 1, torch.logaddexp(s_, s_[0:1] + lcost[c - 1, i]), s_)
            if real and i > 0:
                s_ = torch.where((ch == 3) & pm[None], torch.logaddexp(s_, s_[0:1]), s_)
            d_ = int(dbin[i])
            tg = torch.logaddexp(_wt_batch(s_[..., 0], d_, K0 + ob), _wt_batch(s_[..., 1], d_, K0))     # (4, S)
            if real:
                tg = torch.cat([tg[0:2], torch.logaddexp(tg[2], tg[0] + loglat)[None], tg[3:4]], 0)
            M = msk[int(punct[i])]
            T[i + 1] = torch.where(M, torch.logaddexp(T[i + 1], tg[:, :, None, None]), T[i + 1])
            T[i] = E if last else cur + l1w[None, :, :, None]
        if T[n] is not E:
            done.append(torch.logsumexp(T[n].reshape(4, -1), 1))
        T[n] = E
        if not last:
            for i in range(n):
                if T[i] is E:
                    continue
                a = torch.logsumexp(T[i], dim=-1)                  # b reset: (4, S, 2); drain 1 s
                a = torch.cat([torch.logaddexp(a[:, 10], torch.logsumexp(a[:, :10], 1))[:, None], a[:, 11:],
                               torch.full((4, 10, 2), NEG, device=dev)], 1)
                T[i] = torch.stack([a, torch.full_like(a, NEG)], -1)
    D = torch.logsumexp(torch.stack(done), 0)
    return torch.exp(D[1:] - D[0])


