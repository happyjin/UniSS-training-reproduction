"""sst_hsmm_v1 speech-timeline hidden semi-Markov lattice (reports/sst_hsmm_v1/prereg.zh-CN.md section 2).

Per sample: chunks c = 1..C (audio <= c s; the last chunk is the whole source, time X), target tokens i = 0..n-1 with chunk
lower bound g_i, speech duration d_i (100 ms bins) and punctuation flag p_i. State inside a chunk: (i tokens written,
s speech backlog / silence bin, f last written token is punctuation); s bins k = 0..110 <-> s = -3.0 + 0.1 k seconds
(k >= 30: backlog, k < 30: silent for |s|); i = 0 means nothing written yet (no pause can be counted).

Chunk c (time t_c = c, or X / 1000 for the last chunk), tokens swept in order: token i is writable iff c >= g_i;
  write prob  w = sigmoid(l(c, i) + gate(s, f, started, c / C)),  l = log(1 - P_wait) - log P_wait
  emission    log p(y_i) - log(1 - P_wait)
  pause       started and s <= -0.3 and f = 0 -> cost lam_P
  update      s <- max(s, 0) + d_i, f <- p_i;   latency cost lam_L * min(t_c + s, X)
a state that does not write waits: prob 1 - w (also when c < g_i: forced wait, prereg change 4), then s <- max(s - 1, -3)
at the chunk boundary. Last chunk: every
remaining token is written (w = 1, no transition term). All-written states are absorbed.
eos (prereg change 5): the caller appends the end-of-turn token <|im_end|> as token n-1 with g = C, d = 0, p = 1; it is
written only in the last chunk, pays no pause cost, and every earlier chunk in which all real tokens are written pays
the forced wait log(1 - w) of the end token (the model learns to wait, not to write, once it has said what it heard).
loss(sample) = -(1/n) log sum_z exp(score(z))   (exact forward, log space, autograd).
viterbi(...) = the same lattice in max-plus with uniform transitions (P0c); returns the chunk of every token.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

S, K0, PAUSE_K = 111, 30, 27          # bins; k = 30 is s = 0; s <= -0.3  <=>  k <= 27
NEG = -1e30


def s_of_bins(device):
    return -3.0 + 0.1 * torch.arange(S, device=device, dtype=torch.float32)


class Gate(nn.Module):
    """g_psi(s, f, started, c/C): 4 -> 32 -> 1 tanh MLP, last layer zero-initialised"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def table(self, cfrac: float, device) -> torch.Tensor:
        """-> (S, 2 f, 2 started) gate values"""
        s = s_of_bins(device)
        out = torch.empty(S, 2, 2, device=device)
        feats = []
        for f in (0, 1):
            for st in (0, 1):
                feats.append(torch.stack([s / 8.0, torch.full_like(s, float(f)), torch.full_like(s, float(st)),
                                          torch.full_like(s, cfrac)], -1))
        g = self.net(torch.stack(feats, 0))[..., 0]           # (4, S)
        return g.view(2, 2, S).permute(2, 0, 1)


def _write_targets(src: torch.Tensor, dbin: int, viterbi: bool):
    """src (S, 2) scores of the writing mass by source (s, f) -> (S,) scores at the target bin (f merged)"""
    red = (lambda x: x.max(dim=-1).values) if viterbi else (lambda x: torch.logsumexp(x, dim=-1))
    neg = red(src[:K0].reshape(-1)) if K0 else None            # silent -> s' = d
    pos = red(src[K0:])                                        # (S - K0,) backlog s -> s + d
    tgt = torch.full((S,), NEG, device=src.device, dtype=src.dtype)
    hi = S - 1
    # positives shifted by dbin, clamped at the top bin
    k_src = torch.arange(K0, S, device=src.device)
    k_tgt = torch.clamp(k_src + dbin, max=hi)
    if viterbi:
        tgt = tgt.scatter_reduce(0, k_tgt, pos, reduce="amax", include_self=True)
    else:
        m = torch.full((S,), NEG, device=src.device, dtype=src.dtype)
        m = m.scatter_reduce(0, k_tgt, pos, reduce="amax", include_self=True)
        ex = torch.zeros(S, device=src.device, dtype=src.dtype).scatter_add(0, k_tgt, torch.exp(pos - m[k_tgt]))
        tgt = torch.where(ex > 0, m + torch.log(ex.clamp_min(1e-38)), tgt)
    kd = min(K0 + dbin, hi)
    comb = torch.logaddexp(tgt[kd], neg) if not viterbi else torch.maximum(tgt[kd], neg)
    onehot = torch.arange(S, device=src.device) == kd
    return torch.where(onehot, comb, tgt)


def lattice(lp_tok, lp_wait, g, dbin, punct, X_ms, lam_L, lam_P, gate: Gate | None, viterbi=False, uniform=False,
            eos=False):
    """lp_tok, lp_wait: (C, n) log p(y_i) and log P_wait at pass c (row c - 1). g: chunk lower bounds (1-based).
    dbin: token durations in 100 ms bins (ints). punct: 0/1. Returns (score, path-or-None).
    score = log sum (forward) or max (viterbi) over all paths; uniform: transitions 0 (P0c). No in-place ops on graph
    tensors: the state is a list of (S, 2) tensors, one per i."""
    C, n = lp_tok.shape
    dev = lp_tok.device
    X = X_ms / 1000.0
    s_pos = s_of_bins(dev).clamp_min(0.0)
    add = (lambda a, b: torch.maximum(a, b)) if viterbi else (lambda a, b: torch.logaddexp(a, b))
    red = (lambda x: x.max()) if viterbi else (lambda x: torch.logsumexp(x.reshape(-1), 0))
    empty = torch.full((S, 2), NEG, device=dev)
    A = [empty] * (n + 1)
    init = torch.full((S, 2), NEG, device=dev)
    init[K0, 0] = 0.0
    A[0] = init
    pen = torch.zeros(S, 2, device=dev)
    pen[: PAUSE_K + 1, 0] = lam_P
    E = torch.zeros(C, n, device=dev, requires_grad=viterbi)
    done = []
    for c in range(1, C + 1):
        last = c == C
        t_c = X if last else float(c)
        lat = lam_L * torch.clamp(t_c + s_pos, max=X)                       # (S,) by target bin
        gt = gate.table(c / C, dev) if (gate is not None and not uniform and not last) else None
        for i in range(n):
            if not last and c < g[i]:
                # prereg change 4: a forced wait (token not yet writable) still pays its transition log(1 - w)
                if not uniform and lp_wait is not None and A[i] is not empty:
                    lwt = lp_wait[c - 1, i]
                    l1 = torch.log1p(-torch.exp(lwt).clamp(max=1 - 1e-6))
                    z = (l1 - lwt) + (gt[:, :, 1 if i > 0 else 0] if gt is not None else 0.0)
                    if not torch.is_tensor(z) or z.dim() == 0:
                        z = torch.as_tensor(z, device=dev).expand(S, 2)
                    A[i] = A[i] + F.logsigmoid(-z)
                continue
            cur = A[i]
            if last or uniform:
                lw = 0.0
                l1w = None if last else 0.0
                em = lp_tok[c - 1, i] if (uniform or lp_wait is None) else lp_tok[c - 1, i] - torch.log1p(
                    -torch.exp(lp_wait[c - 1, i]).clamp(max=1 - 1e-6))
            else:
                lwt = lp_wait[c - 1, i]
                l1 = torch.log1p(-torch.exp(lwt).clamp(max=1 - 1e-6))
                z = (l1 - lwt) + (gt[:, :, 1 if i > 0 else 0] if gt is not None else 0.0)
                if not torch.is_tensor(z) or z.dim() == 0:
                    z = torch.as_tensor(z, device=dev).expand(S, 2)
                lw, l1w = F.logsigmoid(z), F.logsigmoid(-z)
                em = lp_tok[c - 1, i] - l1
            src = cur + lw + em + E[c - 1, i]
            if i > 0 and not (eos and i == n - 1):     # prereg change 5: the end token pays no pause cost
                src = src - pen
            tgt = _write_targets(src, int(dbin[i]), viterbi) - lat
            f2 = int(punct[i])
            upd = torch.stack([A[i + 1][:, 0], A[i + 1][:, 1]], 1)
            col = add(upd[:, f2], tgt)
            A[i + 1] = torch.stack([col, upd[:, 1]], 1) if f2 == 0 else torch.stack([upd[:, 0], col], 1)
            A[i] = empty if last else cur + l1w
        done.append(red(A[n]))
        A[n] = empty
        if not last:   # chunk boundary: s -> max(s - 1, -3): bin k -> k - 10, bins k < 10 clamp to 0
            for i in range(n):
                a = A[i]
                if a is empty:
                    continue
                low = a[:10].max(dim=0).values if viterbi else torch.logsumexp(a[:10], dim=0)
                body = a[10:]
                head = add(body[0], low)
                A[i] = torch.cat([head[None], body[1:], torch.full((10, 2), NEG, device=dev)], 0)
    sc = torch.stack(done)
    total = sc.max() if viterbi else torch.logsumexp(sc, 0)
    if not viterbi:
        return total, None
    total.backward()
    path = [int(E.grad[:, i].argmax()) + 1 for i in range(n)]
    return total.detach(), path


def path_stats(path, dbin, punct, X_ms):
    """speech dynamics of a chunk assignment -> (latency proxy mean_i min(e_i, X) s, in-phrase pauses)"""
    X = X_ms / 1000.0
    s, f, started, lat, pauses = 0.0, 0, False, [], 0
    prev_c = 1
    C = math.ceil(X_ms / 1000)
    for i, c in enumerate(path):
        while prev_c < c:
            s = max(s - 1.0, -3.0); prev_c += 1
        t_c = X if c == C else float(c)
        if started and s <= -0.3 + 1e-9 and f == 0:
            pauses += 1
        s = max(s, 0.0) + 0.1 * dbin[i]
        f, started = int(punct[i]), True
        lat.append(min(t_c + s, X))
    return (sum(lat) / max(len(lat), 1)), pauses


def path_laal(path, dbin, punct, X_ms, ref_len=None):
    """LAAL-form version of path_stats (analysis only): d_i = time token i has been spoken (t_c + backlog after it, not
    capped), LAAL of Papi et al. over tokens (si_metrics.laal: tau cutoff, |Y| = max(n, ref_len)) -> (seconds, pauses)"""
    from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal
    X = X_ms / 1000.0
    s, f, started, d, pauses = 0.0, 0, False, [], 0
    prev_c = 1
    C = math.ceil(X_ms / 1000)
    for i, c in enumerate(path):
        while prev_c < c:
            s = max(s - 1.0, -3.0); prev_c += 1
        t_c = X if c == C else float(c)
        if started and s <= -0.3 + 1e-9 and f == 0:
            pauses += 1
        s = max(s, 0.0) + 0.1 * dbin[i]
        f, started = int(punct[i]), True
        d.append((t_c + s) * 1000.0)
    return laal(d, X_ms, max(len(d), ref_len or 0)) / 1000.0, pauses


def lattice_risk(lp_tok, lp_wait, g, dbin, punct, X_ms, lam_L, lam_P, gate: Gate | None, beta=1.0, eos=True):
    """prereg change 7: exact expected cost under the policy pi_beta (write prob sigmoid(beta * (l + gate))) on the same
    speech-timeline lattice, by the expectation semiring in log space: every state carries A = log P (policy mass) and
    Q = log (P * M) (M = mean accumulated cost, >= 0). cost = sum_i -log p~(y_i) (detached, p~ = token prob renormalised
    without wait) + lam_L * min(t_c + s, X) per written token + lam_P per pause (the end token: no pause cost).
    A token with c < g_i is not writable and waits with probability 1 (the read just ends; no mass is lost).
    Returns E[cost] (scalar tensor; gradients flow to lp_wait and the gate only)."""
    C, n = lp_tok.shape
    dev = lp_tok.device
    X = X_ms / 1000.0
    s_pos = s_of_bins(dev).clamp_min(0.0)
    empty = torch.full((S, 2), NEG, device=dev)
    A = [empty] * (n + 1)
    Q = [empty] * (n + 1)
    init = torch.full((S, 2), NEG, device=dev)
    init[K0, 0] = 0.0
    A[0] = init
    pen = torch.zeros(S, 2, device=dev)
    pen[: PAUSE_K + 1, 0] = lam_P
    lpt = lp_tok.detach()
    doneA, doneQ = [], []

    def addcost(a, q, k):            # P*M -> P*(M + k), k >= 0 (tensor broadcastable or float)
        k = torch.as_tensor(k, device=dev, dtype=a.dtype)
        return torch.logaddexp(q, a + torch.log(k.clamp_min(1e-30))) if bool((k > 0).any()) else q

    for c in range(1, C + 1):
        last = c == C
        t_c = X if last else float(c)
        lat = lam_L * torch.clamp(t_c + s_pos, max=X)
        gt = gate.table(c / C, dev) if (gate is not None and not last) else None
        for i in range(n):
            if not last and c < g[i]:
                continue                                     # not writable: waits with prob 1
            if A[i] is empty:
                continue
            curA, curQ = A[i], Q[i]
            l1 = torch.log1p(-torch.exp(lp_wait[c - 1, i]).clamp(max=1 - 1e-6))
            cem = -(lpt[c - 1, i] - l1.detach())              # -log p~(y_i), detached
            if last:
                lw, l1w = 0.0, None
            else:
                z = (l1 - lp_wait[c - 1, i]) + (gt[:, :, 1 if i > 0 else 0] if gt is not None else 0.0)
                if not torch.is_tensor(z) or z.dim() == 0:
                    z = torch.as_tensor(z, device=dev).expand(S, 2)
                lw, l1w = F.logsigmoid(beta * z), F.logsigmoid(-beta * z)
            sA, sQ = curA + lw, curQ + lw
            k = cem.clamp_min(0.0) + (pen if (i > 0 and not (eos and i == n - 1)) else 0.0)
            sQ = addcost(sA, sQ, k if torch.is_tensor(k) and k.dim() else torch.full((S, 2), float(k), device=dev))
            tA = _write_targets(sA, int(dbin[i]), False)
            tQ = _write_targets(sQ, int(dbin[i]), False)
            tQ = torch.logaddexp(tQ, tA + torch.log(lat.clamp_min(1e-30)))
            f2 = int(punct[i])
            uA = torch.stack([A[i + 1][:, 0], A[i + 1][:, 1]], 1)
            uQ = torch.stack([Q[i + 1][:, 0], Q[i + 1][:, 1]], 1)
            cA, cQ = torch.logaddexp(uA[:, f2], tA), torch.logaddexp(uQ[:, f2], tQ)
            A[i + 1] = torch.stack([cA, uA[:, 1]], 1) if f2 == 0 else torch.stack([uA[:, 0], cA], 1)
            Q[i + 1] = torch.stack([cQ, uQ[:, 1]], 1) if f2 == 0 else torch.stack([uQ[:, 0], cQ], 1)
            if last:
                A[i], Q[i] = empty, empty
            else:
                A[i], Q[i] = curA + l1w, curQ + l1w
        doneA.append(torch.logsumexp(A[n].reshape(-1), 0))
        doneQ.append(torch.logsumexp(Q[n].reshape(-1), 0))
        A[n], Q[n] = empty, empty
        if not last:
            for i in range(n):
                for arr in (A, Q):
                    a = arr[i]
                    if a is empty:
                        continue
                    low = torch.logsumexp(a[:10], dim=0)
                    head = torch.logaddexp(a[10], low)
                    arr[i] = torch.cat([head[None], a[11:], torch.full((10, 2), NEG, device=dev)], 0)
    PA = torch.logsumexp(torch.stack(doneA), 0)
    PQ = torch.logsumexp(torch.stack(doneQ), 0)
    return torch.exp(PQ - PA)


def _wt_batch(src: torch.Tensor, dbin: int, kf: int):
    """src (B, S, 2) log-mass by source (s, f) -> (B, S) at the target bin, s' = max(s, s(kf)) + d (f merged); kf >= K0"""
    B = src.shape[0]
    hi = S - 1
    low = torch.logsumexp(src[:, :kf].reshape(B, -1), 1)          # (B,) below the floor -> floor + d
    pos = torch.logsumexp(src[:, kf:], dim=-1)                     # (B, S - kf): bin kf + j -> kf + j + d (clamped)
    m = kf + dbin
    if m >= hi:
        out = torch.cat([torch.full((B, hi), NEG, device=src.device, dtype=src.dtype), torch.logsumexp(pos, 1)[:, None]], 1)
    else:
        out = torch.cat([torch.full((B, m), NEG, device=src.device, dtype=src.dtype), pos[:, : hi - m],
                         torch.logsumexp(pos[:, hi - m:], 1)[:, None]], 1)
    onehot = torch.arange(S, device=src.device) == min(m, hi)
    return torch.where(onehot, torch.logaddexp(out, low[:, None]), out)


def lattice_risk_block(lp_tok, lp_wait, dis, g, dbin, punct, X_ms, gate: Gate | None, ob: int, beta=1.0, eos=True):
    """prereg change 8 (P5): the policy pi_beta of lattice_risk on the same speech-timeline lattice, with the block bit b
    (b = 1 after a write in the current chunk; every chunk boundary resets b = 0) and the Talker per-block overhead ob bins:
    a block's first token (b = 0) starts its speech at max(s, ob) -> s' = max(s, ob) + d_i and pauses iff started, f = 0,
    s <= -0.3 + ob; a continuing token (b = 1): s' = max(s, 0) + d_i (no pause possible). Three expected costs by the
    expectation semiring (channel 0: A = log P; channels 1..3: log P*M_k):
      q: sum over real tokens of dis[c-1, i] (1 if the prefix model's greedy non-wait token at chunk c != y_i, detached)
      L: sum over real tokens of min(t_c + s', X)      P: number of pauses (real tokens only; the end token pays nothing)
    Returns tensor (E[q], E[L], E[P]) (gradients to lp_wait and the gate only). lp_tok: shape only.
    Checked against brute-force path enumeration (max abs error < 1e-6)."""
    C, n = lp_tok.shape
    dev = lp_tok.device
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
    disd = dis.detach().float()
    done = []
    for c in range(1, C + 1):
        last = c == C
        t_c = X if last else float(c)
        loglat = torch.log(torch.clamp(t_c + s_pos, max=X).clamp_min(1e-6))
        gt = gate.table(c / C, dev) if (gate is not None and not last) else None
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
                lw, l1w = F.logsigmoid(beta * z), F.logsigmoid(-beta * z)
                s_ = cur + lw[None, :, :, None]
            if real and float(disd[c - 1, i]) > 0:
                s_ = torch.where(ch == 1, torch.logaddexp(s_, s_[0:1]), s_)
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


# ---- prereg change 11 (P6): multiplier-conditioned gate ----
LAM_LO, LAM_HI = 0.003, 1.0                     # log-uniform range of lam_L, lam_P during training


def lam_z(lam: float) -> float:
    """log lam -> [-1, 1] over [LAM_LO, LAM_HI]"""
    lo, hi = math.log(LAM_LO), math.log(LAM_HI)
    return (math.log(lam) - 0.5 * (lo + hi)) / (0.5 * (hi - lo))


class Gate2(nn.Module):
    """g_psi(s, f, started, z_L, z_P) = MLP(x) + w . x  (5 -> 32 -> 1 tanh MLP, last layer and skip zero-initialised;
    z = lam_z(lam) of the sample's multipliers, held in self.z; no c/C (non-causal)). Same table() API as Gate."""
    def __init__(self, bias0: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.constant_(self.net[2].bias, bias0)
        self.skip = nn.Linear(5, 1, bias=False)
        nn.init.zeros_(self.skip.weight)
        self.z = (0.0, 0.0)

    def forward(self, x):
        return self.net(x) + self.skip(x)

    def table(self, cfrac: float, device) -> torch.Tensor:
        """-> (S, 2 f, 2 started) gate values at the current self.z (cfrac ignored)"""
        s = s_of_bins(device)
        feats = []
        for f in (0, 1):
            for st in (0, 1):
                feats.append(torch.stack([s / 8.0, torch.full_like(s, float(f)), torch.full_like(s, float(st)),
                                          torch.full_like(s, float(self.z[0])), torch.full_like(s, float(self.z[1]))], -1))
        g = self(torch.stack(feats, 0))[..., 0]
        return g.view(2, 2, S).permute(2, 0, 1)
