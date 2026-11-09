"""TurnCRF (reports/mist_w3po_v1/crf_novel_ideas.zh-CN.md §3; CPU stage §3.4 / §11): an overlap-capable semi-CRF
speaker-turn layer over the frozen diarizer's per-frame posteriors P in [0,1]^{T x K} (80 ms frames, K = 8 slots), and
its linear-chain ablation. Non-end-to-end: the potentials see P only.

Label set C = {empty} U {k} U {{k,k'}} (|C| = 37 for K = 8, at most 2 active slots). A segmentation z is a sequence of
MAXIMAL segments [i, j) with labels c_1..c_M, consecutive labels different, lengths <= L. Score

    S(z) = start[type c_1] + sum_m s(i_m, j_m, c_m) + sum_{m>=2} T_{i_m}(c_{m-1}, c_m) + end[type c_M]
    s(i, j, c) = sum_{t=i}^{j-1} u_t^c + d^{type c}_{j-i}
    u_t^c = a_type * log(prod_{k in c} P_tk prod_{k not in c} (1 - P_tk)) + b_type       (P clipped to [1e-4, 1-1e-4])
    T_i(c', c) = A[relation(c', c)] + w_enter * sum_{k in c \\ c'} dP_ik + w_leave * sum_{k in c' \\ c} (-dP_ik),
                 dP_i = P_i - P_{i-1} (causal; -inf for c' == c: segments are maximal)

The content term is SUMMED over the segment (not averaged as written in §3.2): with a = 1, b = 0, flat durations and
zero transitions the semi-CRF is then exactly the "independent P" model renormalised over |c| <= 2, which is the
residual starting point §3.2 asks for. Parameters are shared across slots by type / relation (slot indices are arbitrary).

Streaming (§3.2): the prefix distribution at tau is right-censored: the last, still-open segment uses the survival
potential dbar_q = logsumexp_{q' >= q} d_{q'} (q' <= L) and has no end score.

Linear-chain ablation: frame states c in C, same content / relation / boundary / start / end potentials, plus a
self-transition A_self[type] (geometric durations); the streaming prefix simply drops the end score.

Algorithms (all exact): batched log-partition (autograd; closed and censored, optionally constrained to reference frame
labels y, i.e. log Z_y = sum over the segmentations consistent with y), explicit forward-backward frame marginals
mu_t^c and slot marginals pi_t^k = sum_{c ni k} mu_t^c, Viterbi. Tests: test_turn_crf.py (brute-force enumeration).
"""
from __future__ import annotations

import itertools
import math

import torch
from torch import nn

FRAME_S = 0.08
EPS_P = 1e-4
NEG = -1.0e4          # finite "minus infinity" for masks (keeps logsumexp gradients finite)
N_TYPES = 3           # empty, single, pair

REL_NAMES = ["empty->single", "empty->pair", "single->empty", "single->other_single", "single->pair_containing",
             "single->pair_not_containing", "pair->empty", "pair->single_contained", "pair->single_not_contained",
             "pair->pair_sharing_one", "pair->pair_disjoint"]
N_REL = len(REL_NAMES)


def relation(c: tuple, c2: tuple) -> int:
    """Relation-type id of the change c -> c2 (-1 when c == c2: not a segment boundary)."""
    if c == c2:
        return -1
    s1, s2 = set(c), set(c2)
    if not c:
        return 0 if len(c2) == 1 else 1
    if len(c) == 1:
        if not c2:
            return 2
        if len(c2) == 1:
            return 3
        return 4 if s1 <= s2 else 5
    if not c2:
        return 6
    if len(c2) == 1:
        return 7 if s2 <= s1 else 8
    return 9 if len(s1 & s2) == 1 else 10


class LabelSpace:
    """C = {()} U {(k,)} U {(k, k')}; M (C x K) membership, types (C,), rel (C x C) relation ids (-1 on the diagonal)."""

    def __init__(self, K: int = 8, max_active: int = 2):
        labels = [()] + [(k,) for k in range(K)]
        if max_active >= 2:
            labels += [tuple(p) for p in itertools.combinations(range(K), 2)]
        self.K, self.labels, self.C = K, labels, len(labels)
        self.index = {c: i for i, c in enumerate(labels)}
        self.M = torch.zeros(self.C, K, dtype=torch.float64)
        for i, c in enumerate(labels):
            for k in c:
                self.M[i, k] = 1.0
        self.types = torch.tensor([len(c) for c in labels], dtype=torch.long)
        self.rel = torch.tensor([[relation(c, c2) for c2 in labels] for c in labels], dtype=torch.long)

    def label_of(self, slots) -> int:
        return self.index[tuple(sorted(int(k) for k in slots))]

    def slots_from_labels(self, y: torch.Tensor) -> torch.Tensor:
        """(..., ) label ids -> (..., K) 0/1 slot activity."""
        return self.M[y]


# ----------------------------------------------------------------------------------------------------------------
# potentials
# ----------------------------------------------------------------------------------------------------------------

def content_ell(P: torch.Tensor, space: LabelSpace, eps: float = EPS_P) -> torch.Tensor:
    """ell_t^c = log(prod_{k in c} P prod_{k not in c} (1 - P)), P (..., T, K) -> (..., T, C)."""
    P = P.clamp(eps, 1.0 - eps)
    lp, lq = torch.log(P), torch.log1p(-P)
    return lq.sum(-1, keepdim=True) + (lp - lq) @ space.M.to(P.dtype).T


def boundary_feats(P: torch.Tensor, space: LabelSpace) -> tuple[torch.Tensor, torch.Tensor]:
    """enter[..., t, c', c] = sum_{k in c \\ c'} dP_tk, leave[..., t, c', c] = sum_{k in c' \\ c} (-dP_tk); dP_0 = 0."""
    dP = torch.cat([torch.zeros_like(P[..., :1, :]), P[..., 1:, :] - P[..., :-1, :]], -2)
    M = space.M.to(P.dtype)
    s = dP @ M.T                                                   # (..., T, C)
    inter = torch.einsum("pk,nk,...tk->...tpn", M, M, dP)          # sum over k in c' & c
    enter = s.unsqueeze(-2) - inter
    leave = -(s.unsqueeze(-1) - inter)
    return enter, leave


def duration_table(d_tab: torch.Tensor, d_tail: torch.Tensor, L: int) -> torch.Tensor:
    """d (N_TYPES, L) for q = 1..L: the free table for q <= Q, a linear (geometric) tail beyond Q. d_tail is the total
    change of the tail over q = Q..L (slope d_tail / (L - Q) per frame), so one optimiser step on it moves the
    long-duration potentials by as much as one step on a table entry."""
    Q = d_tab.shape[1]
    if L <= Q:
        return d_tab[:, :L]
    extra = torch.arange(1, L - Q + 1, dtype=d_tab.dtype) / (L - Q)
    return torch.cat([d_tab, d_tab[:, -1:] + d_tail[:, None] * extra], 1)


def survival(d: torch.Tensor) -> torch.Tensor:
    """dbar_q = logsumexp_{q' >= q} d_{q'} along the last axis."""
    return torch.logcumsumexp(d.flip(-1), -1).flip(-1)


# ----------------------------------------------------------------------------------------------------------------
# semi-CRF: batched log-partition with autograd (training / tests)
# ----------------------------------------------------------------------------------------------------------------

def _mismatch_cum(y: torch.Tensor, C: int, dtype) -> torch.Tensor:
    """(B, T) label ids (-1 = unconstrained frame) -> (B, T+1, C) cumulative count of frames whose label != c."""
    mis = (y.unsqueeze(-1) != torch.arange(C)).to(dtype) * (y.unsqueeze(-1) >= 0).to(dtype)
    return torch.cat([mis.new_zeros(mis.shape[0], 1, C), mis.cumsum(1)], 1)


def semi_logZ(u, tr, start_c, end_c, d_c, ds_c, lengths, tau=None, y=None):
    """Batched exact semi-CRF log-partition.

    u (B, T, C) content; tr (B, T, C, C) with tr[:, j, c', c] the boundary potential between frame j-1 and j (index 0
    unused); start_c / end_c (C,); d_c / ds_c (Lmax, C) duration / survival potentials for q = 1..Lmax; lengths (B,)
    true lengths (rows are right-padded); tau (B,) optional censoring points 1 <= tau <= length; y (B, T) optional
    reference labels -> constrained partitions Z_y (sum over segmentations whose frame labels equal y).
    Returns (logZ closed at lengths (B,), logZ censored at tau (B,) or None).
    """
    B, T, C = u.shape
    L = d_c.shape[0]
    # G_i = beta_i - U_i (beta_i: log-sum of the prefixes whose next segment starts at i with label c), so that
    # alpha_j = U_j + logsumexp_i (G_i + d_{j-i}); tensors are unbound once so that autograd never back-propagates
    # through per-step slices of a (B, T, ...) tensor (a slice's backward materialises the whole tensor).
    U = torch.cat([u.new_zeros(B, 1, C), u.cumsum(1)], 1)
    Us, trs = U.unbind(1), tr.unbind(1)
    Mis = _mismatch_cum(y, C, u.dtype) if y is not None else None
    G = [start_c.expand(B, C)]
    closed = []
    cens_rows, cens_vals = [], []
    d_rev, ds_rev = d_c.flip(0), ds_c.flip(0)
    for j in range(1, T + 1):
        qm = min(L, j)
        x = torch.stack(G[j - qm:], 1)                                          # (B, qm, C): start j-qm..j-1
        if Mis is not None:
            x = x + NEG * ((Mis[:, j:j + 1] - Mis[:, j - qm:j]) > 0.5).to(u.dtype)
        a_j = torch.logsumexp(x + d_rev[L - qm:], 1) + Us[j]                   # q = qm..1
        closed.append(torch.logsumexp(a_j + end_c, -1))
        if tau is not None:
            rows = (tau == j).nonzero().flatten()
            if len(rows):
                v = torch.logsumexp(x[rows] + ds_rev[L - qm:], 1) + Us[j][rows]
                cens_rows.append(rows)
                cens_vals.append(torch.logsumexp(v, -1))
        if j < T:
            G.append(torch.logsumexp(a_j.unsqueeze(-1) + trs[j], 1) - Us[j])
    logZ = torch.stack(closed, 1).gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
    logZc = None
    if tau is not None:
        rows = torch.cat(cens_rows)
        logZc = u.new_zeros(B).index_put((rows,), torch.cat(cens_vals))
    return logZ, logZc


# ----------------------------------------------------------------------------------------------------------------
# semi-CRF: explicit forward-backward and Viterbi for one sequence (no autograd)
# ----------------------------------------------------------------------------------------------------------------

def _seg_mask(y, C, dtype):
    if y is None:
        return None
    return _mismatch_cum(y.unsqueeze(0), C, dtype)[0]


@torch.no_grad()
def semi_forward_backward(u, tr, start_c, end_c, d_c, ds_c, censored=False, y=None):
    """One sequence: u (T, C), tr (T, C, C). Returns (logZ, mu (T, C) frame label marginals). censored: the last
    segment is open (survival potential, no end score)."""
    T, C = u.shape
    L = d_c.shape[0]
    d_last = ds_c if censored else d_c
    e_last = torch.zeros_like(end_c) if censored else end_c
    U = torch.cat([u.new_zeros(1, C), u.cumsum(0)], 0)
    Mis = _seg_mask(y, C, u.dtype)
    beta = u.new_full((T, C), -math.inf)
    alpha = u.new_full((T + 1, C), -math.inf)
    beta[0] = start_c
    for j in range(1, T + 1):
        qm = min(L, j)
        x = beta[j - qm:j] + (U[j] - U[j - qm:j])
        if Mis is not None:
            x = x + NEG * ((Mis[j] - Mis[j - qm:j]) > 0.5).to(u.dtype)
        dd = (d_last if j == T else d_c)[:qm].flip(0)
        alpha[j] = torch.logsumexp(x + dd, 0)
        if j < T:
            beta[j] = torch.logsumexp(alpha[j].unsqueeze(-1) + tr[j], 0)
    logZ = torch.logsumexp(alpha[T] + e_last, 0)
    bout = u.new_full((T + 1, C), -math.inf)
    bin_ = u.new_full((T, C), -math.inf)
    bout[T] = e_last
    for i in range(T - 1, -1, -1):
        qm = min(L, T - i)
        dur = d_c[:qm].clone()
        if i + qm == T:
            dur[qm - 1] = d_last[qm - 1]
        x = (U[i + 1:i + qm + 1] - U[i]) + dur + bout[i + 1:i + qm + 1]
        if Mis is not None:
            x = x + NEG * ((Mis[i + 1:i + qm + 1] - Mis[i]) > 0.5).to(u.dtype)
        bin_[i] = torch.logsumexp(x, 0)
        if i >= 1:
            bout[i] = torch.logsumexp(tr[i] + bin_[i].unsqueeze(0), 1)
    S = torch.exp(beta + bin_ - logZ)                  # segments starting at i
    E = torch.exp(alpha[1:] + bout[1:] - logZ)         # E[j-1]: segments ending at j
    mu = S.cumsum(0) - torch.cat([S.new_zeros(1, C), E.cumsum(0)[:-1]], 0)
    return logZ, mu.clamp(0.0, 1.0)


@torch.no_grad()
def semi_viterbi(u, tr, start_c, end_c, d_c, ds_c, censored=False):
    """One sequence -> (best frame labels (T,), best score)."""
    T, C = u.shape
    L = d_c.shape[0]
    d_last = ds_c if censored else d_c
    e_last = torch.zeros_like(end_c) if censored else end_c
    U = torch.cat([u.new_zeros(1, C), u.cumsum(0)], 0)
    beta = u.new_full((T, C), -math.inf)
    alpha = u.new_full((T + 1, C), -math.inf)
    bq = torch.zeros(T + 1, C, dtype=torch.long)
    bc = torch.zeros(T, C, dtype=torch.long)
    beta[0] = start_c
    for j in range(1, T + 1):
        qm = min(L, j)
        x = beta[j - qm:j] + (U[j] - U[j - qm:j]) + (d_last if j == T else d_c)[:qm].flip(0)
        v, arg = x.max(0)
        alpha[j], bq[j] = v, qm - arg                                   # row r <-> q = qm - r
        if j < T:
            v2, arg2 = (alpha[j].unsqueeze(-1) + tr[j]).max(0)
            beta[j], bc[j] = v2, arg2
    v, c = (alpha[T] + e_last).max(0)
    c = int(c)
    out = torch.empty(T, dtype=torch.long)
    j = T
    while j > 0:
        q = int(bq[j, c])
        out[j - q:j] = c
        i = j - q
        if i > 0:
            c = int(bc[i, c])
        j = i
    return out, float(v)


# ----------------------------------------------------------------------------------------------------------------
# linear-chain CRF
# ----------------------------------------------------------------------------------------------------------------

def chain_logZ(u, tr, start_c, end_c, lengths, tau=None, y=None):
    """Batched linear-chain log-partition; tr (B, T, C, C) includes the self-transition on the diagonal."""
    B, T, C = u.shape
    if y is not None:
        u = u + NEG * ((y.unsqueeze(-1) != torch.arange(C)) & (y.unsqueeze(-1) >= 0)).to(u.dtype)
    us, trs = u.unbind(1), tr.unbind(1)
    a = start_c + us[0]
    closed = [torch.logsumexp(a + end_c, -1)]
    cens_rows, cens_vals = [], []
    for t in range(1, T + 1):
        if tau is not None:
            rows = (tau == t).nonzero().flatten()
            if len(rows):
                cens_rows.append(rows)
                cens_vals.append(torch.logsumexp(a[rows], -1))
        if t == T:
            break
        a = torch.logsumexp(a.unsqueeze(-1) + trs[t], 1) + us[t]
        closed.append(torch.logsumexp(a + end_c, -1))
    logZ = torch.stack(closed, 1).gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
    logZc = None
    if tau is not None:
        rows = torch.cat(cens_rows)
        logZc = u.new_zeros(B).index_put((rows,), torch.cat(cens_vals))
    return logZ, logZc


@torch.no_grad()
def chain_forward_backward(u, tr, start_c, end_c, censored=False, y=None):
    T, C = u.shape
    if y is not None:
        u = u + NEG * ((y.unsqueeze(-1) != torch.arange(C)) & (y.unsqueeze(-1) >= 0)).to(u.dtype)
    e_last = torch.zeros_like(end_c) if censored else end_c
    al = u.new_empty(T, C)
    be = u.new_empty(T, C)
    al[0] = start_c + u[0]
    for t in range(1, T):
        al[t] = torch.logsumexp(al[t - 1].unsqueeze(-1) + tr[t], 0) + u[t]
    logZ = torch.logsumexp(al[T - 1] + e_last, 0)
    be[T - 1] = e_last
    for t in range(T - 2, -1, -1):
        be[t] = torch.logsumexp(tr[t + 1] + (u[t + 1] + be[t + 1]).unsqueeze(0), 1)
    return logZ, torch.exp(al + be - logZ).clamp(0.0, 1.0)


@torch.no_grad()
def chain_viterbi(u, tr, start_c, end_c, censored=False):
    T, C = u.shape
    e_last = torch.zeros_like(end_c) if censored else end_c
    a = start_c + u[0]
    bp = torch.zeros(T, C, dtype=torch.long)
    for t in range(1, T):
        v, arg = (a.unsqueeze(-1) + tr[t]).max(0)
        a, bp[t] = v + u[t], arg
    v, c = (a + e_last).max(0)
    out = torch.empty(T, dtype=torch.long)
    c = int(c)
    for t in range(T - 1, -1, -1):
        out[t] = c
        c = int(bp[t, c])
    return out, float(v)


# ----------------------------------------------------------------------------------------------------------------
# reference path scores (with maximal segments a frame-label sequence has exactly one segmentation, so log Z_y is the
# score of that path; equal to the constrained DP, see test_turn_crf.py, at O(T) cost)
# ----------------------------------------------------------------------------------------------------------------

def _path_index(y, lengths):
    B, T = y.shape
    ar = torch.arange(T).unsqueeze(0).expand(B, T)
    valid = ar < lengths.unsqueeze(1)
    change = torch.zeros(B, T, dtype=torch.bool)
    change[:, 1:] = (y[:, 1:] != y[:, :-1]) & valid[:, 1:]
    newrun = change.clone()
    newrun[:, 0] = True
    runstart = torch.cummax(torch.where(newrun, ar, torch.zeros_like(ar)), 1).values
    nxt = torch.zeros_like(change)
    nxt[:, :-1] = change[:, 1:]
    return ar, valid, change, runstart, nxt


def _gather_terms(u, tr, y):
    B, T, C = u.shape
    yc = y.clamp(min=0)
    cont = u.gather(2, yc.unsqueeze(-1)).squeeze(-1)
    yprev = torch.cat([yc[:, :1], yc[:, :-1]], 1)
    trv = tr.reshape(B, T, C * C).gather(2, (yprev * C + yc).unsqueeze(-1)).squeeze(-1)
    return yc, cont, trv


def semi_path_score(u, tr, start_c, end_c, d_c, ds_c, y, lengths, tau=None):
    """Batched score of the reference segmentation (closed at lengths; right-censored prefix at tau)."""
    L = d_c.shape[0]
    yc, cont, trv = _gather_terms(u, tr, y)
    ar, valid, change, runstart, nxt = _path_index(y, lengths)
    q = ar - runstart + 1
    too_long = NEG * ((q > L) & valid).any(1).to(u.dtype)
    dsel = d_c[(q - 1).clamp(max=L - 1), yc]
    endflag = valid & (nxt | (ar == (lengths - 1).unsqueeze(1)))
    y_last = yc.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
    z = u.new_zeros(())
    closed = (start_c[yc[:, 0]] + torch.where(valid, cont, z).sum(1) + torch.where(change, trv, z).sum(1)
              + torch.where(endflag, dsel, z).sum(1) + end_c[y_last] + too_long)
    if tau is None:
        return closed, None
    pv = ar < tau.unsqueeze(1)
    last = (tau - 1).unsqueeze(1)
    ql = (last - runstart.gather(1, last) + 1).squeeze(1)
    yl = yc.gather(1, last).squeeze(1)
    too_long_c = NEG * (((q > L) & pv).any(1)).to(u.dtype)
    cens = (start_c[yc[:, 0]] + torch.where(pv, cont, z).sum(1) + torch.where(change & pv, trv, z).sum(1)
            + torch.where(nxt & (ar < last), dsel, z).sum(1) + ds_c[(ql - 1).clamp(max=L - 1), yl] + too_long_c)
    return closed, cens


def chain_path_score(u, tr, start_c, end_c, y, lengths, tau=None):
    yc, cont, trv = _gather_terms(u, tr, y)
    ar, valid, _, _, _ = _path_index(y, lengths)
    step = valid & (ar >= 1)
    y_last = yc.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
    z = u.new_zeros(())
    closed = start_c[yc[:, 0]] + torch.where(valid, cont, z).sum(1) + torch.where(step, trv, z).sum(1) + end_c[y_last]
    if tau is None:
        return closed, None
    pv = ar < tau.unsqueeze(1)
    cens = start_c[yc[:, 0]] + torch.where(pv, cont, z).sum(1) + torch.where(step & pv, trv, z).sum(1)
    return closed, cens


# ----------------------------------------------------------------------------------------------------------------
# parameterised models
# ----------------------------------------------------------------------------------------------------------------

class _TurnBase(nn.Module):
    """Shared content / relation / boundary parameters (all shared across slots by type)."""

    def __init__(self, K: int = 8, max_active: int = 2):
        super().__init__()
        self.space = LabelSpace(K, max_active)
        dt = torch.float64
        self.a = nn.Parameter(torch.ones(N_TYPES, dtype=dt))
        self.b = nn.Parameter(torch.zeros(N_TYPES, dtype=dt))
        self.A = nn.Parameter(torch.zeros(N_REL, dtype=dt))
        self.start = nn.Parameter(torch.zeros(N_TYPES, dtype=dt))
        self.end = nn.Parameter(torch.zeros(N_TYPES, dtype=dt))
        self.w = nn.Parameter(torch.zeros(2, dtype=dt))           # w_enter, w_leave

    def _content(self, P):
        ty = self.space.types
        return self.a[ty] * content_ell(P, self.space) + self.b[ty]

    def _trans(self, P, diag):
        """(..., T, C, C) boundary potentials; diag (C,) on c' == c."""
        enter, leave = boundary_feats(P, self.space)
        rel = self.space.rel
        base = torch.where(rel >= 0, self.A[rel.clamp(min=0)], diag.unsqueeze(0).expand_as(rel))
        return base + self.w[0] * enter + self.w[1] * leave

    def slot_marginals(self, mu: torch.Tensor) -> torch.Tensor:
        return mu @ self.space.M.to(mu.dtype)


class SemiTurnCRF(_TurnBase):
    kind = "semi"

    def __init__(self, K: int = 8, max_active: int = 2, L: int = 257, Q: int = 50):
        super().__init__(K, max_active)
        self.L, self.Q = L, Q
        self.d_tab = nn.Parameter(torch.zeros(N_TYPES, Q, dtype=torch.float64))
        self.d_tail = nn.Parameter(torch.zeros(N_TYPES, dtype=torch.float64))

    def potentials(self, P: torch.Tensor) -> dict:
        P = P.to(torch.float64)
        ty = self.space.types
        d = duration_table(self.d_tab, self.d_tail, self.L)            # (3, L)
        ds = survival(d)
        diag = torch.full((self.space.C,), NEG, dtype=torch.float64)
        return {"u": self._content(P), "tr": self._trans(P, diag), "start_c": self.start[ty], "end_c": self.end[ty],
                "d_c": d[ty].T.contiguous(), "ds_c": ds[ty].T.contiguous()}

    def log_partitions(self, P, lengths, tau=None, y=None):
        p = self.potentials(P)
        return semi_logZ(p["u"], p["tr"], p["start_c"], p["end_c"], p["d_c"], p["ds_c"], lengths, tau, y)

    def nll_terms(self, P, lengths, y, tau):
        p = self.potentials(P)
        lz, lzc = semi_logZ(p["u"], p["tr"], p["start_c"], p["end_c"], p["d_c"], p["ds_c"], lengths, tau)
        sy, syc = semi_path_score(p["u"], p["tr"], p["start_c"], p["end_c"], p["d_c"], p["ds_c"], y, lengths, tau)
        return lz, lzc, sy, syc

    @torch.no_grad()
    def marginals(self, P, censored=False):
        p = self.potentials(P)
        return semi_forward_backward(p["u"], p["tr"], p["start_c"], p["end_c"], p["d_c"], p["ds_c"], censored)

    @torch.no_grad()
    def viterbi(self, P, censored=False):
        p = self.potentials(P)
        return semi_viterbi(p["u"], p["tr"], p["start_c"], p["end_c"], p["d_c"], p["ds_c"], censored)


class ChainTurnCRF(_TurnBase):
    kind = "chain"

    def __init__(self, K: int = 8, max_active: int = 2):
        super().__init__(K, max_active)
        self.A_self = nn.Parameter(torch.zeros(N_TYPES, dtype=torch.float64))

    def potentials(self, P: torch.Tensor) -> dict:
        P = P.to(torch.float64)
        ty = self.space.types
        return {"u": self._content(P), "tr": self._trans(P, self.A_self[ty]), "start_c": self.start[ty],
                "end_c": self.end[ty]}

    def log_partitions(self, P, lengths, tau=None, y=None):
        p = self.potentials(P)
        return chain_logZ(p["u"], p["tr"], p["start_c"], p["end_c"], lengths, tau, y)

    def nll_terms(self, P, lengths, y, tau):
        p = self.potentials(P)
        lz, lzc = chain_logZ(p["u"], p["tr"], p["start_c"], p["end_c"], lengths, tau)
        sy, syc = chain_path_score(p["u"], p["tr"], p["start_c"], p["end_c"], y, lengths, tau)
        return lz, lzc, sy, syc

    @torch.no_grad()
    def marginals(self, P, censored=False):
        p = self.potentials(P)
        return chain_forward_backward(p["u"], p["tr"], p["start_c"], p["end_c"], censored)

    @torch.no_grad()
    def viterbi(self, P, censored=False):
        p = self.potentials(P)
        return chain_viterbi(p["u"], p["tr"], p["start_c"], p["end_c"], censored)


def window_nll(model, P, lengths, y, tau, constrained: str = "path"):
    """Per-window training loss (B,): (log Z - log Z_y)/T + (log Z~_tau - log Z~_tau,y)/tau (closed full window plus
    one right-censored prefix; §3.2). Rows are right-padded (y = -1 on padding). constrained = "path" scores the unique
    reference segmentation directly, "dp" runs the y-constrained DP (identical value; tests)."""
    if constrained == "path":
        logZ, logZc, logZy, logZyc = model.nll_terms(P, lengths, y, tau)
    else:
        B = P.shape[0]
        lz, lzc = model.log_partitions(torch.cat([P, P], 0), torch.cat([lengths, lengths], 0),
                                       torch.cat([tau, tau], 0), torch.cat([torch.full_like(y, -1), y], 0))
        logZ, logZc, logZy, logZyc = lz[:B], lzc[:B], lz[B:], lzc[B:]
    full = (logZ - logZy) / lengths.to(logZ.dtype)
    pref = (logZc - logZyc) / tau.to(logZ.dtype)
    return full + pref, full, pref


def build_model(kind: str, **kw):
    return SemiTurnCRF(**kw) if kind == "semi" else ChainTurnCRF(**kw)
