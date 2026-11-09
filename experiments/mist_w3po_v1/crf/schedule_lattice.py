"""Scheme 1' (reports/mist_w3po_v1/crf_novel_ideas.zh-CN.md §10.2-§10.3): the latency-constrained latent-schedule
lattice of one window's fixed target stream S, as a semi-Markov CRF over the committed-token count after each read.

A window has R reads and S has N tokens. A schedule is beta_0 = 0 <= beta_1 <= ... <= beta_R = N (empty blocks
allowed); read r (0-based here: reads 0..R-1, state s_r = beta_{r+1}) commits the block S[s_{r-1}:s_r) and then its
end token e_r (<wait>, or <eos> at the last read), with the score

    psi_r(b, b') = sum_{j=b}^{b'-1} tok_lp[r, j] + end_lp[r, b'] - (lam / N) * sum_{j=b}^{b'-1} lag[r, j]

  tok_lp[r, j]  log p(S_j | x<=t_r, c, S^{<j})
  end_lp[r, b]  log p(e_r | x<=t_r, c, S^{<b}),  b = 0..N
  lag[r, j]     t_r - tau_j (s), so sum_r psi_r = log p(S, z | x) - lam * Lat(z), Lat(z) = (1/N) sum_j lag[r(j), j]

Admissibility: token j can only be committed at a read r with lo[j] <= r <= hi[j] (both inclusive; a token whose
tok_lp is -inf is inadmissible there too), and a block holds at most W tokens. Every function works on a padded batch
(Lattice) in float64 on any device; Lattice.single wraps one window. With prefix sums every read costs O(N * W), so a
window costs O(R * N * W).

  marginals()     log Z, w_tok[r, j] = P(s_{r-1} <= j < s_r), w_end[r, b] = P(s_r = b), E[Lat] (forward-backward)
  viterbi()       the MAP schedule
  sample()        exact forward-filtering backward-sampling
  solve_lambda()  global lam* by bisection so that the (weighted) mean E[Lat] over a set of windows hits a target

Infeasible windows (no schedule satisfies lo/hi/W) get log Z ~ NEG (finite, so gradients stay finite); check
`feasible`. -inf in end_lp is allowed (clamped to NEG); tok_lp = -inf marks the (r, j) inadmissible.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

NEG = -1e30          # "minus infinity" that keeps sums and gradients finite (float64)
DTYPE = torch.float64


@dataclass
class Lattice:
    """A padded batch of windows. Padding: tokens j >= n_tok and reads r >= n_read are inadmissible; padded reads
    commit nothing and score 0, so a window's log Z is read at its own final state n_tok after the last padded read."""
    tok_lp: torch.Tensor    # [B, R, N]
    end_lp: torch.Tensor    # [B, R, N + 1]
    lo: torch.Tensor        # [B, N] long, inclusive
    hi: torch.Tensor        # [B, N] long, inclusive
    lag: torch.Tensor       # [B, R, N] seconds
    n_tok: torch.Tensor     # [B] long
    n_read: torch.Tensor    # [B] long
    W: int                  # block-length cap (tokens)

    @property
    def B(self) -> int:
        return self.tok_lp.shape[0]

    @property
    def R(self) -> int:
        return self.tok_lp.shape[1]

    @property
    def N(self) -> int:
        return self.tok_lp.shape[2]

    @property
    def device(self):
        return self.tok_lp.device

    def to(self, device) -> "Lattice":
        return replace(self, **{k: getattr(self, k).to(device) for k in
                                ("tok_lp", "end_lp", "lo", "hi", "lag", "n_tok", "n_read")})

    @staticmethod
    def single(tok_lp, end_lp, lo, hi, lag, W: int | None = None, device=None) -> "Lattice":
        return pack([{"tok_lp": tok_lp, "end_lp": end_lp, "lo": lo, "hi": hi, "lag": lag}], W=W, device=device)


def _t(x, dtype, device):
    return torch.as_tensor(x, dtype=dtype, device=device)


def pack(windows: list[dict], W: int | None = None, device=None) -> Lattice:
    """windows: dicts with tok_lp [R, N], end_lp [R, N+1], lo [N], hi [N], lag [R, N] (arrays or tensors).
    W None = no cap (max N)."""
    Rs = [int(_t(w["end_lp"], DTYPE, "cpu").shape[0]) for w in windows]
    Ns = [int(_t(w["end_lp"], DTYPE, "cpu").shape[1]) - 1 for w in windows]
    B, R, N = len(windows), max(Rs), max(max(Ns), 1)
    dev = device if device is not None else (windows[0]["tok_lp"].device if torch.is_tensor(windows[0]["tok_lp"])
                                             else "cpu")
    tok = torch.zeros((B, R, N), dtype=DTYPE, device=dev)
    end = torch.zeros((B, R, N + 1), dtype=DTYPE, device=dev)
    lag = torch.zeros((B, R, N), dtype=DTYPE, device=dev)
    lo = torch.full((B, N), R, dtype=torch.long, device=dev)
    hi = torch.full((B, N), -1, dtype=torch.long, device=dev)
    for i, w in enumerate(windows):
        r, n = Rs[i], Ns[i]
        if n:
            tok[i, :r, :n] = _t(w["tok_lp"], DTYPE, dev).reshape(r, n)
            lag[i, :r, :n] = _t(w["lag"], DTYPE, dev).reshape(r, n)
            lo[i, :n] = _t(w["lo"], torch.long, dev)
            hi[i, :n] = _t(w["hi"], torch.long, dev)
        end[i, :r, : n + 1] = _t(w["end_lp"], DTYPE, dev)
    return Lattice(tok, end, lo, hi, lag, torch.tensor(Ns, device=dev), torch.tensor(Rs, device=dev),
                   int(W) if W is not None else N)


def _lam(lat: Lattice, lam) -> torch.Tensor:
    lam = torch.as_tensor(lam, dtype=DTYPE, device=lat.device)
    return lam.expand(lat.B) if lam.dim() == 0 else lam


def _prep(lat: Lattice, lam):
    """Per read: admissibility, prefix sums P[b, r, k] of the token scores, the longest block m[b, r, k] that may end
    at state k, and the clamped end scores."""
    B, R, N = lat.B, lat.R, lat.N
    dev = lat.device
    r_idx = torch.arange(R, device=dev)[None, :, None]
    j_idx = torch.arange(N, device=dev)[None, None, :]
    adm = ((lat.lo[:, None, :] <= r_idx) & (r_idx <= lat.hi[:, None, :])
           & (j_idx < lat.n_tok[:, None, None]) & (r_idx < lat.n_read[:, None, None]) & torch.isfinite(lat.tok_lp))
    lam_n = _lam(lat, lam) / lat.n_tok.clamp(min=1).to(DTYPE)
    lag = torch.where(adm, lat.lag, torch.zeros((), dtype=DTYPE, device=dev))
    s = torch.where(adm, lat.tok_lp, torch.zeros((), dtype=DTYPE, device=dev)) - lam_n[:, None, None] * lag
    P = F.pad(torch.cumsum(s, -1), (1, 0))
    last_bad = torch.where(~adm, j_idx.expand(B, R, N), torch.full((), -1, device=dev, dtype=torch.long)).cummax(-1).values
    run = F.pad(j_idx - last_bad, (1, 0))                                   # consecutive admissible tokens ending at k-1
    m = run.clamp(max=lat.W)
    k_idx = torch.arange(N + 1, device=dev)[None, None, :]
    end = lat.end_lp.clamp(min=NEG)
    pad_read = r_idx >= lat.n_read[:, None, None]
    end = torch.where(pad_read, torch.zeros((), dtype=DTYPE, device=dev), end)
    end = torch.where(k_idx > lat.n_tok[:, None, None], torch.full((), NEG, dtype=DTYPE, device=dev), end)
    return adm, P, m, end, lag


def _win_back(u: torch.Tensor, W: int) -> torch.Tensor:
    """u [B, K] -> [B, K, W+1] with out[..., k, w] = u[k - w] (NEG before 0)."""
    return F.pad(u, (W, 0), value=NEG).unfold(-1, W + 1, 1).flip(-1)


def _win_fwd(u: torch.Tensor, W: int, fill) -> torch.Tensor:
    """u [B, K] -> [B, K, W+1] with out[..., k, w] = u[k + w] (fill past the end)."""
    return F.pad(u, (0, W), value=fill).unfold(-1, W + 1, 1)


def _start(lat: Lattice) -> torch.Tensor:
    a = torch.full((lat.B, lat.N + 1), NEG, dtype=DTYPE, device=lat.device)
    a[:, 0] = 0.0
    return a


def forward(lat: Lattice, lam=0.0, prep=None):
    """alpha [B, R, N+1] (log-sum of the scores of the first r+1 reads ending in state k) and log Z [B].
    Differentiable in tok_lp, end_lp, lag and lam."""
    adm, P, m, end, _ = prep if prep is not None else _prep(lat, lam)
    W = lat.W
    w_rng = torch.arange(W + 1, device=lat.device)
    prev = _start(lat)
    alphas = []
    for r in range(lat.R):
        win = _win_back(prev - P[:, r], W)
        win = torch.where(w_rng[None, None, :] <= m[:, r, :, None], win, torch.full((), NEG, dtype=DTYPE,
                                                                                    device=lat.device))
        prev = P[:, r] + end[:, r] + torch.logsumexp(win, -1)
        alphas.append(prev)
    alpha = torch.stack(alphas, 1)
    logZ = alpha[:, -1].gather(-1, lat.n_tok[:, None])[:, 0]
    return alpha, logZ


def backward(lat: Lattice, lam=0.0, prep=None):
    """gamma [B, R, N+1]: log-sum of the scores of reads r+1.. given state k after read r; and log Z from it."""
    adm, P, m, end, _ = prep if prep is not None else _prep(lat, lam)
    W, R = lat.W, lat.R
    w_rng = torch.arange(W + 1, device=lat.device)
    g = torch.full((lat.B, lat.N + 1), NEG, dtype=DTYPE, device=lat.device)
    g.scatter_(-1, lat.n_tok[:, None], 0.0)
    gammas = [g]
    for r in range(R - 1, -1, -1):
        v = P[:, r] + end[:, r] + g
        win = _win_fwd(v, W, NEG)
        mw = _win_fwd(m[:, r], W, -1)
        win = torch.where(w_rng[None, None, :] <= mw, win, torch.full((), NEG, dtype=DTYPE, device=lat.device))
        g = -P[:, r] + torch.logsumexp(win, -1)
        if r > 0:
            gammas.append(g)
    gamma = torch.stack(gammas[::-1], 1)
    return gamma, g[:, 0]


def marginals(lat: Lattice, lam=0.0) -> dict:
    """Exact posterior quantities under q_lam(z) ~ exp(sum_r psi_r):
    logZ [B], w_tok [B, R, N], w_end [B, R, N+1] (zero on padded reads), exp_lat [B] (E[Lat], s),
    feasible [B] bool, logZ_back [B] (the backward pass's log Z, a consistency check).
    Infeasible windows get zero weights and exp_lat NaN."""
    with torch.no_grad():
        prep = _prep(lat, lam)
        adm, P, m, end, lag = prep
        alpha, logZ = forward(lat, lam, prep)
        gamma, logZ_b = backward(lat, lam, prep)
        post = torch.exp(alpha + gamma - logZ[:, None, None])
        real = (torch.arange(lat.R, device=lat.device)[None, :] < lat.n_read[:, None])
        Fc = torch.cumsum(post, -1)
        Fprev = torch.cat([torch.ones_like(Fc[:, :1]), Fc[:, :-1]], 1)
        w_tok = (Fprev - Fc)[..., : lat.N].clamp(min=0.0)
        w_tok = torch.where(adm, w_tok, torch.zeros((), dtype=DTYPE, device=lat.device))
        w_end = torch.where(real[:, :, None], post, torch.zeros((), dtype=DTYPE, device=lat.device))
        feasible = logZ > NEG / 2
        zero = torch.zeros((), dtype=DTYPE, device=lat.device)
        w_tok = torch.where(feasible[:, None, None], w_tok, zero)      # infeasible windows: no weights
        w_end = torch.where(feasible[:, None, None], w_end, zero)
        exp_lat = (w_tok * lag).sum((1, 2)) / lat.n_tok.clamp(min=1).to(DTYPE)
        exp_lat = torch.where(feasible, exp_lat, torch.full((), float("nan"), dtype=DTYPE, device=lat.device))
    return {"logZ": logZ, "logZ_back": logZ_b, "w_tok": w_tok, "w_end": w_end, "exp_lat": exp_lat,
            "feasible": feasible}


def viterbi(lat: Lattice, lam=0.0):
    """MAP schedule: states [B, R] (s_r = committed count after read r; padded reads repeat n_tok) and its score."""
    with torch.no_grad():
        adm, P, m, end, _ = _prep(lat, lam)
        W = lat.W
        w_rng = torch.arange(W + 1, device=lat.device)
        prev = _start(lat)
        bps = []
        for r in range(lat.R):
            win = _win_back(prev - P[:, r], W)
            win = torch.where(w_rng[None, None, :] <= m[:, r, :, None], win,
                              torch.full((), NEG, dtype=DTYPE, device=lat.device))
            best, arg = win.max(-1)
            prev = P[:, r] + end[:, r] + best
            bps.append(arg)
        score = prev.gather(-1, lat.n_tok[:, None])[:, 0]
        s = lat.n_tok.clone()
        states = torch.empty((lat.B, lat.R), dtype=torch.long, device=lat.device)
        for r in range(lat.R - 1, -1, -1):
            states[:, r] = s
            s = s - bps[r].gather(-1, s[:, None])[:, 0]
    return states, score


def sample(lat: Lattice, lam=0.0, n: int = 1, generator: torch.Generator | None = None):
    """Exact FFBS samples from q_lam: states [B, n, R]."""
    with torch.no_grad():
        prep = _prep(lat, lam)
        adm, P, m, end, _ = prep
        alpha, _ = forward(lat, lam, prep)
        W, B, R = lat.W, lat.B, lat.R
        w_rng = torch.arange(W + 1, device=lat.device)
        s = lat.n_tok[:, None].expand(B, n).clone()
        out = torch.empty((B, n, R), dtype=torch.long, device=lat.device)
        start = _start(lat)
        for r in range(R - 1, -1, -1):
            out[:, :, r] = s
            prev = alpha[:, r - 1] if r > 0 else start
            win = _win_back(prev - P[:, r], W)                                          # [B, N+1, W+1]
            win = torch.where(w_rng[None, None, :] <= m[:, r, :, None], win,
                              torch.full((), NEG, dtype=DTYPE, device=lat.device))
            lg = win.gather(1, s[:, :, None].expand(B, n, W + 1))                       # [B, n, W+1]
            u = torch.rand(lg.shape, dtype=DTYPE, device=lat.device, generator=generator).clamp(1e-300, 1 - 1e-16)
            w = (lg - torch.log(-torch.log(u))).argmax(-1)
            s = s - w
    return out


def blocks_ok(lat: Lattice, states: torch.Tensor) -> torch.Tensor:
    """states [B, R] -> bool [B]: the schedule is admissible (monotone, ends at n_tok, every block within m)."""
    adm, P, m, end, _ = _prep(lat, 0.0)
    prev = torch.cat([torch.zeros_like(states[:, :1]), states[:, :-1]], 1)
    blen = states - prev
    ok = (blen >= 0).all(1) & (states[:, -1] == lat.n_tok)
    mk = m.gather(-1, states.clamp(0, lat.N)[:, :, None])[..., 0]
    return ok & (blen <= mk).all(1)


def schedule_score(lat: Lattice, states: torch.Tensor, lam=0.0) -> torch.Tensor:
    """sum_r psi_r of the schedule states [B, R] (no admissibility check: see blocks_ok)."""
    adm, P, m, end, _ = _prep(lat, lam)
    prev = torch.cat([torch.zeros_like(states[:, :1]), states[:, :-1]], 1)
    g = lambda T, idx: T.gather(-1, idx[:, :, None])[..., 0]   # noqa: E731
    return (g(P, states) - g(P, prev) + g(end, states)).sum(1)


def schedule_lat(lat: Lattice, states: torch.Tensor) -> torch.Tensor:
    """Lat(z) [B] of schedules states [B, R]: (1/N) sum of lag[r(j), j]."""
    L = F.pad(torch.cumsum(lat.lag, -1), (1, 0))
    prev = torch.cat([torch.zeros_like(states[:, :1]), states[:, :-1]], 1)
    g = lambda T, idx: T.gather(-1, idx[:, :, None])[..., 0]   # noqa: E731
    real = (torch.arange(lat.R, device=lat.device)[None, :] < lat.n_read[:, None])
    d = torch.where(real, g(L, states) - g(L, prev), torch.zeros((), dtype=DTYPE, device=lat.device))
    return d.sum(1) / lat.n_tok.clamp(min=1).to(DTYPE)


def expected_lat(lats: list[Lattice], lam: float, weights: list[torch.Tensor] | None = None) -> float:
    """(Weighted) mean E[Lat] over a list of batches at one lam; infeasible windows are left out."""
    num, den = 0.0, 0.0
    for i, lat in enumerate(lats):
        mg = marginals(lat, lam)
        e = torch.nan_to_num(mg["exp_lat"], nan=0.0)
        w = torch.ones_like(e) if weights is None else torch.as_tensor(weights[i], dtype=DTYPE, device=e.device)
        w = w * mg["feasible"].to(DTYPE)
        num += float((e * w).sum())
        den += float(w.sum())
    return num / max(den, 1e-300)


def solve_lambda(lats: list[Lattice], target: float, weights: list | None = None, lam_lo: float = -1.0,
                 lam_hi: float = 1.0, tol: float = 1e-7, max_abs: float = 1e5, max_iter: int = 200) -> dict:
    """Global lam* with mean E_{q_lam}[Lat] = target (E[Lat] is non-increasing in lam: d/dlam = -Var[Lat] per window).
    Brackets by doubling, then bisects. Returns {"lam", "exp_lat", "bracketed", "iters"}."""
    f = lambda x: expected_lat(lats, x, weights) - target   # noqa: E731
    lo, hi = min(lam_lo, -1e-3), max(lam_hi, 1e-3)
    flo, fhi = f(lo), f(hi)
    while flo < 0 and abs(lo) < max_abs:
        lo *= 2.0
        flo = f(lo)
    while fhi > 0 and abs(hi) < max_abs:
        hi *= 2.0
        fhi = f(hi)
    if flo < 0 or fhi > 0:
        lam = lo if flo < 0 else hi
        return {"lam": lam, "exp_lat": f(lam) + target, "bracketed": False, "iters": 0}
    it = 0
    while hi - lo > tol * max(1.0, abs(lo) + abs(hi)) and it < max_iter:
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm > 0:
            lo = mid
        else:
            hi = mid
        it += 1
        if abs(fm) < 1e-10:
            lo = hi = mid
            break
    lam = 0.5 * (lo + hi)
    return {"lam": lam, "exp_lat": f(lam) + target, "bracketed": True, "iters": it}
