"""Scheme 2's alignment CRF (reports/mist_w3po_v1/talker_crf_prereg_v1.zh-CN.md §2.3): a monotonic semi-Markov
alignment of a block's M text units (its text tokens) to its T speech-code frames.

Every frame belongs to one unit, the units in order, each unit >= 1 and <= D frames. Segment score of unit m over
frames (t-d, t]:  sum of e[t', m] + omega_m(d) + s_m * d,  omega_m(d) = -(d - mu_m)^2 / (2 sigma_m^2),
mu_m = T * len_m / sum(len), sigma_m = max(2, mu_m / 2). alpha[t, m] = log-sum over alignments of frames [0, t) whose
last segment is unit m. Z_cover = alpha[T, M-1] (every unit covered); Z_all = logsumexp_m alpha[T, m] (the speech
says units 0..m only, any m). P_cover = Z_cover / Z_all. E_q[d_m] under the cover posterior = d log Z_cover / d s_m.
Batched over rows (padded T / M), exact, differentiable (autograd through the recursion).
"""
from __future__ import annotations

import torch

NEG = -1e30


def duration_prior(T: torch.Tensor, lens: torch.Tensor, mask_m: torch.Tensor, D: int) -> tuple[torch.Tensor, torch.Tensor]:
    """omega [B, D, M] for d = 1..D, and mu [B, M]. T [B] frames, lens [B, M] unit lengths (chars, >= 1)."""
    lens = torch.where(mask_m, lens.clamp(min=1).float(), torch.zeros_like(lens, dtype=torch.float))
    mu = T.float()[:, None] * lens / lens.sum(1, keepdim=True).clamp(min=1)
    sig = torch.clamp(mu / 2, min=2.0)
    d = torch.arange(1, D + 1, device=T.device, dtype=torch.float)[None, :, None]
    om = -(d - mu[:, None, :]) ** 2 / (2 * sig[:, None, :] ** 2)
    return om, mu


def forward(e: torch.Tensor, T: torch.Tensor, M: torch.Tensor, lens: torch.Tensor, D: int = 250,
            s: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """e [B, Tmax, Mmax] frame-unit scores; T [B], M [B] true sizes; lens [B, Mmax]; s [B, Mmax] duration tilt (or None).
    Returns (log Z_cover [B], log Z_all [B]); -inf (NEG) where infeasible."""
    B, Tm, Mm = e.shape
    dev = e.device
    ar_m = torch.arange(Mm, device=dev)
    mask_m = ar_m[None, :] < M[:, None]
    ar_t = torch.arange(Tm, device=dev)
    e = torch.where((ar_t[None, :, None] < T[:, None, None]) & mask_m[:, None, :], e, torch.zeros_like(e))
    C = torch.cat([torch.zeros(B, 1, Mm, device=dev, dtype=e.dtype), torch.cumsum(e, dim=1)], dim=1)   # [B, Tm+1, Mm]
    D = max(1, min(D, Tm))
    om, _ = duration_prior(T, lens, mask_m, D)
    if s is not None:
        om = om + s[:, None, :] * torch.arange(1, D + 1, device=dev, dtype=e.dtype)[None, :, None]
    om = torch.where(mask_m[:, None, :], om, torch.full_like(om, NEG))
    init = torch.full((B, Mm), NEG, device=dev, dtype=e.dtype)
    alphas = [init]   # alpha[0]: nothing covered yet (no last unit); the start lives in the shifted column below
    zero_col = torch.zeros(B, 1, device=dev, dtype=e.dtype)
    neg_col = torch.full((B, 1), NEG, device=dev, dtype=e.dtype)
    for t in range(1, Tm + 1):
        dmax = min(t, D)
        A = torch.stack(alphas[t - dmax: t][::-1], dim=1)           # [B, dmax, Mm]: alpha[t-d] for d = 1..dmax
        start = torch.cat([zero_col if t - d == 0 else neg_col for d in range(1, dmax + 1)], dim=1)   # [B, dmax]
        P = torch.cat([start[:, :, None], A[:, :, :-1]], dim=2)      # unit m's segment follows unit m-1 (or the start)
        Ct = C[:, t, :][:, None, :]
        Cprev = C[:, t - dmax: t, :].flip(1)                         # C[t-d] for d = 1..dmax
        seg = Ct - Cprev + om[:, :dmax, :]
        alphas.append(torch.logsumexp(P + seg, dim=1).clamp(min=NEG))
    Aall = torch.stack(alphas, dim=1)                                  # [B, Tm+1, Mm]
    end = Aall[torch.arange(B, device=dev), T]                         # [B, Mm] at each row's own T
    end = torch.where(mask_m, end, torch.full_like(end, NEG))
    z_cover = end[torch.arange(B, device=dev), (M - 1).clamp(min=0)]
    z_all = torch.logsumexp(end, dim=1)
    return z_cover, z_all


def losses(e, T, M, lens, label, pr: bool = True, D: int = 250):
    """-> (alignment loss per row: -log P_cover (label 1) / -log(1 - P_cover) (label 0) / 0 (label < 0 or
    infeasible), duration posterior-regularisation per row (label 1 rows; 0 elsewhere), P_cover per row (detached))."""
    s = torch.zeros(e.shape[0], e.shape[2], device=e.device, dtype=e.dtype, requires_grad=pr)
    zc, za = forward(e, T, M, lens, D, s if pr else None)
    ok = (zc > NEG / 2) & (za > NEG / 2)
    logp = torch.where(ok, zc - za, torch.zeros_like(zc)).clamp(max=0.0)
    p = logp.exp()
    l1 = -logp
    l0 = -torch.log1p(-p.clamp(max=1 - 1e-6))
    al = torch.where(label == 1, l1, torch.where(label == 0, l0, torch.zeros_like(l1)))
    al = torch.where(ok | (label == 0), al, torch.zeros_like(al))
    prl = torch.zeros_like(al)
    pos = ok & (label == 1)
    if pr and pos.any():
        g = torch.autograd.grad(torch.where(pos, zc, torch.zeros_like(zc)).sum(), s, create_graph=True)[0]   # E[d_m]
        mask_m = torch.arange(e.shape[2], device=e.device)[None, :] < M[:, None]
        _, mu = duration_prior(T, lens, mask_m, 1)
        r = torch.where(mask_m, (g - mu) / mu.clamp(min=1.0), torch.zeros_like(g))
        prl = torch.where(pos, (r ** 2).sum(1) / M.clamp(min=1).float(), torch.zeros_like(al))
    return al, prl, torch.where(ok, p, torch.zeros_like(p)).detach()
