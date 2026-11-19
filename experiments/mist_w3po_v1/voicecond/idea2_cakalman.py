"""Idea 2: linear-Gaussian CA-Kalman (plan §3.1-3.5, prereg §4): parameters, exact-posterior EM, causal filter.

Model (window index k):   z_k = s_{j(k)} + A n_k + eps_k,  eps_k ~ N(0, R / pi_k)          (speaker windows only)
                          v_k = B n_k + zeta_k,           zeta_k ~ N(0, diag(S))          (every window; no intercept)
                          n_k = nbar + nt_k,  nbar ~ N(mu_nb, Sig_nb),  nt_k = rho nt_{k-1} + eta_k, Var(nt) = I
                          s_j ~ N(mu0, Sig0)
EM (fit_em): non-overlapping windows; per stream the exact Gaussian posterior of (s_1..s_J, nbar, nt_1..nt_K) in
information form (dense Cholesky, float64 on GPU) = RTS smoothing; closed-form M step; Q fixed by the gauge
Var(nt) = I (unconstrained (rho, Q) step followed by the exact re-gauge T = P_inf^{-1/2}). Marginal log-likelihood logged.
Filter (CAKalmanStream): covariance form, hop 0.5 s, rho_f = rho^(0.5/L), Q_f = (1 - rho_f^2) I, R_f = (L/0.5) R,
S_f = (L/0.5) S; slots are added when first observed (equivalent to carrying their independent prior from the start).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

D = 192


@dataclass
class Params:
    A: np.ndarray        # [D, r]
    B: np.ndarray        # [m, r]
    S: np.ndarray        # [m]   diag
    R: np.ndarray        # [D, D]
    rho: float
    mu_nb: np.ndarray    # [r]
    Sig_nb: np.ndarray   # [r, r]
    mu0: np.ndarray      # [D]
    Sig0: np.ndarray     # [D, D]
    L: float = 2.0       # window length the params were fitted with (non-overlapping hop = L)
    hist: list = field(default_factory=list)

    @property
    def r(self):
        return self.A.shape[1]

    def save(self, path):
        np.savez(path, A=self.A, B=self.B, S=self.S, R=self.R, rho=self.rho, mu_nb=self.mu_nb, Sig_nb=self.Sig_nb,
                 mu0=self.mu0, Sig0=self.Sig0, L=self.L, hist=np.array(self.hist, dtype=np.float64))

    @staticmethod
    def load(path):
        z = np.load(path)
        return Params(z["A"], z["B"], z["S"], z["R"], float(z["rho"]), z["mu_nb"], z["Sig_nb"], z["mu0"], z["Sig0"],
                      float(z["L"]), list(z["hist"]))


def _t(x, dev):
    return torch.as_tensor(np.asarray(x), dtype=torch.float64, device=dev)


# ------------------------------------------------------------------------------------------------ EM
class _Stream:
    """one stream's EM data (non-overlapping windows): z [K, D], v [K, m], slot [K] (-1 = no z obs), pi [K]"""

    def __init__(self, z, v, slot, pi):
        self.z, self.v = z.astype(np.float64), v.astype(np.float64)
        sl = np.asarray(slot)
        ids = sorted(set(sl[sl >= 0].tolist()))
        self.J = len(ids)
        m = {s: i for i, s in enumerate(ids)}
        self.j = np.array([m.get(int(s), -1) for s in sl])
        self.pi = np.where(self.j >= 0, pi, 0.0).astype(np.float64)
        self.K = len(sl)


def posterior(p: Params, st: _Stream, dev, want_cov=True):
    """exact posterior of x = [s_1..s_J, nbar, nt_1..nt_K]; returns (mean, cov, loglik)"""
    r, K, J = p.r, st.K, st.J
    A, B = _t(p.A, dev), _t(p.B, dev)
    Ri = torch.linalg.inv(_t(p.R, dev))
    Ri = 0.5 * (Ri + Ri.T)
    Si = 1.0 / _t(p.S, dev)
    S0i = torch.linalg.inv(_t(p.Sig0, dev))
    Snbi = torch.linalg.inv(_t(p.Sig_nb, dev))
    mu0, munb = _t(p.mu0, dev), _t(p.mu_nb, dev)
    z, v, pi = _t(st.z, dev), _t(st.v, dev), _t(st.pi, dev)
    jj = torch.as_tensor(st.j, device=dev)
    RA = Ri @ A
    ARA = A.T @ RA
    BSB = B.T @ (Si[:, None] * B)
    N = J * D + r + K * r
    o_n, o_t = J * D, J * D + r
    Lam = torch.zeros(N, N, dtype=torch.float64, device=dev)
    eta = torch.zeros(N, dtype=torch.float64, device=dev)
    has = jj >= 0
    pz = pi[:, None] * z                                   # [K, D]
    # speakers
    for j in range(J):
        sel = jj == j
        pj = pi[sel].sum()
        a = slice(j * D, (j + 1) * D)
        Lam[a, a] = S0i + pj * Ri
        eta[a] = S0i @ mu0 + Ri @ pz[sel].sum(0)
        Lam[a, o_n:o_n + r] = pj * RA
        Lam[o_n:o_n + r, a] = (pj * RA).T
    # speaker - nt coupling
    if J > 0:
        Lsn = torch.zeros(J, D, K, r, dtype=torch.float64, device=dev)
        kk = torch.nonzero(has).squeeze(1)
        Lsn[jj[kk], :, kk, :] = pi[kk][:, None, None] * RA[None]
        Lsn = Lsn.reshape(J * D, K * r)
        Lam[:J * D, o_t:] = Lsn
        Lam[o_t:, :J * D] = Lsn.T
    # nbar
    Lam[o_n:o_n + r, o_n:o_n + r] = Snbi + K * BSB + pi.sum() * ARA
    eta[o_n:o_n + r] = Snbi @ munb + B.T @ (Si * v.sum(0)) + A.T @ (Ri @ pz.sum(0))
    cpl = BSB[None] + pi[:, None, None] * ARA[None]        # [K, r, r]
    Lam[o_n:o_n + r, o_t:] = cpl.permute(1, 0, 2).reshape(r, K * r)
    Lam[o_t:, o_n:o_n + r] = cpl.reshape(K * r, r)
    # nt chain
    rho = p.rho
    c = 1.0 / (1.0 - rho ** 2)
    dg = torch.full((K,), c * (1 + rho ** 2), dtype=torch.float64, device=dev)
    if K == 1:
        dg[0] = 1.0
    else:
        dg[0] = dg[-1] = c
    I = torch.eye(r, dtype=torch.float64, device=dev)
    Ltt = torch.zeros(K, r, K, r, dtype=torch.float64, device=dev)
    ar = torch.arange(K, device=dev)
    Ltt[ar, :, ar, :] = dg[:, None, None] * I[None] + cpl
    if K > 1:
        Ltt[ar[1:], :, ar[:-1], :] = -c * rho * I
        Ltt[ar[:-1], :, ar[1:], :] = -c * rho * I
    Lam[o_t:, o_t:] = Ltt.reshape(K * r, K * r)
    eta[o_t:] = ((v * Si) @ B + pz @ RA).reshape(-1)
    Lam = 0.5 * (Lam + Lam.T)
    Lc = torch.linalg.cholesky(Lam)
    mean = torch.cholesky_solve(eta[:, None], Lc).squeeze(1)
    cov = torch.cholesky_inverse(Lc) if want_cov else None
    # marginal log-likelihood
    logdet_lam = 2 * torch.log(torch.diagonal(Lc)).sum()
    nz = int(has.sum())
    ldR = torch.linalg.slogdet(_t(p.R, dev))[1]
    quad = (pi[has, None] * z[has] * (z[has] @ Ri)).sum() + (v * v * Si).sum()
    quad = quad + J * mu0 @ (S0i @ mu0) + munb @ (Snbi @ munb) - eta @ mean
    ldP0 = J * torch.linalg.slogdet(_t(p.Sig0, dev))[1] + torch.linalg.slogdet(_t(p.Sig_nb, dev))[1] \
        + r * (K - 1) * math.log(1 - rho ** 2)
    ldobs = nz * ldR - D * torch.log(pi[has]).sum() + K * torch.log(_t(p.S, dev)).sum()
    nobs = nz * D + K * v.shape[1]
    ll = -0.5 * (quad + logdet_lam + ldP0 + ldobs + nobs * math.log(2 * math.pi))
    return mean, cov, float(ll)


def fit_em(streams, p: Params, dev, iters=30, tol=1e-6, log=print):
    """streams: list of (z [K, D], v [K, m], slot [K], pi [K]) on non-overlapping windows"""
    data = [_Stream(*s) for s in streams]
    prev = None
    for it in range(iters + 1):
        r = p.r
        m = p.B.shape[0]
        acc = {k: torch.zeros(s, dtype=torch.float64, device=dev) for k, s in {
            "zn": (D, r), "nnz": (r, r), "ee": (D, D), "vn": (m, r), "nnv": (r, r), "vv": (m,), "tt1": (), "t1t1": (),
            "Qa": (r, r), "Qb": (r, r), "Qc": (r, r), "nb": (r,), "nbnb": (r, r)}.items()}
        A, B = _t(p.A, dev), _t(p.B, dev)
        Nz = Nv = Nq = 0
        LL = 0.0
        stats = []
        for st in data:
            mean, cov, ll = posterior(p, st, dev)
            LL += ll
            K, J = st.K, st.J
            o_n, o_t = J * D, J * D + r
            mnb = mean[o_n:o_n + r]
            mt = mean[o_t:].reshape(K, r)
            mn = mnb[None] + mt                                            # [K, r]
            Cnb = cov[o_n:o_n + r, o_n:o_n + r]
            Ctt = cov[o_t:, o_t:].reshape(K, r, K, r)
            ar = torch.arange(K, device=dev)
            Ctk = Ctt[ar, :, ar, :]                                        # [K, r, r]
            Cnt = cov[o_n:o_n + r, o_t:].reshape(r, K, r).permute(1, 0, 2)  # [K, r(nb), r(t)]
            Cnn = Cnb[None] + Ctk + Cnt + Cnt.transpose(1, 2)               # Cov(n_k)
            Enn = mn[:, :, None] * mn[:, None, :] + Cnn
            v = _t(st.v, dev)
            acc["vn"] += v.T @ mn
            acc["nnv"] += Enn.sum(0)
            red = {"v": v, "mn": mn, "Cnn": Cnn}
            Nv += K
            # dynamics
            if K > 1:
                Ett1 = mt[1:, :, None] * mt[:-1, None, :] + Ctt[ar[1:], :, ar[:-1], :]
                Et1t1 = mt[:-1, :, None] * mt[:-1, None, :] + Ctt[ar[:-1], :, ar[:-1], :]
                Ett = mt[1:, :, None] * mt[1:, None, :] + Ctt[ar[1:], :, ar[1:], :]
                acc["tt1"] += torch.diagonal(Ett1, dim1=1, dim2=2).sum()
                acc["t1t1"] += torch.diagonal(Et1t1, dim1=1, dim2=2).sum()
                acc["Qa"] += Ett.sum(0); acc["Qb"] += Ett1.sum(0); acc["Qc"] += Et1t1.sum(0)
                Nq += K - 1
            acc["nb"] += mnb
            acc["nbnb"] += mnb[:, None] * mnb[None] + Cnb
            # speaker windows
            has = np.nonzero(st.j >= 0)[0]
            if len(has):
                hk = torch.as_tensor(has, device=dev)
                jk = torch.as_tensor(st.j[has], device=dev)
                pi = _t(st.pi[has], dev)
                z = _t(st.z[has], dev)
                ms = mean[:J * D].reshape(J, D)[jk]                          # [n, D]
                Csnb = cov[:J * D, o_n:o_n + r].reshape(J, D, r)[jk]          # [n, D, r]
                Cst = cov[:J * D, o_t:].reshape(J, D, K, r)[jk, :, hk, :]      # [n, D, r]
                Csn = Csnb + Cst
                acc["zn"] += ((pi[:, None] * (z - ms)).T @ mn[hk]) - (pi[:, None, None] * Csn).sum(0)
                acc["nnz"] += (pi[:, None, None] * Enn[hk]).sum(0)
                Nz += len(has)
                Css = torch.zeros(D, D, dtype=torch.float64, device=dev)
                for j in range(J):           # sum_k pi_k Cov(s_j): static per speaker
                    Css += pi[jk == j].sum() * cov[j * D:(j + 1) * D, j * D:(j + 1) * D]
                red.update({"hk": hk, "pi": pi, "z": z, "ms": ms, "Csn_w": (pi[:, None, None] * Csn).sum(0),
                            "Cnn_w": (pi[:, None, None] * Cnn[hk]).sum(0), "Css_w": Css})
            stats.append(red)
            del mean, cov, Ctt
        # convergence
        p.hist.append(LL)
        log(f"[em] iter {it} loglik {LL:.6e} (Nz {Nz}, Nv {Nv})")
        if prev is not None and abs(LL - prev) < tol * abs(prev):
            log(f"[em] converged at iter {it}")
            break
        prev = LL
        if it == iters:
            break
        # M step
        A_new = acc["zn"] @ torch.linalg.inv(acc["nnz"])
        B_new = acc["vn"] @ torch.linalg.inv(acc["nnv"])
        ee = torch.zeros(D, D, dtype=torch.float64, device=dev)
        vv = torch.zeros(m, dtype=torch.float64, device=dev)
        for red in stats:
            mn, Cnn = red["mn"], red["Cnn"]
            ev = red["v"] - mn @ B_new.T
            vv += (ev * ev).sum(0) + torch.einsum("ar,krs,as->a", B_new, Cnn, B_new)
            if "hk" not in red:
                continue
            pi = red["pi"]
            e = red["z"] - red["ms"] - mn[red["hk"]] @ A_new.T
            ee += (pi[:, None] * e).T @ e + red["Css_w"] + A_new @ red["Cnn_w"] @ A_new.T
            X = red["Csn_w"] @ A_new.T
            ee += X + X.T
        R_new = ee / Nz
        R_new = 0.5 * (R_new + R_new.T)
        S_new = (vv / Nv).clamp_min(1e-8)
        ns = len(data)
        mu_nb = acc["nb"] / ns
        Sig_nb = acc["nbnb"] / ns - mu_nb[:, None] * mu_nb[None]
        rho = float((acc["tt1"] / acc["t1t1"]).clamp(0.0, 0.999))
        Q = (acc["Qa"] - rho * (acc["Qb"] + acc["Qb"].T) + rho ** 2 * acc["Qc"]) / max(Nq, 1)
        Pinf = 0.5 * (Q + Q.T) / (1 - rho ** 2)
        ev_, U = torch.linalg.eigh(Pinf)
        ev_ = ev_.clamp_min(1e-10)
        T = U @ torch.diag(ev_ ** -0.5) @ U.T                                 # Pinf^{-1/2}
        Ti = U @ torch.diag(ev_ ** 0.5) @ U.T
        p = Params(A=(A_new @ Ti).cpu().numpy(), B=(B_new @ Ti).cpu().numpy(), S=S_new.cpu().numpy(),
                   R=R_new.cpu().numpy(), rho=rho, mu_nb=(T @ mu_nb).cpu().numpy(),
                   Sig_nb=(T @ (0.5 * (Sig_nb + Sig_nb.T)) @ T.T).cpu().numpy(), mu0=p.mu0, Sig0=p.Sig0, L=p.L,
                   hist=p.hist)
        log(f"[em]   rho {rho:.4f}  tr R {np.trace(p.R):.4f}  |A| {np.linalg.norm(p.A):.4f}  |B| {np.linalg.norm(p.B):.4f}"
            f"  |mu_nb| {np.linalg.norm(p.mu_nb):.3f}")
    return p


# ------------------------------------------------------------------------------------------------ filter
class CAKalmanStream:
    """causal filter over 0.5 s-hop windows. step(z or None, slot, pi, v) after each window; mean(slot) / trP(slot).
    use_v=False drops observation 2; nuisance=False is the prior-shrinkage-only filter (A = 0, no n state)."""

    def __init__(self, p: Params, hop=0.5, dev="cpu", use_v=True, nuisance=True):
        self.dev = torch.device(dev)
        self.use_v, self.nuis = use_v, nuisance
        f = p.L / hop
        self.rho = p.rho ** (hop / p.L)
        self.r = p.r if nuisance else 0
        self.A, self.B = _t(p.A, self.dev), _t(p.B, self.dev)
        self.Rf = _t(p.R, self.dev) * f
        self.Sf = _t(p.S, self.dev) * f
        self.mu0, self.Sig0 = _t(p.mu0, self.dev), _t(p.Sig0, self.dev)
        r = self.r
        if nuisance:   # state [nbar, nt, s_1, ...]
            self.m = torch.cat([_t(p.mu_nb, self.dev), torch.zeros(r, dtype=torch.float64, device=self.dev)])
            self.P = torch.block_diag(_t(p.Sig_nb, self.dev), torch.eye(r, dtype=torch.float64, device=self.dev))
        else:
            self.m = torch.zeros(0, dtype=torch.float64, device=self.dev)
            self.P = torch.zeros(0, 0, dtype=torch.float64, device=self.dev)
        self.slots = {}
        self.k = 0

    def _add_slot(self, s):
        n = self.m.shape[0]
        self.slots[s] = n
        self.m = torch.cat([self.m, self.mu0])
        P = torch.zeros(n + D, n + D, dtype=torch.float64, device=self.dev)
        P[:n, :n] = self.P
        P[n:, n:] = self.Sig0
        self.P = P

    def step(self, z, slot, pi, v):
        r = self.r
        if self.nuis:   # predict nt
            a = slice(r, 2 * r)
            if self.k > 0:
                self.m[a] *= self.rho
                self.P[a, :] *= self.rho
                self.P[:, a] *= self.rho
                self.P[a, a] += (1 - self.rho ** 2) * torch.eye(r, dtype=torch.float64, device=self.dev)
        self.k += 1
        Hs, ys, Rs = [], [], []
        n = self.m.shape[0]
        if self.use_v and self.nuis and v is not None:
            H = torch.zeros(self.B.shape[0], n, dtype=torch.float64, device=self.dev)
            H[:, :r] = self.B
            H[:, r:2 * r] = self.B
            Hs.append(H); ys.append(_t(v, self.dev)); Rs.append(torch.diag(self.Sf))
        if z is not None and slot is not None and slot >= 0 and pi > 0:
            if slot not in self.slots:
                self._add_slot(slot)
                n = self.m.shape[0]
                Hs = [torch.cat([h, torch.zeros(h.shape[0], D, dtype=torch.float64, device=self.dev)], 1) for h in Hs]
            H = torch.zeros(D, n, dtype=torch.float64, device=self.dev)
            o = self.slots[slot]
            H[:, o:o + D] = torch.eye(D, dtype=torch.float64, device=self.dev)
            if self.nuis:
                H[:, :r] = self.A
                H[:, r:2 * r] = self.A
            Hs.append(H); ys.append(_t(z, self.dev).reshape(-1)); Rs.append(self.Rf / pi)
        if not Hs:
            return
        H = torch.cat(Hs)
        y = torch.cat(ys)
        Rn = torch.block_diag(*Rs)
        PHt = self.P @ H.T
        Sinn = H @ PHt + Rn
        Sinn = 0.5 * (Sinn + Sinn.T)
        Lc = torch.linalg.cholesky(Sinn)
        Kt = torch.cholesky_solve(PHt.T, Lc)        # = Sinn^{-1} H P  -> K = Kt.T
        self.m = self.m + Kt.T @ (y - H @ self.m)
        self.P = self.P - PHt @ Kt
        self.P = 0.5 * (self.P + self.P.T)

    def mean(self, slot):
        if slot not in self.slots:
            return self.mu0.cpu().numpy()
        o = self.slots[slot]
        return self.m[o:o + D].cpu().numpy()

    def trP(self, slot):
        if slot not in self.slots:
            return float(torch.trace(self.Sig0))
        o = self.slots[slot]
        return float(torch.trace(self.P[o:o + D, o:o + D]))
