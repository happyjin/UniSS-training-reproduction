"""SerialCRF v2' with a latency budget (plan B, user 2026-11-14 ~10:20 Prague; reports/mist_w3po_v1/
joint_baseline_vs_method_20261114.zh-CN.md "解决方案" B). Additive: serial_v2p_train.py is not changed; install()
replaces its module attributes posterior / run_step in the running process.

Why: v2''s posterior q(rho, o) ∝ exp(tau sum_r psi_{r,o_r}) over the lattice {e, e + 1} (every item written at its
label read e or one read later) has no latency term (lam = 0). More audio makes the same text more likely, so the EM
moves mass to the deferred paths: the trained model writes later (dev text LAAL +0.4 s against the strict baseline).
PMC ([e - 1, e + 1] + an e-path budget) went the other way and lost quality. Plan B keeps v2''s lattice (no earlier
writes) and adds the cost of each deferral:

  Lat(path) = sum over deferred items c of units(c) x (t_{rho(c)+1} - t_{rho(c)}) / 1000   [unit-seconds]
            units = target units of the chunk text (CJK characters + Latin words, the LAAL units)
  q_lam(rho, o) ∝ exp(tau sum_r psi_{r,o_r} - lam Lat(rho))
  per window: lam* = 0 if E_{q_0}[Lat] <= 1e-9 or ratio >= 1, else the lam with E_{q_lam}[Lat] = ratio x E_{q_0}[Lat]
  (bisection; E_q[Lat] decreases in lam). ratio = V2P_LAT_RATIO, a single pre-registered value.

The cost is a function of the read's set state (deferred = B_r minus the on-time set D), so it enters the edge terms:
E'_r[i, j] = E_r[i, j] - lam cost_r[j]; forward-backward on E'; the order split inside an edge uses the unmodified
E_r (the cost does not depend on the order). With lam = 0 the result is serial_v2p_train.posterior exactly (tested).
Per step the trainer's metrics get lat_ratio, lat_lam_mean, lat_lam_max, lat_E0 (sum over the step's windows of
E_{q_0}[Lat]) and lat_E (after lam), lat_bound (windows with lam > 0).
"""
from __future__ import annotations

import math
import re

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V
from experiments.mist_w3po_v1.crf import precheck_serial_v2p as VP
from experiments.mist_w3po_v1.crf import serial_v2p_train as V2T

UNIT = re.compile(r"[一-鿿]|[A-Za-z0-9']+")
STEP: list[dict] = []


def units(text: str) -> int:
    return len(UNIT.findall(text))


def chunk_costs(rec: dict, lat: dict) -> dict[int, float]:
    """deferrable chunk -> its deferral cost in unit-seconds."""
    t = rec["t_ms"]
    return {c: units(rec["chunks"][c]["text"]) * (t[r + 1] - t[r]) / 1000.0
            for c, r in enumerate(lat["rho"]) if r < lat["R"] - 1}


def state_costs(rec: dict, lat: dict) -> list[np.ndarray]:
    cc = chunk_costs(rec, lat)
    out = []
    for r in range(lat["R"]):
        out.append(np.array([sum(cc.get(c, 0.0) for c in lat["B"][r] if c not in set(s)) for s in lat["states"][r]],
                            dtype=np.float64))
    return out


def shifted(E: list[np.ndarray], sc: list[np.ndarray], lam: float) -> list[np.ndarray]:
    return [e - lam * c[None, :] for e, c in zip(E, sc)] if lam else E


def exp_lat(E: list[np.ndarray], sc: list[np.ndarray], lam: float) -> tuple[float, dict]:
    fb = V.forward_backward(shifted(E, sc, lam), 1.0)
    return float(sum(float(np.dot(m, c)) for m, c in zip(fb["marg"], sc))), fb


def solve(E: list[np.ndarray], sc: list[np.ndarray], ratio: float, iters: int = 60) -> dict:
    L0, fb0 = exp_lat(E, sc, 0.0)
    if ratio >= 1.0 or L0 <= 1e-9:
        return {"lam": 0.0, "L0": L0, "L": L0, "fb": fb0}
    target = ratio * L0
    lo, hi = 0.0, 1.0
    L, fb = exp_lat(E, sc, hi)
    while L > target and hi < 1e6:
        lo, hi = hi, hi * 2.0
        L, fb = exp_lat(E, sc, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        Lm, fbm = exp_lat(E, sc, mid)
        if Lm > target:
            lo = mid
        else:
            hi, L, fb = mid, Lm, fbm
    return {"lam": hi, "L0": L0, "L": L, "fb": fb}


def posterior_lat(w, pl: dict, res: dict, tau: float, ratio: float) -> dict:
    """serial_v2p_train.posterior with q_lam (module docstring); same return keys plus lam, lat0, lat."""
    full = []
    for k, q in enumerate(pl["reqs"]):
        m = len(q["forced_ids"])
        full.append(res[(w.idx, k)] if q["scored"] else (np.zeros(m), np.zeros(m + 1)))
    reads = VP.assemble_p(pl, full)
    E = VP.edge_terms(reads, tau, "joint")
    sc = state_costs(w.rec, pl["lat"])
    sol = solve(E, sc, ratio)
    lam, fb = sol["lam"], sol["fb"]
    Es = shifted(E, sc, lam)
    cend = [np.zeros(len(q["forced_ids"]) + 1) for q in pl["reqs"]]
    exp_nf, n_real, worst = 0.0, 0, 0.0
    for r, (rd, pr) in enumerate(zip(reads, pl["reads"])):
        C, Nf = VP.mats(rd)
        a = fb["alpha"][r - 1] if r > 0 else np.zeros(1)
        qe = np.exp(a[:, None] + Es[r] + fb["beta"][r][None, :] - fb["logZ"])
        qC = qe * np.exp(tau * C - E[r])
        if Nf is not None and not np.isnan(Nf).all():
            real = ~np.isnan(Nf)
            qN = np.where(real, qe * np.exp(tau * np.where(real, Nf, 0.0) - E[r]), 0.0)
            n_real += 1
            exp_nf += float(qN.sum())
        else:
            qN = np.zeros_like(qC)
        worst = max(worst, abs(float(qC.sum() + qN.sum()) - 1.0))
        for cx in pr["ctx"]:
            i = cx["i"]
            for j, (k, b) in cx["cells"]["new"].items():
                cend[k][b] += qC[i, j]
            for j, (k, b, _bn, _s, _nl) in cx["nf"].items():
                cend[k][b] += qN[i, j]
    if worst > 1e-6:
        raise RuntimeError(f"v2'-lat posterior of {w.rec['id']}: per-read mass off by {worst:.2e}")
    R = pl["lat"]["R"]
    q_label = math.exp(tau * VP.path_score_p(reads, [0] * R, [0] * R) - fb["logZ"])
    defer = V.deferral_marginals(pl["lat"], fb)
    out = {"cend": cend, "logZ": fb["logZ"], "q_label": q_label, "exp_nf": exp_nf, "n_real": n_real,
           "exp_defer": float(sum(defer.values())), "fb": fb, "reads": reads,
           "lam": lam, "lat0": sol["L0"], "lat": sol["L"]}
    STEP.append({"lam": lam, "lat0": sol["L0"], "lat": sol["L"]})
    return out


def install(ratio: float) -> None:
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"V2P_LAT_RATIO must be in [0, 1], got {ratio}")
    orig_step = V2T.run_step

    def posterior(w, pl, res, tau=V2T.TAU):
        return posterior_lat(w, pl, res, tau, ratio)

    def run_step(*a, **k):
        STEP.clear()
        out = orig_step(*a, **k)
        lams = [s["lam"] for s in STEP]
        out["stats"].update({"lat_ratio": ratio, "lat_lam_mean": float(np.mean(lams)) if lams else 0.0,
                             "lat_lam_max": float(max(lams)) if lams else 0.0,
                             "lat_E0": float(sum(s["lat0"] for s in STEP)), "lat_E": float(sum(s["lat"] for s in STEP)),
                             "lat_bound": int(sum(1 for s in STEP if s["lam"] > 0))})
        return out

    V2T.posterior = posterior
    V2T.run_step = run_step
