"""T2: the latent write-time CRF (reports/mist_w3po_v1/thinker_crf_t_prereg_v1.zh-CN.md §T2).

v2''s set-state chain (crf/precheck_serial_v2.py, crf/precheck_serial_v2p.py, crf/serial_v2p_train.py; imported,
unchanged) with the feasible write sets widened from {rho_lab, rho_lab + 1} to an interval per item, plus a latency
potential whose weight lambda is the posterior-regularisation dual (Ganchev et al. 2010).

Data: sched-format records (crf/build_piece_sched.py) whose chunks are <= 4-word pieces in S order, S sorted by
(e_read, source end, source start); beta_heur = the e_read label path, so S.chunk_reads(rec) == e_read; commit_ms =
source end + 1 s uncapped, so V.kap(rec, e, .) == S.kappa(rec, e, .); reads every 0.5 s plus the final read.

Lattice (one window, R reads)
  items     the pieces i; speaker k_i (slot); per-speaker order = kap order restricted to the speaker.
  bounds    e_i = rho_lab (the T0 earliest writable read, or the S_prop read under the fallback);
            h_i = min(R - 1, max(u_i + 1, e_i + 1)), u_i = the S_prop deadline read (the first 0.5 s read at or after
            source end + 1 s, else the final read). The user's {u_i <= r - 1} ⊆ A_r ⊆ {e_i <= r} is empty when
            e_i > u_i + 1; the union with v2''s {e_i, e_i + 1} keeps it feasible and makes T1's lattice (v2' on the same
            labels) a sub-lattice of T2's. e and h are asserted non-decreasing along each speaker's order.
            mode "v2p": h_i = min(R - 1, e_i + 1) (= v2' exactly); mode "label": h_i = e_i (the label path only).
  state     A_r = the items written after read r: per speaker a prefix of c_k items,
            c_k in [#{i of k : h_i <= r}, #{i of k : e_i <= r}]  (so {h_i <= r} ⊆ A_r ⊆ {e_i <= r}; A_{R-1} = I).
            Enumerated as V.read_states does in the collapsed case: the free speakers (an interval of > 1 count) in the
            kap order of their first free item, counts descending; index 0 = {e_i <= r} (the label path).
  edge      A' -> A iff A ⊇ A'. W = kap(A \\ A'), then e_r (<wait>; <eos> at the final read, <wait> barred there).
  history   kappa(A_{r-1}) = kap order of the set (a written item sits at its label position in later histories).
  order     o_r in {canon, newfirst} (VP): carried = the items of W with e_i <= r - 1; N = the items of W with e_i = r
            whose slot is no carried item's slot; a real choice iff both are non-empty; newfirst = kap(N) ⊕ kap(W \\ N).
  potential psi_{r,o}(A' -> A) = log p_theta(W in order o ⊕ e_r | x <= t_r, kappa(A')) (a function of the edge, as v2');
            forced reads (one context, one state) are not scored (psi := 0, a path constant).
  latency   L_r(A' -> A) = sum_{i in W} n_words_i (t_r - src_end_i) / 1000   (seconds x words, text timeline)
  posterior q_lam(z) ∝ exp(sum_r [tau psi_{r,o_r} - lam L_r]),  tau = 0.5 (v2''s pre-registered tempering).
            Forward-backward over the set states with the per-edge logsumexp over the order, exact, CPU float64; edge-
            order marginals q(r, A' -> A, o) = exp(alpha + E_r + beta - log Z) x the order conditional
            exp(tau psi_o - logsumexp_o tau psi_o) (lam does not depend on the order).
  lam       per optimiser step, on the step's global batch: lam* >= 0 with sum_w E_{q_lam,w}[Lat] = sum_w E_{q_T1,w}[Lat]
            (0 if the constraint does not bind), where q_T1 is the same model's tempered v2' posterior on T1's sub-lattice
            (states with c_k >= #{e_i <= r - 1}, lam = 0): T2 is held at T1's expected latency and may only move the mass
            to a better placement. E_q[Lat](lam) is non-increasing (d/dlam = -Var_q[Lat]); bracket by doubling from 1 (cap
            1e3, flagged), then 50 bisection steps.
  M-step    v2''s posterior-weighted cross-entropy (serial_v2p_train.set_coefs; q without gradient).

Exactness: with h = min(R - 1, e + 1) the plan (requests, read dicts, cells) is VP.plan_window_p's field for field and
the lam = 0 posterior is serial_v2p_train.posterior's bit for bit; with h = e the rows are C0's rows; log Z, the
edge-order marginals, the per-sequence weights and E_q[Lat] equal a brute-force enumeration over (rho assignment x
per-read order) built without the lattice (crf/test_latent_write_crf.py).

    python -m experiments.mist_w3po_v1.crf.latent_write_crf --sizes --sched <dir> --out <new json> [--procs 48]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_serial as S
from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V

TAU = 0.5
LAM_START, LAM_MAX, BISECT_ITERS = 1.0, 1e3, 50
MODES = ("t2", "v2p", "label")


# ------------------------------------------------------------------------------------------------ lattice
def n_words(c: dict) -> int:
    return int(c["n_words"]) if "n_words" in c else len(c["text"].split())


def bounds(rec: dict, mode: str = "t2") -> tuple[list[int], list[int]]:
    """(e, h) per chunk; e = rho_lab from beta_heur (== chunk e_read when present)."""
    e = S.chunk_reads(rec)
    R = rec["n_read"]
    if rec["chunks"] and "e_read" in rec["chunks"][0]:
        assert e == [c["e_read"] for c in rec["chunks"]], rec["id"]
    if mode == "t2":
        h = [min(R - 1, max(c["u_read"] + 1, x + 1)) for c, x in zip(rec["chunks"], e)]
    elif mode == "v2p":
        h = [min(R - 1, x + 1) for x in e]
    elif mode == "label":
        h = list(e)
    else:
        raise ValueError(mode)
    return e, h


def lattice(rec: dict, mode: str = "t2", h: list[int] | None = None) -> dict:
    """States per read as count vectors over the window's speakers (sorted slots), with their item sets."""
    e, h_ = bounds(rec, mode)
    h = h_ if h is None else list(h)
    R = rec["n_read"]
    n = len(rec["chunks"])
    assert all(x <= y <= R - 1 for x, y in zip(e, h)), rec["id"]
    order = V.kap(rec, e, range(n))
    pos = {c: p for p, c in enumerate(order)}
    spk = sorted({rec["chunks"][c]["slot"] for c in range(n)})
    items = {k: [c for c in order if rec["chunks"][c]["slot"] == k] for k in spk}
    for k in spk:
        it = items[k]
        assert all(e[a] <= e[b] and h[a] <= h[b] for a, b in zip(it, it[1:])), (rec["id"], k)
    states, sets = [], []
    for r in range(R):
        lo = {k: sum(1 for c in items[k] if h[c] <= r) for k in spk}
        hi = {k: sum(1 for c in items[k] if e[c] <= r) for k in spk}
        assert all(lo[k] <= hi[k] for k in spk)
        free = sorted([k for k in spk if hi[k] > lo[k]], key=lambda k: pos[items[k][lo[k]]])
        cs, ss = [], []
        for js in itertools.product(*[range(hi[k], lo[k] - 1, -1) for k in free]):
            cnt = dict(lo)
            cnt.update(zip(free, js))
            cv = tuple(cnt[k] for k in spk)
            cs.append(cv)
            ss.append(frozenset(c for k, x in zip(spk, cv) for c in items[k][:x]))
        states.append(cs)
        sets.append(ss)
    assert len(states[R - 1]) == 1 and len(sets[R - 1][0]) == n
    return {"rho": e, "h": h, "R": R, "spk": spk, "items": items, "states": states, "sets": sets, "mode": mode}


def prev_of(lat: dict, r: int) -> tuple[list, list]:
    if r > 0:
        return lat["states"][r - 1], lat["sets"][r - 1]
    return [tuple(0 for _ in lat["spk"])], [frozenset()]


def covers(reach: list[tuple[int, tuple]]) -> list[tuple[tuple, list[tuple[int, int]]]]:
    """V.covers over (state index, write tuple) pairs: forwards longest first, each covering the reachable states whose
    write tuple is a prefix of it; every state exactly once (= V.covers when every state is reachable)."""
    idx = {s: j for j, s in reach}
    done, out = set(), []
    for j, s in sorted(reach, key=lambda x: (-len(x[1]), x[0])):
        if s in done:
            continue
        ent = [(idx[s[:k]], k) for k in range(len(s) + 1) if s[:k] in idx and s[:k] not in done]
        done.update(s[:k] for _, k in ent)
        out.append((s, ent))
    return out


def plan_window(rec: dict, lat: dict | None = None, skip_forced: bool = False, mode: str = "t2") -> dict:
    """The forwards of one window: canonical requests of all reads, then the newfirst requests (VP.plan_window_p's
    layout). Besides VP's fields: pl["W"][(r, i)] = {j: write tuple}, pl["L"][r] latency [n_prev, n_cur] (0 off the
    edges), pl["edge"][r] bool, pl["t1"][r] = (prev mask, cur mask) of T1's sub-lattice."""
    lat = lat or lattice(rec, mode)
    R, e, t_ms = lat["R"], lat["rho"], rec["t_ms"]
    ch = rec["chunks"]
    reads, reqs, Wd, Lm, Em, T1 = [], [], {}, [], [], []
    for r in range(R):
        ps, pset = prev_of(lat, r)
        cs, cset = lat["states"][r], lat["sets"][r]
        forced = len(ps) == 1 and len(cs) == 1
        rd = {"r": r, "final": r == R - 1, "forced": forced, "n_prev": len(ps), "n_cur": len(cs), "ctx": []}
        reads.append(rd)
        L = np.zeros((len(ps), len(cs)))
        Ed = np.zeros((len(ps), len(cs)), bool)
        for i, (cp, Ap) in enumerate(zip(ps, pset)):
            reach = []
            for j, (c, A) in enumerate(zip(cs, cset)):
                if all(x >= y for x, y in zip(c, cp)):
                    Wt = tuple(V.kap(rec, e, A - Ap))
                    reach.append((j, Wt))
                    Ed[i, j] = True
                    L[i, j] = sum(n_words(ch[x]) * (t_ms[r] - ch[x]["src_end_ms"]) / 1000.0 for x in Wt)
            Wd[(r, i)] = dict(reach)
            if forced and skip_forced:
                continue
            hist = V.kap(rec, e, Ap)
            pend = V.kap(rec, e, [x for x in range(len(ch)) if x not in Ap and e[x] <= r - 1])
            h_ids = S.ids_of(rec, hist)
            cx = {"i": i, "hist": hist, "pend": pend, "hist_len": len(h_ids), "cells": {"new": {}}, "first_req": None}
            for s, ent in covers(reach):
                items = list(s)
                lens = [ch[x]["tok1"] - ch[x]["tok0"] for x in items]
                offs = [int(x) for x in np.cumsum([0] + lens)]
                q = len(reqs)
                reqs.append({"r": r, "proto": "new", "hist_ids": h_ids, "forced_ids": S.ids_of(rec, items),
                             "items": items, "offsets": offs})
                if cx["first_req"] is None:
                    cx["first_req"] = q
                for j, k in ent:
                    cx["cells"]["new"][j] = (q, offs[k])
            assert len(cx["cells"]["new"]) == len(reach)
            rd["ctx"].append(cx)
        Lm.append(L)
        Em.append(Ed)
        lo1 = [sum(1 for c in lat["items"][k] if e[c] <= r - 1) for k in lat["spk"]]
        lo0 = [sum(1 for c in lat["items"][k] if e[c] <= r - 2) for k in lat["spk"]]
        T1.append((np.array([all(x >= y for x, y in zip(c, lo0)) for c in ps]) if r > 0 else np.ones(1, bool),
                   np.array([all(x >= y for x, y in zip(c, lo1)) for c in cs])))
    for rd in reads:                                            # newfirst requests (VP.plan_window_p's loop)
        r = rd["r"]
        for cx in rd["ctx"]:
            seqs = []
            for j, Wt in sorted(Wd[(r, cx["i"])].items()):
                carried = [x for x in Wt if e[x] <= r - 1]
                if not carried:
                    continue
                psl = {ch[x]["slot"] for x in carried}
                N = [x for x in Wt if e[x] == r and ch[x]["slot"] not in psl]
                if not N:
                    continue
                ns = set(N)
                seqs.append((j, tuple(N) + tuple(V.kap(rec, e, [x for x in Wt if x not in ns])), len(N)))
            h_ids = S.ids_of(rec, cx["hist"])
            cells, done = {}, set()
            for j, s, nl in sorted(seqs, key=lambda x: (-len(x[1]), x[0])):
                if j in done:
                    continue
                items = list(s)
                lens = [ch[x]["tok1"] - ch[x]["tok0"] for x in items]
                offs = [int(x) for x in np.cumsum([0] + lens)]
                q = len(reqs)
                reqs.append({"r": r, "proto": "new", "kind": "nf", "hist_ids": h_ids,
                             "forced_ids": S.ids_of(rec, items), "items": items, "offsets": offs})
                for j2, s2, nl2 in seqs:
                    if j2 not in done and s[: len(s2)] == s2:
                        cells[j2] = (q, offs[len(s2)], offs[nl2], s2, nl2)
                        done.add(j2)
            assert len(cells) == len(seqs)
            cx["nf"] = cells
        if rd["forced"]:
            assert not rd["ctx"] or not any(cx["nf"] for cx in rd["ctx"])
    return {"lat": lat, "reads": reads, "reqs": reqs, "W": Wd, "L": Lm, "edge": Em, "t1": T1}


# ------------------------------------------------------------------------------------------------ potentials
def assemble(plan: dict, res: list) -> list[dict]:
    """Per read: C [n_prev, n_cur] (canonical psi, -inf off the edges), Nf (newfirst psi, nan where no real choice;
    None when the read has none). Same arithmetic as V.assemble / VP.assemble_p."""
    out = []
    for rd in plan["reads"]:
        C = np.full((rd["n_prev"], rd["n_cur"]), -np.inf)
        Nf = None
        for cx in rd["ctx"]:
            for j, (q, b) in cx["cells"]["new"].items():
                tl, ec = res[q]
                C[cx["i"], j] = float(np.sum(tl[:b]) + ec[b])
        if any(cx.get("nf") for cx in rd["ctx"]):
            Nf = np.full((rd["n_prev"], rd["n_cur"]), np.nan)
            for cx in rd["ctx"]:
                for j, (q, b, _bn, _s, _nl) in cx["nf"].items():
                    tl, ec = res[q]
                    Nf[cx["i"], j] = float(np.sum(tl[:b]) + ec[b])
        out.append({"r": rd["r"], "final": rd["final"], "forced": rd["forced"], "C": C, "Nf": Nf})
    return out


def edge_terms(reads: list[dict], tau: float) -> list[np.ndarray]:
    """VP.edge_terms(mode joint): logsumexp_o(tau psi_o) per edge (-inf off the edges)."""
    out = []
    for rd in reads:
        C, Nf = rd["C"], rd["Nf"]
        if Nf is None:
            out.append(tau * C)
            continue
        Nn = np.where(np.isnan(Nf), -np.inf, Nf)
        out.append(np.logaddexp(tau * C, tau * Nn))
    return out


def with_lam(Eo: list[np.ndarray], L: list[np.ndarray], lam: float) -> list[np.ndarray]:
    return Eo if lam == 0.0 else [x - lam * l for x, l in zip(Eo, L)]


def exp_lat(E: list[np.ndarray], L: list[np.ndarray], fb: dict | None = None) -> tuple[float, dict]:
    """E_q[Lat] under the chain with edge terms E (and its forward-backward)."""
    if fb is None:
        with np.errstate(divide="ignore"):           # masked sub-lattices: states off it have alpha = beta = -inf
            fb = V.forward_backward(E, 1.0)
    s = 0.0
    for r, (M, Lr) in enumerate(zip(E, L)):
        a = fb["alpha"][r - 1] if r > 0 else np.zeros(1)
        qe = np.exp(a[:, None] + M + fb["beta"][r][None, :] - fb["logZ"])
        s += float((qe * Lr).sum())
    return s, fb


def full_res(w, pl: dict, res: dict) -> list:
    out = []
    for k, q in enumerate(pl["reqs"]):
        m = len(q["forced_ids"])
        out.append(res[(w.idx, k)] if q["scored"] else (np.zeros(m), np.zeros(m + 1)))
    return out


def window_parts(w, pl: dict, res: dict, tau: float = TAU) -> dict:
    """Everything of one window that does not depend on lam: psi, the lam-free edge terms, T1's target E_{q_T1}[Lat],
    the label path's Lat and sum of words."""
    reads = assemble(pl, full_res(w, pl, res))
    Eo = edge_terms(reads, tau)
    E1 = [np.where(pm[:, None] & cm[None, :], x, -np.inf) for x, (pm, cm) in zip(Eo, pl["t1"])]
    tgt, _ = exp_lat(E1, pl["L"])
    lab = sum(float(Lr[0, 0]) for Lr in pl["L"])
    sw = float(sum(n_words(c) for c in w.rec["chunks"]))
    return {"reads": reads, "Eo": Eo, "L": pl["L"], "edge": pl["edge"], "target": tgt, "lat_label": lab,
            "sum_words": sw}


def solve_lambda(parts: list[dict]) -> tuple[float, dict]:
    """lam* >= 0 with sum_w E_{q_lam,w}[Lat] = sum_w target_w (step level); 0 if not binding."""
    tgt = sum(p["target"] for p in parts)

    def f(lam):
        return sum(exp_lat(with_lam(p["Eo"], p["L"], lam), p["L"])[0] for p in parts)
    f0 = f(0.0)
    info = {"target": tgt, "Eq_lat_lam0": f0, "binding": f0 > tgt, "bracketed": True, "capped": False, "evals": 1}
    if f0 <= tgt:
        return 0.0, info
    lo, hi = 0.0, LAM_START
    while True:
        fh = f(hi)
        info["evals"] += 1
        if fh <= tgt:
            break
        lo = hi
        if hi >= LAM_MAX:
            info.update(bracketed=False, capped=True)
            return LAM_MAX, info
        hi = min(2.0 * hi, LAM_MAX)
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if f(mid) > tgt:
            lo = mid
        else:
            hi = mid
        info["evals"] += 1
    return hi, info


def posterior_from_parts(w, pl: dict, parts: dict, tau: float, lam: float) -> dict:
    """serial_v2p_train.posterior's body with E = E_order - lam L (cend from the edge marginal x the order
    conditional exp(tau psi_o - E_order)); lam = 0 is bit-identical to it on a v2' lattice."""
    reads, Eo, edge = parts["reads"], parts["Eo"], parts["edge"]
    E = with_lam(Eo, parts["L"], lam)
    fb = V.forward_backward(E, 1.0)
    cend = [np.zeros(len(q["forced_ids"]) + 1) for q in pl["reqs"]]
    exp_nf, n_real, worst, eq_lat, exp_defer = 0.0, 0, 0.0, 0.0, 0.0
    lat, e = pl["lat"], pl["lat"]["rho"]
    by_e = [[x for x in range(len(e)) if e[x] == r] for r in range(lat["R"])]
    for r, (rd, pr) in enumerate(zip(reads, pl["reads"])):
        C, Nf = rd["C"], rd["Nf"]
        a = fb["alpha"][r - 1] if r > 0 else np.zeros(1)
        qe = np.exp(a[:, None] + E[r] + fb["beta"][r][None, :] - fb["logZ"])
        if edge[r].all():
            qC = qe * np.exp(tau * C - Eo[r])
        else:
            with np.errstate(invalid="ignore"):
                qC = qe * np.where(edge[r], np.exp(tau * C - Eo[r]), 0.0)
        if Nf is not None and not np.isnan(Nf).all():
            real = ~np.isnan(Nf)
            with np.errstate(invalid="ignore", over="ignore"):
                qN = np.where(real, qe * np.exp(tau * np.where(real, Nf, 0.0) - Eo[r]), 0.0)
            n_real += 1
            exp_nf += float(qN.sum())
        else:
            qN = np.zeros_like(qC)
        worst = max(worst, abs(float(qC.sum() + qN.sum()) - 1.0))
        eq_lat += float((qe * parts["L"][r]).sum())
        mg = fb["marg"][r]
        for j, A in enumerate(lat["sets"][r]):
            exp_defer += float(mg[j]) * sum(1 for x in by_e[r] if x not in A)
        for cx in pr["ctx"]:
            i = cx["i"]
            for j, (k, b) in cx["cells"]["new"].items():
                cend[k][b] += qC[i, j]
            for j, (k, b, _bn, _s, _nl) in cx["nf"].items():
                cend[k][b] += qN[i, j]
    if worst > 1e-6:
        raise RuntimeError(f"T2 posterior of {w.rec['id']}: per-read mass off by {worst:.2e}")
    s_lab = 0.0
    for rd in reads:
        s_lab += float(rd["C"][0, 0])
    q_label = math.exp(tau * s_lab - lam * parts["lat_label"] - fb["logZ"])
    return {"cend": cend, "logZ": fb["logZ"], "q_label": q_label, "exp_nf": exp_nf, "n_real": n_real,
            "exp_defer": exp_defer, "fb": fb, "reads": reads, "Eq_lat": eq_lat, "target": parts["target"],
            "lat_label": parts["lat_label"], "sum_words": parts["sum_words"], "lam": lam}


def posterior(w, pl: dict, res: dict, tau: float = TAU, lam: float = 0.0) -> dict:
    return posterior_from_parts(w, pl, window_parts(w, pl, res, tau), tau, lam)


# ------------------------------------------------------------------------------------------------ CPU size check
def _plan_sizes(rec: dict) -> dict:
    from experiments.mist_w3po_v1.crf import train_crf_sft as T

    w = T.make_window(0, rec, True)
    out = {"id": rec["id"], "R": w.R}
    for mode in ("t2", "v2p"):
        lat = lattice(rec, mode)
        pl = plan_window(rec, lat, skip_forced=False, mode=mode)
        forced = {rd["r"] for rd in pl["reads"] if rd["forced"]}
        sc = [q for q in pl["reqs"] if q["r"] not in forced]
        out[mode] = {"p1": len(sc), "p2": len(pl["reqs"]),
                     "tok1": sum(len(q["hist_ids"]) + len(q["forced_ids"]) for q in sc),
                     "tok2": sum(len(q["hist_ids"]) + len(q["forced_ids"]) for q in pl["reqs"]),
                     "max_states": max(len(s) for s in lat["states"]),
                     "edges": int(sum(int(x.sum()) for x in pl["edge"]))}
    return out


def _stat(v) -> dict:
    v = np.asarray(v, np.float64)
    return {"mean": round(float(v.mean()), 2), "p50": float(np.median(v)), "p95": round(float(np.percentile(v, 95)), 1),
            "max": float(v.max())}


def run_sizes(args) -> None:
    from multiprocessing import Pool

    from experiments.mist_w3po_v1.crf import train_crf_sft as T

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    recs = []
    for f in sorted(Path(args.sched).glob("shard_*.jsonl")):
        recs += [json.loads(l) for l in open(f, encoding="utf-8")]
    recs = [r for r in recs if r["n_tok"] > 0]
    recs.sort(key=lambda r: r["id"])
    plan_ = T.plan_steps([r["n_read"] for r in recs], args.seed, args.steps, args.rows_per_step)
    used = sorted({i for s in plan_ for i in s})
    with Pool(args.procs) as pool:
        res = pool.map(_plan_sizes, [recs[i] for i in used], chunksize=32)
    by = dict(zip(used, res))
    rep = {"windows_used": len(used), "steps": args.steps, "rows_per_step": args.rows_per_step, "seed": args.seed,
           "per_step": {}, "per_window_max_states": {}, "edges_per_window": {}}
    for mode in ("t2", "v2p"):
        rep["per_step"][mode] = {k: _stat([sum(by[i][mode][k] for i in s) for s in plan_])
                                 for k in ("p1", "p2", "tok1", "tok2")}
        rep["per_window_max_states"][mode] = _stat([by[i][mode]["max_states"] for i in used])
        rep["edges_per_window"][mode] = _stat([by[i][mode]["edges"] for i in used])
    rep["read_rows_per_step"] = _stat([sum(by[i]["R"] for i in s) for s in plan_])
    # lam solve on dummy scores, a sample of steps (seeded): CPU time of the posterior + solve per step
    rng = np.random.default_rng(args.seed)
    tm, lams = [], []
    for si in sorted(rng.choice(len(plan_), size=min(args.time_steps, len(plan_)), replace=False).tolist()):
        wins = [T.make_window(i, recs[i], True) for i in plan_[si]]
        t1 = time.time()
        parts, pls = {}, {}
        for w in wins:
            pl = plan_window(w.rec, skip_forced=False)
            forced = {rd["r"] for rd in pl["reads"] if rd["forced"]}
            for q in pl["reqs"]:
                q["scored"] = q["r"] not in forced
            dres = V.dummy_res(pl)
            res_ = {(w.idx, k): v for k, (v, q) in enumerate(zip(dres, pl["reqs"])) if q["scored"]}
            pls[w.idx] = pl
            parts[w.idx] = window_parts(w, pl, res_, TAU)
        t_parts = time.time() - t1
        lam, info = solve_lambda(list(parts.values()))
        for w in wins:
            posterior_from_parts(w, pls[w.idx], parts[w.idx], TAU, lam)
        tm.append({"step": si, "plan_parts_s": round(t_parts, 3), "total_s": round(time.time() - t1, 3),
                   "evals": info["evals"]})
        lams.append(lam)
    rep["lambda_solve_dummy"] = {"steps": len(tm), "total_s": _stat([x["total_s"] for x in tm]),
                                 "plan_parts_s": _stat([x["plan_parts_s"] for x in tm]),
                                 "evals": _stat([x["evals"] for x in tm]), "lam": _stat(lams)}
    rep["cpu_s"] = round(time.time() - t0, 1)
    rep["sched"] = args.sched
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", action="store_true")
    ap.add_argument("--sched", default="eval_outputs/mist_w3po_v1/crf/sched_piece_t0_v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--rows-per-step", type=int, default=64)
    ap.add_argument("--time-steps", type=int, default=30)
    args = ap.parse_args()
    if args.sizes:
        run_sizes(args)


if __name__ == "__main__":
    main()
