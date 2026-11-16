"""TS-PA causal replay (reports/mist_w3po_v1/tspa_causal_replay_prereg_v1.zh-CN.md; post-hoc hypothesis).

Would a Talker trained on the TS-PA target (PL-EM + starvation potential), with its present inputs only (backlog
token, text, history), pause less than one trained on the PL-EM target? Both targets are imitated by the same causal
conditional-logit policy pi(c | ctx) over a block's candidates (features: seconds per unit, faster-speed flag, CTC f,
lpbar, and the first two crossed with the backlog token's seconds, sentence end, log W, first block; columns
standardised, L2 1e-3 on the standardised weights), fitted to q(c | b_n = the rule path's state) on the fit set, then
replayed on held-out streams by an exact forward pass over the backlog distribution with CTC best-of-2 on top:
pi2(c) = pi(c) [pi(c) + 2 sum_{f(c')<f(c)} pi(c') + sum_{c' != c, f(c')=f(c)} pi(c')].

    python -m experiments.mist_w3po_v1.talker.tspa_causal_replay --fit-cands ... --fit-win ... \\
        [--eval-cands ... --eval-win ... | --fold even|odd] [--lang cmn --units char] --procs 60 --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.talker import playback_lattice as PL
from experiments.mist_w3po_v1.talker import talker_rows as TR
from experiments.mist_w3po_v1.talker import tspa_precheck_p1 as T

NB = 8
L2 = 1e-3
BOOT = 2000


def bin_s(state: int) -> int:
    return min(7, int(state * 80 / 500))


def _load(args) -> list[dict]:
    path, W, lang, units = args
    TR.LANG, TR.UNITS = (TR.LANG_ENG if lang == "eng" else TR.LANG_CMN), units
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if not (r["blocks"] and all(b["cands"] for b in r["blocks"])):
            continue
        w = W[r["id"].split("__S")[0]]
        bl = r["blocks"]
        out.append({
            "id": r["id"], "meeting": w["meeting"],
            "a": [TR.availability(w, b["write_ms"]) for b in bl],
            "dur": [[len(c["codes"]) * 20 for c in b["cands"]] for b in bl],
            "feat": [[TR.feat(c) for c in b["cands"]] for b in bl],
            "f": [[-(PL.NLL_CAP if c["nll"] is None else min(float(c["nll"]), PL.NLL_CAP)) for c in b["cands"]] for b in bl],
            "lp": [[(c["lp"] or 0.0) / max(1, c["ntok"]) for c in b["cands"]] for b in bl],
            "fast": [[int(c["v"] != b["speed_base"]) for c in b["cands"]] for b in bl],
            "words": [TR.n_units(b["text"]) for b in bl],
            "texts": [b["text"] for b in bl],
            "chosen": [int(b["chosen"]) for b in bl]})
    return out


def load(pattern: str, win: str, lang: str, units: str, procs: int) -> list[T.TSStream]:
    W = TR.windows(win)
    files = sorted(glob.glob(pattern))
    with mp.get_context("fork").Pool(min(len(files), procs)) as pool:
        recs = [r for part in pool.map(_load, [(f, W, lang, units) for f in files]) for r in part]
    out = []
    for r in recs:
        st = T.make_stream(r)
        st.meeting, st.fast = r["meeting"], [np.asarray(x, dtype=np.float64) for x in r["fast"]]
        st.se = [float(T.sent_end(t)) for t in r["texts"]]
        out.append(st)
    return out


# ---------------------------------------------------------------- features
def phi(st: T.TSStream, n: int, b_s: float) -> np.ndarray:
    """C x 12 for block n with backlog-token seconds b_s."""
    W = max(1, st.words[n])
    x1 = st.dur[n] / 1000.0 / W
    x2 = st.fast[n]
    lw, se, first = np.log(W), st.se[n], float(n == 0)
    return np.stack([x1, x2, st.f[n], st.lp[n], x1 * b_s, x2 * b_s, x1 * se, x2 * se, x1 * lw, x2 * lw,
                     x1 * first, x2 * first], axis=1)


# ---------------------------------------------------------------- solving the targets on the fit set
def solve(streams, procs: int) -> dict:
    k = max(1, procs)
    T._G["chunks"] = [streams[i::k] for i in range(k)]
    T._G["pool"] = mp.get_context("fork").Pool(k)
    rule = np.zeros(9)
    for st in streams:
        rule += T.path_sums(st, st.chosen)[0]
    budget = T.lat_of(rule) - TR.BUDGET_DROP_S
    lam_pl = T.bisect(lambda x: T.lat_of(T.corpus(x, 0.0)), lambda v: v <= budget)
    target = T.TARGET * T.corpus(lam_pl, 0.0)[1]

    def kappa_of(lam):
        return T.bisect(lambda x: T.corpus(lam, x)[1], lambda v: v <= target * (1 + T.TOL_P))

    def h(lam):
        return T.lat_of(T.corpus(lam, kappa_of(lam)))

    if h(0.0) <= budget:
        lam_ts = 0.0
    elif h(T.CAP) > budget:
        lam_ts = T.CAP
    else:
        lo, hi = 0.0, T.CAP
        for _ in range(T.ITERS):
            mid = (lo + hi) / 2
            if h(mid) > budget:
                lo = mid
            else:
                hi = mid
        lam_ts = hi
    kap = kappa_of(lam_ts)
    s_ts = T.corpus(lam_ts, kap)
    feasible = bool(T.lat_of(s_ts) <= budget + T.TOL_L and s_ts[1] <= target * (1 + T.TOL_P) and lam_ts < T.CAP
                    and kap < T.CAP)
    return {"budget_s": float(budget), "lam_pl": float(lam_pl), "lam_ts": float(lam_ts), "kappa_ts": float(kap),
            "feasible": feasible}


def _targets(args):
    i, lam_pl, lam_ts, kap = args
    X, Qp, Qt = [], [], []
    for st in T._G["chunks"][i]:
        ss = st.path_states(st.chosen)
        st.kappa = 0.0
        bp = st.forward_backward(lam_pl)[3]
        qp = [st.conditional(n, ss[n], lam_pl, bp) for n in range(len(st.a))]
        st.kappa = kap
        bt = st.forward_backward(lam_ts)[3]
        qt = [st.conditional(n, ss[n], lam_ts, bt) for n in range(len(st.a))]
        st.kappa = 0.0
        for n in range(len(st.a)):
            X.append(phi(st, n, 0.5 * bin_s(ss[n])))
            Qp.append(qp[n])
            Qt.append(qt[n])
    return X, Qp, Qt


def pad(X, Q, C=8):
    n = len(X)
    Xp = np.zeros((n, C, X[0].shape[1]))
    Qp = np.zeros((n, C))
    M = np.zeros((n, C), dtype=bool)
    for i, (x, q) in enumerate(zip(X, Q)):
        c = len(q)
        Xp[i, :c], Qp[i, :c], M[i, :c] = x, q, True
    return Xp, Qp, M


def fit(X, Q, M, mu, sd) -> tuple[np.ndarray, dict]:
    from scipy.optimize import minimize

    Z = np.where(M[..., None], (X - mu) / sd, 0.0)

    def obj(th):
        s = np.where(M, Z @ th, -np.inf)
        m = s.max(1, keepdims=True)
        lse = np.log(np.exp(s - m).sum(1, keepdims=True)) + m
        lp = np.where(M, s - lse, 0.0)
        p = np.exp(lp) * M
        val = -(Q * lp).sum() / len(Q) + L2 * th @ th
        g = -(((Q - p * Q.sum(1, keepdims=True))[..., None] * Z).sum((0, 1))) / len(Q) + 2 * L2 * th
        return val, g

    r = minimize(obj, np.zeros(Z.shape[2]), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    th = r.x
    s = np.where(M, Z @ th, -np.inf)
    pi = np.exp(s - s.max(1, keepdims=True))
    pi = pi / pi.sum(1, keepdims=True)
    ent_q = -(Q * np.log(np.where(Q > 0, Q, 1.0))).sum() / len(Q)
    stats = {"xent": round(float(r.fun - L2 * th @ th), 4), "target_entropy": round(float(ent_q), 4),
             "argmax_agree": round(float((pi.argmax(1) == Q.argmax(1)).mean()), 4), "converged": bool(r.success),
             "theta": th.round(4).tolist()}
    return th, stats


# ---------------------------------------------------------------- replay
def best_of_2(pi: np.ndarray, f: np.ndarray) -> np.ndarray:
    lower = (f[None, :] < f[:, None]).astype(np.float64)     # lower[c, c'] = f(c') < f(c)
    tie = (f[None, :] == f[:, None]).astype(np.float64)
    np.fill_diagonal(tie, 0.0)
    return pi * (pi + 2 * lower @ pi + tie @ pi)


def policy(st, n, b_s, th, mu, sd):
    z = ((phi(st, n, b_s) - mu) / sd) @ th
    p = np.exp(z - z.max())
    return p / p.sum()


def replay_stream(st: T.TSStream, th, mu, sd, bo2: bool = True) -> np.ndarray:
    """exact expectations [sum l, -, pauses, mid pauses, sum D (ms), sum f, sum lp, blocks, words]."""
    S = st.S
    alpha = np.zeros(S)
    alpha[0] = 1.0
    acc = np.zeros(9)
    bins = np.array([bin_s(s) for s in range(S)])
    for n in range(len(st.a)):
        pis = np.zeros((NB, len(st.dur[n])))
        for b in range(NB):
            p = policy(st, n, 0.5 * b, th, mu, sd)
            pis[b] = best_of_2(p, st.f[n]) if bo2 else p
        P = alpha[:, None] * pis[bins]                            # S x C
        acc[0] += float((P * st.lat[n]).sum())
        pp = float((P * st.pz[n]).sum())
        acc[2] += pp
        acc[3] += pp if st.om[n] == 1.0 else 0.0
        pc = P.sum(0)
        acc[4] += float(pc @ st.dur[n])
        acc[5] += float(pc @ st.f[n])
        acc[6] += float(pc @ st.lp[n])
        if n + 1 < len(st.a):
            nxt = np.zeros(S)
            np.add.at(nxt, st.nxt[n].ravel(), P.ravel())
            alpha = nxt
    acc[7], acc[8] = len(st.a), sum(st.words)
    return acc


_R: dict = {}


def _replay(i):
    out = {}
    for st in _R["chunks"][i]:
        m = out.setdefault(st.meeting, {k: np.zeros(9) for k in ("pl", "ts", "pl_nobo2", "ts_nobo2", "q_pl", "q_ts")})
        for key, (th, bo2) in {"pl": (_R["th_pl"], True), "ts": (_R["th_ts"], True),
                               "pl_nobo2": (_R["th_pl"], False), "ts_nobo2": (_R["th_ts"], False)}.items():
            m[key] += replay_stream(st, th, _R["mu"], _R["sd"], bo2)
        m["q_pl"] += T._sums(st, _R["lam_pl"], 0.0)
        m["q_ts"] += T._sums(st, _R["lam_ts"], _R["kappa_ts"])
        st.kappa = 0.0
    return out


def boot_ratio(per: dict, a: str, b: str, idx: int = 3, rng=None) -> list[float]:
    ms = sorted(per)
    A = np.array([[per[m][a][idx], per[m][a][4]] for m in ms])
    B = np.array([[per[m][b][idx], per[m][b][4]] for m in ms])
    rng = rng or np.random.default_rng(20261116)
    r = []
    for _ in range(BOOT):
        i = rng.integers(0, len(ms), len(ms))
        r.append((B[i, 0].sum() / B[i, 1].sum()) / (A[i, 0].sum() / A[i, 1].sum()) - 1)
    return [round(float(np.percentile(r, 2.5)), 4), round(float(np.percentile(r, 97.5)), 4)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-cands", required=True)
    ap.add_argument("--fit-win", required=True)
    ap.add_argument("--eval-cands")
    ap.add_argument("--eval-win")
    ap.add_argument("--fold", choices=("even", "odd"), help="same set: fit on this fold of meetings, replay the other")
    ap.add_argument("--lang", choices=("eng", "cmn"), default="eng")
    ap.add_argument("--units", choices=("word", "char"), default="word")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    TR.LANG, TR.UNITS = (TR.LANG_ENG if a.lang == "eng" else TR.LANG_CMN), a.units
    allfit = load(a.fit_cands, a.fit_win, a.lang, a.units, a.procs)
    if a.fold:
        ms = sorted({st.meeting for st in allfit})
        keep = set(ms[0::2] if a.fold == "even" else ms[1::2])
        fit_s = [st for st in allfit if st.meeting in keep]
        ev_s = [st for st in allfit if st.meeting not in keep]
    else:
        fit_s = allfit
        ev_s = load(a.eval_cands, a.eval_win, a.lang, a.units, a.procs)
    print(json.dumps({"fit_streams": len(fit_s), "eval_streams": len(ev_s), "secs": round(time.time() - t0, 1)}),
          flush=True)
    sol = solve(fit_s, a.procs)
    print(json.dumps({**sol, "secs": round(time.time() - t0, 1)}), flush=True)
    parts = T._G["pool"].map(_targets, [(i, sol["lam_pl"], sol["lam_ts"], sol["kappa_ts"])
                                        for i in range(len(T._G["chunks"]))])
    T._G["pool"].close()
    X = [x for p in parts for x in p[0]]
    Xp, Qpl, M = pad(X, [q for p in parts for q in p[1]])
    _, Qts, _ = pad(X, [q for p in parts for q in p[2]])
    flat = Xp[M]
    mu, sd = flat.mean(0), flat.std(0) + 1e-9
    th_pl, st_pl = fit(Xp, Qpl, M, mu, sd)
    th_ts, st_ts = fit(Xp, Qts, M, mu, sd)
    print(json.dumps({"fit_pl": st_pl, "fit_ts": st_ts, "secs": round(time.time() - t0, 1)}), flush=True)
    k = max(1, a.procs)
    _R.update({"chunks": [ev_s[i::k] for i in range(k)], "th_pl": th_pl, "th_ts": th_ts, "mu": mu, "sd": sd,
               "lam_pl": sol["lam_pl"], "lam_ts": sol["lam_ts"], "kappa_ts": sol["kappa_ts"]})
    with mp.get_context("fork").Pool(k) as pool:
        res = pool.map(_replay, range(k))
    per: dict = {}
    for r in res:
        for m, d in r.items():
            tgt = per.setdefault(m, {kk: np.zeros(9) for kk in d})
            for kk, v in d.items():
                tgt[kk] += v
    tot = {kk: sum(per[m][kk] for m in per) for kk in next(iter(per.values()))}
    R = {kk: T.summary(v) for kk, v in tot.items()}
    for kk in R:
        R[kk].pop("weighted_pauses", None)
    rel_mid = R["ts"]["mid_pauses_per_min"] / R["pl"]["mid_pauses_per_min"] - 1
    go = {"1_mid_pauses_-15%": bool(R["ts"]["mid_pauses_per_min"] <= 0.85 * R["pl"]["mid_pauses_per_min"]),
          "2_latency_+0.10": bool(R["ts"]["latency_s"] <= R["pl"]["latency_s"] + 0.10),
          "3_f_-0.05": bool(R["ts"]["f_mean"] >= R["pl"]["f_mean"] - 0.05)}
    resj = {"fit_cands": a.fit_cands, "eval_cands": a.eval_cands or f"fold {a.fold} -> other", "lang": a.lang,
            "fit_streams": len(fit_s), "eval_streams": len(ev_s), "eval_meetings": len(per), **sol,
            "fit_pl": st_pl, "fit_ts": st_ts, "rows": R, "rel_mid_pauses_ts_vs_pl": round(float(rel_mid), 4),
            "rel_mid_ci95": boot_ratio(per, "pl", "ts", 3), "rel_all_ci95": boot_ratio(per, "pl", "ts", 2),
            "rel_mid_oracle_q": round(R["q_ts"]["mid_pauses_per_min"] / R["q_pl"]["mid_pauses_per_min"] - 1, 4),
            "per_meeting": {m: {kk: [round(float(x), 3) for x in v] for kk, v in d.items()} for m, d in per.items()},
            "go": go, "GO": bool(all(go.values())), "secs": round(time.time() - t0, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(resj, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in resj.items() if k != "per_meeting"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
