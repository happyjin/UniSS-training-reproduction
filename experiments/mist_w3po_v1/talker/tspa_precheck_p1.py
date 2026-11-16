"""TS-PA (two-sided playback alignment) CPU precheck P1 (reports/mist_w3po_v1/tspa_precheck_prereg_v1.zh-CN.md).

The PL-EM playback lattice (playback_lattice.Stream, talker_rows.stream_of, unchanged) plus a starvation potential:
for blocks n < N-1, G_n(s, c) = a_{n+1} - (a_n + 80 s + D_c) ms is the silence after block n when it starts with
backlog state s and is spoken as candidate c; P_n = 1[G_n >= 500 ms] is one pause of the paper's metric
(talker_block_pauses); w_n = 1 inside a sentence (block text does not end in .?!。？！), 0.25 after one.

    psi_n(s, c) = feat_c - lam * l_n(s, c) - kappa * w_n * P_n(s, c)

PL-EM reference: kappa = 0, lam* for the budget b = rule path latency - 0.3 s. TS-PA: the same budget plus
E_q[sum w P] <= 0.75 * E_{q_PL}[sum w P]; (lam, kappa) by nested bisection (kappa*(lam) the smallest kappa meeting the
pause bound; lam the smallest with latency(lam, kappa*(lam)) <= b). Corpus sums run in forked workers.

    python -m experiments.mist_w3po_v1.talker.tspa_precheck_p1 --cands '<dir>/cands_*.jsonl' --win <windows> \\
        [--lang cmn --units char] --procs 64 --out <json>     (refuses to overwrite)
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

PAUSE_MS = 500.0
W_AFTER = 0.25
SENT_END = set(".?!。？！")
TARGET = 0.75
CAP = 200.0
ITERS = 40
TOL_L = 2e-3
TOL_P = 1e-3


def sent_end(text: str) -> bool:
    t = (text or "").rstrip().rstrip("\"'”’)]）】」")
    return bool(t) and t[-1] in SENT_END


class TSStream(PL.Stream):
    """PL.Stream + pause indicators pz[n] (S x C, zeros for the last block) and weights om[n]."""

    kappa = 0.0

    def setup_gaps(self, texts: list[str]) -> None:
        sv = np.arange(self.S, dtype=np.float64) * self.delta
        self.om, self.pz, self.gap = [], [], []
        for n in range(len(self.a)):
            self.om.append(W_AFTER if sent_end(texts[n]) else 1.0)
            if n + 1 < len(self.a):
                g = self.a[n + 1] - (self.a[n] + sv[:, None] + self.dur[n][None, :])
                self.gap.append(g)
                self.pz.append((g >= PAUSE_MS).astype(np.float64))
            else:
                self.gap.append(None)
                self.pz.append(np.zeros((self.S, len(self.dur[n]))))

    def psi(self, n: int, lam: float) -> np.ndarray:
        return self.feat[n][None, :] - lam * self.lat[n] - self.kappa * self.om[n] * self.pz[n]


def make_stream(rec: dict) -> TSStream:
    st = TSStream(rec["a"], rec["dur"], rec["feat"], rec["words"])
    st.setup_gaps(rec["texts"])
    st.f = [np.asarray(x) for x in rec["f"]]
    st.lp = [np.asarray(x) for x in rec["lp"]]
    st.chosen = rec["chosen"]
    return st


def _load_file(args) -> list[dict]:
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
            "id": r["id"],
            "a": [TR.availability(w, b["write_ms"]) for b in bl],
            "dur": [[len(c["codes"]) * 20 for c in b["cands"]] for b in bl],
            "feat": [[TR.feat(c) for c in b["cands"]] for b in bl],
            "f": [[-(PL.NLL_CAP if c["nll"] is None else min(float(c["nll"]), PL.NLL_CAP)) for c in b["cands"]] for b in bl],
            "lp": [[(c["lp"] or 0.0) / max(1, c["ntok"]) for c in b["cands"]] for b in bl],
            "words": [TR.n_units(b["text"]) for b in bl],
            "texts": [b["text"] for b in bl],
            "chosen": [int(b["chosen"]) for b in bl]})
    return out


# ---------------------------------------------------------------- forked corpus sums
_G: dict = {}


def _sums(st: TSStream, lam: float, kappa: float) -> np.ndarray:
    """[E sum l, E sum w P, E sum P, E sum P_mid, E sum D (ms), E sum f, E sum lp, blocks, words]"""
    st.kappa = kappa
    _, marg, el, _ = st.forward_backward(lam)
    wp = p = pm = d = f = lp = 0.0
    for n, q in enumerate(marg):
        qp = float((q * st.pz[n]).sum())
        p += qp
        wp += st.om[n] * qp
        pm += qp if st.om[n] == 1.0 else 0.0
        qc = q.sum(0)
        d += float(qc @ st.dur[n])
        f += float(qc @ st.f[n])
        lp += float(qc @ st.lp[n])
    return np.array([el, wp, p, pm, d, f, lp, len(st.a), sum(st.words)])


def _work(args):
    i, lam, kappa = args
    return sum((_sums(st, lam, kappa) for st in _G["chunks"][i]), np.zeros(9))


def corpus(lam: float, kappa: float) -> np.ndarray:
    parts = _G["pool"].map(_work, [(i, lam, kappa) for i in range(len(_G["chunks"]))])
    return np.sum(parts, axis=0)


def lat_of(s: np.ndarray) -> float:
    return s[0] / s[8]


def bisect(f, ok, hi: float = CAP, iters: int = ITERS) -> float:
    if ok(f(0.0)):
        return 0.0
    if not ok(f(hi)):
        return hi
    lo = 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if ok(f(mid)):
            hi = mid
        else:
            lo = mid
    return hi


def summary(s: np.ndarray) -> dict:
    mins = s[4] / 60000.0
    return {"latency_s": round(lat_of(s), 4), "pauses_per_min": round(s[2] / mins, 3),
            "mid_pauses_per_min": round(s[3] / mins, 3), "weighted_pauses": round(s[1], 2),
            "f_mean": round(s[5] / s[7], 4), "lpbar_mean": round(s[6] / s[7], 4),
            "rate_units_per_s": round(s[8] / (s[4] / 1000.0), 4), "speech_min": round(mins, 2)}


# ---------------------------------------------------------------- paths
def path_sums(st: TSStream, path: list[int]) -> tuple[np.ndarray, int, int]:
    ss = st.path_states(path)
    el = p = pm = d = f = lp = 0.0
    unfill = 0
    for n, (s, c) in enumerate(zip(ss, path)):
        el += st.lat[n][s, c]
        d += st.dur[n][c]
        f += st.f[n][c]
        lp += st.lp[n][c]
        if st.pz[n][s, c] > 0:
            p += 1
            pm += st.om[n] == 1.0
            unfill += bool(st.pz[n][s].min() > 0)
    return np.array([el, 0.0, p, pm, d, f, lp, len(st.a), sum(st.words)]), unfill, int(p)


def _paths(args):
    i, lam_pl, lam_ts, kappa_ts = args
    acc = {k: np.zeros(9) for k in ("rule", "vit_pl", "vit_ts")}
    unf = tot = 0
    for st in _G["chunks"][i]:
        acc["rule"] += path_sums(st, st.chosen)[0]
        st.kappa = 0.0
        x, u, t = path_sums(st, st.viterbi(lam_pl))
        acc["vit_pl"] += x
        unf += u
        tot += t
        st.kappa = kappa_ts
        acc["vit_ts"] += path_sums(st, st.viterbi(lam_ts))[0]
        st.kappa = 0.0
    return acc, unf, tot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cands", required=True)
    ap.add_argument("--win", required=True)
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
    W = TR.windows(a.win)
    files = sorted(glob.glob(a.cands))
    with mp.get_context("fork").Pool(min(len(files), a.procs)) as pool:
        recs = [r for part in pool.map(_load_file, [(f, W, a.lang, a.units) for f in files]) for r in part]
    del W
    streams = [make_stream(r) for r in recs]
    del recs
    print(json.dumps({"loaded_streams": len(streams), "blocks": sum(len(s.a) for s in streams),
                      "secs": round(time.time() - t0, 1)}), flush=True)
    k = max(1, a.procs)
    _G["chunks"] = [streams[i::k] for i in range(k)]
    _G["pool"] = mp.get_context("fork").Pool(k)

    rule = np.zeros(9)
    for st in streams:
        rule += path_sums(st, st.chosen)[0]
    budget = lat_of(rule) - TR.BUDGET_DROP_S
    # PL-EM reference
    lam_pl = bisect(lambda x: lat_of(corpus(x, 0.0)), lambda v: v <= budget)
    s_pl = corpus(lam_pl, 0.0)
    target = TARGET * s_pl[1]
    print(json.dumps({"budget_s": budget, "lam_pl": lam_pl, "pl": summary(s_pl), "target_wp": target,
                      "secs": round(time.time() - t0, 1)}), flush=True)
    hist = []

    def kappa_of(lam):
        return bisect(lambda x: corpus(lam, x)[1], lambda v: v <= target * (1 + TOL_P))

    def h(lam):
        kap = kappa_of(lam)
        s = corpus(lam, kap)
        hist.append((round(lam, 6), round(kap, 6), round(lat_of(s), 5), round(float(s[1]), 3)))
        print(json.dumps({"lam": lam, "kappa": kap, "latency_s": lat_of(s), "wp": float(s[1]),
                          "secs": round(time.time() - t0, 1)}), flush=True)
        return lat_of(s), kap

    if h(0.0)[0] <= budget:
        lam_ts = 0.0
    elif h(CAP)[0] > budget:
        lam_ts = CAP
    else:
        lo, hi = 0.0, CAP
        for _ in range(ITERS):
            mid = (lo + hi) / 2
            if h(mid)[0] > budget:
                lo = mid
            else:
                hi = mid
        lam_ts = hi
    kap_ts = kappa_of(lam_ts)
    s_ts = corpus(lam_ts, kap_ts)
    feasible = (lat_of(s_ts) <= budget + TOL_L and s_ts[1] <= target * (1 + TOL_P) and lam_ts < CAP and kap_ts < CAP)
    parts = _G["pool"].map(_paths, [(i, lam_pl, lam_ts, kap_ts) for i in range(k)])
    _G["pool"].close()
    acc = {key: sum(p[0][key] for p in parts) for key in ("rule", "vit_pl", "vit_ts")}
    unf, tot = sum(p[1] for p in parts), sum(p[2] for p in parts)
    R = {key: summary(v) for key, v in acc.items()}
    R.update({"q_pl": summary(s_pl), "q_ts": summary(s_ts)})
    go = {
        "1_feasible": bool(feasible),
        "2_vit_mid_pauses": bool(R["vit_ts"]["mid_pauses_per_min"] <= TARGET * R["vit_pl"]["mid_pauses_per_min"]),
        "3_vit_latency": bool(R["vit_ts"]["latency_s"] <= R["vit_pl"]["latency_s"] + 0.10),
        "4_f": bool(R["q_ts"]["f_mean"] >= R["q_pl"]["f_mean"] - 0.05)}
    res = {"cands": a.cands, "win": a.win, "lang": a.lang, "units": a.units, "streams": len(streams),
           "blocks": int(acc["rule"][7]), "budget_s": round(budget, 4), "lam_pl": float(lam_pl), "lam_ts": float(lam_ts),
           "kappa_ts": float(kap_ts), "target_weighted_pauses": round(target, 3), "feasible": bool(feasible),
           "unfillable_share_on_pl_viterbi": round(float(unf / max(1, tot)), 4), "pl_viterbi_pauses": int(tot),
           "rows": R, "go": go, "GO": bool(all(go.values())), "dual_history": hist,
           "secs": round(time.time() - t0, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "dual_history"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
