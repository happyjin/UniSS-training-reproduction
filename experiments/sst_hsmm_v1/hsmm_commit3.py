"""sst_hsmm_v1 prereg change 30 (variant B): HSC3 = HSC2's determinacy-constrained HSMM with a RELIABILITY CLOCK prior.
Emissions, observations and the determinacy constraint are hsmm_commit2's. The duration of the unstable phase is measured
on an operational clock: a position first appearing at read c0 has tau_0 = 0 and tau_k = sum_{s=1..k} r_{c0+s}, where
r_t = reliability(): mean 20 ms-frame suppression ratio E_enh / E_raw of read t's new 1 s (floor 0.25). Prior:
P(D = 0) = p0 (stable on first appearance); a switch at read k >= 1 has the mass of the continuous part of D_tau in
(tau_{k-1}, tau_k] (bins of width W up to TMAX, uniform inside a bin, plus a tail beyond TMAX). Clean audio (r = 1) gives
HSC2's read clock; noise slows the clock, so the prior asks for more evidence. Unsupervised EM (no labels) on the h1
N-policy rollouts with r computed from the e1 views.
    fit:  <qwen3omni python> -m experiments.sst_hsmm_v1.hsmm_commit3 fit --roots 'eval_outputs/sst_hsmm_v1/h1/roll_*' --out <json>
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1 import hsmm_commit2 as H2  # noqa: E402

W, TMAX = 0.25, 10.0
NB = int(TMAX / W)              # continuous bins 0..NB-1, tail bin NB
RMIN = 0.25


def reliability(raw_chunk, enh_chunk) -> float:
    """mean over 20 ms frames of the suppression ratio E_enh / E_raw (frames below -60 dBFS raw energy count as 1),
    floored at RMIN (prereg change 30 amendment: the whole-chunk energy ratio was insensitive to noise)"""
    raw = np.asarray(raw_chunk, np.float64); enh = np.asarray(enh_chunk, np.float64)
    F = 320; n = min(len(raw), len(enh)) // F
    if n == 0:
        return 1.0
    er = (raw[: n * F].reshape(n, F) ** 2).mean(1)
    ee = (enh[: n * F].reshape(n, F) ** 2).mean(1)
    s = np.where(er < 1e-6, 1.0, np.clip(ee / (er + 1e-12), 0.0, 1.0))
    return min(1.0, max(RMIN, float(s.mean())))


class Params3(H2.Params):
    def __init__(self, d):
        dd = dict(d)
        dd["dur"] = [1.0] * (H2.DMAX + 1)            # unused by HSC3; keeps H2.Params happy
        super().__init__(dd)
        self.p0 = float(d["p0"])
        self.q = np.array(d["q"], float)               # (NB+1,), continuous bins + tail, sums to 1
        self.cq = np.concatenate([[0.0], np.cumsum(self.q[:NB])])

    def G(self, tau):
        """CDF of the continuous part at tau (tail mass not reached below TMAX)"""
        if tau >= TMAX:
            return float(self.cq[NB])
        b = int(tau / W)
        return float(self.cq[b] + self.q[b] * (tau - b * W) / W)


def _taus(rs):
    """rs[k] = reliability of the read at sequence index k (k = 0: first appearance, its r is not used)"""
    t = [0.0]
    for r in rs[1:]:
        t.append(t[-1] + r)
    return t


def _hyp3(seq, rs, P: Params3):
    T = len(seq)
    rU = np.array([P.lrest(o, "U") for o in seq]); rS = np.array([P.lrest(o, "S") for o in seq])
    aU = np.array([P.la(o, "U") for o in seq]); aS = np.array([P.la(o, "S") for o in seq])
    cU = np.concatenate([[0.0], np.cumsum(rU + aU)])
    cRS = np.concatenate([[0.0], np.cumsum(rS)]); cAS = np.concatenate([[0.0], np.cumsum(aS)])
    tau = _taus(rs)
    hyps = []
    for d in range(T):
        if d == 0:
            m = P.p0
        else:
            m = (1 - P.p0) * max(P.G(tau[d]) - P.G(tau[d - 1]), 0.0)
        if m <= 0:
            continue
        w = math.log(m) + cU[d] + rS[d] + aU[d] + (cRS[T] - cRS[d + 1]) + (cAS[T] - cAS[d + 1])
        hyps.append((d, w))
    sv = (1 - P.p0) * (1 - P.G(tau[T - 1]))
    cens = (math.log(sv) if sv > 0 else -1e30) + cU[T]
    return hyps, cens, tau


def posterior_stable(seq, rs, P: Params3) -> float:
    if not seq:
        return 0.0
    hyps, cens, _ = _hyp3(seq, rs, P)
    lw = [w for _, w in hyps] + [cens]
    m = max(lw)
    z = sum(math.exp(w - m) for w in lw)
    return sum(math.exp(w - m) for _, w in hyps) / z


def _spread(lo, hi, q, out, mass):
    """add `mass` to bin counts over (lo, hi] proportionally to q x overlap (hi = inf -> includes the tail)"""
    bins, wts = [], []
    for b in range(NB):
        a0, a1 = b * W, (b + 1) * W
        ov = min(a1, hi) - max(a0, lo)
        if ov > 0:
            bins.append(b); wts.append(q[b] * ov / W)
    if hi == math.inf:
        bins.append(NB); wts.append(q[NB])
    s = sum(wts)
    if s <= 0:
        return
    for b, w in zip(bins, wts):
        out[b] += mass * w / s


def views_reliability(items_file, views_dir):
    """{item id: {read t_ms: r}} from the raw audio and the wave15-style enhanced-view npy (prefix per read)"""
    import soundfile as sf
    idx = {r["id"]: r for r in map(json.loads, open(Path(views_dir) / "index.jsonl", encoding="utf-8"))}
    out = {}
    for it in map(json.loads, open(items_file, encoding="utf-8")):
        r = idx.get(it["id"])
        if r is None:
            continue
        x, _ = sf.read(it["audio"], dtype="float32", always_2d=True)
        x = x.mean(axis=1)[: int(it["dur_ms"]) * 16]
        v = np.load(Path(views_dir) / r["file"])
        o, rel = 0, {}
        for t in r["reads"]:
            pref = v[o: o + t * 16]; o += t * 16
            rel[t] = reliability(x[(t - 1000) * 16: t * 16], pref[-16000:])
        tf = len(x) // 16                                   # final read: last (partial) chunk, enhanced view absent
        rel[tf] = rel[max(r["reads"])] if r["reads"] else 1.0
        out[it["id"]] = rel
    return out


def sequences_from_blocks(path, zh, rel):
    """as hsmm_commit2.sequences_from_blocks, with the per-read reliability attached: list of (seq, rs)"""
    from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units
    seqs = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        R = rel.get(r["id"])
        if R is None:
            continue
        n, prev, H = 0, None, {}
        for x in r["drafts"][:-1]:
            if "lp0" not in x:
                continue
            d0 = units(x["draft"], zh); d1 = units(x["draft_v1"], zh); ds = units(x["draft_sap"], zh)
            lp0 = [0.0] * n + x["lp0"]; lps = [0.0] * n + x["lps"]
            rt = R.get(x["t"], 1.0)
            for i, o in H2.observations(d0, prev, d1, ds, lp0, lps, n).items():
                H.setdefault(i, ([], []))
                H[i][0].append(o); H[i][1].append(rt)
            c = x["cv_new"]
            for i in range(n, n + c):
                if i in H:
                    seqs.append(H.pop(i))
            n += c
            prev = d0
        fx = r["drafts"][-1]
        dfin = units(fx["draft"], zh)
        rf = R.get(fx["t"], 1.0)
        for i, o in H2.final_observations(dfin, prev, n).items():
            if i in H:
                H[i][0].append(o); H[i][1].append(rf)
        for i in list(H):
            seqs.append(H.pop(i))
    return seqs


def em_fit(seqs, em_init, iters=30):
    p0 = 0.15
    q = np.full(NB + 1, 1.0 / (NB + 1))
    em = em_init
    hist = []
    for it in range(iters):
        d = {"p0": p0, "q": q.tolist(), "em": em}
        P = Params3(d)
        c0 = 1.0; cc = 1.0                                     # Beta(1,1) on p0
        qc = np.ones(NB + 1)                                   # Dirichlet(+1)
        ec = {st: {k: np.ones(H2.CATS[k]) for k in H2.KEYS} for st in ("U", "S")}
        for st in ("U", "S"):
            for k in H2.KEYS:
                ec[st][k][H2.NA[k]] = 0.0
        ll = 0.0
        for seq, rs in seqs:
            T = len(seq)
            hyps, cens, tau = _hyp3(seq, rs, P)
            lw = np.array([w for _, w in hyps] + [cens])
            m = lw.max(); wv = np.exp(lw - m); z = wv.sum(); ll += m + math.log(z); wv /= z
            pS = np.zeros(T)
            for (dd, _), wd in zip(hyps, wv[: len(hyps)]):
                pS[dd:] += wd
                if dd == 0:
                    c0 += wd
                else:
                    cc += wd
                    _spread(tau[dd - 1], tau[dd], P.q, qc, wd)
            cc += wv[-1]
            _spread(tau[T - 1], math.inf, P.q, qc, wv[-1])
            for t, o in enumerate(seq):
                for j, k in enumerate(H2.KEYS):
                    if o[j] == H2.NA[k]:
                        continue
                    if j == 0:
                        ec["U"]["a"][o[0]] += 1.0 - (pS[t - 1] if t > 0 else 0.0)
                    else:
                        ec["S"][k][o[j]] += pS[t]
                        ec["U"][k][o[j]] += 1.0 - pS[t]
        p0 = c0 / (c0 + cc)
        q = qc / qc.sum()
        em = {st: {k: ((v + 1e-9) / (v + 1e-9).sum()).tolist() for k, v in ec[st].items()} for st in ("U", "S")}
        hist.append(ll / max(len(seqs), 1))
        print(f"  em iter {it} mean loglik {hist[-1]:.4f} p0 {p0:.3f}", flush=True)
    return {"p0": p0, "q": q.tolist(), "em": em}, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("fit",))
    ap.add_argument("--roots", required=True)
    ap.add_argument("--init", default=str(REPO / "eval_outputs/sst_hsmm_v1/h3/params.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit(f"拒绝覆盖: {a.out}")
    init = json.load(open(a.init))["dirs"]
    E1 = REPO / "eval_outputs/sst_hsmm_v1/e1"
    res = {"W": W, "TMAX": TMAX, "RMIN": RMIN, "dirs": {}}
    for d in ("en2zh", "zh2en"):
        rel = views_reliability(E1 / "data" / f"items_{d}.jsonl", E1 / "views" / d)
        rs_all = [v for R in rel.values() for v in R.values()]
        seqs = []
        for f in sorted(glob.glob(f"{a.roots}/blocks_{d}_w1.jsonl")):
            seqs += sequences_from_blocks(f, d.endswith("zh"), rel)
        print(d, "sequences", len(seqs), "reliability mean", float(np.mean(rs_all)), "p10", float(np.percentile(rs_all, 10)), flush=True)
        params, hist = em_fit(seqs, init[d]["params"]["em"])
        res["dirs"][d] = {"params": params, "loglik": hist, "n_seq": len(seqs),
                          "rel_mean": float(np.mean(rs_all)), "rel_p10": float(np.percentile(rs_all, 10))}
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
