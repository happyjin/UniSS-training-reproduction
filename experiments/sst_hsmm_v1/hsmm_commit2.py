"""sst_hsmm_v1 prereg change 29: HSC2 = HSC (change 28) with the generative model corrected (hsmm_commit.py not modified).
Per position i a left-to-right two-state HSMM (U for D reads, then S absorbing; explicit pmf D in {0..DMAX}). Per read t:
  a   prefix agreement with read t-1's draft, 0 / 1 / 2 = NA. DETERMINACY CONSTRAINT: a_t is emitted by the state at t-1:
      if that state is S (the prefix was already determined), a_t = 1 (P(a_t = 0) = EPS); from U, a_t ~ table A_U.
  v   cross-view agreement 0 / 1 (2 = NA at the final read)
  sq  SAP support and SAP confidence merged (no double counting): 0 = not supported, 1/2/3 = supported with SAP min token
      prob >= 0.9 / >= 0.5 / else (4 = NA at the final read)
  qN  N's min token prob bin 0/1/2 (3 = NA at the final read)
NA observations add no likelihood. Training sequences (N-policy rollouts, no labels) end at the commit or include the FINAL
read's agreement observation (a only), so the model sees whether a position was revised at the end. Unsupervised EM as
change 28; online posterior P(S at the current read | observations).
    fit:  <qwen3omni python> -m experiments.sst_hsmm_v1.hsmm_commit2 fit --roots 'eval_outputs/sst_hsmm_v1/h1/roll_*' --out <params.json>
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DMAX = 10
KEYS = ("a", "v", "sq", "qN")
CATS = {"a": 3, "v": 3, "sq": 5, "qN": 4}
NA = {"a": 2, "v": 2, "sq": 4, "qN": 3}
EPS = 1e-3


def lcp(a, b) -> int:
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k


def qbin(lp):
    p = math.exp(lp)
    return 0 if p >= 0.9 else (1 if p >= 0.5 else 2)


def observations(d0, prev, d1, ds, lp0, lps, n):
    """per uncommitted position i (n <= i < len(d0)) -> (a, v, sq, qN); lp0 / lps are full-length per-unit lists"""
    agreed = lcp(d0, prev) if prev is not None else 0
    cv, sv = lcp(d0, d1), lcp(d0, ds)
    out = {}
    for i in range(n, len(d0)):
        a = 2 if (prev is None or len(prev) <= i) else (1 if i < agreed else 0)
        sq = (1 + qbin(lps[i])) if (i < sv and i < len(lps)) else 0
        out[i] = (a, int(i < cv), sq, qbin(lp0[i]))
    return out


def final_observations(dfin, prev, n):
    """the final read: only the agreement of the final draft with the last non-final draft (others NA)"""
    agreed = lcp(dfin, prev) if prev is not None else 0
    out = {}
    for i in range(n, len(dfin)):
        a = 2 if (prev is None or len(prev) <= i) else (1 if i < agreed else 0)
        out[i] = (a, 2, 4, 3)
    return out


class Params:
    def __init__(self, d):
        self.dur = np.array(d["dur"], float)
        self.em = {st: {k: np.array(d["em"][st][k], float) for k in KEYS} for st in ("U", "S")}
        self.ldur = np.log(self.dur)
        self.surv = np.array([self.dur[t:].sum() for t in range(DMAX + 2)] + [0.0])   # P(D >= t)
        self.lem = {st: {k: np.log(v) for k, v in self.em[st].items()} for st in ("U", "S")}
        self.la_S = {0: math.log(EPS), 1: math.log(1 - EPS)}

    def lrest(self, o, st):
        """log-lik of v, sq, qN under the state at this read (NA skipped)"""
        return sum(self.lem[st][k][o[j]] for j, k in enumerate(KEYS) if j > 0 and o[j] != NA[k])

    def la(self, o, prev_st):
        """log-lik of a under the state at the previous read (determinacy constraint for S)"""
        if o[0] == 2:
            return 0.0
        return self.la_S[o[0]] if prev_st == "S" else float(self.lem["U"]["a"][o[0]])


def _hyp(seq, P: Params):
    """switch at d = 0..min(T-1, DMAX): U for t < d, S for t >= d; a_t uses the state at t-1 (t = 0: U). Plus the censored
    hypothesis 'U for all T reads' when T <= DMAX."""
    T = len(seq)
    rU = np.array([P.lrest(o, "U") for o in seq]); rS = np.array([P.lrest(o, "S") for o in seq])
    aU = np.array([P.la(o, "U") for o in seq]); aS = np.array([P.la(o, "S") for o in seq])
    cU = np.concatenate([[0.0], np.cumsum(rU + aU)])
    cRS = np.concatenate([[0.0], np.cumsum(rS)]); cAS = np.concatenate([[0.0], np.cumsum(aS)])
    hyps = []
    for d in range(0, min(T - 1, DMAX) + 1):
        w = P.ldur[d] + cU[d] + rS[d] + aU[d] + (cRS[T] - cRS[d + 1]) + (cAS[T] - cAS[d + 1])
        hyps.append((d, w))
    cens = None
    if T <= DMAX:
        sv = P.surv[T]
        cens = (math.log(sv) if sv > 0 else -1e30) + cU[T]
    return hyps, cens


def posterior_stable(seq, P: Params) -> float:
    """P(state at the last observation = S | seq)"""
    if not seq:
        return 0.0
    hyps, cens = _hyp(seq, P)
    lw = [w for _, w in hyps] + ([cens] if cens is not None else [])
    m = max(lw)
    z = sum(math.exp(w - m) for w in lw)
    return sum(math.exp(w - m) for _, w in hyps) / z


def sequences_from_blocks(path, zh):
    """per-position observation sequences of an N-policy rollout (ending at the commit, or with the final read's agreement
    observation appended); diagnostic label 1(d0[:i+1] == final[:i+1]) at the last non-final observation."""
    from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units
    seqs, labels = [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        fin = units(r["translation"], zh)
        n, prev, H, last = 0, None, {}, {}
        for x in r["drafts"][:-1]:
            if "lp0" not in x:
                continue
            d0 = units(x["draft"], zh); d1 = units(x["draft_v1"], zh); ds = units(x["draft_sap"], zh)
            lp0 = [0.0] * n + x["lp0"]; lps = [0.0] * n + x["lps"]
            for i, o in observations(d0, prev, d1, ds, lp0, lps, n).items():
                H.setdefault(i, []).append(o)
                last[i] = int(d0[: i + 1] == fin[: i + 1])
            c = x["cv_new"]
            for i in range(n, n + c):
                if i in H:
                    seqs.append(H.pop(i)); labels.append(last.pop(i))
            n += c
            prev = d0
        dfin = units(r["drafts"][-1]["draft"], zh)
        for i, o in final_observations(dfin, prev, n).items():
            if i in H:
                H[i].append(o)
        for i in list(H):
            seqs.append(H.pop(i)); labels.append(last.pop(i))
    return seqs, labels


def init_params():
    dur = np.full(DMAX + 1, 1.0 / (DMAX + 1))
    em = {"S": {"a": [0.5, 0.5, 1e-9], "v": [0.2, 0.8, 1e-9], "sq": [0.3, 0.4, 0.2, 0.1, 1e-9], "qN": [0.6, 0.3, 0.1, 1e-9]},
          "U": {"a": [0.5, 0.5, 1e-9], "v": [0.5, 0.5, 1e-9], "sq": [0.5, 0.2, 0.15, 0.15, 1e-9], "qN": [0.3, 0.3, 0.4, 1e-9]}}
    return {"dur": dur.tolist(), "em": em}


def em_fit(seqs, iters=30):
    d = init_params()
    hist = []
    for it in range(iters):
        P = Params(d)
        dc = np.ones(DMAX + 1)
        ec = {st: {k: np.ones(CATS[k]) for k in KEYS} for st in ("U", "S")}
        for st in ("U", "S"):
            for k in KEYS:
                ec[st][k][NA[k]] = 0.0
        ll = 0.0
        for seq in seqs:
            T = len(seq)
            hyps, cens = _hyp(seq, P)
            lw = np.array([w for _, w in hyps] + ([cens] if cens is not None else []))
            m = lw.max(); w = np.exp(lw - m); z = w.sum(); ll += m + math.log(z); w /= z
            pS = np.zeros(T)
            for (dd, _), wd in zip(hyps, w[: len(hyps)]):
                dc[dd] += wd
                pS[dd:] += wd
            if cens is not None:
                wc = w[-1]; tail = P.dur[T:]
                if tail.sum() > 0:
                    dc[T:] += wc * tail / tail.sum()
            for t, o in enumerate(seq):
                for j, k in enumerate(KEYS):
                    if o[j] == NA[k]:
                        continue
                    if j == 0:                                  # a: only the U table is free (emitted by the state at t-1)
                        ec["U"]["a"][o[0]] += 1.0 - (pS[t - 1] if t > 0 else 0.0)
                    else:
                        ec["S"][k][o[j]] += pS[t]
                        ec["U"][k][o[j]] += 1.0 - pS[t]
        d = {"dur": (dc / dc.sum()).tolist(),
             "em": {st: {k: ((v + 1e-9) / (v + 1e-9).sum()).tolist() for k, v in ec[st].items()} for st in ("U", "S")}}
        hist.append(ll / max(len(seqs), 1))
        print(f"  em iter {it} mean loglik {hist[-1]:.4f}", flush=True)
    return d, hist


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    o = np.argsort(p, kind="mergesort"); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("fit",))
    ap.add_argument("--roots", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit(f"拒绝覆盖: {a.out}")
    res = {"dmax": DMAX, "keys": KEYS, "eps": EPS, "dirs": {}}
    for d in ("en2zh", "zh2en"):
        seqs, labels = [], []
        for f in sorted(glob.glob(f"{a.roots}/blocks_{d}_w1.jsonl")):
            s, l = sequences_from_blocks(f, d.endswith("zh"))
            seqs += s; labels += l
        print(d, "sequences", len(seqs), "mean length", float(np.mean([len(s) for s in seqs])), flush=True)
        params, hist = em_fit(seqs)
        P = Params(params)
        post = [posterior_stable(s[:-1] if (s and s[-1][1] == 2) else s, P) for s in seqs]   # posterior before the final read
        res["dirs"][d] = {"params": params, "loglik": hist, "n_seq": len(seqs),
                          "diag_auc_vs_final_prefix": auc(labels, post), "diag_pos_rate": float(np.mean(labels))}
        print(d, "diagnostic AUC (not used in training):", res["dirs"][d]["diag_auc_vs_final_prefix"], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
