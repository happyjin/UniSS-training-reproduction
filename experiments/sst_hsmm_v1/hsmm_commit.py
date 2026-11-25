"""sst_hsmm_v1 prereg change 28: HSC = HSMM stability change-point commit.

Every draft position i (prefix d[:i+1]) is a left-to-right two-state hidden semi-Markov chain over the reads after its first
appearance: unstable U for D reads, then stable S (absorbing); D in {0..DMAX} has an explicit non-parametric duration pmf
(per direction). Each read emits a conditionally independent observation tuple (per state / direction tables):
  a  prefix agreement with the previous read's draft: 0 / 1 / 2 = NA (no previous draft covering i)
  v  cross-view agreement, i < LCP(d0, d1): 0 / 1
  s  SAP support, i < LCP(d0, d_sap): 0 / 1
  qN N's minimum token probability over unit i: 0 (>= 0.9) / 1 (>= 0.5) / 2 (else)
  qS SAP's, same bins, 3 = NA when s = 0
A position's sequence ends at its commit or at the final read (right-censored; censoring depends on observed data only).
Unsupervised EM over N-policy rollouts (no labels); online forward filtering gives P(S | observations so far).

    fit:  <qwen3omni python> -m experiments.sst_hsmm_v1.hsmm_commit fit --roots 'eval_outputs/sst_hsmm_v1/h1/roll_*' --out <params.json>
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
CATS = {"a": 3, "v": 2, "s": 2, "qN": 3, "qS": 4}
KEYS = ("a", "v", "s", "qN", "qS")


def lcp(a, b) -> int:
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k


def qbin(lp):
    p = math.exp(lp)
    return 0 if p >= 0.9 else (1 if p >= 0.5 else 2)


def observations(d0, prev, d1, ds, lp0, lps, n):
    """per uncommitted position i (n <= i < len(d0)) -> tuple (a, v, s, qN, qS); lp0 / lps are full-length per-unit lists"""
    agreed = lcp(d0, prev) if prev is not None else 0
    cv, sv = lcp(d0, d1), lcp(d0, ds)
    out = {}
    for i in range(n, len(d0)):
        if prev is None or len(prev) <= i:
            a = 2
        else:
            a = 1 if i < agreed else 0
        s = int(i < sv)
        out[i] = (a, int(i < cv), s, qbin(lp0[i]), qbin(lps[i]) if s and i < len(lps) else 3)
    return out


class Params:
    def __init__(self, d):
        self.dur = np.array(d["dur"], float)                          # (DMAX+1,)
        self.em = {st: {k: np.array(d["em"][st][k], float) for k in KEYS} for st in ("U", "S")}
        self.ldur = np.log(self.dur)
        self.surv = np.array([self.dur[t:].sum() for t in range(DMAX + 2)] + [0.0])   # P(D >= t)
        self.lem = {st: {k: np.log(v) for k, v in self.em[st].items()} for st in ("U", "S")}

    def loglik(self, o, st):
        return sum(self.lem[st][k][o[j]] for j, k in enumerate(KEYS))


def _hyp(seq, P: Params):
    """log weights of the hypotheses for a sequence of T observations: switch at d = 0..min(T-1, DMAX) (U before d, S from d),
    and 'still unstable after T reads' (D >= T, only if T <= DMAX). Returns (list of (d, logw), logw_cens or None)."""
    T = len(seq)
    lU = np.array([P.loglik(o, "U") for o in seq])
    lS = np.array([P.loglik(o, "S") for o in seq])
    cU = np.concatenate([[0.0], np.cumsum(lU)])
    cS = np.concatenate([[0.0], np.cumsum(lS)])
    hyps = []
    for d in range(0, min(T - 1, DMAX) + 1):
        hyps.append((d, P.ldur[d] + cU[d] + (cS[T] - cS[d])))
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
    s = sum(math.exp(w - m) for _, w in hyps)       # every switch hypothesis d <= T-1 means S at the last read
    return s / z


# --------------------------------------------------------------------------------------------- training data from rollouts
def sequences_from_blocks(path, zh):
    """rebuild per-position observation sequences of an N-policy rollout blocks file (rollout records carry lp0 / lps for the
    uncommitted units, draft / draft_v1 / draft_sap, cv_new). Returns list of sequences (each a list of obs tuples) and,
    for diagnostics, the label 1(d0[:i+1] == final[:i+1]) at each sequence's last observation."""
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
            obs = observations(d0, prev, d1, ds, lp0, lps, n)
            for i, o in obs.items():
                H.setdefault(i, []).append(o)
                last[i] = int(d0[: i + 1] == fin[: i + 1])
            c = x["cv_new"]
            for i in range(n, n + c):            # committed now: the sequence ends (censored)
                if i in H:
                    seqs.append(H.pop(i)); labels.append(last.pop(i))
            n += c
            prev = d0
        for i in list(H):                        # end of item: remaining sequences end at the last non-final read
            seqs.append(H.pop(i)); labels.append(last.pop(i))
    return seqs, labels


def init_params():
    dur = np.full(DMAX + 1, 1.0 / (DMAX + 1))
    em = {"S": {"a": [0.1, 0.8, 0.1], "v": [0.2, 0.8], "s": [0.3, 0.7], "qN": [0.7, 0.2, 0.1], "qS": [0.5, 0.15, 0.05, 0.3]},
          "U": {"a": [0.4, 0.3, 0.3], "v": [0.5, 0.5], "s": [0.6, 0.4], "qN": [0.3, 0.3, 0.4], "qS": [0.1, 0.1, 0.2, 0.6]}}
    return {"dur": dur.tolist(), "em": em}


def em_fit(seqs, iters=30):
    d = init_params()
    hist = []
    for it in range(iters):
        P = Params(d)
        dc = np.ones(DMAX + 1)                                         # Dirichlet(+1)
        ec = {st: {k: np.ones(CATS[k]) for k in KEYS} for st in ("U", "S")}
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
            if cens is not None:                                       # censored mass spread over d >= T by the current pmf
                wc = w[-1]; tail = P.dur[T:]
                if tail.sum() > 0:
                    dc[T:] += wc * tail / tail.sum()
            for t, o in enumerate(seq):
                for j, k in enumerate(KEYS):
                    ec["S"][k][o[j]] += pS[t]
                    ec["U"][k][o[j]] += 1.0 - pS[t]
        d = {"dur": (dc / dc.sum()).tolist(),
             "em": {st: {k: (v / v.sum()).tolist() for k, v in ec[st].items()} for st in ("U", "S")}}
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
    res = {"dmax": DMAX, "keys": KEYS, "dirs": {}}
    for d in ("en2zh", "zh2en"):
        seqs, labels = [], []
        for f in sorted(glob.glob(f"{a.roots}/blocks_{d}_w1.jsonl")):
            s, l = sequences_from_blocks(f, d.endswith("zh"))
            seqs += s; labels += l
        print(d, "sequences", len(seqs), "mean length", float(np.mean([len(s) for s in seqs])), flush=True)
        params, hist = em_fit(seqs)
        P = Params(params)
        post = [posterior_stable(s, P) for s in seqs]
        res["dirs"][d] = {"params": params, "loglik": hist, "n_seq": len(seqs),
                          "diag_auc_vs_final_prefix": auc(labels, post), "diag_pos_rate": float(np.mean(labels))}
        print(d, "diagnostic AUC (posterior vs final-prefix label, not used in training):",
              res["dirs"][d]["diag_auc_vs_final_prefix"], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
