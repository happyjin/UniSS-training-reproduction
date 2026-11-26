"""prereg change 38-1: discrimination precheck of the K-sample consistency c on edev (p3h8 policy run with hsc10, h25/dev_0.75_<d>).
Per uncommitted draft position of every non-final read: P (HSMM posterior), c (consistency; -1 = no sample reached i), class
(A/N = inside / beyond the two-read agreement, S/U = inside / beyond SAP support), y = survives (position < lcp(draft, final)).
AUC of P, of c, and of a logistic stack [logit P, c, c missing] fitted with a 2-fold split by item (sha1 % 2), overall and in AU.
Operating point: precision = survival of positions with P >= 0.75; the stacked score's threshold reaching that precision on the
same positions -> number of positions selected, relative to P >= 0.75. GO (prereg 38-1): mean over directions of (AU stacked AUC
- AU P AUC) >= 0.03 and mean relative increase >= +10 %. Writes <out> (refuses to overwrite).
    <qwen3omni python> -m experiments.sst_hsmm_v1.cons_precheck --root eval_outputs/sst_hsmm_v1/h25 --out eval_outputs/sst_hsmm_v1/h25/precheck.json
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units  # noqa: E402


def lcp(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def collect(f, zh):
    X = []
    for r in map(json.loads, open(f, encoding="utf-8")):
        fin = units(r["translation"], zh); prev = None; n = 0
        fold = int(hashlib.sha1(r["id"].encode()).hexdigest(), 16) % 2
        for x in r["drafts"][:-1]:
            d0 = units(x["draft"], zh)
            if "P" in x and "cons" in x:
                ag = lcp(d0, prev) if prev is not None else 0
                sv = lcp(d0, units(x["draft_sap"], zh)); ok = lcp(d0, fin)
                for j, (p, c) in enumerate(zip(x["P"], x["cons"])):
                    i = n + j
                    X.append((fold, p, c, ("A" if i < ag else "N") + ("S" if i < sv else "U"), int(i < ok)))
            prev = d0; n += x["cv_new"]
    return X


def feats(p, c):
    lp = math.log(max(p, 1e-4) / max(1 - p, 1e-4))
    return [lp, max(c, 0.0), float(c < 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit(f"拒绝覆盖: {a.out}")
    res = {}
    for d in ("en2zh", "zh2en"):
        X = collect(Path(a.root) / f"dev_0.75_{d}" / f"blocks_{d}_w1.jsonl", d.endswith("zh"))
        F = np.array([feats(p, c) for _, p, c, _, _ in X]); y = np.array([t[4] for t in X]); fold = np.array([t[0] for t in X])
        P = np.array([t[1] for t in X]); Cc = np.array([max(t[2], 0.0) for t in X]); au = np.array([t[3] == "AU" for t in X])
        S = np.zeros(len(X))
        for k in (0, 1):
            m = LogisticRegression(max_iter=1000).fit(F[fold != k], y[fold != k])
            S[fold == k] = m.predict_proba(F[fold == k])[:, 1]
        sel = P >= 0.75; prec = y[sel].mean()
        order = np.argsort(-S); cum = np.cumsum(y[order]) / np.arange(1, len(S) + 1)
        n_eq = int(np.max(np.where(cum >= prec)[0]) + 1) if (cum >= prec).any() else 0
        r = {"n": int(len(X)), "n_AU": int(au.sum()), "surv": float(y.mean()), "surv_AU": float(y[au].mean()),
             "missing_c": float((np.array([t[2] for t in X]) < 0).mean()),
             "auc_P": roc_auc_score(y, P), "auc_c": roc_auc_score(y, Cc), "auc_stack": roc_auc_score(y, S),
             "auc_P_AU": roc_auc_score(y[au], P[au]), "auc_c_AU": roc_auc_score(y[au], Cc[au]), "auc_stack_AU": roc_auc_score(y[au], S[au]),
             "precision_P075": float(prec), "n_P075": int(sel.sum()), "n_stack_same_precision": n_eq, "rel_increase": n_eq / max(int(sel.sum()), 1) - 1}
        res[d] = r
        print(d, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    dauc = np.mean([res[d]["auc_stack_AU"] - res[d]["auc_P_AU"] for d in res]); inc = np.mean([res[d]["rel_increase"] for d in res])
    res["go"] = bool(dauc >= 0.03 and inc >= 0.10); res["delta_auc_AU"] = float(dauc); res["mean_rel_increase"] = float(inc)
    res["rule"] = "mean(AU stacked AUC - AU P AUC) >= 0.03 and mean relative increase >= 0.10 (prereg 38-1)"
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("GO" if res["go"] else "NO-GO", round(dauc, 4), round(inc, 4), flush=True)


if __name__ == "__main__":
    main()
