"""UniST-Emilia on-policy rollouts (h20, p3h8 policy): AUC of the HSMM posterior P for survival, overall and within classes."""
import json, sys, glob
import numpy as np
sys.path.insert(0, ".")
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units
from sklearn.metrics import roc_auc_score
def lcp(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]: k += 1
    return k
for d in ("en2zh", "zh2en"):
    zh = d.endswith("zh"); P = {"all": [], "AU": [], "AS": [], "NU": []}; Y = {k: [] for k in P}
    for f in sorted(glob.glob(f"eval_outputs/sst_hsmm_v1/h20/roll_{d}_*/blocks_{d}_w1.jsonl")):
        for r in map(json.loads, open(f)):
            fin = units(r["translation"], zh); prev = None; n = 0
            for x in r["drafts"][:-1]:
                d0 = units(x["draft"], zh)
                if "P" in x:
                    ag = lcp(d0, prev) if prev is not None else 0; sv = lcp(d0, units(x["draft_sap"], zh)); ok = lcp(d0, fin)
                    for j, p in enumerate(x["P"]):
                        i = n + j; c = ("A" if i < ag else "N") + ("S" if i < sv else "U"); y = int(i < ok)
                        for k in ("all", c):
                            if k in P: P[k].append(p); Y[k].append(y)
                prev = d0; n += x["cv_new"]
    print(d, " ".join(f"{k}: n={len(Y[k])} surv={np.mean(Y[k]):.2f} AUC={roc_auc_score(Y[k], P[k]):.3f} meanP={np.mean(P[k]):.2f}" for k in P))
