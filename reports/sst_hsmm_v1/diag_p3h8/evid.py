"""evidence rates at uncommitted draft positions, per cell (test p3h8) and UniST dev (h19n2 dev_0.75): a = agrees with previous
draft, v = agrees with enhanced-view draft, s = inside SAP-verifier support (lcp with draft_sap), P0 = posterior at the first
uncommitted position, P_age1 = mean P at positions seen for the first time, r = raw/enh energy ratio, new = LA2-offered units/read."""
import json, sys, collections as C
import numpy as np
sys.path.insert(0, ".")
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units
def lcp(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]: k += 1
    return k
src = {d: [("test", f"eval_outputs/sst_hsmm_v1/p3h8/thinker_{d}/blocks_{d}_w1.jsonl"),
           ("dev", f"eval_outputs/sst_hsmm_v1/h19n2/dev_0.75_{d}/blocks_{d}_w1.jsonl")] for d in ("en2zh", "zh2en")}
for d, L in src.items():
    zh = d.endswith("zh")
    A = C.defaultdict(lambda: C.defaultdict(list))
    for tag, f in L:
        for r in map(json.loads, open(f)):
            cell = r["id"].split("__")[0] if tag == "test" else "UniST-dev"
            a = A[cell]; prev = None; n = 0; seen = 0
            for x in r["drafts"][:-1]:
                d0 = units(x["draft"], zh); d1 = units(x["draft_v1"], zh); ds = units(x["draft_sap"], zh)
                if "P" in x and len(d0) > n:
                    ag = lcp(d0, prev) if prev is not None else 0; cv = lcp(d0, d1); sv = lcp(d0, ds)
                    for i in range(n, len(d0)):
                        a["a"].append(i < ag); a["v"].append(i < cv); a["s"].append(i < sv)
                    P = x["P"]
                    if P: a["P0"].append(P[0])
                    for j, i in enumerate(range(n, len(d0))):
                        if i >= seen and j < len(P): a["P_age1"].append(P[j])
                    a["r"].append(x["r"]); a["new"].append(x["la2_new"]); a["commit"].append(x["cv_new"])
                seen = max(seen, len(d0)); prev = d0; n += x["cv_new"]
    print(f"\n== {d}")
    print("cell        " + " ".join(f"{k:>7s}" for k in ("a", "v", "s", "P0", "P_age1", "r", "new", "commit")))
    for cell in ("UniST-dev", "rsc", "rsn", "obclean", "obid2", "obood1"):
        a = A[cell]
        print(f"{cell:10s}  " + " ".join(f"{np.mean(a[k]):7.3f}" for k in ("a", "v", "s", "P0", "P_age1", "r", "new", "commit")))
