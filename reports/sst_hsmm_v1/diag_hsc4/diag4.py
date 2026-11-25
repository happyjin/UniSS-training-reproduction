# calibration of the HSC4 posterior on UNCOMMITTED positions: P(S) bin vs empirical "prefix survives into final output"
import json, sys, collections
import numpy as np
sys.path.insert(0, ".")
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units
from experiments.sst_hsmm_v1.hsmm_commit2 import lcp
B = [0, .02, .1, .3, .5, .7, .9, 1.01]
for d in ("en2zh", "zh2en"):
    zh = d.endswith("zh")
    H = collections.defaultdict(lambda: np.zeros((len(B) - 1, 2)))
    for r in map(json.loads, open(f"eval_outputs/sst_hsmm_v1/p3n1/thinker_{d}/blocks_{d}_w1.jsonl", encoding="utf-8")):
        c = r["id"].split("__")[0]
        if c.startswith("o3"): continue
        g = "RealSI" if c.startswith("rs") else "OBSI"
        fin = units(r["translation"], zh); n = 0
        for x in r["drafts"][:-1]:
            dr = units(x["draft"], zh); ok = lcp(dr, fin); cut = n + x["cv_new"]; P = x.get("P", [])
            for j, p in enumerate(P):
                i = n + j
                if i < cut: continue
                b = np.searchsorted(B, p, side="right") - 1
                H[g][b] += (int(i < ok), 1)
            n = cut
    for g, h in sorted(H.items()):
        print(f"{d} {g:6s} " + "  ".join(f"[{B[k]:.2f},{B[k+1]:.2f}) {h[k,0]/max(h[k,1],1):.2f} n={int(h[k,1])}" for k in range(len(B) - 1)))
