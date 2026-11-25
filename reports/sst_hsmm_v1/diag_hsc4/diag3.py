# per cell: how predictive is each observation of "prefix up to i survives into the final output" (analysis only)
import json, glob, sys, collections
import numpy as np
sys.path.insert(0, ".")
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units
from experiments.sst_hsmm_v1.hsmm_commit2 import lcp
root = sys.argv[1]
def cell(i): return i.split("__")[0]
for d in ("en2zh", "zh2en"):
    zh = d.endswith("zh")
    C = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in map(json.loads, open(f"eval_outputs/sst_hsmm_v1/{root}/thinker_{d}/blocks_{d}_w1.jsonl", encoding="utf-8")):
        c = cell(r["id"])
        if c.startswith("o3"): continue
        fin = units(r["translation"], zh); n = 0; prev = None
        for x in r["drafts"][:-1]:
            dr = units(x["draft"], zh); d1 = units(x["draft_v1"], zh); ds = units(x["draft_sap"], zh)
            agreed = lcp(dr, prev) if prev is not None else 0; cv = lcp(dr, d1); sv = lcp(dr, ds); ok = lcp(dr, fin)
            for i in range(n, len(dr)):
                a = "new" if (prev is None or i >= len(prev)) else ("a1" if i < agreed else "a0")
                key = f"{a}_v{int(i < cv)}_s{int(i < sv)}"
                C[c][key][0] += int(i < ok); C[c][key][1] += 1
            n += x["cv_new"]; prev = dr
    print(f"===== {d} {root}: P(prefix survives | obs) [count]")
    keys = ["a1_v1_s1", "a1_v1_s0", "a1_v0_s1", "a1_v0_s0", "a0_v1_s1", "new_v1_s1", "a0_v0_s0", "new_v0_s0"]
    print("cell     " + " ".join(f"{k:>15s}" for k in keys))
    for c in sorted(C):
        print(f"{c:8s} " + " ".join(f"{C[c][k][0]/max(C[c][k][1],1):.2f}[{C[c][k][1]:5d}]" for k in keys))
