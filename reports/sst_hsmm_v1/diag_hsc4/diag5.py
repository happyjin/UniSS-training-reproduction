# training-data (open UniST, h1 N-policy rollouts) calibration of the joint HSMM h9 posterior, by #reads observed
import json, sys, collections
import numpy as np
sys.path.insert(0, ".")
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units
from experiments.sst_hsmm_v1 import hsmm_joint as HN, hsmm_commit2 as H2
from pathlib import Path
P = HN.NParams(json.load(open("eval_outputs/sst_hsmm_v1/h9/params.json"))["params"])
E1 = Path("eval_outputs/sst_hsmm_v1/e1")
B = [0, .02, .1, .3, .5, .7, .9, 1.01]
for d in ("en2zh", "zh2en"):
    zh = d.endswith("zh")
    relv = HN.relv_from_views(E1 / "data" / f"items_{d}.jsonl", E1 / "views" / d)
    cal = np.zeros((len(B) - 1, 2)); byT = collections.defaultdict(lambda: np.zeros((len(B) - 1, 2)))
    for line in open(f"eval_outputs/sst_hsmm_v1/h1/roll_{d}_0/blocks_{d}_w1.jsonl", encoding="utf-8"):
        r = json.loads(line); RV = relv.get(r["id"])
        if RV is None: continue
        fin = units(r["translation"], zh); n, prev, H = 0, None, {}
        for x in r["drafts"][:-1]:
            if "lp0" not in x: continue
            d0 = units(x["draft"], zh); d1 = units(x["draft_v1"], zh); ds = units(x["draft_sap"], zh)
            lp0 = [0.0] * n + x["lp0"]; lps = [0.0] * n + x["lps"]; rr, vv = RV.get(x["t"], (1.0, 1.0))
            ok = H2.lcp(d0, fin)
            for i, o in H2.observations(d0, prev, d1, ds, lp0, lps, n).items():
                h = H.setdefault(i, ([], [])); h[1].append(HN.inputs_vec(rr, vv, len(h[0]), x["t"], zh)); h[0].append(o)
                if i >= n + x["cv_new"]:                       # uncommitted at this read
                    p = HN.posterior_stable(h[0], h[1], P); b = np.searchsorted(B, p, side="right") - 1
                    y = int(i < ok); cal[b] += (y, 1); byT[min(len(h[0]), 4)][b] += (y, 1)
            for i in range(n, n + x["cv_new"]): H.pop(i, None)
            n += x["cv_new"]; prev = d0
    print(f"{d} all   " + "  ".join(f"[{B[k]:.2f},{B[k+1]:.2f}) {cal[k,0]/max(cal[k,1],1):.2f} n={int(cal[k,1])}" for k in range(len(B) - 1)))
    for T in sorted(byT):
        h = byT[T]
        print(f"{d} T={T}{'+' if T==4 else ' '} " + "  ".join(f"{h[k,0]/max(h[k,1],1):.2f}({int(h[k,1])})" for k in range(len(B) - 1)))
