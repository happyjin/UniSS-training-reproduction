"""uncommitted positions by evidence class -> share and survival (unit kept in the final output); test p3h8 cells + UniST dev.
classes: AS agreed & SAP-supported, AU agreed & unsupported, NS not agreed & supported, NU neither. Also: for AU units that
survive, reads until SAP supports them."""
import json, sys, collections as C
import numpy as np
sys.path.insert(0, ".")
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units
def lcp(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]: k += 1
    return k
for d in ("en2zh", "zh2en"):
    zh = d.endswith("zh")
    A = C.defaultdict(lambda: C.defaultdict(list))
    import glob
    for tag, f in [("test", f"eval_outputs/sst_hsmm_v1/p3h8/thinker_{d}/blocks_{d}_w1.jsonl")] + [("emilia", g) for g in sorted(glob.glob(f"eval_outputs/sst_hsmm_v1/h20/roll_{d}_*/blocks_{d}_w1.jsonl"))] + [
                   ("dev", f"eval_outputs/sst_hsmm_v1/h19n2/dev_0.75_{d}/blocks_{d}_w1.jsonl")]:
        for r in map(json.loads, open(f)):
            cell = r["id"].split("__")[0] if tag == "test" else ("UniST-emilia" if tag == "emilia" else "CV-dev")
            fin = units(r["translation"], zh); prev = None; n = 0
            dr = r["drafts"]
            sup = [lcp(units(x["draft"], zh), units(x["draft_sap"], zh)) for x in dr[:-1]]
            for j, x in enumerate(dr[:-1]):
                d0 = units(x["draft"], zh)
                ag = lcp(d0, prev) if prev is not None else 0; sv = sup[j]; ok = lcp(d0, fin)
                for i in range(n, len(d0)):
                    c = ("A" if i < ag else "N") + ("S" if i < sv else "U")
                    A[cell][c].append(i < ok)
                    if c == "AU" and i < ok:
                        later = next((k - j for k in range(j + 1, len(sup)) if sup[k] > i), None)
                        A[cell]["wait"].append(later if later is not None else len(sup) - j)
                prev = d0; n += x["cv_new"]
    print(f"\n== {d}: share of uncommitted positions / survival rate")
    print("cell        " + " ".join(f"{c:>13s}" for c in ("AS", "AU", "NS", "NU")) + "   AU-surviving: reads until SAP supports")
    for cell in ("CV-dev", "UniST-emilia", "rsc", "rsn", "obclean", "obid2", "obood1"):
        a = A[cell]; tot = sum(len(a[c]) for c in ("AS", "AU", "NS", "NU"))
        print(f"{cell:10s}  " + " ".join(f"{len(a[c])/tot:6.2f}/{np.mean(a[c]):5.2f}" for c in ("AS", "AU", "NS", "NU")) + f"   {np.mean(a['wait']):.2f}")
