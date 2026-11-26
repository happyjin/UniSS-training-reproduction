"""UniST dev only: AU share / AU survival / SAP lag (reads) by speech-density quartile (final units per source second)."""
import json, sys, collections as C
import numpy as np
sys.path.insert(0, ".")
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units
def lcp(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]: k += 1
    return k
for d in ("en2zh", "zh2en"):
    zh = d.endswith("zh"); rows = []
    for r in map(json.loads, open(f"eval_outputs/sst_hsmm_v1/h19n2/dev_0.75_{d}/blocks_{d}_w1.jsonl")):
        fin = units(r["translation"], zh); dr = r["drafts"]; dur = dr[-1]["t"] / 1000
        sup = [lcp(units(x["draft"], zh), units(x["draft_sap"], zh)) for x in dr[:-1]]
        prev = None; n = 0; au = surv = tot = 0; waits = []
        for j, x in enumerate(dr[:-1]):
            d0 = units(x["draft"], zh); ag = lcp(d0, prev) if prev is not None else 0; ok = lcp(d0, fin)
            for i in range(n, len(d0)):
                tot += 1
                if i < ag and i >= sup[j]:
                    au += 1; surv += i < ok
                    if i < ok:
                        waits.append(next((k - j for k in range(j + 1, len(sup)) if sup[k] > i), len(sup) - j))
            prev = d0; n += x["cv_new"]
        if tot: rows.append((len(fin) / max(dur, 1e-3), au / tot, surv / au if au else np.nan, np.mean(waits) if waits else np.nan))
    rows = np.array(rows); q = np.quantile(rows[:, 0], [0.25, 0.5, 0.75])
    print(f"== {d} dev ({len(rows)} items): density quartile -> units/s, AU share, AU survival, SAP lag (reads)")
    for k, (lo, hi) in enumerate(zip([-1, *q], [*q, 1e9])):
        m = (rows[:, 0] > lo) & (rows[:, 0] <= hi)
        print(f"  Q{k+1}  {np.nanmean(rows[m,0]):5.2f}  {np.nanmean(rows[m,1]):.3f}  {np.nanmean(rows[m,2]):.3f}  {np.nanmean(rows[m,3]):.2f}")
