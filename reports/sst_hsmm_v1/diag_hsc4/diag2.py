import json, glob, sys, collections
import numpy as np
sys.path.insert(0, ".")
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units
from experiments.sst_hsmm_v1.hsmm_commit2 import lcp
R = "eval_outputs/sst_hsmm_v1"; T = "eval_outputs/mist_w3po_v1/test_robust"
def cell(i): return i.split("__")[0]
for d in ("en2zh", "zh2en"):
    zh = d.endswith("zh")
    ref = {}
    for f in glob.glob(f"{R}/p3n1/data/items_{d}_*.jsonl"):
        for r in map(json.loads, open(f, encoding="utf-8")): ref[r["id"]] = len(units(r["translation"], zh))
    print(f"===== {d}  output length / reference length")
    for s, p in {"B_b": f"{T}/B/thinker_{d}", "N": f"{T}/UE/thinker_{d}", "LA2": f"{R}/p3x/thinker_{d}", "HSC2": f"{R}/p3z2/thinker_{d}", "HSC4": f"{R}/p3n1/thinker_{d}"}.items():
        L = collections.defaultdict(list)
        for r in map(json.loads, open(f"{p}/blocks_{d}_w1.jsonl", encoding="utf-8")):
            if r["id"] in ref: L[cell(r["id"])].append((len(units(r["translation"], zh)), ref[r["id"]]))
        print(f"  {s:5s} " + "  ".join(f"{c} {np.mean([a for a,b in v]):.1f}/{np.mean([b for a,b in v]):.1f}" for c, v in sorted(L.items())))
    # HSC4 commit anatomy
    st = collections.defaultdict(lambda: collections.Counter())
    for r in map(json.loads, open(f"{R}/p3n1/thinker_{d}/blocks_{d}_w1.jsonl", encoding="utf-8")):
        if r["id"] not in ref: continue
        c = cell(r["id"]); n = 0; prev = None; seen = {}
        for x in r["drafts"][:-1]:
            dr = units(x["draft"], zh); d1 = units(x["draft_v1"], zh)
            agreed = lcp(dr, prev) if prev is not None else 0
            cv = lcp(dr, d1); base = max(n, min(agreed, cv)); cut = n + x["cv_new"]
            C = st[c]; C["reads"] += 1; C["committed"] += x["cv_new"]
            C["base"] += base - n
            trig = x.get("theta_t", 0.8) < 0.79; C["trig"] += int(trig)
            for i in range(base, cut):
                C["extra"] += 1
                C["extra_beyond_agreed"] += int(i >= agreed)
                C["extra_beyond_cv"] += int(i >= cv)
                C["extra_sap"] += int(i < x["sv0"])
                C["extra_first_seen"] += int(prev is None or i >= len(prev))
                C["extra_trig"] += int(trig)
            # how much a strict theta (no pause rule) would have committed
            P = x.get("P", []); k = 0
            while k < len(P) and P[k] >= 0.8: k += 1
            C["noTrig_cut"] += max(base, n + k) - n
            n = cut; prev = dr
    print(f"  HSC4 anatomy (units per item-read totals)")
    for c, C in sorted(st.items()):
        e = max(C["extra"], 1)
        print(f"  {c:8s} reads {C['reads']} committed {C['committed']} base {C['base']} extra {C['extra']} "
              f"[beyond_agreed {C['extra_beyond_agreed']/e:.2f} beyond_cv {C['extra_beyond_cv']/e:.2f} sap {C['extra_sap']/e:.2f} first_seen {C['extra_first_seen']/e:.2f} on_trig {C['extra_trig']/e:.2f}] "
              f"trig_reads {C['trig']/C['reads']:.2f} noTrig_commit {C['noTrig_cut']}")
