import json, glob, sys, collections
import numpy as np
sys.path.insert(0, ".")
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units
R = "eval_outputs/sst_hsmm_v1"; T = "eval_outputs/mist_w3po_v1/test_robust"
SYS = {"B_b": f"{T}/B/thinker_{{d}}/blocks_{{d}}_w1.jsonl", "N": f"{T}/UE/thinker_{{d}}/blocks_{{d}}_w1.jsonl",
       "LA2": f"{R}/p3x/thinker_{{d}}/blocks_{{d}}_w1.jsonl", "SVCJM": f"{R}/p3w/thinker_{{d}}/blocks_{{d}}_w1.jsonl",
       "HSC2": f"{R}/p3z2/thinker_{{d}}/blocks_{{d}}_w1.jsonl", "HSC4": f"{R}/p3n1/thinker_{{d}}/blocks_{{d}}_w1.jsonl"}
PLAY = {"B_b": f"{R}/p3n1/play/B_b_{{d}}.jsonl", "N": f"{R}/p3n1/play/N_{{d}}.jsonl", "HSC4": f"{R}/p3n1/play/NEW_{{d}}.jsonl",
        "HSC2": f"{R}/p3z2/play/NEW_{{d}}.jsonl", "SVCJM": f"{R}/p3w/play/NEW_{{d}}.jsonl", "LA2": f"{R}/p3x/play/NEW_{{d}}.jsonl"}
def cell(i): return i.split("__")[0]
for d in ("en2zh", "zh2en"):
    zh = d.endswith("zh")
    print(f"===== {d}")
    print("sys   cell     blocks/it  units/blk(nonfin)  small<=2  firstW  tailU  NP_in  PT_in  D2")
    for s, p in SYS.items():
        f = glob.glob(p.format(d=d))
        if not f: print(s, "missing", p.format(d=d)); continue
        pl = {r["id"]: r for r in map(json.loads, open(PLAY[s].format(d=d)))}
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in map(json.loads, open(f[0], encoding="utf-8")):
            c = cell(r["id"]); b = r["blocks"]
            nf = [len(units(x["text"].strip(), zh)) for x in b if not x["final"] and x["text"].strip()]
            fin = [len(units(x["text"].strip(), zh)) for x in b if x["final"]]
            A = agg[c]
            A["nb"].append(len(nf)); A["ub"] += nf; A["small"] += [int(u <= 2) for u in nf]
            A["fw"].append(b[0]["write_ms"] / 1000 if b else 0); A["tail"].append(sum(fin))
            q = pl.get(r["id"])
            if q and "NP_in" in q: A["np"].append(q["NP_in"]); A["pt"].append(q["PT_in"]); A["d2"].append(q["D2"] / 1000)
        for c in sorted(k for k in agg if k != "o3en"):
            A = agg[c]
            print(f"{s:5s} {c:8s} {np.mean(A['nb']):6.2f} {np.mean(A['ub']):10.2f} {np.mean(A['small']):9.2f} {np.mean(A['fw']):7.2f} {np.mean(A['tail']):6.2f} {np.mean(A['np']):6.2f} {np.mean(A['pt']):6.2f} {np.mean(A['d2']):6.2f}")
