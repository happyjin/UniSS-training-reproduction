"""Collect 5-cell test results (vs B_b) for the ICLR draft. Usage: python3 collect.py out.json root1 root2 ..."""
import json, sys
D = "eval_outputs/sst_hsmm_v1"
MET = ["asr", "text_bleu", "d2", "d1", "EndOffset", "NP_in", "PT_in", "NC", "sim", "pesq"]
out, roots = {}, sys.argv[2:]
def val(m, side):
    return m.get(side, m.get(side + "_mean", m.get(side + "_median")))
def dif(m):
    return m["diff"] if "diff" in m else m["paired_median_diff"]
for r in roots:
    try:
        s = json.load(open(f"{D}/{r}/summary.json"))
    except FileNotFoundError:
        print("missing", r); continue
    res = {}
    for key in ("NEW_vs_B_b", "N_vs_B_b"):
        for d in ("en2zh", "zh2en"):
            cells = {k: v[key] for k, v in s["cells"].items() if k.startswith(d)}
            agg = {}
            for m in MET:
                xs = [c[m] for c in cells.values() if m in c]
                if not xs: continue
                agg[m] = {"arm": sum(val(x, "arm") for x in xs) / len(xs), "ref": sum(val(x, "ref") for x in xs) / len(xs),
                          "diff": sum(dif(x) for x in xs) / len(xs),
                          "cells": {k.split("/", 1)[1]: {"diff": dif(c[m]), "ci95": c[m].get("ci95"), "arm": val(c[m], "arm")} for k, c in cells.items() if m in c}}
            res.setdefault(key, {})[d] = agg
    out[r] = res
json.dump(out, open(sys.argv[1], "w"), indent=1)
for r in out:
    for d in ("en2zh", "zh2en"):
        a = out[r]["NEW_vs_B_b"][d]
        print(f"{r:10s} {d} " + " ".join(f"{m}={a[m]['diff']:+.3f}" for m in MET if m in a))
b = out[roots[0]]
for d in ("en2zh", "zh2en"):
    print("B_b abs", d, " ".join(f"{m}={b['NEW_vs_B_b'][d][m]['ref']:.3f}" for m in MET if m in b['NEW_vs_B_b'][d]))
    print("N  diff", d, " ".join(f"{m}={b['N_vs_B_b'][d][m]['diff']:+.3f}" for m in MET if m in b['N_vs_B_b'][d]))
