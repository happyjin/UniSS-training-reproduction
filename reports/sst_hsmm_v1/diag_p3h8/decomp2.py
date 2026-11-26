"""draft coverage (draft units / final units at t/X), first-at-final share, queue wait, zero-commit reads, pause-gap sizes."""
import json, sys, collections as C
import numpy as np
sys.path.insert(0, ".")
from experiments.voice_onset_v1.playout import si_units
T = "eval_outputs/mist_w3po_v1/test_robust"; P = "eval_outputs/sst_hsmm_v1/p3h8"
SYS = {"B_b": (f"{T}/B/b/talker_{{d}}/results.jsonl", f"{T}/B/thinker_{{d}}/blocks_{{d}}_w1.jsonl"),
       "NEW": (f"{P}/new/talker_{{d}}/results.jsonl", f"{P}/thinker_{{d}}/blocks_{{d}}_w1.jsonl")}
def rd(p): return {r["id"]: r for r in map(json.loads, open(p))}
def lcp(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]: k += 1
    return k
for d in ("en2zh", "zh2en"):
    lang = "cmn" if d.endswith("zh") else "eng"
    A = C.defaultdict(lambda: C.defaultdict(list))
    for s, (tp, bp) in SYS.items():
        R = rd(tp.format(d=d)); B = rd(bp.format(d=d))
        for k, r in R.items():
            if k not in B: continue
            cell = k.split("__")[0]; a = A[(s, cell)]; X = r["source_duration_ms"]
            fin = si_units(r["translated_text"], lang); N = max(len(fin), 1)
            dr = B[k]["drafts"]
            cov = {q: [] for q in (0.5, 0.75)}
            for x in dr[:-1]:
                du = si_units(x["draft"], lang)
                for q in cov:
                    if abs(x["t"] / X - q) <= 0.125: cov[q].append(len(du) / N)
            for q in cov:
                if cov[q]: a[f"len@{q}"].append(np.mean(cov[q]))
            L = [lcp(si_units(x["draft"], lang), fin) for x in dr[:-1]]
            a["first_at_final"].append(1 - (max(L) if L else 0) / N)     # share of final units never correctly drafted before the final read
            bl = [b for b in r["blocks"] if si_units(b["text"], lang)]
            qw = [(b["start_ms"] - b["write_ms"]) / 1000 for b in bl if b["write_ms"] < X]
            if qw: a["queue_wait"].append(np.mean(qw))
            # per non-final read commit counts (thinker blocks)
            wr = {b["write_ms"]: len(si_units(b["text"], lang)) for b in B[k]["blocks"] if not b.get("final")}
            reads = [x["t"] for x in dr[:-1] if x["t"] >= min(wr) ] if wr else []
            if reads:
                cnt = [wr.get(t, 0) for t in reads]
                a["zero_reads"].append(np.mean([c == 0 for c in cnt])); a["units_per_commit"].append(np.mean([c for c in cnt if c > 0]))
            # gaps between consecutive speech blocks (stored rendering)
            end = None
            for b in bl:
                if end is not None and b["start_ms"] - end > 300: a["gap_s"].append((b["start_ms"] - end) / 1000)
                end = b["start_ms"] + b["dur_ms"]
    keys = ("len@0.5", "len@0.75", "first_at_final", "queue_wait", "zero_reads", "units_per_commit")
    print(f"\n== {d}  (len@q = draft units / final units at t/X≈q; first_at_final = share of units first drafted correctly at the final read)")
    print("sys  cell     " + " ".join(f"{x:>15s}" for x in keys) + "   gap>0.3s: median p75 share>0.6s")
    for cell in ("rsc", "rsn", "obclean", "obid2", "obood1"):
        for s in SYS:
            a = A[(s, cell)]; g = np.array(a["gap_s"])
            print(f"{s:4s} {cell:8s}" + " ".join(f"{np.mean(a[x]):15.3f}" for x in keys) + f"   {np.median(g):.2f} {np.percentile(g,75):.2f} {np.mean(g>0.6):.2f}")
