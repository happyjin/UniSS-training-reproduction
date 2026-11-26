"""D2 decomposition B_b / NEW(p3h8) / N per cell + oracle-commit counterfactuals on each system's own thinker trajectory."""
import json, sys, collections as C
import numpy as np
sys.path.insert(0, ".")
from experiments.voice_onset_v1.playout import LD, si_units
T = "eval_outputs/mist_w3po_v1/test_robust"; P = "eval_outputs/sst_hsmm_v1/p3h8"
SYS = {"B_b": (f"{T}/B/b/talker_{{d}}/results.jsonl", f"{T}/B/thinker_{{d}}/blocks_{{d}}_w1.jsonl", "B_b"),
       "NEW": (f"{P}/new/talker_{{d}}/results.jsonl", f"{P}/thinker_{{d}}/blocks_{{d}}_w1.jsonl", "NEW"),
       "W17": ("eval_outputs/sst_hsmm_v1/p3x/new/talker_{d}/results.jsonl", "eval_outputs/sst_hsmm_v1/p3x/thinker_{d}/blocks_{d}_w1.jsonl", "W17")}
def rd(p): return {r["id"]: r for r in map(json.loads, open(p))}
def lcp(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]: k += 1
    return k
def d2_from(commit, dur, X, Tn):
    """units with commit times (made prefix-monotone) and per-unit speech durations -> D2 (ms) and pauses>300ms"""
    c = np.maximum.accumulate(np.asarray(commit, float))
    d, end, i, pauses, first = [], 0.0, 0, 0, True
    while i < len(c):
        j = i
        while j < len(c) and c[j] == c[i]: j += 1
        st = max(c[i], end)
        if not first and st - end > 300: pauses += 1
        first = False
        tot = sum(dur[i:j]); acc = 0.0
        for k in range(i, j):
            acc += dur[k]; d.append(st + acc)
        end = st + tot; i = j
    return LD.laal_unitwise(d, X, Tn), pauses, end - X
for d in ("en2zh", "zh2en"):
    lang = "cmn" if d.endswith("zh") else "eng"
    A = C.defaultdict(lambda: C.defaultdict(list))
    for s, (tp, bp, pn) in SYS.items():
        R = rd(tp.format(d=d)); B = rd(bp.format(d=d)) if bp else {}
        PL = rd(f"eval_outputs/sst_hsmm_v1/p3x/play/NEW_{d}.jsonl" if pn == "W17" else f"{P}/play/{pn}_{d}.jsonl")
        for k, r in R.items():
            if k not in PL or PL[k].get("D2") is None: continue
            cell = k.split("__")[0]; a = A[(s, cell)]
            X = r["source_duration_ms"]; Tn = len(si_units(r["translation_ref"], lang))
            bl = [b for b in r["blocks"] if si_units(b["text"], lang)]
            if not bl: continue
            w, dur = [], []
            for b in bl:
                n = len(si_units(b["text"], lang)); w += [b["write_ms"]] * n; dur += [b["dur_ms"] / n] * n
            a["D2play"].append(PL[k]["D2"]); a["NP"].append(PL[k]["NP_in"]); a["EO"].append(PL[k]["EndOffset"])
            a["TL"].append(LD.laal_unitwise([min(x, X) for x in w], X, Tn))      # text-arrival LAAL (write times)
            rec, _, _ = d2_from(w, dur, X, Tn); a["D2rec"].append(rec)
            a["msu"].append(np.mean(dur)); a["nu"].append(len(w)); a["fin"].append(np.mean([x >= X for x in w]))
            a["preX"].append(np.mean([x < X for x in w]))
            if k in B and B[k].get("drafts"):
                fin = si_units(r["translated_text"] if "translated_text" in r else B[k]["translation"], lang)
                if len(fin) != len(w): a["skip"].append(1); continue
                ts = [x["t"] for x in B[k]["drafts"]]; L = [lcp(si_units(x["draft"], lang), fin) for x in B[k]["drafts"]]
                L[-1] = len(fin)
                S = list(L)
                for i in range(len(S) - 2, -1, -1): S[i] = min(S[i], S[i + 1])
                stab = [next(t for t, s_ in zip(ts, S) if s_ > i) for i in range(len(fin))]
                frst = [next(t for t, l_ in zip(ts, L) if l_ > i) for i in range(len(fin))]
                a["gap_pol"].append(np.mean(np.array(w) - np.array(stab)) / 1000)   # commit - stable (s)
                a["gap_thk"].append(np.mean(np.array(stab) - np.array(frst)) / 1000)  # stable - first correct (s)
                o, op, _ = d2_from(stab, dur, X, Tn); a["D2stab"].append(o); a["NPstab"].append(op)
                f, fp, _ = d2_from(frst, dur, X, Tn); a["D2first"].append(f)
                _, np_, _ = d2_from(w, dur, X, Tn); a["NPrec"].append(np_)
    keys = ("D2play", "D2rec", "TL", "msu", "nu", "fin", "preX", "EO", "NP", "NPrec", "gap_pol", "gap_thk", "D2stab", "NPstab", "D2first")
    print(f"\n== {d}  (D2/TL ms; msu = ms per spoken unit; fin = share of units written at/after source end; gaps s)")
    print("sys  cell     n  " + " ".join(f"{x:>7s}" for x in keys))
    for cell in ("rsc", "rsn", "obclean", "obid2", "obood1"):
        for s in SYS:
            a = A[(s, cell)]
            if not a: continue
            print(f"{s:4s} {cell:8s}{len(a['D2play']):4d} " + " ".join(f"{np.mean(a[x]):7.2f}" if a.get(x) else f"{'-':>7s}" for x in keys), f"skip={len(a.get('skip', []))}")
