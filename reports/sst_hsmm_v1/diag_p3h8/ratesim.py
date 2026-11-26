"""talker-side headroom on p3h8's stored renderings: re-time units with modified speech durations (no re-synthesis):
uniform speed-up, backlog-only speed-up (blocks that start after a queue wait), tail-only (blocks written at/after source end)."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from experiments.voice_onset_v1.playout import LD, si_units
def sim(blocks, X, Tn, lang, mode, s):
    end, d, pauses, first = 0.0, [], 0, True
    for b in blocks:
        n = len(si_units(b["text"], lang))
        if n == 0: continue
        st = max(b["write_ms"], end)
        f = 1.0
        if mode == "uniform": f = s
        elif mode == "backlog" and st > b["write_ms"] + 200: f = s
        elif mode == "tail" and b["write_ms"] >= X: f = s
        dur = b["dur_ms"] / f
        if not first and st - end > 300: pauses += 1
        first = False
        d += [st + (k + 1) / n * dur for k in range(n)]
        end = st + dur
    return LD.laal_unitwise(d, X, Tn), pauses, (end - X) / 1000
for dname in ("en2zh", "zh2en"):
    lang = "cmn" if dname.endswith("zh") else "eng"
    R = [json.loads(l) for l in open(f"eval_outputs/sst_hsmm_v1/p3h8/new/talker_{dname}/results.jsonl")]
    rows = []
    for r in R:
        bl = [b for b in r["blocks"] if si_units(b["text"], lang)]
        if not bl: continue
        X = r["source_duration_ms"]; Tn = len(si_units(r["translation_ref"], lang))
        base = sim(bl, X, Tn, lang, "none", 1)
        rows.append([base] + [sim(bl, X, Tn, lang, m, s) for m, s in (("uniform", 1.1), ("uniform", 1.2), ("backlog", 1.2), ("tail", 1.2), ("tail", 1.35))])
    A = np.array(rows)   # (items, configs, 3)
    names = ("uniform1.1", "uniform1.2", "backlog1.2", "tail1.2", "tail1.35")
    print(f"{dname}: base D2 {A[:,0,0].mean():.0f} ms, pauses {A[:,0,1].mean():.2f}, EndOffset {A[:,0,2].mean():.2f} s")
    for j, nm in enumerate(names, 1):
        print(f"   {nm:11s} dD2 {A[:,j,0].mean()-A[:,0,0].mean():+.0f} ms  dPauses {A[:,j,1].mean()-A[:,0,1].mean():+.2f}  dEndOffset {A[:,j,2].mean()-A[:,0,2].mean():+.2f} s")
