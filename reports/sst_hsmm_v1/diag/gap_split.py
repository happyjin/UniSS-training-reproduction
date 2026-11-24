"""diag (iteration 2): where is N's gap to the offline ceiling? split test items by the fraction of characters N commits
before its final read; corpus BLEU of offline vs N vs ref per group; also BLEU of offline when its output begins with N's prefix."""
import glob, json, sys
from pathlib import Path
sys.path.insert(0, ".")
import sacrebleu
from experiments.sst_hsmm_v1 import p3_score as S
from experiments.mist_w3po_v1.voicecond.combo_score import jl
for d in S.DIRS:
    meta = {m["id"]: m for m in jl(S.T / "data" / f"meta_{d}.jsonl")}
    ref = {r["id"]: r["translation"] for r in jl(S.T / "data" / f"items_{d}.jsonl")}
    off = {r["id"]: r["translation"] for p in sorted(glob.glob(f"eval_outputs/sst_hsmm_v1/ceiling/offline_{d}_*.jsonl")) for r in jl(p)}
    Nb = {r["id"]: r for r in jl(S.sysd("N")["thinker"](d))}
    ids = [i for i in sorted(set(off) & set(Nb) - S.DROP) if i in meta and f"{meta[i]['set']}/{meta[i]['cond']}" in S.CELLS
           and not (meta[i]["set"] == "obsi" and meta[i]["pair"] in S.EXCL[d])]
    tok = "zh" if d == "en2zh" else "13a"
    grp = {}
    for i in ids:
        bl = Nb[i].get("blocks", [])
        pre = "".join(b["text"] for b in bl if not b["final"]); tot = Nb[i]["translation"]
        f = len(pre) / max(1, len(tot))
        g = "0" if f == 0 else ("<.5" if f < .5 else ">=.5")
        same = off[i].replace(" ", "").startswith(pre.replace(" ", ""))
        grp.setdefault(g, []).append((i, same))
    for g in ("0", "<.5", ">=.5"):
        L = grp.get(g, []); I = [i for i, _ in L]
        if not I: continue
        b = lambda h: sacrebleu.corpus_bleu([h[i] for i in I], [[ref[i] for i in I]], tokenize=tok).score
        sm = sum(s for _, s in L) / len(L)
        print(f"{d} pre_frac {g:5s} n {len(I):4d} offline {b(off):5.2f} N {b({i: Nb[i]['translation'] for i in I}):5.2f} offline_startswith_Nprefix {sm:.2f}")
