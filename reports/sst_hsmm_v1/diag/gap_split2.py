"""diag 2: within items with a committed prefix, split by whether the offline output begins with N's prefix."""
import glob, sys
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
    G = {"same": [], "diff": []}
    nb = []
    for i in ids:
        pre = "".join(b["text"] for b in Nb[i]["blocks"] if not b["final"])
        if not pre: continue
        nb.append(sum(1 for b in Nb[i]["blocks"] if not b["final"]))
        G["same" if off[i].replace(" ", "").startswith(pre.replace(" ", "")) else "diff"].append(i)
    for g, I in G.items():
        b = lambda h: sacrebleu.corpus_bleu([h[i] for i in I], [[ref[i] for i in I]], tokenize=tok).score
        print(f"{d} {g} n {len(I)} offline {b(off):.2f} N {b({i: Nb[i]['translation'] for i in I}):.2f}")
    print(d, "mean nonfinal blocks", sum(nb) / len(nb))
