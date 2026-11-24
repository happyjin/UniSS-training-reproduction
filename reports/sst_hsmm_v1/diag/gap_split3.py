"""diag 3: prefix-forced full-audio continuation of N's committed prefix vs N vs offline, per prefix-agreement group."""
import glob, json, sys
sys.path.insert(0, ".")
import sacrebleu
from experiments.sst_hsmm_v1 import p3_score as S
from experiments.mist_w3po_v1.voicecond.combo_score import jl
F = {r["key"]: r["translation"] for p in glob.glob("eval_outputs/sst_hsmm_v1/pfx_test/forced_*.jsonl") for r in jl(p)}
for d in S.DIRS:
    meta = {m["id"]: m for m in jl(S.T / "data" / f"meta_{d}.jsonl")}
    ref = {r["id"]: r["translation"] for r in jl(S.T / "data" / f"items_{d}.jsonl")}
    off = {r["id"]: r["translation"] for p in sorted(glob.glob(f"eval_outputs/sst_hsmm_v1/ceiling/offline_{d}_*.jsonl")) for r in jl(p)}
    Nb = {r["id"]: r for r in jl(S.sysd("N")["thinker"](d))}
    ids = [i for i in sorted(set(off) & set(Nb) - S.DROP) if i in meta and f"{meta[i]['set']}/{meta[i]['cond']}" in S.CELLS
           and not (meta[i]["set"] == "obsi" and meta[i]["pair"] in S.EXCL[d]) and f"{d}|{i}" in F]
    tok = "zh" if d == "en2zh" else "13a"
    fo = {i: F[f"{d}|{i}"] for i in ids}
    G = {"all": ids, "same": [], "diff": []}
    for i in ids:
        pre = "".join(b["text"] for b in Nb[i]["blocks"] if not b["final"])
        G["same" if off[i].replace(" ", "").startswith(pre.replace(" ", "")) else "diff"].append(i)
    for g, I in G.items():
        b = lambda h: sacrebleu.corpus_bleu([h[i] for i in I], [[ref[i] for i in I]], tokenize=tok).score
        print(f"{d} {g:4s} n {len(I)} offline {b(off):.2f} forced {b(fo):.2f} N {b({i: Nb[i]['translation'] for i in I}):.2f} "
              f"N==forced {sum(fo[i].replace(' ','') == Nb[i]['translation'].replace(' ','') for i in I)/len(I):.2f}")
