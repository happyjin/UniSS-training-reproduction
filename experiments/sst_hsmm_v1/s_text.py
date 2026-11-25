"""prereg change 23: text BLEU of PC-MBR B / A vs the original blocks (test: per p3_score cell vs N via combo_score.text_compare;
dev: corpus BLEU per direction vs the wave17 LA2 control). -> s/s_text.json"""
import json, sys
sys.path.insert(0, ".")
import sacrebleu
from pathlib import Path
from experiments.sst_hsmm_v1 import p3_score as S
from experiments.mist_w3po_v1.voicecond.combo_score import LANG, jl, text_compare
R = Path("eval_outputs/sst_hsmm_v1/s")
out = {}
for d in S.DIRS:
    meta = {m["id"]: m for m in jl(S.T / "data" / f"meta_{d}.jsonl")}
    ref = {r["id"]: r["translation"] for r in jl(S.T / "data" / f"items_{d}.jsonl")}
    N = {r["id"]: r["translation"] for r in jl(S.sysd("N")["thinker"](d))}
    for v in ("b", "a"):
        X = {r["id"]: r["translation"] for r in jl(R / f"test_{v}" / f"blocks_{d}_w1.jsonl")}
        diffs = []
        for cell in S.CELLS:
            items = sorted(i for i in set(X) & set(N) - S.DROP if i in meta and f"{meta[i]['set']}/{meta[i]['cond']}" == cell
                           and not (meta[i]["set"] == "obsi" and meta[i]["pair"] in S.EXCL[d]))
            c = text_compare(meta, ref, items, {i: {"text": X[i]} for i in items}, {i: {"text": N[i]} for i in items}, LANG[d], "stream")
            out[f"test/{v}/{d}/{cell}"] = c
            diffs.append(c["diff"] if "diff" in c else c)
        print("test", v, d, [round(x, 2) if isinstance(x, float) else x for x in diffs], flush=True)
    dv = {r["id"]: r for r in jl(f"eval_outputs/mist_w3po_v1/wave2_S1/data/items_{d}.jsonl")}
    C = {r["id"]: r["translation"] for r in jl(f"eval_outputs/mist_w3po_v1/wave17/eval_enc_wise05/thinker_{d}/blocks_{d}_w1.jsonl")}
    tok = "zh" if d == "en2zh" else "13a"
    ids = sorted(set(C) & set(dv))
    bl = lambda H: sacrebleu.corpus_bleu([H[i] for i in ids], [[dv[i]["translation"] for i in ids]], tokenize=tok).score
    res = {"ctrl": bl(C)}
    for v in ("b", "a"):
        res[v] = bl({r["id"]: r["translation"] for r in jl(R / f"dev_{v}" / f"blocks_{d}_w1.jsonl")})
    out[f"dev/{d}"] = res
    print("dev", d, {k: round(x, 2) for k, x in res.items()}, flush=True)
json.dump(out, open(R / "s_text.json", "x"), indent=1)
