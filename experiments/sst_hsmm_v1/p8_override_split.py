import json, sacrebleu
from experiments.sst_hsmm_v1 import p3_score as S
S.R = S.REPO / "eval_outputs/sst_hsmm_v1/p3l"
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
for d in ["en2zh", "zh2en"]:
    tok = "zh" if d == "en2zh" else "13a"
    ref = {r["id"]: r["translation"] for r in jl(S.T / "data" / f"items_{d}.jsonl")}
    N = {r["id"]: r["translation"] for r in jl(S.sysd("N")["thinker"](d))}
    P = {r["id"]: r for r in jl(S.R / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}
    for grp in ["realsi", "obsi"]:
        ids = [i for i in P if i in N and i in ref and (i.startswith("rs") == (grp == "realsi"))]
        for name, sel in [("over>0", lambda r: r["n_over"] > 0), ("over=0", lambda r: r["n_over"] == 0)]:
            s = [i for i in ids if sel(P[i])]
            if not s: continue
            bp = sacrebleu.corpus_bleu([P[i]["translation"] for i in s], [[ref[i] for i in s]], tokenize=tok).score
            bn = sacrebleu.corpus_bleu([N[i] for i in s], [[ref[i] for i in s]], tokenize=tok).score
            same = sum(P[i]["translation"] == N[i] for i in s) / len(s)
            print(f"{d} {grp:6s} {name}: n={len(s):4d} P8 {bp:5.2f} N {bn:5.2f} diff {bp-bn:+.2f} identical {same:.2f}")
