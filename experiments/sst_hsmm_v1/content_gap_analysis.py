"""sst_hsmm_v1: why the HSMM Thinker translates worse (report only, existing test outputs). P7' (p3j, HSMM content) vs
P8 (p3l, N content, same HSMM timing family) vs N. Split: items written entirely at the final read (full audio, timing
cannot matter) vs the rest; length ratio, punctuation-free BLEU, chrF, Latin-in-Chinese, truncation."""
import json, re, sys, unicodedata
import sacrebleu
from experiments.sst_hsmm_v1 import p3_score as S
E = S.REPO / "eval_outputs/sst_hsmm_v1"
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
P = lambda s: "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))
def stats(h, r, tok):
    b = sacrebleu.corpus_bleu(h, [r], tokenize=tok)
    return dict(n=len(h), bleu=round(b.score, 2), ratio=round(b.sys_len / max(b.ref_len, 1), 3),
                nop=round(sacrebleu.corpus_bleu([P(x) for x in h], [[P(x) for x in r]], tokenize=tok).score, 2),
                chrf=round(sacrebleu.corpus_chrf(h, [r]).score, 2))
res = {}
for d in S.DIRS:
    tok = "zh" if d == "en2zh" else "13a"
    meta = {m["id"]: m for m in jl(S.T / "data" / f"meta_{d}.jsonl")}
    ref = {r["id"]: r["translation"] for r in jl(S.T / "data" / f"items_{d}.jsonl")}
    N = {r["id"]: r["translation"] for r in jl(S.sysd("N")["thinker"](d))}
    H = {r["id"]: r for r in jl(E / "p3j" / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}
    C = {r["id"]: r for r in jl(E / "p3l" / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}
    for grp in ("realsi", "obsi"):
        ids = [i for i in H if i in C and i in N and i in ref and meta.get(i, {}).get("set") == grp and i not in S.DROP
               and not (grp == "obsi" and meta[i]["pair"] in S.EXCL[d])]
        allfin = lambda r: all(b["final"] for b in r["blocks"])
        for name, sel in (("final_only_both", lambda i: allfin(H[i]) and allfin(C[i])), ("rest", lambda i: not (allfin(H[i]) and allfin(C[i]))), ("all", lambda i: True)):
            s = [i for i in ids if sel(i)]
            r = [ref[i] for i in s]
            row = {k: stats([X[i] if k == "N" else X[i]["translation"] for i in s], r, tok) for k, X in (("N", N), ("P7'_HSMMcontent", H), ("P8_Ncontent", C))}
            if d == "en2zh":
                lat = lambda xs: round(sum(bool(re.search(r"[A-Za-z]{3,}", x)) for x in xs) / max(len(xs), 1), 3)
                row["latin_rate"] = {"N": lat([N[i] for i in s]), "P7'": lat([H[i]["translation"] for i in s]), "P8": lat([C[i]["translation"] for i in s])}
            short = lambda xs: round(sum(len(x) < 0.5 * len(ref[i]) for x, i in zip(xs, s)) / max(len(s), 1), 3)
            row["short_rate"] = {"N": short([N[i] for i in s]), "P7'": short([H[i]["translation"] for i in s]), "P8": short([C[i]["translation"] for i in s])}
            res[f"{d}/{grp}/{name}"] = row
            print(d, grp, name, json.dumps(row, ensure_ascii=False), flush=True)
out = E / "content_gap_analysis.json"
with open(out, "x") as fh:
    json.dump(res, fh, indent=1, ensure_ascii=False)
