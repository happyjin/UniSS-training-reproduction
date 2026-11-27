import json, collections, re, sys, unicodedata, glob
sys.path.insert(0, ".")
from evaluation.text_metrics import remove_punctuation
from opencc import OpenCC
cc = OpenCC("t2s")
from sacrebleu.metrics import BLEU
bl = BLEU(tokenize="zh")
C = {}
def nc(t):
    if t not in C: C[t] = " ".join(c for c in remove_punctuation(cc.convert(unicodedata.normalize("NFKC", t))) if not c.isspace())
    return C[t]
LATW = re.compile(r"[A-Za-z][A-Za-z0-9\-\.']*")
D = collections.defaultdict(dict)
for l in open("eval_outputs/sst_hsmm_v1/pv3/comet/dump.jsonl"):
    r = json.loads(l)
    if r["dir"] == "en2zh": D[r["sys"]][(r["cell"], r["id"])] = r
keys = sorted(D["p3hk"])
def cb(s, sub, k="text"):
    return bl.corpus_score([nc(D[s][x][k]) for x in sub], [[nc(D[s][x]["ref"]) for x in sub]]).score
lat = [x for x in keys if LATW.search(D["p3hk"][x]["ref"])]
nol = [x for x in keys if not LATW.search(D["p3hk"][x]["ref"])]
print("items with Latin in ref: %d / %d (%.1f%%)" % (len(lat), len(keys), 100 * len(lat) / len(keys)))
for s in ("B_b", "p3hk", "p3hk2", "p3pfd1", "p3p2fxff", "p3pfe1", "p3pfh3", "p3pfe1d1", "p3pfh4", "p3d3s05"):
    tl = cb(s, lat); tn = cb(s, nol); al = cb(s, lat, "asr"); an = cb(s, nol, "asr")
    # term recall: ref latin words found verbatim (case-insens.) in hyp
    hit = tot = 0
    for x in lat:
        h = D[s][x]["text"].lower()
        for w in LATW.findall(D[s][x]["ref"]):
            tot += 1; hit += w.lower() in h
    print(f"{s:9s} text BLEU: ref-has-Latin {tl:6.2f} | no-Latin {tn:6.2f} || ASR BLEU: Latin {al:6.2f} | no-Latin {an:6.2f} || term recall {100*hit/tot:5.1f}%")
# per cell latin share
print()
for cell in sorted({c for c, _ in keys}):
    sub = [x for x in keys if x[0] == cell]
    L = sum(len(re.findall('[A-Za-z]', D['p3hk'][x]['ref'])) for x in sub) / sum(len(nc(D['p3hk'][x]['ref']).split()) for x in sub) * 100
    print(cell, "ref latin ch/100 %.2f" % L, " text BLEU H3-p3hk %+.2f" % (cb("p3pfh3", sub) - cb("p3hk", sub)), " ASR %+.2f" % (cb("p3pfh3", sub, "asr") - cb("p3hk", sub, "asr")))
# training data style
for d in ("en2zh",):
    rows = [json.loads(l) for l in open(f"eval_outputs/sst_hsmm_v1/data/train_{d}.jsonl")]
    refs = [r["ref"] for r in rows]
    L = sum(len(re.findall('[A-Za-z]', t)) for t in refs) / sum(len(nc(t).split()) for t in refs) * 100
    print(f"\nUniST train {d} refs: n={len(refs)} latin ch/100 = {L:.2f}; share with any Latin word {100*sum(1 for t in refs if LATW.search(t))/len(refs):.1f}%")
    srcs = [r.get("src_text") or r.get("src") or "" for r in rows]
    print("example train rows:", [(s[:80], t[:60]) for s, t in zip(srcs[:3], refs[:3])])
