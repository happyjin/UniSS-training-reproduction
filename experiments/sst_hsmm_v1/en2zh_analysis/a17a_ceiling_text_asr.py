# 2026-11-28 en2zh BLEU-vs-COMET analysis (report only, existing test outputs, CPU): qwen3omni python experiments/sst_hsmm_v1/en2zh_analysis/a17a_ceiling_text_asr.py
import json, glob, sys, collections
sys.path.insert(0, ".")
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu
D0 = "eval_outputs/sst_hsmm_v1"
rows = [json.loads(l) for l in open(f"{D0}/pv8/comet/dump.jsonl", encoding="utf-8")]
X = collections.defaultdict(dict)
for r in rows: X[(r["sys"], r["dir"])][r["id"]] = r
for d in ("en2zh", "zh2en"):
    bl = Bleu("cmn" if d == "en2zh" else "eng")
    ceil = {r["id"]: r["translation"] for f in glob.glob(f"{D0}/ceiling/offline_{d}_*.jsonl") for r in map(json.loads, open(f, encoding="utf-8"))}
    base = X[("p3hk", d)]; cells = sorted({r["cell"] for r in base.values()})
    def m(get):
        sc = []
        for c in cells:
            ids = [i for i, r in base.items() if r["cell"] == c and i in ceil]
            sc.append(bl.score(bl.stats([get(i) for i in ids], [base[i]["ref"] for i in ids])))
        return sum(sc) / len(sc), sc
    print(d, "items", len(base), "with ceiling", sum(i in ceil for i in base))
    v, sc = m(lambda i: ceil[i]); print(f"  ceiling(text)  {v:6.2f}  " + " ".join(f"{x:5.1f}" for x in sc))
    for s in ("B_b", "N", "p3hk", "p3hk2", "p3pfg2"):
        if (s, d) not in X: continue
        t, sct = m(lambda i: X[(s, d)][i]["text"]); a, sca = m(lambda i: X[(s, d)][i]["asr"])
        print(f"  {s:7s} text {t:6.2f} asr {a:6.2f} loss {t-a:5.2f} | text " + " ".join(f"{x:5.1f}" for x in sct) + " | asr " + " ".join(f"{x:5.1f}" for x in sca))
    print("  cells", cells)
