"""commit-lock vs drafter: per item, share of the final translation written before the final read; BLEU by bucket; paired items."""
import json, collections, re, sys, glob, unicodedata
sys.path.insert(0, ".")
from evaluation.text_metrics import remove_punctuation
from sacrebleu.metrics import BLEU
bl = BLEU(tokenize="zh")
def nz(t): return " ".join(c for c in remove_punctuation(unicodedata.normalize("NFKC", t)) if not c.isspace())
D = {}
for l in open("eval_outputs/sst_hsmm_v1/pv3/comet/dump.jsonl"):
    r = json.loads(l)
    if r["dir"] == "en2zh": D[(r["sys"], r["id"])] = r
ids = sorted({i for (s, i) in D if s == "p3hk"})
def blocks(sys_):
    B = {}
    for f in glob.glob(f"eval_outputs/sst_hsmm_v1/{sys_}/thinker_en2zh_*/blocks_*.jsonl"):
        for l in open(f):
            r = json.loads(l); B[r["id"]] = r
    return B
SYS = ["p3hk", "p3hk2", "p3p2fxff", "p3pfe1", "p3pfh3", "p3pfe1d1", "p3pfd1"]
EARLY = {}
for s in SYS:
    B = blocks(s); E = {}
    for i in ids:
        r = B.get(i)
        if r is None: continue
        tot = len(nz(r["translation"]).split()); early = sum(len(nz(b["text"]).split()) for b in r["blocks"] if not b["final"])
        E[i] = early / max(1, tot)
    EARLY[s] = E
    print(s, "items", len(E), "mean early share %.3f" % (sum(E.values()) / len(E)), "share of items with nothing early %.3f" % (sum(1 for v in E.values() if v == 0) / len(E)))
def cb(sys_, sub):
    H = [nz(D[(sys_, i)]["text"]) for i in sub]; R = [nz(D[(sys_, i)]["ref"]) for i in sub]
    return bl.corpus_score(H, [R]).score
print("\n# pairs vs p3hk: items where BOTH commit nothing before the final read (= both outputs are whole-audio translations)")
for s in SYS[1:]:
    both0 = [i for i in ids if EARLY["p3hk"].get(i) == 0 and EARLY[s].get(i) == 0]
    s_early = [i for i in ids if EARLY["p3hk"].get(i, 1) > 0 or EARLY[s].get(i, 1) > 0]
    print(f"{s:9s} n0={len(both0):4d}  whole-audio subset text BLEU {s} {cb(s, both0):6.2f} vs p3hk {cb('p3hk', both0):6.2f} (d {cb(s, both0)-cb('p3hk', both0):+.2f}) | "
          f"rest n={len(s_early)}: {cb(s, s_early):6.2f} vs {cb('p3hk', s_early):6.2f} (d {cb(s, s_early)-cb('p3hk', s_early):+.2f})")
print("\n# by early-share bucket of the method (text BLEU method vs p3hk on the same items)")
for s in ("p3pfe1", "p3pfh3", "p3p2fxff"):
    for lo, hi in ((0, 1e-9), (1e-9, 0.34), (0.34, 0.67), (0.67, 1.01)):
        sub = [i for i in ids if i in EARLY[s] and lo <= EARLY[s][i] < hi]
        if sub: print(f"{s:9s} early [{lo:.2f},{hi:.2f}) n={len(sub):4d}  {cb(s, sub):6.2f} vs p3hk {cb('p3hk', sub):6.2f}  d {cb(s, sub)-cb('p3hk', sub):+.2f}")
