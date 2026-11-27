"""what goes wrong in the early-committed part: char precision of early text vs tail, length excess, self-repetition"""
import json, collections, re, sys, glob, unicodedata
sys.path.insert(0, ".")
from evaluation.text_metrics import remove_punctuation
def nz(t): return [c for c in remove_punctuation(unicodedata.normalize("NFKC", t)) if not c.isspace()]
DIR = sys.argv[1] if len(sys.argv) > 1 else "en2zh"
D = {}
for l in open("eval_outputs/sst_hsmm_v1/pv3/comet/dump.jsonl"):
    r = json.loads(l)
    if r["dir"] == DIR: D[(r["sys"], r["id"])] = r
ids = sorted({i for (s, i) in D if s == "p3hk"})
def grams(c, n): return collections.Counter(tuple(c[k:k + n]) for k in range(len(c) - n + 1))
def clip(h, r, n=1):
    H, R = grams(h, n), grams(r, n)
    return sum(min(v, R[g]) for g, v in H.items()), sum(H.values())
def rep4(c):
    G = grams(c, 4); return sum(v - 1 for v in G.values() if v > 1)
SYS = ["p3hk", "p3hk2", "p3pfd1", "p3p2fxff", "p3pfe1", "p3pfh3", "p3pfe1d1", "p3pfh4", "p3d3s05"]
print(f"[{DIR}] sys       | early: share  p1   p2  | tail: p1   p2  | len/ref | rep4/100 (ref %.2f)" % (
    100 * sum(rep4(nz(D[("p3hk", i)]["ref"])) for i in ids) / sum(len(nz(D[("p3hk", i)]["ref"])) for i in ids)))
for s in SYS:
    B = {}
    for f in glob.glob(f"eval_outputs/sst_hsmm_v1/{s}/thinker_{DIR}_*/blocks_*.jsonl"):
        for l in open(f):
            r = json.loads(l); B[r["id"]] = r
    if not B: continue
    a = collections.defaultdict(int)
    for i in ids:
        r = B[i]; ref = nz(D[(s, i)]["ref"])
        early = sum((nz(b["text"]) for b in r["blocks"] if not b["final"]), [])
        tail = sum((nz(b["text"]) for b in r["blocks"] if b["final"]), [])
        # early part vs the whole reference; tail vs whole reference minus what early already matched
        for n in (1, 2):
            m, t = clip(early, ref, n); a[f"em{n}"] += m; a[f"et{n}"] += t
            m2, t2 = clip(early + tail, ref, n); a[f"tm{n}"] += m2 - m; a[f"tt{n}"] += t2 - t
        a["early"] += len(early); a["all"] += len(early) + len(tail); a["ref"] += len(ref)
        a["rep"] += rep4(early + tail)
    f = lambda m, t: 100 * a[m] / max(1, a[t])
    print(f"{s:9s} |       {a['early']/a['all']:.3f} {f('em1','et1'):5.1f} {f('em2','et2'):5.1f} |    {f('tm1','tt1'):5.1f} {f('tm2','tt2'):5.1f} | {a['all']/a['ref']:.3f}  | {100*a['rep']/a['all']:.2f}")
