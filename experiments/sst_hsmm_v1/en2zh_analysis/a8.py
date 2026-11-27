import json, collections, re, sys, unicodedata
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
LATW = re.compile(r"[A-Za-z]")
D = collections.defaultdict(dict)
for l in open("eval_outputs/sst_hsmm_v1/pv3/comet/dump.jsonl"):
    r = json.loads(l)
    if r["dir"] == "en2zh": D[r["sys"]][(r["cell"], r["id"])] = r
keys = sorted(D["p3hk"]); cells = sorted({c for c, _ in keys})
def m5(get, k):
    return sum(bl.corpus_score([nc(get(x)[k]) for x in keys if x[0] == c], [[nc(D["p3hk"][x]["ref"]) for x in keys if x[0] == c]]).score for c in cells) / len(cells)
for s in ("p3pfh3", "p3pfe1", "p3p2fxff"):
    for k in ("asr", "text"):
        base = m5(lambda x: D["p3hk"][x], k); full = m5(lambda x: D[s][x], k)
        swapL = m5(lambda x: D["p3hk"][x] if LATW.search(D["p3hk"][x]["ref"]) else D[s][x], k)
        print(f"{s:9s} {k:4s}: gap vs p3hk {full-base:+.2f} | Latin-ref items swapped to p3hk -> {swapL-base:+.2f}  (Latin items explain {100*(full-swapL)/(full-base):.0f}%)")
# lexical drift on non-Latin RealSI items: bigrams H3 overuses vs p3hk that are NOT in the ref
sub = [x for x in keys if not LATW.search(D["p3hk"][x]["ref"])]
def bg(t): t = nc(t).replace(" ", ""); return collections.Counter(t[i:i+2] for i in range(len(t) - 1))
over = collections.Counter(); under = collections.Counter()
for x in sub:
    R, A, B = bg(D["p3hk"][x]["ref"]), bg(D["p3hk"][x]["text"]), bg(D["p3pfh3"][x]["text"])
    for g in set(A) | set(B):
        d = B[g] - A[g]
        if d > 0 and R[g] == 0: over[g] += d
        if d < 0 and R[g] > 0: under[g] += -d
print("\nH3 adds (not in ref):", over.most_common(30))
print("\nH3 drops (were in ref):", under.most_common(30))
