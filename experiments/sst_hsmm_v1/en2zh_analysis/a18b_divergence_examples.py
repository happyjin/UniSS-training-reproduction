import collections, glob, json, sys, random
sys.path.insert(0, ".")
from evaluation.text_metrics import normalize_for_bleu
E0 = "eval_outputs/sst_hsmm_v1"; d = "en2zh"
U = lambda t: tuple(normalize_for_bleu(t, "cmn").split())
ceil = {r["id"]: r["translation"] for f in glob.glob(f"{E0}/ceiling/offline_{d}_*.jsonl") for r in map(json.loads, open(f, encoding="utf-8"))}
dump = {r["id"]: r for r in map(json.loads, open(f"{E0}/pv8/comet/dump.jsonl", encoding="utf-8")) if r["sys"] == "p3hk" and r["dir"] == d}
rows = {r["id"]: r for f in glob.glob(f"{E0}/p3hk/thinker_{d}/blocks_*.jsonl") for r in map(json.loads, open(f, encoding="utf-8"))}
def lcp(a, b):
    j = 0
    while j < len(a) and j < len(b) and a[j] == b[j]: j += 1
    return j
def dtype(a, b, j):
    for s in range(1, 13):
        for u in range(0, 7):
            v = s - u
            if 0 <= v <= 6 and len(a[j+u:j+u+2]) == 2 and a[j+u:j+u+2] == b[j+v:j+v+2]:
                return ("local_sub" if u == v else "local_indel"), "".join(a[j:j+u]), "".join(b[j:j+v])
    return "structural", "".join(a[j:j+8]), "".join(b[j:j+8])
ex = collections.defaultdict(list); ins = collections.Counter(); dele = collections.Counter()
for i in dump:
    if i not in rows or i not in ceil: continue
    blocks = [b for b in rows[i]["blocks"] if not b["final"]]
    e_ = U("".join(b["text"] for b in blocks)); c_ = U(ceil[i])
    if e_ == c_[:len(e_)]: continue
    p = lcp(e_, c_); t, a, b = dtype(e_, c_, p)
    if t == "local_indel":
        if a and not b: ins[a] += 1
        elif b and not a: dele[b] += 1
        else: ins[f"{a}->{b}"] += 1
    ex[t].append((i, "".join(c_[max(0,p-6):p]), a, b, dump[i]["src"][:110], "".join(U(dump[i]["ref"]))[:60]))
print("local_indel: early has extra (E-only) top:", ins.most_common(25))
print("local_indel: ceiling has extra (C-only) top:", dele.most_common(25))
random.seed(1)
for t in ("local_indel", "structural"):
    print("\n####", t)
    for x in random.sample(ex[t], 10):
        print(f"  ctx[{x[1]}] early[{x[2]}] ceiling[{x[3]}] | SRC: {x[4]} | REF: {x[5]}")
