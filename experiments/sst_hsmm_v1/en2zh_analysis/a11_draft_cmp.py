"""compare per-read draft and commit lengths of two systems on the same test items (non-final reads)."""
import glob, json, sys
from collections import defaultdict
A, B, d = sys.argv[1], sys.argv[2], sys.argv[3]
def load(s):
    R = {}
    for f in glob.glob(f"eval_outputs/sst_hsmm_v1/{s}/thinker_{d}_*/blocks_*.jsonl"):
        for l in open(f):
            r = json.loads(l); R[r["id"]] = r
    return R
RA, RB = load(A), load(B)
ids = sorted(set(RA) & set(RB))
zh = d == "en2zh"
U = (lambda s: [c for c in s if not c.isspace()]) if zh else (lambda s: s.split())
def lcp(a, b):
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]: k += 1
    return k
st = defaultdict(float)
for i in ids:
    fa, fb = U(RA[i]["translation"]), U(RB[i]["translation"])
    DA = {x["t"]: x for x in RA[i]["drafts"]}; DB = {x["t"]: x for x in RB[i]["drafts"]}
    ca = cb = 0  # committed units so far
    ba = {b["write_ms"]: b for b in RA[i]["blocks"]}; bb = {b["write_ms"]: b for b in RB[i]["blocks"]}
    for t in sorted(set(DA) & set(DB)):
        if DA[t].get("final") or t == max(DA): break
        da, db = U(DA[t]["draft"]), U(DB[t]["draft"])
        ca += len(U(ba[t]["text"])) if t in ba else 0
        cb += len(U(bb[t]["text"])) if t in bb else 0
        st["reads"] += 1
        st["draftA"] += len(da); st["draftB"] += len(db)
        st["commA"] += ca; st["commB"] += cb
        st["commB_beyond_draftA"] += max(0, cb - len(da))       # B committed more than A's whole draft holds
        st["commB_beyond_commA"] += max(0, cb - ca)
        st["draftA_final_lcp"] += lcp(da, fa); st["draftB_final_lcp"] += lcp(db, fb)
n = st["reads"]
print(d, A, "vs", B, "items", len(ids), "reads", int(n))
for k in ("draftA", "draftB", "commA", "commB", "commB_beyond_commA", "commB_beyond_draftA", "draftA_final_lcp", "draftB_final_lcp"):
    print(f"  {k:22s} {st[k]/n:7.2f}")
