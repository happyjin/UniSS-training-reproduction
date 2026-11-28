"""overnight 2026-11-28 diagnosis (report only): how do consecutive drafts change in the uncommitted region?
for consecutive non-final reads t -> t+1 (units as a13), j = lcp(d_t, d_t+1), only counted when j >= committed(t+1)-free region:
  extend     : d_t is a prefix of d_t+1 (draft only grew)
  closure    : d_t[j] is punctuation and d_t+1[j] is content (draft had closed, now continues)
  local_sub  : the drafts re-synchronise within 6 units with the same span length (u == v): a same-length swap = word choice
  local_indel: re-synchronise within 6 units, different span lengths (local insertion / deletion)
  structural : no re-synchronisation within 6 units
also reports how often a local_sub is undone later (d_t+2 goes back to d_t's span) = flicker between near-ties.
    python3 experiments/sst_hsmm_v1/en2zh_analysis/a14_draft_flips.py <sys> [<sys> ...]
"""
import collections, glob, json, re, sys
PUN = set("。！？，、,.!?;；:：")


def classify(a, b):
    j = 0
    while j < len(a) and j < len(b) and a[j] == b[j]: j += 1
    if j >= len(a): return "extend", j, None
    if a[j] in PUN and (j >= len(b) or b[j] not in PUN): return "closure", j, None
    for s in range(1, 13):
        for u in range(0, 7):
            v = s - u
            if v < 0 or v > 6: continue
            if a[j + u:j + u + 2] == b[j + v:j + v + 2] and len(a[j + u:j + u + 2]) == 2:
                return ("local_sub" if u == v else "local_indel"), j, (a[j:j + u], b[j:j + v])
    return "structural", j, None


for d in ("en2zh", "zh2en"):
    U = (lambda s: [c for c in s if not c.isspace()]) if d == "en2zh" else (lambda s: re.findall(r"[\w'’-]+|[^\w\s]", s.lower()))
    for s in sys.argv[1:]:
        st = collections.Counter(); ex = []
        for f in glob.glob(f"eval_outputs/sst_hsmm_v1/{s}/thinker_{d}_*/blocks_*.jsonl"):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                ts = sorted(x["t"] for x in r["drafts"] if not x.get("final"))
                Dd = {x["t"]: U(x["draft"]) for x in r["drafts"]}
                for k in range(len(ts) - 1):
                    a, b = Dd[ts[k]], Dd[ts[k + 1]]
                    c, j, span = classify(a, b)
                    st[c] += 1; st["n"] += 1
                    if c == "local_sub":
                        st["sub_len"] += len(span[0])
                        if k + 2 < len(ts):
                            c2 = Dd[ts[k + 2]]
                            if c2[j:j + len(span[0])] == span[0]: st["flicker"] += 1
                        if len(ex) < 12 and d == "en2zh": ex.append(("".join(span[0]), "".join(span[1])))
        n = st["n"]
        print(f"{d} {s:8s} pairs {n:6d} | " + " ".join(f"{k} {100*st[k]/n:4.1f}%" for k in ("extend", "closure", "local_sub", "local_indel", "structural"))
              + f" | sub len {st['sub_len']/max(1,st['local_sub']):.2f} flicker-back {100*st['flicker']/max(1,st['local_sub']):.1f}%")
        if ex: print("   e.g.", " ; ".join(f"{x}->{y}" for x, y in ex))
