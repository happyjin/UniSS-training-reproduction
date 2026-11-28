"""overnight 2026-11-28 diagnosis (report only): at every non-final read, where and how does the draft first leave the
system's own final translation?  units: en2zh = non-space chars, zh2en = words/punct tokens.
  backlog   = lcp(draft, final) - committed   (correct but not yet committed: what a perfect commit rule could still write)
  type of the first divergence (draft[lcp] vs final[lcp]):
    exhausted : draft ends at lcp (nothing more drafted)
    closure   : draft[lcp] is punctuation (。！？，,.!?;；) while the final goes on with content
    insertion : draft[lcp:lcp+2] reappears later in final[lcp+1:lcp+25] -> the final put new material BEFORE the drafted
                continuation (re-ordering: later source content lands earlier in the target)
    rewrite   : anything else (lexical / structural rewrite)
    python3 experiments/sst_hsmm_v1/en2zh_analysis/a13_divergence_types.py <sys> [<sys> ...]
"""
import collections, glob, json, re, sys
PUN = set("。！？，、,.!?;；:：")
for d in ("en2zh", "zh2en"):
    U = (lambda s: [c for c in s if not c.isspace()]) if d == "en2zh" else (lambda s: re.findall(r"[\w'’-]+|[^\w\s]", s.lower()))
    for s in sys.argv[1:]:
        st = collections.Counter()
        for f in glob.glob(f"eval_outputs/sst_hsmm_v1/{s}/thinker_{d}_*/blocks_*.jsonl"):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                fin = U(r["translation"])
                bw = {b["write_ms"]: b for b in r["blocks"]}
                com = 0
                ts = sorted(x["t"] for x in r["drafts"])
                Dd = {x["t"]: x for x in r["drafts"]}
                for t in ts[:-1]:
                    if Dd[t].get("final"): break
                    com += len(U(bw[t]["text"])) if t in bw else 0
                    dr = U(Dd[t]["draft"])
                    k = 0
                    while k < len(dr) and k < len(fin) and dr[k] == fin[k]: k += 1
                    st["reads"] += 1; st["draft"] += len(dr); st["comm"] += com; st["lcp"] += k
                    st["backlog"] += max(0, k - com)
                    if k >= len(fin): st["t_done"] += 1; continue
                    if k >= len(dr): st["t_exhausted"] += 1; continue
                    if dr[k] in PUN and fin[k] not in PUN: st["t_closure"] += 1; continue
                    nxt = dr[k:k + 2]
                    win = fin[k + 1:k + 25]
                    if len(nxt) == 2 and any(win[j:j + 2] == nxt for j in range(len(win) - 1)): st["t_insertion"] += 1; continue
                    st["t_rewrite"] += 1
        n = st["reads"]
        typ = " ".join(f"{k[2:]} {100*st[k]/n:4.1f}%" for k in ("t_exhausted", "t_closure", "t_insertion", "t_rewrite", "t_done"))
        print(f"{d} {s:8s} reads {n:6d} | draft {st['draft']/n:5.2f} comm {st['comm']/n:5.2f} lcp {st['lcp']/n:5.2f} "
              f"backlog {st['backlog']/n:4.2f} ({st['backlog']/max(1,st['comm']):.2f} of comm) | {typ}")
