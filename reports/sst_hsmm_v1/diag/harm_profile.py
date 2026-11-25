"""diag (iteration 3): profile of harmful commits in CR-LA training rollouts (r/regret.jsonl): by commit index, read time,
commit length, punctuation end; regret mass share; simple-feature predictability (logistic regression AUC, 5-fold)."""
import json, sys
import numpy as np
from collections import defaultdict
R = [json.loads(l) for l in open("eval_outputs/sst_hsmm_v1/r/regret.jsonl")]
rows = []
for x in R:
    zh = x["dir"] == "en2zh"; prev = ""
    for k, e in enumerate(x["events"]):
        new = e["pre"][len(prev):]
        rows.append(dict(d=x["dir"], k=k, t=e["t"], frac=e["t"] / x["dur_ms"], nnew=len(new) if zh else len(new.split()),
                         npre=len(e["pre"]) if zh else len(e["pre"].split()), punct=int(new.strip()[-1:] in "，。,.!?！？；;：:、"), r=e["r"]))
        prev = e["pre"]
for d in ("en2zh", "zh2en"):
    L = [r for r in rows if r["d"] == d]
    tot = -sum(min(0, r["r"]) for r in L)
    print(d, "events", len(L), "harm", sum(r["r"] < 0 for r in L), "sum r", round(sum(r["r"] for r in L) / len(R), 2), "per rollout")
    for key, f in (("k", lambda r: min(r["k"], 3)), ("t", lambda r: min(r["t"] // 1000, 6)), ("frac", lambda r: round(min(r["frac"], .99) * 4) / 4),
                   ("nnew", lambda r: min(r["nnew"], 8)), ("punct", lambda r: r["punct"])):
        g = defaultdict(list)
        for r in L: g[f(r)].append(r["r"])
        print(" ", key, {k: (len(v), round(np.mean([x < 0 for x in v]), 2), round(-sum(min(0, x) for x in v) / tot, 2), round(np.mean(v), 2)) for k, v in sorted(g.items())})
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
for d in ("en2zh", "zh2en"):
    L = [r for r in rows if r["d"] == d]
    X = np.array([[r["k"], r["t"] / 1000, r["nnew"], r["npre"], r["punct"]] for r in L]); y = np.array([r["r"] < 0 for r in L])
    print(d, "AUC harm (text/time feats)", round(cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5, scoring="roc_auc").mean(), 3))
