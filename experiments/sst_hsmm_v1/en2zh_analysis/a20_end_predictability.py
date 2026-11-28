# dev only (pfb0 base spec heal@0.75): can the last non-final read (source ends within the next 1000 ms) be predicted from the draft?
# f1 = log-prob of the draft's closing punctuation token (draft ends with 。？！.?!), f2 = draft content unchanged vs previous read
import glob, json, numpy as np
D = "eval_outputs/sst_hsmm_v1/pfb0/dev"
def auc(s, y):
    s, y = np.asarray(s), np.asarray(y); o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
CL = set("。？！.?!")
for dv in ("edev", "cvdev"):
    for d in ("en2zh", "zh2en"):
        dur = {r["id"]: r["dur_ms"] for r in map(json.loads, open(f"{D}/{dv}/data/items_{d}.jsonl"))}
        S1, S2, Y = [], [], []
        for f in glob.glob(f"{D}/{dv}/run_{d}_*/blocks_{d}_heal_0.75.jsonl"):
            for r in map(json.loads, open(f)):
                prev = None
                for x in r["drafts"]:
                    if x["t"] >= dur[r["id"]]: continue
                    dr = x.get("draft") or ""; lp = x.get("lp0") or []
                    core = dr.rstrip("".join(CL))
                    S1.append(lp[-1] if lp and dr and dr[-1] in CL else -20.0)
                    S2.append(1.0 if prev is not None and core == prev else 0.0); prev = core
                    Y.append(1 if dur[r["id"]] - x["t"] <= 1000 else 0)
        Y = np.array(Y); S1 = np.array(S1)
        print(f"{dv} {d}: non-final reads {len(Y)}, last-read share {Y.mean():.3f} | AUC closure-logprob {auc(S1, Y):.3f} | AUC unchanged {auc(np.array(S2), Y):.3f}",
              "| P(last | closure lp > log 0.9) %.3f  recall %.3f" % (Y[S1 > np.log(0.9)].mean(), (Y[S1 > np.log(0.9)].sum() / Y.sum())))
