"""prereg change 45 (D2 free precheck, report only): the blocking-aware threshold applied to the P0-R forced-commit forks (p0r).
For each group (item, fork read) the rule's extra units beyond the n2 cut are computed from the null clone's posterior P at the fork
read; the fork with the largest available forced count <= extra is taken (0 -> null); dD2 / d sentence BLEU vs the null clone,
summed per sentence (first-order), averaged over all sentences of both halves. kappa in {0.10, 0.15, 0.20}, theta 0.75."""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1.p0rl_analyze import playback  # noqa: E402
from experiments.voice_onset_v1.playout import si_units  # noqa: E402
import sacrebleu
R = REPO / "eval_outputs/sst_hsmm_v1/p0r"
res = {}
for d in ("en2zh", "zh2en"):
    zh = d.endswith("zh"); lang = "cmn" if zh else "eng"; tk = "zh" if zh else "13a"
    items = {r["id"]: r for r in map(json.loads, open(R / "data" / f"items_{d}.jsonl", encoding="utf-8"))}
    forks = defaultdict(dict)
    for f in glob.glob(str(R / f"run_{d}_*" / f"forks_{d}.jsonl")):
        for r in map(json.loads, open(f, encoding="utf-8")):
            forks[(r["id"], r["fork_t"])][r["fork_k"]] = r
    def st(row, it):
        d2, _, _ = playback(row["blocks"], lang, it["dur_ms"], len(si_units(it["translation"], lang)))
        return d2, sacrebleu.sentence_bleu(row["translation"], [it["translation"]], tokenize=tk).score
    cache = {}
    for kap in (0.10, 0.15, 0.20):
        tot_d2 = tot_bl = 0.0; taken = 0
        for (iid, t), cl in forks.items():
            if 0 not in cl:
                continue
            f0 = cl[0]["drafts"][0]; P = f0.get("P", []); n = f0["forced"]["n"]; c0 = f0["forced"]["cut0"]
            j = c0 - n
            while j < len(P):
                b = 0
                while j + 1 + b < len(P) and P[j + 1 + b] >= 0.75:
                    b += 1
                if P[j] >= max(0.5, 1 - kap * (1 + b)):
                    j += 1 + b
                else:
                    break
            extra = j - (c0 - n)
            if extra <= 0:
                continue
            avail = {}
            for k, r in cl.items():
                m = r["drafts"][0]["forced"]["cut"] - r["drafts"][0]["forced"]["cut0"]
                if k > 0 and m > 0:
                    avail.setdefault(m, k)
            ms = [m for m in avail if m <= extra]
            if not ms:
                continue
            k = avail[max(ms)]
            key = (iid, t, k)
            if key not in cache:
                it = items[iid]
                a, b_ = st(cl[k], it), st(cl[0], it)
                cache[key] = (a[0] - b_[0], a[1] - b_[1])
            tot_d2 += cache[key][0]; tot_bl += cache[key][1]; taken += 1
        ns = len(items)
        res[f"{d}_kappa{kap}"] = {"dd2_per_sent": tot_d2 / ns, "dbleu_per_sent": tot_bl / ns, "groups_taken": taken}
        print(d, kap, res[f"{d}_kappa{kap}"], flush=True)
out = R / "p45_pandora_pre.json"
assert not out.exists()
json.dump(res, open(out, "x"), indent=1)
