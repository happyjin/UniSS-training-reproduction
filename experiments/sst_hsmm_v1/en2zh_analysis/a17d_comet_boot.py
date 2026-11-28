# 2026-11-28 en2zh BLEU-vs-COMET analysis (report only, existing test outputs, CPU): qwen3omni python experiments/sst_hsmm_v1/en2zh_analysis/a17d_comet_boot.py
"""change 56: paired bootstrap (1000, per-cell resampling, mean of 5 cell means) of COMET vs p3hk from pv8/comet/comet.json.rows.jsonl.
    python3 experiments/sst_hsmm_v1/pv3_comet_boot.py > eval_outputs/sst_hsmm_v1/pv3/comet_boot.txt"""
import collections, json, random
R = collections.defaultdict(dict)
for l in open("eval_outputs/sst_hsmm_v1/pv8/comet/comet.json.rows.jsonl"):
    r = json.loads(l); R[(r["sys"], r["dir"])][(r["cell"], r["id"])] = (r["comet_asr"], r["comet_text"])
random.seed(0)
def boot(a, b, d, k):
    A, B = R[(a, d)], R[(b, d)]
    cells = collections.defaultdict(list)
    for x in A:
        if x in B: cells[x[0]].append(x)
    stat = lambda S: sum(sum(A[x][k] - B[x][k] for x in s) / len(s) for s in S.values()) / len(S)
    bs = sorted(stat({c: [random.choice(v) for _ in v] for c, v in cells.items()}) for _ in range(1000))
    return stat(cells), bs[25], bs[974]
for d in ("en2zh", "zh2en"):
    for s in ("p3pfg2", "p3pfg3", "p3pfe1", "p3pfh3", "p3pfe1d1", "p3p2fxff", "p3pfd1", "p3d3s05", "p3pfs2", "p3pfk1", "p3hk2", "N", "B_b"):
        if (s, d) in R:
            print(d, s, "vs p3hk:", " | ".join("%s %+.4f [%+.4f,%+.4f]" % ((n,) + boot(s, "p3hk", d, k)) for k, n in ((0, "ASR"), (1, "text"))))
