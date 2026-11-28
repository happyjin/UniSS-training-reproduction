"""overnight 2026-11-28: paired bootstrap (1000, per-cell item resampling, mean of the 5 cell corpus BLEUs) of ASR-BLEU and text BLEU
(official wave2_s1_score.Bleu) of systems vs p3hk, from a pf_comet_dump file (rows: sys, dir, cell, id, ref, text, asr).
Report only (verdicts use the point estimates of the change 55 rule).
    <qwen3omni python> experiments/sst_hsmm_v1/pv8_bleu_boot.py <dump.jsonl> <sys> [<sys> ...]"""
import collections, json, random, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu  # noqa: E402
D = collections.defaultdict(dict)
for l in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(l); D[(r["sys"], r["dir"])][(r["cell"], r["id"])] = r
rng = np.random.default_rng(0)
for d in ("en2zh", "zh2en"):
    bl = Bleu("cmn" if d == "en2zh" else "eng")
    for s in sys.argv[2:]:
        A, B = D[(s, d)], D[("p3hk", d)]
        keys = sorted(set(A) & set(B))
        cells = collections.defaultdict(list)
        for x in keys:
            cells[x[0]].append(x)
        out = []
        for k in ("asr", "text"):
            st = {}
            for c, ks in cells.items():
                st[c] = (bl.stats([A[x][k] for x in ks], [A[x]["ref"] for x in ks]), bl.stats([B[x][k] for x in ks], [B[x]["ref"] for x in ks]))
            point = np.mean([bl.score(a) - bl.score(b) for a, b in st.values()])
            bs = []
            for _ in range(1000):
                v = []
                for c, (a, b) in st.items():
                    idx = rng.integers(0, len(a), len(a))
                    v.append(bl.score(a[idx]) - bl.score(b[idx]))
                bs.append(np.mean(v))
            bs = sorted(bs)
            out.append("%s %+.2f [%+.2f, %+.2f]" % (k, point, bs[25], bs[974]))
        print(d, s, "vs p3hk:", " | ".join(out), flush=True)
