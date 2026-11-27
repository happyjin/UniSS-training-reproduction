"""portfolio 2026-11-27: paired per-cell bootstrap of segment COMET (pf_comet_score rows) between two systems (2000 resamples, seed 0).
    python3 experiments/sst_hsmm_v1/pf_comet_pair.py <comet.json.rows.jsonl> <X> <Y> [key=comet_asr|comet_text]"""
import json, random, sys
from collections import defaultdict
rows, X, Y = sys.argv[1], sys.argv[2], sys.argv[3]
key = sys.argv[4] if len(sys.argv) > 4 else "comet_asr"
S = defaultdict(dict)
for l in open(rows):
    r = json.loads(l)
    if r["sys"] in (X, Y):
        S[(r["dir"], r["cell"])].setdefault(r["id"], {})[r["sys"]] = r[key]
rng = random.Random(0)
for (d, c), v in sorted(S.items()):
    diff = [x[X] - x[Y] for x in v.values() if X in x and Y in x]
    n = len(diff); m = sum(diff) / n
    bs = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    lo, hi = bs[49], bs[1949]
    print(f"{d} {c:13s} n={n} {key} {X}-{Y} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]{' *' if lo > 0 or hi < 0 else ''}")
