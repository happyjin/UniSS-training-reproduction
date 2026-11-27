"""portfolio 2026-11-27 method C1 (prereg change 55): group-conditional conformal risk control of the commit threshold, calibrated on
the B0 dev runs (pfb0, wave17 + h14, edev + cvdev, grid mode@0.85..0.60, --shadow-final). A unit committed at a non-final read is an
error when its index is >= LCP(units(final translation), units(shadow final draft)). Groups = direction x tercile of the read's
reliability r_t (cut points from all non-final reads of the mode@0.75 runs). alpha(direction) = pooled error rate of mode@0.75.
Per group: the smallest theta in the grid with (n/(n+1)) * R_hat + 1/(n+1) <= alpha (n = committed units of the group at that theta);
none -> the largest theta. Writes {dir: {edges, theta, alpha, table}} for hsc22 --theta-map. CPU only; refuses to overwrite.
    python3 experiments/sst_hsmm_v1/pf_c1_calib.py --ref eval_outputs/sst_hsmm_v1/pfb0 --out eval_outputs/sst_hsmm_v1/pfc1_theta.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units  # noqa: E402

MODE = {"en2zh": "heal", "zh2en": "base"}


def lcp(a, b):
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k


def events(root, d, theta):
    """(r, committed units, errors) per non-final commit event of the mode@theta runs (both devsets)"""
    zh = d.endswith("zh")
    out = []
    for f in sorted(glob.glob(str(Path(root) / "dev" / "*" / f"run_{d}_*" / f"blocks_{d}_{MODE[d]}_{theta:.2f}.jsonl"))):
        for r in map(json.loads, open(f, encoding="utf-8")):
            fin = units(r["translation"], zh)
            sh = r["drafts"][-1].get("shadow")
            if sh is None:
                raise SystemExit(f"no shadow in {f}")
            ok = lcp(fin, units(sh, zh))
            n = 0
            for x in r["drafts"][:-1]:
                c = int(x.get("cv_new", 0) or 0)
                if c and "r" in x:
                    out.append((float(x["r"]), c, sum(1 for i in range(n, n + c) if i >= ok)))
                n += c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"refuse to overwrite {a.out}")
    grid = sorted({float(re.search(r"_(\d\.\d\d)\.jsonl$", f).group(1))
                   for f in glob.glob(str(Path(a.ref) / "dev" / "*" / "run_en2zh_0" / "blocks_en2zh_*.jsonl"))})
    res = {}
    for d in ("en2zh", "zh2en"):
        rs = []
        for f in sorted(glob.glob(str(Path(a.ref) / "dev" / "*" / f"run_{d}_*" / f"blocks_{d}_{MODE[d]}_0.75.jsonl"))):
            for r in map(json.loads, open(f, encoding="utf-8")):
                rs += [float(x["r"]) for x in r["drafts"][:-1] if "r" in x]
        edges = [float(np.quantile(rs, 1 / 3)), float(np.quantile(rs, 2 / 3))]
        grp = lambda r_: sum(1 for e in edges if r_ >= e)
        E = {th: events(a.ref, d, th) for th in grid}
        e75 = E[0.75]
        alpha = sum(x[2] for x in e75) / max(1, sum(x[1] for x in e75))
        table, theta = defaultdict(dict), []
        for g in range(3):
            pick = None
            for th in grid:
                n = sum(x[1] for x in E[th] if grp(x[0]) == g)
                e = sum(x[2] for x in E[th] if grp(x[0]) == g)
                rh = e / max(1, n)
                bound = n / (n + 1) * rh + 1 / (n + 1)
                table[g][f"{th:.2f}"] = {"n": n, "err": e, "risk": round(rh, 4), "bound": round(bound, 4)}
                if pick is None and bound <= alpha:
                    pick = th
            theta.append(pick if pick is not None else max(grid))
        res[d] = {"edges": edges, "theta": theta, "alpha": alpha, "grid": grid, "table": table}
        print(d, json.dumps({"edges": [round(x, 3) for x in edges], "theta": theta, "alpha": round(alpha, 4)}), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
