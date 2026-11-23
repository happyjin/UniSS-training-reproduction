"""sst_hsmm_v1 P5 budgets (prereg change 8, rule fixed before any P5 number). Playout proxy (proxy.item_proxy) of every
HSMM dev system that has a P2-style summary (p2, p2b, p2c, p2d_t070, p2d_t085, p2d_t095), per direction OLS fits
    D2_diff = a + b * L_dev        NP_in = a' + b' * P_dev
-> L* = (D2_TARGET - a) / b, P* = (NP_TARGET_REL * NP_in(ctrl) - a') / b' with D2_TARGET = -0.10 s, NP_TARGET_REL = 0.70;
budgets relative to P1a on dev (= p2): dL = L* - L_dev(p2), rho = P* / P_dev(p2). Writes <out> (refuses existing).
    qwen3omni PYTHONPATH=. python -m experiments.sst_hsmm_v1.calib_budget --out eval_outputs/sst_hsmm_v1/p5/budget.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.sst_hsmm_v1 import proxy as PX

REPO = Path(__file__).resolve().parents[2]
SYSTEMS = ["p2", "p2b", "p2c", "p2d_t070", "p2d_t085", "p2d_t095"]
D2_TARGET, NP_TARGET_REL = -0.10, 0.70


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    tk = PX.tok_fn()
    data = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"
    dur = {json.loads(l)["id"]: json.loads(l)["dur_ms"] for d in ("en2zh", "zh2en") for l in open(data / f"items_{d}.jsonl")}
    res = {}
    for d in ("en2zh", "zh2en"):
        pts = []
        for name in SYSTEMS:
            root = REPO / "eval_outputs/sst_hsmm_v1" / name
            sm = json.load(open(root / "summary.json"))["pooled"][d]
            v = PX.system(root / f"thinker_{d}" / f"blocks_{d}_w1.jsonl", d, tk, lambda x: dur[x["id"]])
            L = float(np.mean([x[0] for x in v.values()])); P = float(np.mean([x[1] for x in v.values()]))
            pts.append({"sys": name, "L": L, "P": P, "d2_diff": sm["d2"]["diff"], "np_in": sm["NP_in"]["arm"],
                        "np_ctrl": sm["NP_in"]["ref"]})
        L_, D_ = np.array([p["L"] for p in pts]), np.array([p["d2_diff"] for p in pts])
        P_, N_ = np.array([p["P"] for p in pts]), np.array([p["np_in"] for p in pts])
        b, a0 = np.polyfit(L_, D_, 1); b2, a2 = np.polyfit(P_, N_, 1)
        r2 = lambda x, y, s, i: 1 - float(((y - (i + s * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        if b <= 0 or b2 <= 0:
            raise SystemExit(f"{d}: non-positive slope (b={b}, b'={b2}): proxy unusable, stop")
        np_ctrl = pts[0]["np_ctrl"]
        Ls, Ps = (D2_TARGET - a0) / b, (NP_TARGET_REL * np_ctrl - a2) / b2
        if Ps <= 0:
            raise SystemExit(f"{d}: P* = {Ps} <= 0: pause budget infeasible, stop")
        p2 = pts[0]
        res[d] = {"dL": Ls - p2["L"], "rho": Ps / p2["P"], "L_star": Ls, "P_star": Ps, "fit_L": [a0, b, r2(L_, D_, b, a0)],
                  "fit_P": [a2, b2, r2(P_, N_, b2, a2)], "np_ctrl": np_ctrl, "points": pts}
        print(d, json.dumps({k: res[d][k] for k in ("dL", "rho", "L_star", "P_star", "fit_L", "fit_P")}), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
