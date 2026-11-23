"""sst_hsmm_v1 post-hoc analysis (after P2 NO-GO; report only, CPU). On P0b's 1200 held-out items, with the teacher-forced
emissions of a checkpoint (step 0 = p0/em + initial gate; step k = p1/diag/ckpt_k/em + that gate), replays the INFERENCE
decision rule of e2e_si_thinker (per 1 s read, token i is written iff w = sigmoid(l + gate(s, f, started, c/C)) >= 0.5,
the read ends at the first wait; s follows the mean token duration SST_TOKMS; last read writes everything) and compares
it with the lattice's Viterbi path (model's own transitions). Separates the decision-rule mismatch from exposure to
the model's own prefix (which teacher forcing removes).
--rule survival: a candidate decision rule consistent with the semi-Markov hazard (median write time), analysis only.
    qwen3omni OMP_NUM_THREADS=1 PYTHONPATH=. python -m experiments.sst_hsmm_v1.analysis_greedy_tf --steps 0 2500 --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from experiments.sst_hsmm_v1 import hsmm as H

REPO = Path(__file__).resolve().parents[2]
P0 = REPO / "eval_outputs/sst_hsmm_v1/p0"
P1 = REPO / "eval_outputs/sst_hsmm_v1/p1"
DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
TOKS = {"en2zh": 0.3456, "zh2en": 0.3325}
LAM = (0.5, 0.5)
_G = {}
RULE = "threshold"
LAT = "proxy"                       # --lat laal: LAAL-form latency (hsmm.path_laal) instead of path_stats' absolute mean


def _init(gs):
    torch.set_num_threads(1)
    g = H.Gate()
    g.load_state_dict(gs)
    _G["W"] = [gs[k].double().numpy() for k in ("net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias")]
    _G["gate"] = g


def gval(s, f, st, cf):
    W1, b1, W2, b2 = _G["W"]
    return float(np.tanh(W1 @ np.array([s / 8.0, f, st, cf]) + b1) @ W2.T + b2)


def _one(job):
    r, f = job
    z = np.load(f)
    lp, lw = z["lp_tok"].astype(np.float64), z["lp_wait"].astype(np.float64)
    C = lp.shape[0]
    n = len(r["tok"])
    eos = lp.shape[1] == n + 1
    ell = np.log1p(-np.minimum(np.exp(lw), 1 - 1e-6)) - lw
    tok_s = TOKS[r["dir"]]
    # greedy replay of the inference rule
    s, fl, st, i, path, wfirst = 0.0, 0, 0, 0, [], []
    surv = 1.0                      # --rule survival: P(pending token not yet written) under the HSMM hazard
    for c in range(1, C + 1):
        if c == C:
            path += [C] * (n - i)
            break
        while i < n:
            gv = gval(s, fl, st, c / C)
            w = 1 / (1 + math.exp(-(ell[c - 1, i] + gv)))
            if c == r["g"][i]:
                wfirst.append(w)
            if RULE == "survival":
                if surv * (1 - w) > 0.5:      # write at the median of the token's write-time distribution
                    surv *= 1 - w
                    break
                surv = 1.0
            elif w < 0.5:
                break
            path.append(c)
            p = r["punct"][i]
            s = min(max(s, 0.0) + (0.0 if p else tok_s), 8.0)
            fl, st, i = int(p), 1, i + 1
        if i == n:
            break
        s = max(s - 1.0, -3.0)
    # tokens whose g_i is at a chunk the greedy path passed after writing them still need wfirst; fill from the
    # teacher-forced emissions at chunk g_i with a neutral state (s = 0, f = 0, started = i > 0)
    while len(wfirst) < n:
        k = len(wfirst)
        c = r["g"][k]
        wfirst.append(1.0 if c >= C else 1 / (1 + math.exp(-(ell[c - 1, k] + gval(0.0, 0, int(k > 0), c / C)))))
    dbin = [int(round(x / 100)) for x in r["d_ms"]]
    PS = H.path_laal if LAT == "laal" else H.path_stats
    lat_g, pau_g = PS(path, dbin, r["punct"], r["X"])
    g, db, pu = (r["g"] + [C], dbin + [0], r["punct"] + [1]) if eos else (r["g"], dbin, r["punct"])
    _, vp = H.lattice(torch.tensor(lp, dtype=torch.float32), torch.tensor(lw, dtype=torch.float32), g, db, pu, r["X"],
                      LAM[0], LAM[1], _G["gate"], viterbi=True, eos=eos)
    vp = vp[:n]
    lat_v, pau_v = PS(vp, dbin, r["punct"], r["X"])
    lat_e, pau_e = PS(list(r["g"]), dbin, r["punct"], r["X"])          # earliest possible: every token at g_i
    nf = [c < C for c in path]
    return {"dir": r["dir"], "lat_g": lat_g, "pau_g": pau_g, "lat_v": lat_v, "pau_v": pau_v, "lat_e": lat_e, "pau_e": pau_e,
            "early_g": float(np.mean([c < gi for c, gi in zip(path, r["g"])])),
            "dch_g": float(np.mean([c - gi for c, gi in zip(path, r["g"])])),
            "dch_v": float(np.mean([c - gi for c, gi in zip(vp, r["g"])])),
            "nonfinal_g": float(np.mean(nf)), "nonfinal_v": float(np.mean([c < C for c in vp])),
            "wfirst": [w for w, gi in zip(wfirst, r["g"]) if gi < C]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--rule", choices=("threshold", "survival"), default="threshold",
                    help="threshold: inference rule (w >= 0.5); survival: write when the pending token's survival <= 0.5")
    ap.add_argument("--lat", choices=("proxy", "laal"), default="proxy")
    a = ap.parse_args()
    global RULE, LAT
    RULE, LAT = a.rule, a.lat
    if Path(a.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    rows = {}
    for d in ("en2zh", "zh2en"):
        for l in open(DATA / f"heldout_{d}.jsonl", encoding="utf-8"):
            x = json.loads(l)
            rows[x["id"]] = x
    p0 = json.load(open(P0 / "summary.json"))
    res = {}
    for step in a.steps:
        if step == 0:
            g = H.Gate()
            g.net[2].bias.data.fill_(float(p0["P0b"]["gate_bias_init"]))
            gs, em = g.state_dict(), P0 / "em"
        else:
            gs = torch.load(P1 / "train" / f"ckpt_{step:05d}.pt", map_location="cpu", weights_only=False)["gate"]
            em = P1 / "diag" / f"ckpt_{step:05d}" / "em"
        jobs = [(rows[f.stem.split("__", 1)[1]], str(f)) for f in sorted(em.glob("*.npz"))]
        with mp.get_context("fork").Pool(a.procs, _init, (gs,)) as pool:
            per = pool.map(_one, jobs, chunksize=4)
        out = {}
        for d in ("en2zh", "zh2en", "all"):
            v = [x for x in per if d == "all" or x["dir"] == d]
            wf = np.concatenate([x["wfirst"] for x in v])
            out[d] = {"n": len(v), **{k: round(float(np.mean([x[k] for x in v])), 4) for k in
                      ("lat_g", "lat_v", "lat_e", "pau_e", "pau_g", "pau_v", "early_g", "dch_g", "dch_v", "nonfinal_g", "nonfinal_v")},
                      "wfirst_ge_0.5": round(float((wf >= 0.5).mean()), 4),
                      "wfirst_0.1_0.5": round(float(((wf >= 0.1) & (wf < 0.5)).mean()), 4),
                      "wfirst_lt_0.1": round(float((wf < 0.1).mean()), 4),
                      "wfirst_median": round(float(np.median(wf)), 4)}
        res[str(step)] = out
        print(step, json.dumps(out["all"]), flush=True)
    with open(a.out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
