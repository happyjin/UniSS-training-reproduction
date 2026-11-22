"""sst_hsmm_v1 P1a checkpoint diagnostic, CPU part (reports/sst_hsmm_v1/prereg.zh-CN.md change 3; report only).
Per checkpoint (step 0 = p0/em + the gate at its initial bias; step k = p1/diag/ckpt_<k>/em + that checkpoint's gate), on
P0b's 1200 held-out items: tilted loss / token (lambda 0.5 / 0.5, the checkpoint's gate), emission NLL / token at the last
chunk and at chunk g_i, mean P_wait and median l over writable non-final positions, and the Viterbi path under the
model's own transitions (Thinker + gate): latency proxy, in-phrase pauses, mean (chunk - g_i). By direction x condition.
    qwen3omni OMP_NUM_THREADS=1 PYTHONPATH=. python -m experiments.sst_hsmm_v1.p1_diag_score --steps 0 500 1000 --out <json>
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from experiments.sst_hsmm_v1 import hsmm as H

REPO = Path(__file__).resolve().parents[2]
P0 = REPO / "eval_outputs/sst_hsmm_v1/p0"
P1 = REPO / "eval_outputs/sst_hsmm_v1/p1"
DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
LAM_L, LAM_P = 0.5, 0.5
_G = {}


def _init(gate_state):
    torch.set_num_threads(1)
    g = H.Gate()
    g.load_state_dict(gate_state)
    _G["gate"] = g


def _one(job):
    key, r, f = job
    z = np.load(f)
    lp, lw = z["lp_tok"].astype(np.float32), z["lp_wait"].astype(np.float32)
    C, n = lp.shape
    dbin = [int(round(x / 100)) for x in r["d_ms"]]
    gate = _G["gate"]
    tot, _ = H.lattice(torch.tensor(lp), torch.tensor(lw), r["g"], dbin, r["punct"], r["X"], LAM_L, LAM_P, gate)
    _, path = H.lattice(torch.tensor(lp), torch.tensor(lw), r["g"], dbin, r["punct"], r["X"], LAM_L, LAM_P, gate, viterbi=True)
    lat, pauses = H.path_stats(path, dbin, r["punct"], r["X"])
    pw, ell = [], []
    for i, gi in enumerate(r["g"]):
        for c in range(gi, C):
            pw.append(float(np.exp(lw[c - 1, i])))
            ell.append(float(np.log1p(-min(np.exp(lw[c - 1, i]), 1 - 1e-6)) - lw[c - 1, i]))
    return key, {"tilted": float(-tot) / n, "nll_last": float(-lp[C - 1].mean()),
                 "nll_g": float(-np.mean([lp[gi - 1, i] for i, gi in enumerate(r["g"])])),
                 "pw": pw, "ell": ell, "lat": lat, "pauses": pauses,
                 "dchunk": float(np.mean([c - gi for c, gi in zip(path, r["g"])]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--p1-root", default=str(P1), help="P1 run root (train/ckpt_*.pt, diag/ckpt_*/em)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rows = {}
    for d in ("en2zh", "zh2en"):
        for l in open(DATA / f"heldout_{d}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            rows[r["id"]] = r
    p0 = json.load(open(P0 / "summary.json"))
    res = {"lam": [LAM_L, LAM_P], "steps": {}}
    for step in a.steps:
        if step == 0:
            g = H.Gate()
            g.net[2].bias.data.fill_(float(p0["P0b"]["gate_bias_init"]))
            gs, em = g.state_dict(), P0 / "em"
        else:
            root = Path(a.p1_root)
            gs = torch.load(root / "train" / f"ckpt_{step:05d}.pt", map_location="cpu", weights_only=False)["gate"]
            em = root / "diag" / f"ckpt_{step:05d}" / "em"
        files = sorted(em.glob("*.npz"))
        jobs = []
        for f in files:
            cond, rid = f.stem.split("__", 1)
            jobs.append(((rows[rid]["dir"], cond), rows[rid], str(f)))
        with mp.get_context("fork").Pool(a.procs, _init, (gs,)) as pool:
            per = pool.map(_one, jobs, chunksize=4)
        groups = {}
        for key, v in per:
            for gk in (f"{key[0]}/{key[1]}", "all"):
                groups.setdefault(gk, []).append(v)
        sr = {}
        for gk, vs in sorted(groups.items()):
            pw = np.concatenate([v["pw"] for v in vs]) if vs else np.array([])
            ell = np.concatenate([v["ell"] for v in vs]) if vs else np.array([])
            sr[gk] = {"n": len(vs), **{k: round(float(np.mean([v[k] for v in vs])), 4)
                                       for k in ("tilted", "nll_last", "nll_g", "lat", "pauses", "dchunk")},
                      "p_wait_mean": round(float(pw.mean()), 4) if len(pw) else None,
                      "ell_median": round(float(np.median(ell)), 3) if len(ell) else None}
        sr["gate_bias"] = round(float(gs["net.2.bias"]), 4)
        res["steps"][str(step)] = sr
        print(step, json.dumps(sr["all"]), flush=True)
    res["ref_P0c"] = p0.get("P0c", {}).get("la2") if isinstance(p0.get("P0c"), dict) else None
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
