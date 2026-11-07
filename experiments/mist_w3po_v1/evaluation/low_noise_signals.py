"""Two low-noise direction signals attached to every gate report (user decision 2026-11-07 16:25 UTC; they do NOT change
the gate verdict):
  (i)  dev mean STAMP R of the candidate vs v8 ctx0, window by window (same online dev windows, the training reward
       function rl/stamp.credit, diag_reward_noise.one): mean paired difference, paired bootstrap over windows (10k)
       with 95% CI and P(diff <= 0), plus a meeting-cluster bootstrap for comparison;
  (ii) the training-R trend of the RL run: least-squares slope of R over steps (per 10 steps), with its 95% CI, and the
       per-10-step means (last record per step wins: resumes repeat steps).
Writes <cand>/low_noise_signals.json and, if <cand>/gate.json exists, adds the same dict under "low_noise_signals".

    python -m experiments.mist_w3po_v1.evaluation.low_noise_signals <cand decode dir> --run <rl out dir> \
        [--ref eval_outputs/mist_w3po_v1/decode/dev_v8_ctx0]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation.diag_reward_noise import WIN, one, ref_utts, window_utts


def window_R(decode_dir: Path, W, U, workers: int) -> dict:
    cache = decode_dir / "window_R.json"
    if cache.exists():
        return json.loads(cache.read_text())
    H = {h["id"]: h for h in map(json.loads, open(decode_dir / "hyp.jsonl", encoding="utf-8"))}
    jobs = [(w, U[w["id"]], H.get(w["id"], {})) for w in W if U[w["id"]]]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(one, jobs, chunksize=16))
    out = {r["id"]: {"R": r["R"], "meeting": r["meeting"]} for r in rows}
    cache.write_text(json.dumps(out))
    return out


def train_trend(run: Path) -> dict:
    m = {}
    for l in open(run / "metrics.jsonl", encoding="utf-8"):
        r = json.loads(l)
        m[r["step"]] = r
    steps = np.array(sorted(m), dtype=float)
    R = np.array([m[int(s)]["R"] for s in steps])
    X = np.vstack([steps, np.ones_like(steps)]).T
    coef, res, *_ = np.linalg.lstsq(X, R, rcond=None)
    n = len(steps)
    if n > 2:
        s2 = float(np.sum((R - X @ coef) ** 2) / (n - 2))
        se = float(np.sqrt(s2 / np.sum((steps - steps.mean()) ** 2)))
    else:
        se = float("nan")
    per10 = {f"{a}-{a + 9}": float(np.mean([m[s]["R"] for s in range(a, a + 10) if s in m]))
             for a in range(1, int(steps.max()) + 1, 10) if any(s in m for s in range(a, a + 10))}
    return {"steps": n, "slope_per_10_steps": float(coef[0] * 10), "slope_ci95_per_10": [float((coef[0] - 1.96 * se) * 10),
            float((coef[0] + 1.96 * se) * 10)], "R_per_10_steps": per10}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cand")
    ap.add_argument("--run", default="")
    ap.add_argument("--ref", default="eval_outputs/mist_w3po_v1/decode/dev_v8_ctx0")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    W = [json.loads(l) for l in open(WIN, encoding="utf-8")]
    by = ref_utts()
    U = {w["id"]: window_utts(w, by) for w in W}
    A = window_R(Path(a.ref), W, U, a.workers)
    B = window_R(Path(a.cand), W, U, a.workers)
    ids = sorted(set(A) & set(B))
    d = np.array([B[i]["R"] - A[i]["R"] for i in ids])
    rng = np.random.default_rng(0)
    bw = d[rng.integers(0, len(d), size=(a.n, len(d)))].mean(1)
    meet = sorted({A[i]["meeting"] for i in ids})
    dm = {m: d[[k for k, i in enumerate(ids) if A[i]["meeting"] == m]] for m in meet}
    sums = np.array([dm[m].sum() for m in meet])
    cnts = np.array([len(dm[m]) for m in meet])
    sel = rng.integers(0, len(meet), size=(a.n, len(meet)))
    bm = sums[sel].sum(1) / cnts[sel].sum(1)
    out = {"dev_R": {"ref": a.ref, "windows": len(ids), "mean_R_ref": float(np.mean([A[i]["R"] for i in ids])),
                     "mean_R_cand": float(np.mean([B[i]["R"] for i in ids])), "diff": float(d.mean()),
                     "window_boot_ci95": [float(np.percentile(bw, 2.5)), float(np.percentile(bw, 97.5))],
                     "window_boot_p_le0": float(np.mean(bw <= 0)),
                     "meeting_boot_ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                     "meeting_boot_p_le0": float(np.mean(bm <= 0))}}
    if a.run:
        out["train_R_trend"] = train_trend(Path(a.run))
    cand = Path(a.cand)
    (cand / "low_noise_signals.json").write_text(json.dumps(out, indent=1))
    g = cand / "gate.json"
    if g.exists():
        gj = json.loads(g.read_text())
        gj["low_noise_signals"] = out   # informational; the verdict ("pass") is unchanged
        tmp = cand / ".gate.json.tmp"
        tmp.write_text(json.dumps(gj, indent=1))
        tmp.replace(g)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
