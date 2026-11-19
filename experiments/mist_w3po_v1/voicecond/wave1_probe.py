"""Robust wave 1, K4 P4a: linear rate probe on the T1c mid-layer state, and the P4b clamp targets.

fit   features = wave1_talker feats (layer-14 state averaged over the code positions of each cvt teacher sequence, g*);
      target = speaking rate in target-text tokens per second of speech (n_text / (n_codes * 0.02)).
      Ridge on standardised features, fitted on cvt TRAIN only (zh2en_train + en2zh_train, 8000); the ridge strength by
      generalised cross-validation on the train set (alphas 10^-2 .. 10^4, 13 values; no dev data). Evaluated on cvt dev
      (800): pooled R^2 = 1 - SS_res / SS_tot (gate: >= 0.5), also per direction (each about its own mean).
      Rate axis in the raw hidden space: r = (w / s) / ||w / s||; rate_hat = scale * r'h + b0.
      -> <out>/probe.npz {r, scale, b0, alpha}, <out>/p4a.json
rhat  P4b clamp target per item: rhat = mean over the TTS seeds of r' hbar_B (hbar = the job's mean mid state over its
      generated code positions, wave1_talker gen hmid.npz of arm B, unclamped) -> <out>.json
    <python> -m experiments.mist_w3po_v1.voicecond.wave1_probe fit --feats <dir>[,<dir>...] --out <dir>
    <python> -m experiments.mist_w3po_v1.voicecond.wave1_probe rhat --probe <dir>/probe.npz --gen <B gen dir> --out <json>
Refuses existing outputs.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def cmd_fit(a):
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    zs = [np.load(Path(d) / "feats.npz") for d in a.feats.split(",")]
    ids = np.concatenate([z["id"] for z in zs])
    sets = np.concatenate([z["set"] for z in zs])
    X = np.concatenate([z["feat"] for z in zs]).astype(np.float64)
    y = np.concatenate([z["n_text"] / (z["nc"] * 0.02) for z in zs]).astype(np.float64)
    assert len(set(ids.tolist())) == len(ids)
    tr = np.array([s.endswith("_train") for s in sets])
    dv = ~tr
    m, s = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = (X - m) / s
    ym = y[tr].mean()
    U, S, Vt = np.linalg.svd(Z[tr], full_matrices=False)
    Uy = U.T @ (y[tr] - ym)
    n = int(tr.sum())
    best = None
    gcv = {}
    for al in np.logspace(-2, 4, 13):
        f = S ** 2 / (S ** 2 + al)
        fit = U @ (f * Uy)
        res = (y[tr] - ym) - fit
        df = f.sum()
        g = float((res ** 2).mean() / (1 - df / n) ** 2)
        gcv[f"{al:.3g}"] = g
        if best is None or g < best[0]:
            best = (g, al)
    al = best[1]
    w = Vt.T @ ((S / (S ** 2 + al)) * Uy)
    pred = Z @ w + ym
    res = {"n_train": n, "n_dev": int(dv.sum()), "alpha": float(al), "gcv": gcv,
           "R2_train": r2(y[tr], pred[tr]), "R2_dev": r2(y[dv], pred[dv]),
           "R2_dev_per_dir": {d: r2(y[dv & np.char.startswith(sets.astype(str), d)], pred[dv & np.char.startswith(sets.astype(str), d)])
                              for d in ("zh2en", "en2zh")},
           "rate_dev": {"mean": float(y[dv].mean()), "std": float(y[dv].std())},
           "rate_dev_per_dir": {d: {"mean": float(y[dv & np.char.startswith(sets.astype(str), d)].mean()),
                                    "std": float(y[dv & np.char.startswith(sets.astype(str), d)].std())} for d in ("zh2en", "en2zh")}}
    res["gate_R2_dev_ge_0.5"] = bool(res["R2_dev"] >= 0.5)
    wr = w / s
    scale = float(np.linalg.norm(wr))
    r = wr / scale
    b0 = float(ym - (w * m / s).sum())
    assert np.allclose(scale * (X @ r) + b0, pred, atol=1e-6 * max(1.0, np.abs(pred).max()))
    out.mkdir(parents=True)
    np.savez(out / "probe.npz", r=r.astype(np.float32), scale=scale, b0=b0, alpha=al)
    (out / "p4a.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def cmd_rhat(a):
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    z = np.load(a.probe)
    g = np.load(Path(a.gen) / "hmid.npz")
    acc = defaultdict(list)
    for k, h in zip(g["key"], g["h"]):
        arm, seed, iid = str(k).split("__", 2)
        assert arm == "B"
        acc[iid].append(float(h.astype(np.float64) @ z["r"].astype(np.float64)))
    rh = {i: float(np.mean(v)) for i, v in acc.items()}
    out.write_text(json.dumps(rh, indent=1))
    pr = np.array([float(z["scale"]) * v + float(z["b0"]) for v in rh.values()])
    print(json.dumps({"n": len(rh), "pred_rate_B_mean": float(pr.mean()), "pred_rate_B_std": float(pr.std())}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fit")
    s.add_argument("--feats", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("rhat")
    s.add_argument("--probe", required=True)
    s.add_argument("--gen", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"fit": cmd_fit, "rhat": cmd_rhat}[a.cmd](a)


if __name__ == "__main__":
    main()
