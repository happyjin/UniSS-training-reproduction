"""sst_hsmm_v1 prereg change 27 (E2): train the SDC survival head on the E1 rollout features.
Input: <root>/roll_<dir>_<k>/feats_<dir>.npz (X, y, ids). Split by item id (sha1 % 10 == 0 -> validation). MLP sdc.Head,
BCE, Adam lr 1e-3, batch 4096, <= 30 epochs, early stop on validation loss (patience 3). One head for both directions
(direction is a feature). Saves <out>/head.pt + head.json (AUC, base rates, calibration bins). Refuses to overwrite.
    <qwen3omni python> -m experiments.sst_hsmm_v1.train_sdc --root eval_outputs/sst_hsmm_v1/e1 --out eval_outputs/sst_hsmm_v1/e2
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1 import sdc as SD  # noqa: E402


def auc(y, p):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "head.pt").exists():
        sys.exit(f"拒绝覆盖已有输出: {out}")
    out.mkdir(parents=True, exist_ok=True)
    X, y, ids = [], [], []
    for f in sorted(glob.glob(f"{a.root}/roll_*/feats_*.npz")):
        z = np.load(f)
        X.append(z["X"]); y.append(z["y"]); ids.extend(z["ids"].tolist())
    X = np.concatenate(X).astype(np.float32); y = np.concatenate(y).astype(np.float32)
    val = np.array([int(hashlib.sha1(i.encode()).hexdigest(), 16) % 10 == 0 for i in ids])
    print(json.dumps({"rows": len(y), "pos_rate": float(y.mean()), "val_rows": int(val.sum())}), flush=True)
    torch.manual_seed(20261125)
    h = SD.Head()
    opt = torch.optim.Adam(h.parameters(), lr=1e-3)
    Xt, yt = torch.from_numpy(X[~val]), torch.from_numpy(y[~val])
    Xv, yv = torch.from_numpy(X[val]), torch.from_numpy(y[val])
    bce = torch.nn.BCEWithLogitsLoss()
    best, bad, best_state = 1e9, 0, None
    for ep in range(30):
        h.train()
        perm = torch.randperm(len(yt))
        for s in range(0, len(perm), 4096):
            b = perm[s: s + 4096]
            loss = bce(h(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        h.eval()
        with torch.no_grad():
            vl = float(bce(h(Xv), yv))
        print(f"epoch {ep} val_bce {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, {k: v.clone() for k, v in h.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    h.load_state_dict(best_state)
    h.eval()
    with torch.no_grad():
        pv = torch.sigmoid(h(Xv)).numpy()
    bins = []
    for lo in np.arange(0, 1, 0.1):
        m = (pv >= lo) & (pv < lo + 0.1)
        if m.sum():
            bins.append([round(float(lo), 1), int(m.sum()), round(float(yv.numpy()[m].mean()), 3)])
    # simple baselines on validation: rules as survival estimators (precision of the units they would commit)
    yvn = yv.numpy()
    rules = {"la2 (i < agreed)": X[val][:, 20] > 0.5, "cv (i < cv)": X[val][:, 21] > 0.5, "sap (i < sv)": X[val][:, 22] > 0.5,
             "la2&cv (N)": (X[val][:, 20] > 0.5) & (X[val][:, 21] > 0.5)}
    rule_prec = {k: [int(m.sum()), round(float(yvn[m].mean()), 4) if m.sum() else None] for k, m in rules.items()}
    meta = {"rows": len(y), "val_rows": int(val.sum()), "pos_rate": float(y.mean()), "val_bce": best, "val_auc": auc(yvn, pv),
            "calibration": bins, "rule_n_precision": rule_prec, "nf": SD.NF}
    torch.save(h.state_dict(), out / "head.pt")
    (out / "head.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
