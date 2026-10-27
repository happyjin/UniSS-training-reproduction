"""Fit the bayesrisk survival posterior (docs/m2_test_time_bayes_model.zh-CN.md, sec. 2-3) on training-domain reads.

Input: the --br-dump files of si_thinker_stream --policy bayesrisk run on the op_v3 dev items (UniST rows held out of
the adapter's training, data/si_thinker_oracle_prefix_v1/dev_items_*.jsonl). Each read gives the chain labels of its
stable length S. Per direction:
  * theta's posterior N(mu, Sigma): Laplace around the MAP under the prior N(m0, S0), m0 = the ABC logit with weight 1;
  * 2-fold cross-validation by sentence: NLL / ECE / AUC of the ABC posterior alone (theta = m0) and of the fit (A1);
  * dispersion: mean (sum of a sentence's Pearson residuals)^2 / its rows -- 1 when its reads are independent (A4);
  * Psi (between-group variance of the bias and ABC weights): empirical Bayes, Laplace marginal likelihood on a grid,
    over the source corpora of the items (the training domain has no talks); --psi fixes it instead.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.fit_bayesrisk_posterior --dumps 'dir/*.jsonl' --out model.json
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import re
import zlib
from pathlib import Path

import numpy as np

from experiments.uniss_qwen3_omni_v1.evaluation.si_bayesrisk import FEATS, label_rows, laplace_update, prior_model

TALK_DIMS = (FEATS.index("bias"), FEATS.index("abc"))


def load(pattern: str, direction: str, items: str):
    corpus = {}
    for line in open(items.format(d=direction), encoding="utf-8"):
        r = json.loads(line)
        corpus[r["id"]] = re.sub(r"^\d+_", "", Path(r["audio"]).stem).split("_")[0]
    X, y, sid, grp = [], [], [], []
    for f in sorted(glob.glob(pattern)):
        if f"_{direction}_" not in Path(f).name:
            continue
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            R, lab = label_rows(np.asarray(r["X"], float).reshape(-1, len(FEATS)), int(r["S"]))
            X.append(R)
            y.append(lab)
            sid += [r["id"]] * len(lab)
            grp += [corpus.get(r["id"], "?")] * len(lab)
    return np.concatenate(X), np.concatenate(y), np.asarray(sid), np.asarray(grp)


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    nll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    bins = np.minimum((p * 10).astype(int), 9)
    ece = float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any()))
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = y.sum(), len(y) - y.sum()
    auc = float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))
    return {"nll": round(nll, 4), "ece": round(ece, 4), "auc": round(auc, 4), "n": int(len(y))}


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit(X, y, m0, S0):
    return laplace_update(m0, S0, X, y)


def group_evidence(X, y, grp, mu, psi):
    """Laplace log marginal likelihood of the groups' data under delta_g ~ N(0, diag(psi)) on TALK_DIMS."""
    total = 0.0
    for g in np.unique(grp):
        m = grp == g
        Xg, yg = X[m][:, TALK_DIMS], y[m]
        base = X[m] @ mu
        P0 = np.diag(np.maximum(psi, 1e-6))
        d = np.zeros(2)
        for _ in range(30):
            s = sig(base + Xg @ d)
            grad = np.linalg.solve(P0, d) - Xg.T @ (yg - s)
            H = np.linalg.inv(P0) + (Xg * (s * (1 - s))[:, None]).T @ Xg
            step = np.linalg.solve(H, grad)
            d -= step
            if np.abs(step).max() < 1e-7:
                break
        s = np.clip(sig(base + Xg @ d), 1e-9, 1 - 1e-9)
        H = np.linalg.inv(P0) + (Xg * (s * (1 - s))[:, None]).T @ Xg
        ll = (yg * np.log(s) + (1 - yg) * np.log(1 - s)).sum()
        lp = -0.5 * d @ np.linalg.solve(P0, d) - 0.5 * np.linalg.slogdet(2 * np.pi * P0)[1]
        total += ll + lp + 0.5 * np.linalg.slogdet(2 * np.pi * np.linalg.inv(H))[1]
    return float(total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", required=True, help="glob of --br-dump jsonl files")
    ap.add_argument("--items", default="data/si_thinker_oracle_prefix_v1/dev_items_{d}.jsonl")
    ap.add_argument("--prior-sd", type=float, default=3.0, help="S0 = prior-sd^2 I")
    ap.add_argument("--psi", default="", help="fix Psi on (bias, abc) as 'a,b' instead of empirical Bayes")
    ap.add_argument("--min-group-rows", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    m0 = prior_model()["mu"]
    S0 = np.eye(len(FEATS)) * args.prior_sd ** 2
    res = {"feats": list(FEATS), "dirs": {}}
    for d in ("en2zh", "zh2en"):
        X, y, sid, grp = load(args.dumps, d, args.items)
        if not len(y):
            continue
        fold = np.array([zlib.crc32(s.encode()) % 2 for s in sid])
        cv_prior, cv_fit, cv_y = [], [], []
        for k in (0, 1):
            tr, te = fold != k, fold == k
            mu_k, _ = fit(X[tr], y[tr], m0, S0)
            cv_prior.append(sig(X[te] @ m0))
            cv_fit.append(sig(X[te] @ mu_k))
            cv_y.append(y[te])
        cv_y = np.concatenate(cv_y)
        mu, Sigma = fit(X, y, m0, S0)
        s = sig(X @ mu)
        r = (y - s) / np.sqrt(s * (1 - s))
        per = {}
        for i, v in zip(sid, r):
            a = per.setdefault(i, [0.0, 0])
            a[0] += v
            a[1] += 1
        disp = float(np.mean([a[0] ** 2 for a in per.values()]) / np.mean([a[1] for a in per.values()]))
        groups = {g: int((grp == g).sum()) for g in np.unique(grp)}
        keep = np.isin(grp, [g for g, n in groups.items() if n >= args.min_group_rows])
        psi_src, grid = "fixed", None
        if args.psi:
            psi2 = np.array([float(v) for v in args.psi.split(",")])
        elif len({g for g, n in groups.items() if n >= args.min_group_rows}) >= 3:
            vals = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
            grid = {f"{a},{b}": group_evidence(X[keep], y[keep], grp[keep], mu, np.array([a, b]))
                    for a, b in itertools.product(vals, vals)}
            psi2 = np.array([float(v) for v in max(grid, key=grid.get).split(",")])
            psi_src = "empirical Bayes over source corpora"
        else:
            psi2 = np.array([0.1, 0.05])
            psi_src = "hyper-prior (fewer than 3 source corpora with enough rows)"
        psi = np.zeros(len(FEATS))
        psi[list(TALK_DIMS)] = psi2
        res["dirs"][d] = {"mu": mu.round(5).tolist(), "Sigma": Sigma.tolist(), "psi": psi.tolist(), "psi_source": psi_src,
                          "psi_grid": grid, "rows": int(len(y)), "sentences": len(per), "groups": groups,
                          "dispersion": round(disp, 3),
                          "cv_abc_prior": metrics(np.concatenate(cv_prior), cv_y),
                          "cv_fitted": metrics(np.concatenate(cv_fit), cv_y),
                          "mu_by_feat": dict(zip(FEATS, mu.round(3).tolist())),
                          "sd_by_feat": dict(zip(FEATS, np.sqrt(np.diag(Sigma)).round(4).tolist()))}
        print(d, json.dumps({k: v for k, v in res["dirs"][d].items() if k not in ("Sigma", "mu", "psi_grid")}), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
