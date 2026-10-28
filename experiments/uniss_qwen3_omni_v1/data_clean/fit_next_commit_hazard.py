"""Posterior of the next-commit hazard (docs/m2_bayesrisk_v2_model.zh-CN.md §2.3, §5).

Data: the "next" rows of si_bayesrisk --br-dump (training-domain dev only): at a read that commits, the covariates x
known then (NEXT_BASE, including the policy's bias and lambda) and J, the reads until the next committing read.
Model, per sentence n:
    P(J = j | J >= j, x, eta, c_n) = sigmoid(eta . next_x(x, j) + c_n),   c_n ~ N(0, tau_c^2),   eta ~ N(n0, V0),
with steps j < NEXT_J fitted (the last step absorbs J >= NEXT_J and is not fitted) and right censoring. The sentence
intercept is integrated out by Gauss-Hermite quadrature; tau_c by type-II maximum likelihood (empirical Bayes), eta by
its MAP with a Laplace (inverse Hessian) covariance. A covariate constant over the data (e.g. lambda before any run with
lambda > 0) is not identified: it is left out of the model (coefficient pinned at 0 with zero variance, listed in
"pinned"), so a later policy is evaluated with the hazard of the policy that produced the data (policy iteration),
not with prior noise. Reported: 2-fold cross-validated (salted sentence hash) held-out
NLL per step and AUC against the steps-only model.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.fit_next_commit_hazard --direction zh2en \
        --dump 'eval_outputs/.../dump/*.jsonl' --out next_zh2en.json
"""

from __future__ import annotations

import argparse
import glob
import json
import zlib
from pathlib import Path

import numpy as np
import torch

from experiments.uniss_qwen3_omni_v1.evaluation.si_bayesrisk import NEXT_BASE, NEXT_J, next_x

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(12)   # nodes / weights for E over N(0, 1)
GH_W = GH_W / GH_W.sum()


def load(pattern: str, direction: str):
    rows = []
    for f in sorted(glob.glob(pattern)):
        if f"_{direction}_" not in Path(f).name:
            continue
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("type") == "next":
                rows.append(r)
    return rows


def person_period(rows):
    """One row per fitted step: (sentence index, x, y)."""
    sid, X, y = [], [], []
    ids = sorted({r["id"] for r in rows})
    index = {k: i for i, k in enumerate(ids)}
    for r in rows:
        base = np.asarray(r["x"], float)
        for j in range(1, min(r["J"], NEXT_J - 1) + 1):
            sid.append(index[r["id"]])
            X.append(next_x(base, j))
            y.append(float(j == r["J"] and not r["cens"]))
    return np.asarray(sid), np.stack(X), np.asarray(y), ids


def neg_log_post(eta, log_tau, sid, X, y, n_sent, n0, v0):
    """-log prior(eta) - sum_n log int prod_rows Bernoulli(y | sigmoid(x eta + c)) N(c; 0, tau^2) dc."""
    z = X @ eta                                                     # [R]
    c = torch.exp(log_tau) * torch.as_tensor(GH_X, dtype=z.dtype)   # [Q]
    zz = z[:, None] + c[None, :]
    ll = y[:, None] * torch.nn.functional.logsigmoid(zz) + (1 - y[:, None]) * torch.nn.functional.logsigmoid(-zz)
    per = torch.zeros(n_sent, len(GH_X), dtype=z.dtype).index_add_(0, sid, ll)    # [N, Q]
    marg = torch.logsumexp(per + torch.log(torch.as_tensor(GH_W, dtype=z.dtype))[None], 1)
    prior = 0.5 * (((eta - n0) ** 2) / v0).sum()
    return prior - marg.sum()


def fit(sid, X, y, n_sent, n0, v0, iters=300):
    sid_t, X_t, y_t = torch.as_tensor(sid), torch.as_tensor(X), torch.as_tensor(y)
    eta = torch.tensor(n0, requires_grad=True)
    log_tau = torch.tensor(np.log(0.5), requires_grad=True)
    opt = torch.optim.LBFGS([eta, log_tau], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        f = neg_log_post(eta, log_tau, sid_t, X_t, y_t, n_sent, torch.as_tensor(n0), torch.as_tensor(v0))
        f.backward()
        return f

    opt.step(closure)
    f = lambda e: neg_log_post(e, log_tau.detach(), sid_t, X_t, y_t, n_sent, torch.as_tensor(n0), torch.as_tensor(v0))
    H = torch.autograd.functional.hessian(f, eta.detach())
    Sigma = torch.linalg.inv(H + 1e-9 * torch.eye(len(n0), dtype=H.dtype))
    return eta.detach().numpy(), float(torch.exp(log_tau).item()), Sigma.numpy()


def predict(eta, tau, X):
    """Posterior-mean-eta hazard with the sentence intercept integrated (a new sentence)."""
    z = X @ eta
    return (GH_W[None, :] / (1 + np.exp(-(z[:, None] + tau * GH_X[None, :])))).sum(1)


def auc(p, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction", choices=("en2zh", "zh2en"), required=True)
    ap.add_argument("--dump", required=True, help="glob of si_bayesrisk dump jsonl files")
    ap.add_argument("--prior-sd", type=float, default=2.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    torch.set_default_dtype(torch.float64)
    rows = load(args.dump, args.direction)
    sid, X, y, ids = person_period(rows)
    d = X.shape[1]
    n0 = np.zeros(d)
    h1 = y[X[:, len(NEXT_BASE):].sum(1) == 0].mean()
    n0[0] = float(np.log(h1) - np.log1p(-h1))
    v0 = np.full(d, args.prior_sd ** 2)
    keep = X.std(0) > 1e-9
    keep[0] = True
    keep[len(NEXT_BASE):] = True
    pinned = [n for n, k in zip(list(NEXT_BASE) + [f"j{j}" for j in range(2, NEXT_J + 1)], keep) if not k]
    steps_only = np.zeros(d, bool)
    steps_only[0] = True
    steps_only[len(NEXT_BASE):] = True

    cv = {"full": {"nll": [], "auc": []}, "steps_only": {"nll": [], "auc": []}}
    fold = np.array([zlib.crc32(f"hz|{k}".encode()) % 2 for k in ids])[sid]
    for k in (0, 1):
        tr, te = fold != k, fold == k
        for name, cols in (("full", keep), ("steps_only", steps_only)):
            remap = {s: i for i, s in enumerate(np.unique(sid[tr]))}
            e, t, _ = fit(np.array([remap[s] for s in sid[tr]]), X[tr][:, cols], y[tr], len(remap), n0[cols], v0[cols])
            p = np.clip(predict(e, t, X[te][:, cols]), 1e-9, 1 - 1e-9)
            cv[name]["nll"].append(float(-np.mean(y[te] * np.log(p) + (1 - y[te]) * np.log1p(-p))))
            cv[name]["auc"].append(auc(p, y[te]))
    e_k, tau, S_k = fit(sid, X[:, keep], y, len(ids), n0[keep], v0[keep])
    eta, Sigma = np.zeros(d), np.zeros((d, d))
    eta[keep] = e_k
    Sigma[np.ix_(keep, keep)] = S_k
    res = {"base": list(NEXT_BASE), "n_j": NEXT_J, "direction": args.direction, "mu": eta.tolist(),
           "Sigma": Sigma.tolist(), "tau_c": tau, "prior": {"n0": n0.tolist(), "sd": args.prior_sd},
           "pinned": pinned, "rows": len(rows), "periods": int(len(y)), "sentences": len(ids), "event_rate": float(y.mean()),
           "cv": {m: {k: float(np.mean(v)) for k, v in c.items()} for m, c in cv.items()},
           "coef": dict(zip(list(NEXT_BASE) + [f"j{j}" for j in range(2, NEXT_J + 1)], np.round(eta, 3).tolist())),
           "sd": dict(zip(list(NEXT_BASE) + [f"j{j}" for j in range(2, NEXT_J + 1)], np.round(np.sqrt(np.diag(Sigma)), 3).tolist()))}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("Sigma", "mu")}, indent=1))


if __name__ == "__main__":
    main()
