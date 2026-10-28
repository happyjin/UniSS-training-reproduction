"""Step 0 of the sentence-level test-time filter (docs/m2_bayesrisk_sf_model.zh-CN.md): does a sentence's own revision
history, observed while it is being read, rank the survival of the current draft better than ABC?

Model, per decoding run g of a sentence (one bias, one sentence):
  prior        b_g ~ N(0, tau^2)                                    (the sentence's volatility)
  target       P(S_k >= u+1 | S_k >= u, x, b) = sigmoid(theta . x_{k,u} + b)    (final stability, the decision quantity)
  interim      P(V_{k,u} = 1 | V_{k,u-1} = 1, x, b) = sigmoid(eta . x_{k,u} + gamma b)
               V_{k,u}: unit u of read k's draft is unchanged in read k+1's draft (observed one read later)
  filter       p(b | interim chains of reads < k) on a Gauss-Hermite grid; the decision at read k uses the posterior
               predictive of the target chain under it (and, inside the read, the chain's own conditioning).
theta, eta, gamma: MAP under N(m0, 3^2) (m0: the ABC logit, weight 1), N(1, 1) for gamma; tau: type-II maximum
likelihood (b integrated out by Gauss-Hermite). tau = 0 and theta = m0 is ABC, i.e. v3's posterior.

Evaluation, 2-fold by sentence (the v1 fold): NLL / ECE / AUC on the target chain rows of
  abc       theta = m0, b = 0
  fit       theta fitted without b (the v1 fitted posterior)
  re_prior  the random-effect model, b at its prior (no test-time evidence)
  filter    b filtered on the interim chains of the run's earlier reads (test-time evidence only)
  oracle    b given the target labels of all the run's other reads (not observable at test time: the ceiling)
Gate (model doc §4): filter - fit >= +0.02 AUC with lower NLL.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.fit_sentence_filter --out reports/.../step0.json
"""

from __future__ import annotations

import argparse
import glob
import json
import zlib
from pathlib import Path

import numpy as np
import torch

from experiments.uniss_qwen3_omni_v1.data_clean.fit_bayesrisk_posterior import metrics
from experiments.uniss_qwen3_omni_v1.evaluation.si_bayesrisk import FEATS, label_rows, prior_model

DUMPS = "eval_outputs/uniss_qwen3_omni_v1/bayesrisk_v2/dev_it0b/dump/blocks_{d}_*.jsonl"
GH = 24


def interim_rows(prev: dict, cur: dict):
    """Read prev's uncommitted draft units against the next read's draft: the chain of 'unchanged' labels (1 until the
    first changed unit, which is 0); a next draft that ends earlier censors the chain."""
    X = np.asarray(prev["X"], float).reshape(-1, len(FEATS))
    u0 = cur["n"] - prev["n"]                      # units committed at prev are no longer at risk
    if u0 < 0 or u0 >= prev["K"]:
        return X[:0], np.zeros(0)
    gp, gc = prev["gu"], cur["gu"]
    y = []
    for u in range(u0, prev["K"]):
        j = u - u0
        if j >= len(gc):
            break                                  # censored: the next draft stops before this unit
        same = gp[u] == gc[j]
        y.append(1.0 if same else 0.0)
        if not same:
            break
    return X[u0:u0 + len(y)], np.asarray(y)


def load(direction: str, pattern: str):
    """Runs: list of {sid, reads: [(target X, target y, interim X, interim y)]} in read order."""
    runs = []
    for f in sorted(glob.glob(pattern.format(d=direction))):
        if "mposterior" in Path(f).name:           # the ABC-prior runs only (the v3-like policy)
            continue
        by = {}
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("type") == "next":
                continue
            by.setdefault(r["id"], []).append(r)
        for sid, rs in by.items():
            rs.sort(key=lambda r: r["t"])
            reads = []
            for k, r in enumerate(rs):
                tX, ty = label_rows(np.asarray(r["X"], float).reshape(-1, len(FEATS)), int(r["S"]))
                iX, iy = interim_rows(r, rs[k + 1]) if k + 1 < len(rs) else (tX[:0], np.zeros(0))
                reads.append((tX, ty, iX, iy))
            runs.append({"sid": sid, "file": Path(f).name, "reads": reads})
    return runs


def flat(runs, part):
    """Stack the target (part 0) or interim (part 2) rows with their run index."""
    X, y, g = [], [], []
    for i, run in enumerate(runs):
        for rd in run["reads"]:
            X.append(rd[part])
            y.append(rd[part + 1])
            g.append(np.full(len(rd[part + 1]), i))
    return (torch.tensor(np.concatenate(X)), torch.tensor(np.concatenate(y)), torch.tensor(np.concatenate(g)))


def fit(runs, m0, prior_sd=3.0, random_effect=True, iters=200):
    tX, ty, tg = flat(runs, 0)
    iX, iy, ig = flat(runs, 2)
    G = len(runs)
    z, w = np.polynomial.hermite_e.hermegauss(GH)  # probabilists' Hermite: int f(b) N(b; 0, 1) db
    z, lw = torch.tensor(z), torch.log(torch.tensor(w / w.sum()))
    m0t = torch.tensor(m0)
    theta = m0t.clone().requires_grad_(True)
    eta = m0t.clone().requires_grad_(True)
    gamma = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    log_tau = torch.tensor(np.log(0.5), dtype=torch.float64, requires_grad=True)
    params = [theta, eta, gamma, log_tau] if random_effect else [theta]

    def chain_ll(X, y, gidx, coef, load_b):
        # [rows, nodes] Bernoulli log-likelihood, summed per run -> [G, nodes]
        eta_ = (X @ coef)[:, None] + load_b * z[None, :]
        ll = y[:, None] * torch.nn.functional.logsigmoid(eta_) + (1 - y[:, None]) * torch.nn.functional.logsigmoid(-eta_)
        return torch.zeros(G, GH, dtype=torch.float64).index_add_(0, gidx, ll)

    def loss():
        pen = ((theta - m0t) ** 2).sum() / (2 * prior_sd ** 2)
        if not random_effect:
            s = tX @ theta
            ll = (ty * torch.nn.functional.logsigmoid(s) + (1 - ty) * torch.nn.functional.logsigmoid(-s)).sum()
            return -ll + pen
        tau = torch.exp(log_tau)
        pen = pen + ((eta - m0t) ** 2).sum() / (2 * prior_sd ** 2) + (gamma - 1) ** 2 / 2
        L = chain_ll(tX, ty, tg, theta, tau) + chain_ll(iX, iy, ig, eta, gamma * tau) + lw[None, :]
        return -torch.logsumexp(L, 1).sum() + pen

    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, line_search_fn="strong_wolfe", tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        v = loss()
        v.backward()
        return v

    opt.step(closure)
    return {"theta": theta.detach().numpy(), "eta": eta.detach().numpy(), "gamma": float(gamma),
            "tau": float(torch.exp(log_tau)) if random_effect else 0.0, "neg_log_marg": float(loss())}


def sig(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -30, 30)))


def logsig(v):
    return -np.logaddexp(0, -v)


def chain_pred(X, y, th, b_nodes, logw):
    """P(y_u = 1 | y_<u = 1, evidence) for one read's target chain under node weights logw: the chain's own earlier
    units update b inside the read (they are known at the decision: the action commits a prefix)."""
    out = np.zeros(len(y))
    lw = logw.copy()
    for u in range(len(y)):
        s = X[u] @ th + b_nodes
        w = np.exp(lw - lw.max())
        w /= w.sum()
        out[u] = float(w @ sig(s))
        lw = lw + logsig(s)                         # condition on this unit surviving for the next one
    return out


def evaluate(runs, P, Pfit, m0):
    z, w = np.polynomial.hermite_e.hermegauss(GH)
    b = z * P["tau"]
    lw0 = np.log(w / w.sum())
    zero = np.zeros(1)
    preds = {k: [] for k in ("abc", "fit", "re_prior", "filter", "oracle")}
    ys = []
    for run in runs:
        reads = run["reads"]
        # oracle: b's posterior from the target chains of all the run's reads, read k left out
        tll = []
        for rd in reads:
            s = (rd[0] @ P["theta"])[:, None] + b[None, :]
            tll.append((rd[1][:, None] * logsig(s) + (1 - rd[1][:, None]) * logsig(-s)).sum(0) if len(rd[1]) else np.zeros(GH))
        tsum = np.sum(tll, 0)
        lw_f = lw0.copy()
        for k, (tX, ty, iX, iy) in enumerate(reads):
            if len(ty):
                ys.append(ty)
                preds["abc"].append(chain_pred(tX, ty, m0, zero, np.zeros(1)))
                preds["fit"].append(chain_pred(tX, ty, Pfit["theta"], zero, np.zeros(1)))
                preds["re_prior"].append(chain_pred(tX, ty, P["theta"], b, lw0))
                preds["filter"].append(chain_pred(tX, ty, P["theta"], b, lw_f))
                preds["oracle"].append(chain_pred(tX, ty, P["theta"], b, lw0 + tsum - tll[k]))
            if len(iy):                             # read k's interim chain is observed at read k+1
                s = (iX @ P["eta"])[:, None] + P["gamma"] * b[None, :]
                lw_f = lw_f + (iy[:, None] * logsig(s) + (1 - iy[:, None]) * logsig(-s)).sum(0)
    y = np.concatenate(ys)
    return {k: metrics(np.concatenate(v), y) for k, v in preds.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", default=DUMPS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    torch.set_num_threads(16)
    m0 = prior_model()["mu"]
    res = {"feats": list(FEATS), "dirs": {}}
    for d in ("en2zh", "zh2en"):
        runs = load(d, args.dumps)
        fold = np.array([zlib.crc32(("cv:" + r["sid"]).encode()) % 2 for r in runs])
        cv, fits = [], []
        for k in (0, 1):
            tr = [r for r, f in zip(runs, fold) if f != k]
            te = [r for r, f in zip(runs, fold) if f == k]
            P = fit(tr, m0)
            Pfit = fit(tr, m0, random_effect=False)
            cv.append((evaluate(te, P, Pfit, m0), len(te)))
            fits.append({"tau": round(P["tau"], 4), "gamma": round(P["gamma"], 4)})
        # pooled over the two folds: row-weighted average of the per-fold metrics
        keys = cv[0][0].keys()
        pooled = {}
        for k in keys:
            n = sum(c[0][k]["n"] for c in cv)
            pooled[k] = {m: round(sum(c[0][k][m] * c[0][k]["n"] for c in cv) / n, 4) for m in ("nll", "ece", "auc")}
            pooled[k]["n"] = n
        P = fit(runs, m0)
        nt = sum(len(rd[1]) for r in runs for rd in r["reads"])
        ni = sum(len(rd[3]) for r in runs for rd in r["reads"])
        res["dirs"][d] = {"runs": len(runs), "sentences": len({r["sid"] for r in runs}), "target_rows": nt,
                          "interim_rows": ni, "folds": fits, "cv": pooled, "per_fold": [c[0] for c in cv],
                          "tau": round(P["tau"], 4), "gamma": round(P["gamma"], 4),
                          "theta": dict(zip(FEATS, np.round(P["theta"], 4).tolist())),
                          "eta": dict(zip(FEATS, np.round(P["eta"], 4).tolist())),
                          "gate_auc_gain": round(pooled["filter"]["auc"] - pooled["fit"]["auc"], 4)}
        print(d, json.dumps({k: v for k, v in res["dirs"][d].items() if k not in ("per_fold",)}), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
