"""Step 0b of the test-time filter (docs/m2_bayesrisk_sf_model.zh-CN.md): the step-0 sentence random effect carries
almost no ranking information (even its oracle adds < 0.01 AUC), so the reads' correlation sits on the units: the same
doubtful unit is scored at every read. This measures whether the unit's own history across the run's earlier reads --
observable at the read -- ranks its final survival better than the v1 statistics.

Per unit u of read k at absolute position p (draft_k = committed_k + gu_k):
  age     log1p(number of consecutive earlier reads whose draft agrees with draft_k on the whole prefix up to p)
  flips   log1p(number of earlier reads that drafted position p and disagreed with draft_k at or before p)
  seen    log1p(number of earlier reads that drafted position p)
The likelihood is the v1 survival chain sigmoid(theta . [T, history]); theta MAP under N(m0, 3^2), 2-fold by sentence.
Gate: AUC +0.02 over the fit on T alone, with lower NLL.

The decision view (Prop. 1 of the v2 model: a commit rule thresholds P(S >= a) inside each read): with the held-out
predictions on every unit of every read, commit the longest prefix with P(S >= a) > c for a grid of c, and trace
wrong units E = sum (a - S)+ against withheld stable units W = sum (min(S, K) - a)+ (per 100 reads). The frontier
gain is the relative drop of E at equal W (interpolated), over the W range both models cover.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.history_features_auc --out reports/.../step0b.json
"""

from __future__ import annotations

import argparse
import glob
import json
import zlib
from pathlib import Path

import numpy as np

from experiments.uniss_qwen3_omni_v1.data_clean.fit_bayesrisk_posterior import metrics
from experiments.uniss_qwen3_omni_v1.evaluation.si_bayesrisk import FEATS, label_rows, laplace_update, prior_model

DUMPS = "eval_outputs/uniss_qwen3_omni_v1/bayesrisk_v2/dev_it0b/dump/blocks_{d}_*.jsonl"
HIST = ("age", "flips", "seen")


def lcp(a: list, b: list) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def load(direction: str, pattern: str):
    X, y, sid, bad, full = [], [], [], 0, []
    for f in sorted(glob.glob(pattern.format(d=direction))):
        if "mposterior" in Path(f).name:
            continue
        by = {}
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("type") != "next":
                by.setdefault(r["id"], []).append(r)
        for s, rs in by.items():
            rs.sort(key=lambda r: r["t"])
            committed, drafts = [], []
            for r in rs:
                if len(committed) != r["n"]:       # the dump's committed length must match the reconstruction
                    bad += 1
                    committed = committed[: r["n"]] + ["?"] * max(0, r["n"] - len(committed))
                draft = committed + r["gu"]
                T = np.asarray(r["X"], float).reshape(-1, len(FEATS))
                H = np.zeros((len(r["gu"]), len(HIST)))
                L = [lcp(d, draft) for d in drafts]  # agreement of each earlier draft with this one
                for u in range(len(r["gu"])):
                    p = r["n"] + u
                    age = 0
                    for j in range(len(drafts) - 1, -1, -1):
                        if L[j] > p:
                            age += 1
                        else:
                            break
                    seen = [j for j in range(len(drafts)) if len(drafts[j]) > p]
                    flips = sum(1 for j in seen if L[j] <= p)
                    H[u] = (np.log1p(age), np.log1p(flips), np.log1p(len(seen)))
                full.append((np.concatenate([T, H], 1), int(r["S"]), s))
                R, lab = label_rows(full[-1][0], int(r["S"]))
                X.append(R)
                y.append(lab)
                sid += [s] * len(lab)
                drafts.append(draft)
                committed = committed + r["gu"][: r["a"]]
    return np.concatenate(X), np.concatenate(y), np.asarray(sid), bad, full


def sig(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -30, 30)))


def frontier(full, fold, fits, cols, grid=np.linspace(0.02, 0.995, 120)):
    """Wrong vs withheld units per 100 reads of the threshold rule, held-out predictions."""
    E, W = np.zeros(len(grid)), np.zeros(len(grid))
    for (Xr, S, s), f in zip(full, fold):
        if not len(Xr):
            continue
        q = sig(Xr[:, cols] @ fits[f]) if cols is not None else sig(Xr[:, :len(FEATS)] @ fits[f])
        ge = np.concatenate([[1.0], np.cumprod(q)])             # P(S >= a), a = 0..K
        a = np.array([int(np.max(np.nonzero(ge > c)[0])) for c in grid])
        E += np.maximum(a - S, 0)
        W += np.maximum(min(S, len(Xr)) - a, 0)
    n = len(full) / 100
    return E / n, W / n


def gain(Eb, Wb, Ea, Wa):
    """Mean relative drop of wrong units at equal withheld units over the shared W range."""
    lo, hi = max(Wb.min(), Wa.min()), min(Wb.max(), Wa.max())
    w = np.linspace(lo, hi, 50)
    ob, oa = np.argsort(Wb), np.argsort(Wa)
    eb, ea = np.interp(w, Wb[ob], Eb[ob]), np.interp(w, Wa[oa], Ea[oa])
    return float(np.mean((eb - ea) / np.maximum(eb, 1e-9)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", default=DUMPS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    nf = len(FEATS)
    m0 = np.concatenate([prior_model()["mu"], np.zeros(len(HIST))])
    S0 = np.eye(len(m0)) * 9.0
    res = {"feats": list(FEATS) + list(HIST), "dirs": {}}
    for d in ("en2zh", "zh2en"):
        X, y, sid, bad, full = load(d, args.dumps)
        fold = np.array([zlib.crc32(("cv:" + s).encode()) % 2 for s in sid])
        rfold = np.array([zlib.crc32(("cv:" + s).encode()) % 2 for _, _, s in full])
        heldout = {}
        cols = {"abc": None, "fit_T": list(range(nf)), "fit_T+age": list(range(nf)) + [nf],
                "fit_T+history": list(range(nf + len(HIST)))}
        preds = {k: np.zeros(len(y)) for k in cols}
        for k in (0, 1):
            tr, te = fold != k, fold == k
            for name, c in cols.items():
                if c is None:
                    preds[name][te] = sig(X[te][:, :nf] @ m0[:nf])
                    continue
                mu, _ = laplace_update(m0[c], S0[np.ix_(c, c)], X[tr][:, c], y[tr])
                preds[name][te] = sig(X[te][:, c] @ mu)
                heldout.setdefault(name, [None, None])[k] = mu   # fitted on fold != k, applied to fold k
        cv = {k: metrics(v, y) for k, v in preds.items()}
        heldout["abc"] = [m0[:nf], m0[:nf]]
        fr = {name: frontier(full, rfold, heldout[name], cols[name]) for name in cols}
        fgain = {name: round(gain(*fr["fit_T"], *fr[name]), 4) for name in cols if name != "fit_T"}
        mu, Sig = laplace_update(m0, S0, X, y)
        res["dirs"][d] = {"rows": int(len(y)), "sentences": int(len(set(sid))), "committed_mismatch": bad, "cv": cv,
                          "gate_auc_gain": round(cv["fit_T+history"]["auc"] - cv["fit_T"]["auc"], 4),
                          "frontier_wrong_drop_vs_fit_T": fgain,
                          "frontier": {k: {"wrong_per_100": np.round(v[0][::12], 2).tolist(),
                                           "withheld_per_100": np.round(v[1][::12], 2).tolist()} for k, v in fr.items()},
                          "theta": dict(zip(res["feats"], np.round(mu, 4).tolist())),
                          "sd": dict(zip(res["feats"], np.round(np.sqrt(np.diag(Sig)), 4).tolist())),
                          "history_mean": dict(zip(HIST, np.round(X[:, nf:].mean(0), 3).tolist()))}
        print(d, json.dumps(res["dirs"][d]), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
