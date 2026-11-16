"""TS-PA CPU precheck P2 (reports/mist_w3po_v1/tspa_precheck_prereg_v1.zh-CN.md): can the arrival of the next block be
foreseen causally, beyond what the PA Talker already sees?

Units: blocks n with a successor in their stream (window x output speaker) of MIST's dev output (B's text + rl + PA
Talker). Label y_n = 1[start_{n+1} - (start_n + dur_n) >= 500 ms] (the pause metric, talker layout). Features at the
hand-over time t_n = write_ms_n; diarizer frame f is visible iff 80 f <= t_n - 1040 (Eq. causal's bound):
  base  backlog = start_n - write_ms_n, W_n (units), sentence end of block n
  new   final (written at the last read), window close detected (t_n >= close_detect_ms),
        debt = max(0, t_n - 1040 - 1000 end_s_n) / 1000, pend = 0.08 * sum P[f, k-1] over 1000 end_s_n < 80 f <= t_n - 1040,
        act1 = mean P[f, k-1] over t_n - 2040 < 80 f <= t_n - 1040
Standardised L2 logistic regression (C = 1), 5-fold GroupKFold by meeting, AUC of the out-of-fold probabilities.

    python -m experiments.mist_w3po_v1.talker.tspa_precheck_p2 --run <speech run dir> --win <windows> \\
        [--lang cmn --units char] --out <json>     (refuses to overwrite)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.talker import talker_rows as TR
from experiments.mist_w3po_v1.talker.tspa_precheck_p1 import PAUSE_MS, sent_end

GAMMA_MS = 1040
BASE = ("backlog_s", "units", "sent_end")
NEW = ("final", "closed", "debt_s", "pend_s", "act1")


def _rows(args):
    lines, W, TI, lang, units = args
    TR.LANG, TR.UNITS = (TR.LANG_ENG if lang == "eng" else TR.LANG_CMN), units
    out = []
    for line in lines:
        r = json.loads(line)
        bl = r.get("blocks") or []
        if r.get("error") or len(bl) < 2:
            continue
        wid, k = r["id"].split("__S")
        k = int(k)
        w = W[wid]
        ti = TI.get(r["id"])
        if ti is None or len(ti) != len(bl):
            continue
        P = np.load(w["post"]["lowlat"]).astype(np.float64)
        col = P[:, k - 1] if k - 1 < P.shape[1] else np.zeros(len(P))
        tf = np.arange(len(col)) * 80.0
        for n in range(len(bl) - 1):
            b, nb = bl[n], bl[n + 1]
            t = float(b["write_ms"])
            vis = t - GAMMA_MS
            end_ms = 1000.0 * float(ti[n]["end_s"])
            gap = float(nb["start_ms"]) - (float(b["start_ms"]) + float(b["dur_ms"]))
            out.append({
                "meeting": w["meeting"], "y": int(gap >= PAUSE_MS),
                "mid": int(not sent_end(b.get("text", ""))),
                "backlog_s": (float(b["start_ms"]) - t) / 1000.0,
                "units": TR.n_units(b.get("text", "")),
                "sent_end": int(sent_end(b.get("text", ""))),
                "final": int(bool(b.get("final"))),
                "closed": int(t >= w["close_detect_ms"]),
                "debt_s": max(0.0, vis - end_ms) / 1000.0,
                "pend_s": 0.08 * float(col[(tf > end_ms) & (tf <= vis)].sum()),
                "act1": float(col[(tf > vis - 1000) & (tf <= vis)].mean()) if ((tf > vis - 1000) & (tf <= vis)).any() else 0.0})
    return out


def auc_oof(X, y, groups, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    full = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)
    return float(roc_auc_score(y, p)), full[-1].coef_[0].round(3).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--win", required=True)
    ap.add_argument("--lang", choices=("eng", "cmn"), default="eng")
    ap.add_argument("--units", choices=("word", "char"), default="word")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    run = Path(a.run)
    W = TR.windows(a.win)
    TI = {r["id"]: r["blocks"] for r in map(json.loads, open(run / "talker_in" / "blocks.jsonl", encoding="utf-8"))}
    lines = open(run / "talker" / "results.jsonl", encoding="utf-8").read().splitlines()
    k = max(1, a.procs)
    with mp.get_context("fork").Pool(k) as pool:
        rows = [x for part in pool.map(_rows, [(lines[i::k], W, TI, a.lang, a.units) for i in range(k)]) for x in part]
    y = np.array([r["y"] for r in rows])
    g = np.array([r["meeting"] for r in rows])
    Xb = np.array([[r[c] for c in BASE] for r in rows], dtype=np.float64)
    Xf = np.array([[r[c] for c in BASE + NEW] for r in rows], dtype=np.float64)
    auc_b, coef_b = auc_oof(Xb, y, g)
    auc_f, coef_f = auc_oof(Xf, y, g)
    mid = np.array([r["mid"] for r in rows]) == 1
    from sklearn.metrics import roc_auc_score  # noqa: F401
    res = {"run": str(run), "win": a.win, "units": len(rows), "meetings": int(len(set(g))),
           "pause_rate": round(float(y.mean()), 4), "pause_rate_mid": round(float(y[mid].mean()), 4) if mid.any() else None,
           "auc_base": round(auc_b, 4), "auc_full": round(auc_f, 4), "auc_gain": round(auc_f - auc_b, 4),
           "coef_base": dict(zip(BASE, coef_b)), "coef_full": dict(zip(BASE + NEW, coef_f)),
           "feature_means": {c: round(float(np.mean([r[c] for r in rows])), 4) for c in BASE + NEW},
           "go": {"auc_full_ge_0.75": auc_f >= 0.75, "gain_ge_0.03": (auc_f - auc_b) >= 0.03}}
    res["GO"] = all(res["go"].values())
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
