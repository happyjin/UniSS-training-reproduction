"""Route 1: the commit posterior as prior x likelihood, with no evidence counted twice.

  Z_u        unit u of the read's draft survives the future: draft[:u+1] == oracle[:u+1], the oracle being the
             Thinker continuing the same committed prefix with the whole sentence heard (si_thinker_stream --oracle)
  prior      pi_u = sigmoid(g(x_u)), g the commit head on the draft's own features with the agreement feature
             (13) zeroed -- the prior sees no other draft
  evidence   e_u = the unit's state against each other draft of the moment: the previous read's (real) and drafts
             of this read on the audio minus 250 / 500 / 1000 ms (virtual), each none / agree / contradict / beyond
             its end, with the read window binned -- one cell c(e_u, window) of 4^4 x 4
  likelihood the log-ratio w_c = log P(e in c | Z=1) / P(e in c | Z=0); fitted ("fit": conditional maximum likelihood
             with the prior as offset, so evidence the prior already explains is not counted again) or counted
             ("count": the generative estimate, exact Bayes if e is independent of x given Z); the joint cell, not a
             product over channels, so the four drafts' dependence on one another is not counted as four
  posterior  logit p_u = logit pi_u + w_c(e_u); a read commits the units whose tokens all score >= tau

  prep    oracle dumps -> one npz: features (agreement zeroed), labels, evidence cells, the old posterior's inputs
  fit     the head trained on that npz (run_commit_head.sh, LABEL=final) -> the likelihood table (json); w fitted on
          half the held-out items, NLL / ECE / AUC and the risk-units curve of the old (beta=4) and new posterior
          reported on the other half

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_bayes_lr prep --dumps 'dir/*.jsonl' --out rows.npz
    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_bayes_lr fit --rows rows.npz --head head.pt \
        --old-head head_cons_all.pt --out table.json
"""

from __future__ import annotations

import argparse
import glob
import json
import zlib
from pathlib import Path

import numpy as np
import torch

from experiments.uniss_qwen3_omni_v1.evaluation.si_commit_head import N_FEAT, features, load_head

DROP = (13,)                       # prev_agree: evidence, kept out of the prior
CHANNELS = ("real", 250, 500, 1000)
STATES = 4                         # none, agree, contradict, beyond its end
WINDOW_EDGES = (750, 1500, 3000)   # read-window bins: 0.5 / 1 / 2 / >= 3 s
N_CELLS = STATES ** len(CHANNELS) * (len(WINDOW_EDGES) + 1)


def lcp(a: list, b: list) -> int:
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def unit_cells(units: list[str], others: list, window_ms: int) -> np.ndarray:
    """Each unit's evidence cell: its state against every other draft (None: that draft does not exist)."""
    cell = np.zeros(len(units), dtype=np.int64)
    for j, o in enumerate(others):
        if o is None:
            continue
        k = lcp(units, o)
        s = np.where(np.arange(len(units)) < k, 1, np.where(np.arange(len(units)) < len(o), 2, 3))
        cell += s * STATES ** j
    return cell + STATES ** len(CHANNELS) * int(np.searchsorted(WINDOW_EDGES, window_ms, side="right"))


def prior_features(tok: dict, n_units: int, heard_s: float, n_committed: int, zh: bool) -> np.ndarray:
    x = features(tok, n_units, heard_s, n_committed, 0, zh)
    x[:, list(DROP)] = 0.0
    return x


def load_table(path: str) -> dict:
    t = json.load(open(path, encoding="utf-8"))
    t["w_arr"] = np.asarray(t["w_" + t.get("use", "fit")], dtype=np.float32)
    return t


def token_logits(head, table: dict, tok: dict, units: list[str], others: list, window_ms: int, heard_s: float,
                 n_committed: int, zh: bool) -> np.ndarray:
    """The posterior log-odds of every draft token: the prior plus its unit's log-likelihood ratio."""
    x = prior_features(tok, len(units), heard_s, n_committed, zh)
    if len(x) == 0:
        return np.zeros(0)
    with torch.no_grad():
        logit = head(torch.from_numpy(x)).numpy()
    cell = unit_cells(units, others, window_ms)
    u = np.minimum(np.asarray(tok["unit_of"]), max(len(units) - 1, 0))
    if len(units):
        logit = logit + table["w_arr"][cell[u]]
    return logit


def lr_commit(head, table: dict, tok: dict, units: list[str], others: list, window_ms: int, heard_s: float,
              n_committed: int, zh: bool, tau: float, to_probs=None):
    """to_probs: logits -> probabilities (Route 2's talk-adapted predictive); else the sigmoid. Returns the units
    committed, each unit's probability (its least sure token) and the token logits."""
    from experiments.uniss_qwen3_omni_v1.evaluation.si_commit_head import keep_units

    z = token_logits(head, table, tok, units, others, window_ms, heard_s, n_committed, zh)
    probs = to_probs(z) if to_probs is not None else 1.0 / (1.0 + np.exp(-z))
    if len(probs) == 0:
        return 0, np.ones(len(units)), z
    keep = keep_units(probs, tok["unit_of"], len(units), tau)
    unit_p = np.ones(len(units))
    for p, u in zip(probs, tok["unit_of"]):
        if u < len(units):
            unit_p[u] = min(unit_p[u], float(p))
    return keep, unit_p, z


def _prep_file(path: str):
    zh = "_en2zh_" in path
    out = {k: [] for k in ("x", "y", "cell", "x13", "has_prev", "ids", "unit", "virt")}
    recs = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if not r["lp"] or "oracle_units" not in r:
            continue
        gu = r["draft_units"]
        x = prior_features(r, len(gu), r["heard_s"], r["n_committed"], zh)
        u = np.asarray(r["unit_of"])
        L = lcp(gu, r["oracle_units"])
        virt = r.get("virt")
        others = [r.get("prev_real")] + [(virt or {}).get(str(d)) for d in CHANNELS[1:]]
        cell = unit_cells(gu, others, int(r.get("window_ms", 1000)))
        uc = np.minimum(u, max(len(gu) - 1, 0))
        out["x"].append(x)
        out["y"].append((u < L).astype(np.uint8))
        out["cell"].append(cell[uc] if len(gu) else np.zeros(len(u), np.int64))
        out["x13"].append((u < r["prev_agree"]).astype(np.uint8))
        out["has_prev"].append(np.full(len(u), bool(r.get("prev_units")), dtype=bool))
        out["unit"].append(u.astype(np.int32))
        out["virt"].append(np.full(len(u), virt is not None, dtype=bool))
        out["ids"] += [r["id"]] * len(u)
        recs.append((len(u), len(gu), L))
    if not recs:
        return None
    return {k: (np.concatenate(v) if k != "ids" else v) for k, v in out.items()}, recs, zh


def cmd_prep(args) -> None:
    from multiprocessing import Pool

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    files = sorted({f for pat in args.dumps.split(",") for f in glob.glob(pat)})
    with Pool(min(32, len(files))) as pool:
        parts = [p for p in pool.map(_prep_file, files) if p is not None]
    cat = lambda k: np.concatenate([p[0][k] for p in parts])
    x = cat("x")
    row_ids = [i for p in parts for i in p[0]["ids"]]
    # held out by item (any window, either dump): 1 in --val-every items by a stable hash
    val = np.fromiter((zlib.crc32(i.encode()) % args.val_every == 0 for i in row_ids), dtype=bool, count=len(row_ids))
    rec_len = np.asarray([n for p in parts for n, _, _ in p[1]], dtype=np.int64)
    rec_units = np.asarray([m for p in parts for _, m, _ in p[1]], dtype=np.int32)
    rec_lcp = np.asarray([L for p in parts for _, _, L in p[1]], dtype=np.int32)
    rec_zh = np.concatenate([np.full(len(p[1]), p[2]) for p in parts])
    mean, std = x[~val].mean(0), x[~val].std(0) + 1e-4
    y = cat("y")
    np.savez(out, x=x, y_cons=y, y_final=y, val=val, mean=mean, std=std, cell=cat("cell"), x13=cat("x13"),
             has_prev=cat("has_prev"), unit=cat("unit"), virt=cat("virt"), rec_len=rec_len, rec_units=rec_units,
             rec_lcp=rec_lcp, rec_zh=rec_zh, val_ids=np.array(json.dumps(sorted({i for i, v in zip(row_ids, val) if v}))),
             item_hash=np.fromiter((zlib.crc32(i.encode()) for i in row_ids), dtype=np.uint32, count=len(row_ids)),
             files=np.array(json.dumps(files)))
    print(json.dumps({"files": len(files), "rows": int(len(x)), "reads": int(len(rec_len)), "train": int((~val).sum()),
                      "val": int(val.sum()), "with_virtual": int(cat("virt").sum()), "pos": round(float(y.mean()), 3),
                      "out": str(out)}), flush=True)


def logits_of(head, x: np.ndarray, batch: int = 1 << 20) -> np.ndarray:
    out = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            out.append(head(torch.from_numpy(x[s: s + batch])).numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def fit_cells(g: np.ndarray, y: np.ndarray, cell: np.ndarray, ridge: float, iters: int = 30) -> np.ndarray:
    """w_c maximising sum_i log p(y_i | g_i + w_c(i)) - ridge w_c^2 / 2; one-hot cells, so each w_c on its own."""
    w = np.zeros(N_CELLS)
    yc = np.bincount(cell, weights=y, minlength=N_CELLS)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(g + w[cell])))
        grad = yc - np.bincount(cell, weights=p, minlength=N_CELLS) - ridge * w
        hess = np.bincount(cell, weights=p * (1 - p), minlength=N_CELLS) + ridge
        w = w + grad / hess
    return w


def count_cells(y: np.ndarray, cell: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    n1 = np.bincount(cell, weights=y, minlength=N_CELLS) + alpha
    n0 = np.bincount(cell, weights=1 - y, minlength=N_CELLS) + alpha
    return np.log(n1 / n1.sum()) - np.log(n0 / n0.sum())


def report(p: np.ndarray, y: np.ndarray, starts: np.ndarray, rec_len: np.ndarray, unit: np.ndarray,
           rec_units: np.ndarray, rec_lcp: np.ndarray) -> dict:
    from experiments.uniss_qwen3_omni_v1.evaluation.si_commit_head import auc_score

    eps = 1e-7
    nll = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    bins = np.minimum((p * 15).astype(int), 14)
    ece = float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(15) if (bins == b).any()))
    curve = []
    for tau in np.round(np.arange(0.2, 0.96, 0.05), 2):   # the decision: risk of a wrong commit vs units committed
        low = p < tau
        keep = np.empty(len(rec_len), np.int64)
        for k, (s, n) in enumerate(zip(starts, rec_len)):
            hit = np.flatnonzero(low[s: s + n])
            keep[k] = unit[s + hit[0]] if len(hit) else rec_units[k]
        curve.append((float(tau), round(float((keep > rec_lcp).mean()), 4), round(float(keep.mean()), 3)))
    return {"nll": round(nll, 4), "ece": round(ece, 4), "auc": round(auc_score(y, p), 4), "curve": curve}


def units_at_risk(curve: list, eps: float) -> float | None:
    """Units committed a read at the loosest tau whose risk stays <= eps (linear between grid points)."""
    pts = sorted(((r, k) for _, r, k in curve))
    for (r0, k0), (r1, k1) in zip(pts, pts[1:]):
        if r0 <= eps <= r1:
            return round(k0 + (k1 - k0) * (eps - r0) / max(r1 - r0, 1e-9), 3)
    return None


def cmd_fit(args) -> None:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    d = np.load(args.rows)
    val, virt = d["val"], d["virt"]
    rec_len = d["rec_len"]
    starts = np.concatenate([[0], np.cumsum(rec_len)[:-1]])
    rec_val = val[starts] & virt[starts]   # held-out reads that carry all four drafts
    half = (d["item_hash"][starts] // 7) % 2 == 0   # fit on one half of the held-out items, report on the other
    fit_r, eval_r = np.flatnonzero(rec_val & half), np.flatnonzero(rec_val & ~half)

    def rows_of(rs):
        return np.concatenate([np.arange(starts[k], starts[k] + rec_len[k]) for k in rs])

    fr, er = rows_of(fit_r), rows_of(eval_r)
    head = load_head(args.head)
    x = d["x"]
    g_f, g_e = logits_of(head, x[fr]), logits_of(head, x[er])
    y = d["y_final"].astype(np.float64)
    w_fit = fit_cells(g_f.astype(np.float64), y[fr], d["cell"][fr], args.ridge)
    w_count = count_cells(y[fr], d["cell"][fr])
    old = load_head(args.old_head)
    x_old = x[er].copy()
    x_old[:, 13] = d["x13"][er]
    g_old = logits_of(old, x_old) + np.where(d["has_prev"][er], args.old_beta * np.where(d["x13"][er] > 0, 1.0, -1.0), 0.0)
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    e_starts = np.concatenate([[0], np.cumsum(rec_len[eval_r])[:-1]])
    meta = (e_starts, rec_len[eval_r], d["unit"][er], d["rec_units"][eval_r], d["rec_lcp"][eval_r])
    res = {"fit_reads": int(len(fit_r)), "eval_reads": int(len(eval_r)), "eval_tokens": int(len(er)),
           "old_beta4": report(sig(g_old), y[er], *meta),
           "prior_only": report(sig(g_e), y[er], *meta),
           "post_count": report(sig(g_e + w_count[d["cell"][er]]), y[er], *meta),
           "post_fit": report(sig(g_e + w_fit[d["cell"][er]]), y[er], *meta)}
    for k in ("old_beta4", "prior_only", "post_count", "post_fit"):
        res[k]["units_at_risk"] = {str(e): units_at_risk(res[k]["curve"], e) for e in (0.05, 0.1, 0.2)}
    used = np.bincount(d["cell"][fr], minlength=N_CELLS)
    table = {"channels": [str(c) for c in CHANNELS], "states": ["none", "agree", "contradict", "beyond"],
             "window_edges": list(WINDOW_EDGES), "drop": list(DROP), "virtual": [c for c in CHANNELS if c != "real"],
             "head": str(Path(args.head).resolve()), "rows": str(Path(args.rows).resolve()), "ridge": args.ridge,
             "use": "fit", "w_fit": [round(float(v), 4) for v in w_fit], "w_count": [round(float(v), 4) for v in w_count],
             "cell_rows": used.tolist(), "eval": res}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table), encoding="utf-8")
    for k in ("old_beta4", "prior_only", "post_count", "post_fit"):
        r = res[k]
        print(json.dumps({"posterior": k, "nll": r["nll"], "ece": r["ece"], "auc": r["auc"], "units_at_risk": r["units_at_risk"]}),
              flush=True)
    single = {}   # what one channel says on its own, each window bin: agree vs contradict, in log-odds
    for j, c in enumerate(CHANNELS):
        for b in range(len(WINDOW_EDGES) + 1):
            for s in (1, 2, 3):
                cs = [k for k in range(N_CELLS) if (k // STATES ** j) % STATES == s
                      and k // STATES ** len(CHANNELS) == b and used[k] > 0]
                if cs:
                    single[f"{c}|bin{b}|{table['states'][s]}"] = round(float(np.average(w_fit[cs], weights=used[cs])), 3)
    print(json.dumps({"mean_w_by_channel_state": single}, ensure_ascii=False), flush=True)
    print(f"saved {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--dumps", required=True, help="globs, comma-separated")
    p.add_argument("--out", required=True)
    p.add_argument("--val-every", type=int, default=5, help="1 in this many items held out")
    f = sub.add_parser("fit")
    f.add_argument("--rows", required=True)
    f.add_argument("--head", required=True, help="the prior, trained on --rows (agreement zeroed)")
    f.add_argument("--old-head", required=True, help="the current head (head_cons_all), for the beta=4 comparison")
    f.add_argument("--old-beta", type=float, default=4.0)
    f.add_argument("--ridge", type=float, default=1.0)
    f.add_argument("--out", required=True)
    args = ap.parse_args()
    {"prep": cmd_prep, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
