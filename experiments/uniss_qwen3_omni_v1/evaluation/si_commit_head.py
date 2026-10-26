"""CRISP's commit head: what the vote would have committed, from one greedy draft (C2), with a risk bound (C3).

The vote (si_thinker_stream --policy vote) commits what at least gamma of n
sampled continuations agree on -- n times the generation. The head reads one
greedy draft instead: per token, its log-probability, the top-5 margin and
entropy, where it sits in the draft and in its unit, how much source has been
heard, how much is committed already, whether the previous read's draft said
the same unit, and whether it is punctuation; a small MLP gives the chance the
token's unit is inside the vote's consensus. A read commits the units whose
tokens all score at least tau.

  train      the head on --dump records (label: inside the consensus; or with
             --label final, consistent with the finished translation; or both)
  calibrate  conformal risk control (Angelopoulos et al., ICLR 2024): on held-out
             reads, the loss of a read is 1 when any unit it commits disagrees
             with the finished translation; the loss falls as tau rises, so the
             smallest tau with (n/(n+1)) mean_loss + 1/(n+1) <= eps bounds the
             expected rate of such reads by eps -- the least waiting that keeps it.
             --check picks tau on half the held-out items and measures the risk on the other half.

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_commit_head train --dumps 'dir/*.jsonl' --out head.pt
    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_commit_head calibrate --head head.pt --dumps 'held/*.jsonl' --eps 0.05,0.1

  prep       the same features and labels as one npz, for training the head under Megatron
             (training/pretrain_commit_head.py); the head it exports loads here as any other.
"""

from __future__ import annotations

import argparse
import glob
import json
import random

from pathlib import Path

import numpy as np
import torch
from torch import nn

PUNCT = set(",.;:!?，。；：！？、…—\"'“”‘’()（）")
N_FEAT = 17


def features(tok: dict, n_units: int, heard_s: float, n_committed: int, prev_agree: int, zh: bool) -> np.ndarray:
    lp = np.asarray(tok["lp"], dtype=np.float32)
    n = len(lp)
    if n == 0:
        return np.zeros((0, N_FEAT), dtype=np.float32)
    out = np.zeros((n, N_FEAT), dtype=np.float32)
    cmin, csum = 0.0, 0.0
    for i in range(n):
        top = tok["top"][i] or [lp[i]]
        p = np.exp(np.asarray(top) - max(top))
        p = p / p.sum()
        u = tok["unit_of"][i]
        cmin = lp[i] if i == 0 else min(cmin, lp[i])
        csum += lp[i]
        out[i] = [lp[i], (top[0] - top[1]) if len(top) > 1 else 5.0, float(-(p * np.log(p + 1e-9)).sum()),
                  float(abs(lp[i] - top[0]) < 1e-4), i, i / n, u, u / max(n_units, 1), n_units, heard_s,
                  n_committed / (heard_s + 1e-3), cmin, csum / (i + 1), float(u < prev_agree),
                  float(u >= n_units - 1), float(zh), float(tok["tokens"][i].strip() in PUNCT)]
    return out


class Head(nn.Module):
    def __init__(self, mean=None, std=None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(N_FEAT) if mean is None else torch.as_tensor(mean))
        self.register_buffer("std", torch.ones(N_FEAT) if std is None else torch.as_tensor(std))
        self.net = nn.Sequential(nn.Linear(N_FEAT, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std).squeeze(-1)


def load_head(path: str) -> Head:
    ck = torch.load(path, map_location="cpu")
    h = Head()
    h.load_state_dict(ck["state"])
    return h.eval()


def keep_units(probs: np.ndarray, unit_of: list[int], n_units: int, tau: float) -> int:
    """Units committed: those whose tokens all lie in the leading run of tokens scoring >= tau."""
    for i, p in enumerate(probs):
        if p < tau:
            return unit_of[i] if i < len(unit_of) else n_units
    return n_units


def bayes_units(probs: np.ndarray, unit_of: list[int], n_units: int, lam: float) -> int:
    """The Bayes decision: commit the k units maximising P(the first k agree) + lam * k.

    Committing k costs C_err if any of them is wrong and saves C_lat * delta for each unit not left
    waiting, so the expected loss is C_err (1 - P_k) - C_lat delta k + const and lam = C_lat delta / C_err.
    A unit's agreement is a prefix event (labels are u < cons), so P_k is the running minimum of the
    per-unit scores (a unit scores its least sure token); unlike the threshold rule it may commit past a dip.
    """
    unit_min = [1.0] * n_units
    for p, u in zip(probs, unit_of):
        if u < n_units:
            unit_min[u] = min(unit_min[u], float(p))
    run, best, best_k = 1.0, 1.0, 0
    for u in range(n_units):
        run = min(run, unit_min[u])
        if run + lam * (u + 1) > best:
            best, best_k = run + lam * (u + 1), u + 1
    return best_k


def commit_units(head: Head, tok: dict, n_units: int, heard_s: float, n_committed: int, prev_agree: int, zh: bool,
                 tau: float, decision: str = "threshold", agree_beta: float | None = None,
                 has_prev: bool = False, return_probs: bool = False):
    """agree_beta: the head's score taken as the prior and the previous read's draft as evidence -- a unit both
    reads said gains beta in log-odds, one the previous read said otherwise loses beta (a log likelihood ratio of
    +-beta); with no previous draft to compare (has_prev False) there is no evidence and the prior stands."""
    x = features(tok, n_units, heard_s, n_committed, prev_agree, zh)
    if len(x) == 0:
        return (0, np.ones(n_units)) if return_probs else 0
    with torch.no_grad():
        logit = head(torch.from_numpy(x)).numpy()
    if agree_beta is not None and has_prev:
        logit = logit + agree_beta * np.where(np.asarray(tok["unit_of"]) < prev_agree, 1.0, -1.0)
    probs = 1.0 / (1.0 + np.exp(-logit))
    if decision == "bayes":   # tau is lam here
        keep = bayes_units(probs, tok["unit_of"], n_units, tau)
    else:
        keep = keep_units(probs, tok["unit_of"], n_units, tau)
    if not return_probs:
        return keep
    unit_p = np.ones(n_units)   # a unit scores its least sure token
    for p, u in zip(probs, tok["unit_of"]):
        if u < n_units:
            unit_p[u] = min(unit_p[u], float(p))
    return keep, unit_p


def load_records(pattern: str):
    for f in sorted(glob.glob(pattern)):
        zh = "_en2zh_" in f
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            r["zh"] = zh
            yield r


def labels(r: dict, kind: str) -> np.ndarray:
    u = np.asarray(r["unit_of"])
    cons, fin = u < r["cons"], u < r["final_ok"]
    return (cons if kind == "cons" else fin if kind == "final" else cons & fin).astype(np.float32)


def cmd_train(args) -> None:
    recs = [r for r in load_records(args.dumps) if r["lp"]]
    random.Random(0).shuffle(recs)
    ids = sorted({r["id"] for r in recs})
    held = set(random.Random(1).sample(ids, max(1, int(len(ids) * args.val_frac))))
    def stack(rs):
        xs = [features(r, len(r["draft_units"]), r["heard_s"], r["n_committed"], r["prev_agree"], r["zh"]) for r in rs]
        ys = [labels(r, args.label) for r in rs]
        return np.concatenate(xs), np.concatenate(ys)
    xtr, ytr = stack([r for r in recs if r["id"] not in held])
    xva, yva = stack([r for r in recs if r["id"] in held])
    mean, std = xtr.mean(0), xtr.std(0) + 1e-4
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    head = Head(mean, std).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    pos = float(ytr.mean())
    loss_fn = nn.BCEWithLogitsLoss()
    X, Y = torch.from_numpy(xtr).to(dev), torch.from_numpy(ytr).to(dev)
    g = torch.Generator(device="cpu").manual_seed(0)
    for ep in range(args.epochs):
        perm = torch.randperm(len(X), generator=g)   # a full shuffle each epoch
        for s in range(0, len(X), args.batch):
            k = perm[s: s + args.batch].to(dev)
            loss = loss_fn(head(X[k]), Y[k])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pv = torch.sigmoid(head(torch.from_numpy(xva).to(dev))).cpu().numpy()
        auc = auc_score(yva, pv)
        print(json.dumps({"epoch": ep, "train_loss": round(float(loss), 4), "val_auc": round(auc, 4),
                          "pos_rate": round(pos, 3), "val_n": int(len(yva))}), flush=True)
    torch.save({"state": head.cpu().state_dict(), "label": args.label, "val_ids": sorted(held)}, args.out)
    print(f"saved {args.out}", flush=True)


def auc_score(y, p) -> float:
    order = np.argsort(p)
    y = y[order]
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cmd_calibrate(args) -> None:
    head = load_head(args.head)
    val_ids = set(torch.load(args.head, map_location="cpu").get("val_ids", []))
    recs = [r for r in load_records(args.dumps) if r["lp"] and (not val_ids or r["id"] in val_ids)]
    probs = []
    for r in recs:
        x = features(r, len(r["draft_units"]), r["heard_s"], r["n_committed"], r["prev_agree"], r["zh"])
        with torch.no_grad():
            probs.append(torch.sigmoid(head(torch.from_numpy(x))).numpy())
    taus = np.round(np.arange(0.05, 1.0, 0.01), 2)

    def curve(idx):
        table = []
        for tau in taus:
            keep = [keep_units(probs[i], recs[i]["unit_of"], len(recs[i]["draft_units"]), tau) for i in idx]
            loss = np.asarray([float(k > recs[i]["final_ok"]) for k, i in zip(keep, idx)])
            table.append((float(tau), float(loss.mean()), float(np.mean(keep))))
        return table

    def tau_for(table, n, eps):
        ok = [t for t, risk, _ in table if n / (n + 1) * risk + 1 / (n + 1) <= eps]
        return min(ok) if ok else None

    n = len(recs)
    table = curve(range(n))
    out = {str(e): tau_for(table, n, e) for e in map(float, args.eps.split(","))}
    res = {"reads": n, "tau_for_eps": out}
    if args.check:   # the guarantee itself: tau from half the held-out items, risk measured on the other half
        ids = sorted({r["id"] for r in recs})
        cal_ids = set(random.Random(2).sample(ids, len(ids) // 2))
        cal = [i for i, r in enumerate(recs) if r["id"] in cal_ids]
        test = [i for i, r in enumerate(recs) if r["id"] not in cal_ids]
        tc, tt = curve(cal), dict((t, (risk, k)) for t, risk, k in curve(test))
        res["check"] = {}
        for e in map(float, args.eps.split(",")):
            t = tau_for(tc, len(cal), e)
            res["check"][str(e)] = None if t is None else {"tau": t, "test_risk": round(tt[t][0], 4),
                                                            "test_units": round(tt[t][1], 2)}
    print(json.dumps({**res, "curve": [(t, round(risk, 4), round(k, 2)) for t, risk, k in table[::5]]}), flush=True)
    if args.out:
        json.dump({**res, "curve": table}, open(args.out, "w"))


def _prep_file(path: str):
    zh = "_en2zh_" in path
    xs, yc, yf, ids = [], [], [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if not r["lp"]:
            continue
        xs.append(features(r, len(r["draft_units"]), r["heard_s"], r["n_committed"], r["prev_agree"], zh))
        yc.append(labels(r, "cons"))
        yf.append(labels(r, "final"))
        ids += [r["id"]] * len(r["lp"])
    if not xs:
        return np.zeros((0, N_FEAT), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32), []
    return np.concatenate(xs), np.concatenate(yc), np.concatenate(yf), ids


def cmd_prep(args) -> None:
    """Features and both labels of every dump token in one npz, with the held-out split by item, for Megatron."""
    from multiprocessing import Pool

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    files = sorted({f for pat in args.dumps.split(",") for f in glob.glob(pat)})
    with Pool(min(32, len(files))) as pool:
        parts = pool.map(_prep_file, files)
    x = np.concatenate([p[0] for p in parts])
    y_cons = np.concatenate([p[1] for p in parts]).astype(np.uint8)
    y_final = np.concatenate([p[2] for p in parts]).astype(np.uint8)
    row_ids = [i for p in parts for i in p[3]]
    uniq = sorted(set(row_ids))
    held = set(random.Random(1).sample(uniq, max(1, int(len(uniq) * args.val_frac))))
    val = np.fromiter((i in held for i in row_ids), dtype=bool, count=len(row_ids))
    mean, std = x[~val].mean(0), x[~val].std(0) + 1e-4
    np.savez(out, x=x, y_cons=y_cons, y_final=y_final, val=val, mean=mean, std=std,
             val_ids=np.array(json.dumps(sorted(held))), files=np.array(json.dumps(files)))
    print(json.dumps({"files": len(files), "rows": int(len(x)), "train": int((~val).sum()), "val": int(val.sum()),
                      "items": len(uniq), "pos_cons": round(float(y_cons.mean()), 3),
                      "pos_final": round(float(y_final.mean()), 3), "out": str(out)}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("prep", help="dumps -> one npz of features and labels, for training under Megatron")
    c.add_argument("--dumps", required=True, help="globs, comma-separated")
    c.add_argument("--out", required=True)
    c.add_argument("--val-frac", type=float, default=0.1)
    a = sub.add_parser("train")
    a.add_argument("--dumps", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--label", choices=("cons", "final", "both"), default="cons")
    a.add_argument("--epochs", type=int, default=6)
    a.add_argument("--batch", type=int, default=4096)
    a.add_argument("--lr", type=float, default=2e-3)
    a.add_argument("--val-frac", type=float, default=0.1)
    b = sub.add_parser("calibrate")
    b.add_argument("--head", required=True)
    b.add_argument("--dumps", required=True)
    b.add_argument("--eps", default="0.05,0.1,0.2")
    b.add_argument("--out", default="")
    b.add_argument("--check", action="store_true", help="choose tau on half the held-out items, report the risk on the rest")
    args = ap.parse_args()
    {"train": cmd_train, "calibrate": cmd_calibrate, "prep": cmd_prep}[args.cmd](args)


if __name__ == "__main__":
    main()
