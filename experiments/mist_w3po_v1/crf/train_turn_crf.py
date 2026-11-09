"""TurnCRF CPU stage: train the non-end-to-end semi-CRF / linear-chain CRF (turn_crf.py) on AMI train windows by
maximum marginal likelihood (crf_novel_ideas.zh-CN.md §3.2 loss without the LLM term), one pre-registered setting:

    loss = mean over train windows of (log Z - log Z_y) / T + (log Z~_tau - log Z~_tau,y) / tau
           (closed full window + one right-censored prefix, tau ~ U{1..T-1} redrawn every step from a fixed seed)
    full-batch Adam, lr 0.05 with cosine decay to 0, 150 steps (one step = one pass over all train windows),
    init a = 1, b = 0, durations / transitions / boundary weights 0 (= independent P).

The full-batch gradient is computed by --workers forked CPU processes (1 torch thread each) over fixed shards.
Writes --out (refuses an existing --out): params.pt, train_log.jsonl, config.json, DONE (or FAILED).

    PYTHONPATH=. python -m experiments.mist_w3po_v1.crf.train_turn_crf --kind semi \
        --data eval_outputs/mist_w3po_v1/crf/turn_refs_v1 --out eval_outputs/mist_w3po_v1/crf/turn_v1/semi --workers 56
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments.mist_w3po_v1.crf.turn_crf import build_model, window_nll


def tau_for_step(seed: int, step: int, T: np.ndarray) -> np.ndarray:
    r = np.random.default_rng([seed, step]).random(len(T))
    return 1 + np.floor(r * (T - 1)).astype(np.int64)          # 1 .. T-1


def make_batches(idx: np.ndarray, T: np.ndarray, kind: str, budget: float, max_b: int) -> list[np.ndarray]:
    idx = idx[np.argsort(T[idx], kind="stable")]
    out, cur = [], []
    for i in idx:
        tmax = T[i]
        cost = (len(cur) + 1) * (tmax ** 2 if kind == "semi" else tmax)
        if cur and (cost > budget or len(cur) >= max_b):
            out.append(np.array(cur))
            cur = []
        cur.append(i)
    if cur:
        out.append(np.array(cur))
    return out


def worker(conn, kind, model_kw, P, y, offs, T, idx, seed, budget, max_b):
    torch.set_num_threads(1)
    try:
        model = build_model(kind, **model_kw)
        batches = []
        for bi in make_batches(idx, T, kind, budget, max_b):
            tm = int(T[bi].max())
            Pb = torch.zeros(len(bi), tm, P.shape[1], dtype=torch.float64)
            yb = torch.full((len(bi), tm), -1, dtype=torch.long)
            for r, i in enumerate(bi):
                a, b = offs[i], offs[i + 1]
                Pb[r, :b - a] = torch.from_numpy(P[a:b].astype(np.float64))
                yb[r, :b - a] = torch.from_numpy(y[a:b].astype(np.int64))
            batches.append((bi, Pb, yb, torch.from_numpy(T[bi].astype(np.int64))))
        conn.send(("ready", len(batches)))
        while True:
            msg = conn.recv()
            if msg is None:
                break
            step, state = msg
            model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
            model.zero_grad()
            tau_all = tau_for_step(seed, step, T)
            tot = np.zeros(3)
            for bi, Pb, yb, Tb in batches:
                tau = torch.from_numpy(tau_all[bi])
                nll, full, pref = window_nll(model, Pb, Tb, yb, tau)
                nll.sum().backward()
                tot += [float(nll.sum()), float(full.sum()), float(pref.sum())]
            grads = {n: p.grad.detach().numpy().copy() for n, p in model.named_parameters()}
            conn.send(("ok", tot, grads))
    except Exception:
        conn.send(("err", traceback.format_exc()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["semi", "chain"], required=True)
    ap.add_argument("--data", default="eval_outputs/mist_w3po_v1/crf/turn_refs_v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=257)
    ap.add_argument("--Q", type=int, default=50)
    ap.add_argument("--budget", type=float, default=0, help="per-sub-batch B*T^2 (semi) or B*T (chain); 0 = default")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N windows only")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    torch.set_num_threads(1)
    d = np.load(Path(args.data) / "train.npz")
    T, offs, P, y = d["T"], d["offs"], d["P"], d["y"]
    N = len(T) if not args.limit else min(args.limit, len(T))
    T = T[:N]
    budget = args.budget or (3.0e5 if args.kind == "semi" else 6.0e3)
    model_kw = {"L": args.L, "Q": args.Q} if args.kind == "semi" else {}
    model = build_model(args.kind, **model_kw)
    # LPT shards on the DP cost
    cost = T.astype(np.float64) ** 2 if args.kind == "semi" else T.astype(np.float64)
    loads = np.zeros(args.workers)
    shards = [[] for _ in range(args.workers)]
    for i in np.argsort(-cost, kind="stable"):
        w = int(loads.argmin())
        shards[w].append(i)
        loads[w] += cost[i]
    out.mkdir(parents=True)
    cfg = {**vars(args), "n_windows": int(N), "frames": int(T.sum()), "budget": budget, "model_kw": model_kw,
           "params": {n: list(p.shape) for n, p in model.named_parameters()},
           "loss": "mean_w (logZ-logZ_y)/T + (logZc_tau-logZc_tau,y)/tau, tau~U{1..T-1} per step",
           "optimizer": "Adam(lr, betas=(0.9,0.999)) full batch, cosine decay to 0", "pid": os.getpid()}
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    ctx = mp.get_context("fork")
    conns, procs = [], []
    for w in range(args.workers):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(b, args.kind, model_kw, P, y, offs, T, np.array(shards[w], dtype=np.int64),
                                             args.seed, budget, 512), daemon=True)
        p.start()
        conns.append(a)
        procs.append(p)
    try:
        for c in conns:
            msg = c.recv()
            if msg[0] != "ready":
                raise RuntimeError(msg[1])
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        log = open(out / "train_log.jsonl", "x", encoding="utf-8")
        for step in range(args.steps + 1):          # the last pass only evaluates the final parameters
            t0 = time.time()
            state = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
            for c in conns:
                c.send((step, state))
            tot = np.zeros(3)
            grads = {n: np.zeros_like(p.detach().numpy()) for n, p in model.named_parameters()}
            for c in conns:
                msg = c.recv()
                if msg[0] != "ok":
                    raise RuntimeError(msg[1])
                tot += msg[1]
                for n, g in msg[2].items():
                    grads[n] += g
            rec = {"step": step, "loss": tot[0] / N, "full": tot[1] / N, "prefix": tot[2] / N,
                   "sec": round(time.time() - t0, 2)}
            if step < args.steps:
                lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))
                for g in opt.param_groups:
                    g["lr"] = lr
                for n, p in model.named_parameters():
                    p.grad = torch.from_numpy(grads[n] / N)
                opt.step()
                rec["lr"] = lr
                rec["grad_norm"] = float(math.sqrt(sum(float((g / N) ** 2).sum() if np.ndim(g) == 0 else
                                                       float(((g / N) ** 2).sum()) for g in grads.values())))
            rec.update({n: [round(float(v), 4) for v in p.detach().flatten()]
                        for n, p in model.named_parameters() if n in ("a", "b", "w", "A", "A_self", "start", "end",
                                                                     "d_tail")})
            log.write(json.dumps(rec) + "\n")
            log.flush()
            torch.save({"kind": args.kind, "model_kw": model_kw, "state_dict": model.state_dict(), "step": step},
                       out / "params.pt.tmp")
            os.replace(out / "params.pt.tmp", out / "params.pt")
        log.close()
        (out / "DONE").write_text(json.dumps({"rc": 0, "final": rec}), encoding="utf-8")
    except Exception:
        (out / "FAILED").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        for c in conns:
            try:
                c.send(None)
            except Exception:
                pass
        for p in procs:
            p.join(timeout=30)


if __name__ == "__main__":
    main()
