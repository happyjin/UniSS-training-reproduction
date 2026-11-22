"""voice_onset_v1 S2 training (reports/voice_onset_v1/S0_S1_S2_prereg.zh-CN.md section 4 + change 1): learned hold length
of the player, a left-to-right semi-Markov duration (release hazard h_k over 250 ms steps, J = 8) trained end-to-end by
the exact expected cost  E_p[c(j)] = sum_j p(j) c(j),  c(j) = sum of U_set D2 times / U_total (s) + alpha * NP_in(window).
On-policy: every epoch re-simulates the open dev traces (eval_outputs/voice_onset_v1/s1/dev, queue talker, both
directions) with the deployed rule (first h_k >= 0.5), collects the decision points, then 25 full-batch Adam steps
(lr 1e-3). 40 epochs per alpha, seed 20261122, last epoch kept. alpha: log bisection on [0.01, 10] (<= 12 trainings) to
match the dev mean D2 of the R0 player on the same traces (+- 0.01 s). Cost components are cached per decision state.
    qwen3omni PYTHONPATH=. python -m experiments.voice_onset_v1.s2_train --dev eval_outputs/voice_onset_v1/s1/dev \
        --out eval_outputs/voice_onset_v1/s2/train
Refuses an existing --out.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SEED, EPOCHS, STEPS, LR = 20261122, 40, 25, 1e-3
LANG = {"en2zh": "cmn", "zh2en": "eng"}
_W = {}


def _item(i):
    return _W["items"][i]


def _run(args):
    """simulate items idx with the given policy -> (decisions [(F, Lcost, Ncost)], metrics rows)"""
    idx, mode, state = args
    PO = _W["PO"]
    hz = PO.mlp_numpy({k: torch.tensor(v) for k, v in state.items()}) if state is not None else None
    dec, met = [], []
    for i in idx:
        it = _item(i)
        if mode == "r0":
            segs = PO.simulate(it, PO.r0, dip=True)
        else:
            rec = []
            pol = PO.S2Policy(hz if hz is not None else (lambda F: np.r_[np.full(len(F) - 1, 0.9), 1.0]), rec)
            segs = PO.simulate(it, pol, dip=True)
            if mode == "train":
                for (_, t, x, prev, F, times) in rec:
                    key = (i, t, x, prev)
                    if key not in _W["cache"]:
                        _W["cache"][key] = comp(PO, it, x, times, prev)
                    L, N = _W["cache"][key]
                    dec.append((np.asarray(F, np.float32), L, N))
        m = PO.metrics(it, segs)
        if m is not None:
            met.append((m["D2"] / 1000, m["NP_in"], m["NC"]))
    return dec, met


def comp(PO, it, x, times, prev):
    kmax = it.arrived(times[-1]) - 1
    U = {i for i in range(kmax + 1) if it.P[i]["e"] > x}
    until = max(it.P[i]["e"] for i in U)
    L, N = [], []
    for r in times:
        segs = PO.simulate(it, None, dip=True, x=x, t=r, until=until)
        _, d2 = PO.unit_times(it, segs, pieces=U)
        L.append(sum(d2) / max(it.Utot, 1) / 1000)
        N.append(sum(1 for g, ip, _ in PO.pauses(it, segs, prev) if ip))
    return np.asarray(L), np.asarray(N, dtype=np.float64)


class Hazard(nn.Module):
    def __init__(self, mu, sd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 32), nn.Tanh(), nn.Linear(32, 1))
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))

    def forward(self, F):   # (B, Jmax, 10) -> hazards (B, Jmax)
        return torch.sigmoid(self.net((F - self.mu) / self.sd)[..., 0])


def expected(model, Fp, Lp, Np, mask, alpha):
    h = model(Fp)
    last = mask.sum(1).long() - 1
    h = h.clone()
    h[torch.arange(len(h)), last] = 1.0
    h = h * mask
    surv = torch.cumprod(torch.cat([torch.ones(len(h), 1), 1 - h[:, :-1]], 1), 1)
    p = h * surv
    return (p * (Lp + alpha * Np)).sum(1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=96)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"拒绝覆盖已有输出: {out}")
    out.mkdir(parents=True)
    rows = []
    for d in ("en2zh", "zh2en"):
        for l in open(Path(a.dev) / f"talker_{d}" / "results.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if l.strip() and not r.get("error"):
                rows.append((r, LANG[d]))
    n = len(rows)
    from experiments.voice_onset_v1 import playout as PO
    _W.update(PO=PO, cache={}, items=[PO.Item(r, lang) for r, lang in rows])   # loaded before the fork, shared
    shards = [list(range(k, n, a.procs)) for k in range(a.procs)]
    log = open(out / "log.jsonl", "x")

    def say(o):
        o["time"] = time.strftime("%H:%M:%S")
        log.write(json.dumps(o) + "\n"); log.flush(); print(json.dumps(o), flush=True)

    # items are loaded before the fork; the cost cache is per worker (a state may be recomputed in another worker)
    with mp.get_context("fork").Pool(a.procs) as pool:
        def run(mode, state):
            res = pool.map(_run, [(s, mode, state) for s in shards], chunksize=1)
            dec = [x for r in res for x in r[0]]
            met = np.array([x for r in res for x in r[1]])
            return dec, met
        _, mr0 = run("r0", None)
        target = float(mr0[:, 0].mean())
        _, mnone = run("s2init", None)
        say({"n_items": n, "r0_dev_D2": target, "r0_NP_in": float(mr0[:, 1].mean()), "r0_NC": float(mr0[:, 2].mean()),
             "release_now_dev_D2": float(mnone[:, 0].mean()), "release_now_NP_in": float(mnone[:, 1].mean())})

        def train(alpha):
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            dec, _ = run("train", None)
            F0 = np.concatenate([f for f, _, _ in dec])
            model = Hazard(F0.mean(0), F0.std(0) + 1e-3)
            opt = torch.optim.Adam(model.net.parameters(), lr=LR)
            for ep in range(1, EPOCHS + 1):
                if ep > 1:
                    st = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
                    dec, _ = run("train", st)
                Jm = max(len(f) for f, _, _ in dec)
                B = len(dec)
                Fp = torch.zeros(B, Jm, 10); Lp = torch.zeros(B, Jm); Np = torch.zeros(B, Jm); mask = torch.zeros(B, Jm)
                for b, (f, L, N) in enumerate(dec):
                    m = len(f)
                    Fp[b, :m] = torch.from_numpy(f); Lp[b, :m] = torch.from_numpy(L).float()
                    Np[b, :m] = torch.from_numpy(N).float(); mask[b, :m] = 1
                for _ in range(STEPS):
                    loss = expected(model, Fp, Lp, Np, mask, alpha)
                    opt.zero_grad(); loss.backward(); opt.step()
                if ep % 10 == 0 or ep == 1:
                    say({"alpha": alpha, "epoch": ep, "decisions": B, "loss": float(loss)})
            st = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
            _, met = run("eval", st)
            return model, float(met[:, 0].mean()), float(met[:, 1].mean()), float(met[:, 2].mean())

        lo, hi = math.log(0.01), math.log(10.0)
        best = None
        for it_ in range(12):
            la = (lo + hi) / 2
            alpha = math.exp(la)
            model, d2, npi, nc = train(alpha)
            say({"calib": it_, "alpha": alpha, "dev_D2": d2, "dev_NP_in": npi, "dev_NC": nc, "target": target})
            if best is None or abs(d2 - target) < abs(best[1] - target):
                best = (alpha, d2, npi, nc, {k: v.clone() for k, v in model.state_dict().items()})
            if abs(d2 - target) <= 0.01:
                break
            if d2 < target:
                lo = la
            else:
                hi = la
        alpha, d2, npi, nc, state = best
        torch.save({"state": state, "alpha": alpha, "J": 8, "step_ms": 250}, out / "model.pt")
        json.dump({"alpha": alpha, "dev_D2": d2, "dev_NP_in": npi, "dev_NC": nc, "target_r0_dev_D2": target,
                   "r0_dev_NP_in": float(mr0[:, 1].mean()), "r0_dev_NC": float(mr0[:, 2].mean())},
                  open(out / "summary.json", "x"), indent=1)
        say({"done": True, "alpha": alpha, "dev_D2": d2, "dev_NP_in": npi})


if __name__ == "__main__":
    main()
