"""sst_hsmm_v1 prereg change 32: HSC5 = dependency-aware end-to-end HSMM, ONE model for both directions (direction is an input).
Same switch-time structure, inputs u = inputs_vec(r, vad, age, t, zh) and dynamic prior (p0 = sigmoid(g0(u_0)), hazard
h_k = sigmoid(g(u_k))) as hsmm_joint (HSC4). Changes (the HSC4 diagnosis: evidence double counted within a read and across reads):
  * joint emission of the per-read evidence tuple e_t = (v, sq, qN) (24 classes, NA at the final read) per state, not a product;
  * temporal persistence: log P(e_t | st, e_{t-1}, u_t) = log_softmax(A_st[e_{t-1}] + B_st [1, r, r^2, z])[e_t], e_{t-1} with an
    extra 'first observation' row (25 rows); U's agreement a likewise conditioned on the previous read's a (0 / 1 / none);
  * determinacy constraint unchanged (S at t-1 => a_t = 1, P(a = 0) = EPS).
Exact marginal likelihood (forward over the switch time, censored end), Adam end-to-end on the h1 N-policy rollouts (open UniST,
no labels), mini-batch 4096, 60 epochs, lr 1e-2, seed 20261125, uniform init.
    precheck + final fit (CPU):
      <qwen3omni python> -m experiments.sst_hsmm_v1.hsmm_dep fit --roots 'eval_outputs/sst_hsmm_v1/h1/roll_*' \
          --pre-out <precheck.json> --out <params.json>
    (the final model is fitted on all h1 data only if the precheck is GO; refuses to overwrite)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1 import hsmm_commit2 as H2  # noqa: E402
from experiments.sst_hsmm_v1 import hsmm_joint as HJ  # noqa: E402
from experiments.sst_hsmm_v1.hsmm_joint import EPS, LMAX, _logsig, _lsm, _mlp, inputs_vec, rel_vad, relv_from_views  # noqa: E402,F401

NE = 24          # evidence classes v * 12 + sq * 3 + qN
ENONE = NE       # 'first observation' row of A
ANONE = 2        # previous a absent / NA


def ecode(o):
    return None if o[1] == H2.NA["v"] else o[1] * 12 + o[2] * 3 + o[3]


def prev_codes(seq):
    """per sequence index: (code of e at the previous index or ENONE, previous a or ANONE)"""
    ep, ap, pe, pa = [], [], ENONE, ANONE
    for o in seq:
        ep.append(pe); ap.append(pa)
        e = ecode(o)
        pe = e if e is not None else ENONE
        pa = o[0] if o[0] != H2.NA["a"] else ANONE
    return ep, ap


def _feat(u):
    return np.array([1.0, u[0], u[0] * u[0], u[4]])


class DParams:
    def __init__(self, d):
        self.g0 = {k: np.array(v, float) for k, v in d["g0"].items()}
        self.g = {k: np.array(v, float) for k, v in d["g"].items()}
        self.A = {st: np.array(d[f"A_{st}"], float) for st in ("U", "S")}      # (25, 24)
        self.B = {st: np.array(d[f"B_{st}"], float) for st in ("U", "S")}      # (24, 4)
        self.Aa = np.array(d["Aa"], float)                                      # (3, 2)
        self.Ba = np.array(d["Ba"], float)                                      # (2, 4)


def _terms(seq, inps, P: DParams):
    ep, ap = prev_codes(seq)
    rU, rS, aU, aS = [], [], [], []
    for o, u, e_prev, a_prev in zip(seq, inps, ep, ap):
        f = _feat(u)
        e = ecode(o)
        if e is None:
            rU.append(0.0); rS.append(0.0)
        else:
            rU.append(float(_lsm(P.A["U"][e_prev] + P.B["U"] @ f)[e])); rS.append(float(_lsm(P.A["S"][e_prev] + P.B["S"] @ f)[e]))
        if o[0] == H2.NA["a"]:
            aU.append(0.0); aS.append(0.0)
        else:
            aU.append(float(_lsm(P.Aa[a_prev] + P.Ba @ f)[o[0]]))
            aS.append(math.log(1 - EPS) if o[0] == 1 else math.log(EPS))
    return rU, rS, aU, aS


def _weights(seq, inps, P: DParams):
    T = len(seq)
    rU, rS, aU, aS = _terms(seq, inps, P)
    z0 = _mlp(P.g0, np.array(inps[0]))
    lp0, l1mp0 = _logsig(z0), _logsig(-z0)
    zs = [_mlp(P.g, np.array(u)) for u in inps]
    cU = np.concatenate([[0.0], np.cumsum(np.array(rU) + np.array(aU))])
    cRS = np.concatenate([[0.0], np.cumsum(rS)]); cAS = np.concatenate([[0.0], np.cumsum(aS)])
    c1mh, lw = 0.0, []
    for d in range(T):
        if d == 0:
            pr = lp0
        else:
            pr = l1mp0 + c1mh + _logsig(zs[d])
            c1mh += _logsig(-zs[d])
        lw.append(pr + cU[d] + rS[d] + aU[d] + (cRS[T] - cRS[d + 1]) + (cAS[T] - cAS[d + 1]))
    cens = l1mp0 + c1mh + cU[T]
    return lw, cens


def posterior_stable(seq, inps, P: DParams) -> float:
    """P(S at the last observation | obs, inputs)"""
    if not seq:
        return 0.0
    lw, cens = _weights(seq, inps, P)
    allw = lw + [cens]
    m = max(allw)
    return sum(math.exp(w - m) for w in lw) / sum(math.exp(w - m) for w in allw)


def loglik_np(seq, inps, P: DParams) -> float:
    lw, cens = _weights(seq, inps, P)
    allw = np.array(lw + [cens]); m = allw.max()
    return float(m + math.log(np.exp(allw - m).sum()))


# --------------------------------------------------------------------------------------------------------- data
def walk(blocks_path, zh, relv, held=None):
    """per-position sequences (as hsmm_joint.build_sequences, item id attached) and, for items with held(id) True, evaluation
    instances (prefix obs, prefix inputs, label, #obs) at every non-final read for every position left uncommitted by the
    N-policy rollout; label = the position's prefix survives into the final output."""
    from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units
    seqs, inst = [], []
    for line in open(blocks_path, encoding="utf-8"):
        r = json.loads(line)
        RV = relv.get(r["id"])
        if RV is None:
            continue
        ho = bool(held and held(r["id"]))
        fin = units(r["translation"], zh)
        n, prev, H = 0, None, {}
        for x in r["drafts"][:-1]:
            if "lp0" not in x:
                continue
            d0 = units(x["draft"], zh); d1 = units(x["draft_v1"], zh); ds = units(x["draft_sap"], zh)
            lp0 = [0.0] * n + x["lp0"]; lps = [0.0] * n + x["lps"]
            rr, vv = RV.get(x["t"], (1.0, 1.0))
            ok = H2.lcp(d0, fin)
            c = x["cv_new"]
            for i, o in H2.observations(d0, prev, d1, ds, lp0, lps, n).items():
                h = H.setdefault(i, ([], []))
                h[1].append(inputs_vec(rr, vv, len(h[0]), x["t"], zh)); h[0].append(o)
                if ho and i >= n + c:
                    inst.append((list(h[0]), list(h[1]), int(i < ok), len(h[0])))
            for i in range(n, n + c):
                if i in H:
                    seqs.append((r["id"],) + H.pop(i))
            n += c
            prev = d0
        fx = r["drafts"][-1]
        dfin = units(fx["draft"], zh)
        rr, vv = RV.get(fx["t"], (1.0, 1.0))
        for i, o in H2.final_observations(dfin, prev, n).items():
            if i in H:
                h = H[i]; h[1].append(inputs_vec(rr, vv, len(h[0]), fx["t"], zh)); h[0].append(o)
        seqs += [(r["id"],) + v for v in H.values()]
    return seqs, inst


# ------------------------------------------------------------------------------------------------------------ torch model
def _tensors(seqs):
    import torch
    B = len(seqs)
    L = min(LMAX, max(len(s[0]) for s in seqs))
    E = torch.zeros(B, L, dtype=torch.long); EP = torch.full((B, L), ENONE, dtype=torch.long)
    A = torch.zeros(B, L, dtype=torch.long); AP = torch.full((B, L), ANONE, dtype=torch.long)
    Em = torch.ones(B, L, dtype=torch.bool); Am = torch.ones(B, L, dtype=torch.bool)       # True = no likelihood term
    U = torch.zeros(B, L, 5); T = torch.zeros(B, dtype=torch.long)
    for b, (ob, ip) in enumerate(seqs):
        k = min(len(ob), L)
        ep, ap = prev_codes(ob[:k])
        for j in range(k):
            e = ecode(ob[j])
            if e is not None:
                E[b, j] = e; Em[b, j] = False
            if ob[j][0] != H2.NA["a"]:
                A[b, j] = ob[j][0]; Am[b, j] = False
        EP[b, :k] = torch.tensor(ep); AP[b, :k] = torch.tensor(ap)
        U[b, :k] = torch.tensor(ip[:k], dtype=torch.float32); T[b] = k
    return E, EP, Em, A, AP, Am, U, T


def make_model():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def mlp():
        return nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 1))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.g0, self.g = mlp(), mlp()
            self.A_U = nn.Parameter(torch.zeros(NE + 1, NE)); self.A_S = nn.Parameter(torch.zeros(NE + 1, NE))
            self.B_U = nn.Parameter(torch.zeros(NE, 4)); self.B_S = nn.Parameter(torch.zeros(NE, 4))
            self.Aa = nn.Parameter(torch.zeros(3, 2)); self.Ba = nn.Parameter(torch.zeros(2, 4))

        def forward(self, E, EP, Em, A, AP, Am, u, t):
            r = u[..., 0]; zf = u[..., 4]
            feat = torch.stack([torch.ones_like(r), r, r * r, zf], -1)                     # (b, L, 4)
            rest = {}
            for st, Am_, Bm in (("U", self.A_U, self.B_U), ("S", self.A_S, self.B_S)):
                lg = F.log_softmax(Am_[EP] + feat @ Bm.T, -1)
                rest[st] = torch.where(Em, torch.zeros_like(r), torch.gather(lg, -1, E[..., None])[..., 0])
            la = F.log_softmax(self.Aa[AP] + feat @ self.Ba.T, -1)
            aU = torch.where(Am, torch.zeros_like(r), torch.gather(la, -1, A[..., None])[..., 0])
            aS = torch.where(Am, torch.zeros_like(r),
                             torch.where(A == 1, torch.full_like(r, math.log(1 - EPS)), torch.full_like(r, math.log(EPS))))
            z0 = self.g0(u[:, 0])[..., 0]
            zh = self.g(u)[..., 0]
            lh, l1mh = F.logsigmoid(zh), F.logsigmoid(-zh)
            Lb = E.shape[1]
            valid = torch.arange(Lb)[None, :] < t[:, None]
            l1mh = torch.where(valid, l1mh, torch.zeros_like(l1mh))
            l1mh_ = l1mh.clone(); l1mh_[:, 0] = 0.0
            c1mh = torch.cumsum(l1mh_, 1)
            prior = torch.empty_like(zh)
            prior[:, 0] = F.logsigmoid(z0)
            prior[:, 1:] = F.logsigmoid(-z0)[:, None] + c1mh[:, :-1] + lh[:, 1:]
            zU = torch.cumsum(rest["U"] + aU, 1); cU = torch.cat([torch.zeros_like(zU[:, :1]), zU], 1)
            zS = torch.cumsum(rest["S"], 1); cRS = torch.cat([torch.zeros_like(zS[:, :1]), zS], 1)
            zA = torch.cumsum(aS, 1); cAS = torch.cat([torch.zeros_like(zA[:, :1]), zA], 1)
            totRS, totAS = cRS[:, -1:], cAS[:, -1:]
            w = prior + cU[:, :-1] + rest["S"] + aU + (totRS - cRS[:, 1:]) + (totAS - cAS[:, 1:])
            w = torch.where(valid, w, torch.full_like(w, -1e30))
            cens = F.logsigmoid(-z0) + torch.gather(c1mh, 1, (t - 1)[:, None])[:, 0] + torch.gather(cU, 1, t[:, None])[:, 0]
            return torch.logsumexp(torch.cat([w, cens[:, None]], 1), 1)
    return Model()


def fit(seqs, epochs=60, bs=4096, lr=1e-2, seed=20261125):
    """seqs: list of (obs, inputs)"""
    import torch
    torch.manual_seed(seed)
    X = _tensors(seqs)
    B = len(seqs)
    m = make_model()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(B)
        tot = 0.0
        for s in range(0, B, bs):
            idx = perm[s: s + bs]
            ll = m(*[x[idx] for x in X])
            loss = -ll.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(ll.sum())
        hist.append(tot / B)
        print(f"  epoch {ep} mean loglik {hist[-1]:.4f}", flush=True)
    sd = {k: v.detach().numpy() for k, v in m.state_dict().items()}
    params = {"g0": {"w1": sd["g0.0.weight"].tolist(), "b1": sd["g0.0.bias"].tolist(), "w2": sd["g0.2.weight"][0].tolist(), "b2": float(sd["g0.2.bias"][0])},
              "g": {"w1": sd["g.0.weight"].tolist(), "b1": sd["g.0.bias"].tolist(), "w2": sd["g.2.weight"][0].tolist(), "b2": float(sd["g.2.bias"][0])},
              **{k: sd[k].tolist() for k in ("A_U", "A_S", "B_U", "B_S", "Aa", "Ba")}}
    with torch.no_grad():                                         # numpy filter == torch likelihood check on 50 sequences
        k = min(50, B)
        ll_t = m(*[x[:k] for x in X]).numpy()
        P = DParams(params)
        ll_n = np.array([loglik_np(o[:LMAX], u[:LMAX], P) for o, u in seqs[:k]])
    return params, hist, float(np.abs(ll_t - ll_n).max())


# ------------------------------------------------------------------------------------------------------------ precheck
def ece(p, y, nb=10):
    p = np.asarray(p); y = np.asarray(y)
    b = np.minimum((p * nb).astype(int), nb - 1)
    return float(sum((b == k).sum() / len(p) * abs(p[b == k].mean() - y[b == k].mean()) for k in range(nb) if (b == k).any()))


def report(name, p, y, Tn):
    p = np.asarray(p); y = np.asarray(y); Tn = np.minimum(np.asarray(Tn), 4)
    low = {int(t): (float(y[(Tn == t) & (p < 0.1)].mean()) if ((Tn == t) & (p < 0.1)).any() else None,
                    float(p[(Tn == t) & (p < 0.1)].mean()) if ((Tn == t) & (p < 0.1)).any() else None,
                    int(((Tn == t) & (p < 0.1)).sum())) for t in (1, 2, 3, 4)}
    out = {"ece": ece(p, y), "auc": H2.auc(y, p), "mean_p": float(p.mean()), "mean_y": float(y.mean()), "n": int(len(p)),
           "low_bin_by_T(survival, mean_p, n)": low}
    print(name, json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("fit",))
    ap.add_argument("--roots", required=True)
    ap.add_argument("--pre-out", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=str(REPO / "eval_outputs/sst_hsmm_v1/h9/params.json"))
    a = ap.parse_args()
    for f in (a.pre_out, a.out):
        if Path(f).exists():
            sys.exit(f"拒绝覆盖: {f}")
    held = lambda i: int(hashlib.sha1(i.encode()).hexdigest(), 16) % 5 == 0
    E1 = REPO / "eval_outputs/sst_hsmm_v1/e1"
    seqs, inst = [], []
    for d in ("en2zh", "zh2en"):
        relv = relv_from_views(E1 / "data" / f"items_{d}.jsonl", E1 / "views" / d)
        n0, m0 = len(seqs), len(inst)
        for f in sorted(glob.glob(f"{a.roots}/blocks_{d}_w1.jsonl")):
            s, i = walk(f, d.endswith("zh"), relv, held)
            seqs += s; inst += i
        print(d, "sequences", len(seqs) - n0, "held-out instances", len(inst) - m0, flush=True)
    train = [(o, u) for i, o, u in seqs if not held(i)]
    print("train sequences", len(train), "held-out sequences", sum(held(i) for i, _, _ in seqs), flush=True)
    p80, h80, chk = fit(train)
    print("numpy/torch loglik max abs diff", chk, flush=True)
    P5 = DParams(p80)
    P9 = HJ.NParams(json.load(open(a.ref))["params"])
    y = [x[2] for x in inst]; Tn = [x[3] for x in inst]
    r9 = report("h9", [HJ.posterior_stable(o, u, P9) for o, u, _, _ in inst], y, Tn)
    r5 = report("hsc5_80", [posterior_stable(o, u, P5) for o, u, _, _ in inst], y, Tn)
    go = r5["ece"] <= 0.7 * r9["ece"] and r5["auc"] >= r9["auc"] - 0.005
    res = {"go": bool(go), "rule": "ECE5 <= 0.7 ECE9 and AUC5 >= AUC9 - 0.005", "h9": r9, "hsc5_80": r5,
           "loglik_80": h80, "np_torch_diff": chk, "n_train": len(train)}
    Path(a.pre_out).write_text(json.dumps(res, indent=1))
    print("precheck GO" if go else "precheck NO-GO", flush=True)
    if not go:
        sys.exit(3)
    pall, hall, chk2 = fit([(o, u) for _, o, u in seqs])
    out = {"lmax": LMAX, "eps": EPS, "joint": True, "params": pall, "loglik": hall, "np_torch_diff": chk2, "n_seq": len(seqs)}
    out["dirs"] = {d: {"params": pall} for d in ("en2zh", "zh2en")}      # the same single model for both directions
    Path(a.out).write_text(json.dumps(out))
    print("final fit written", a.out, flush=True)


if __name__ == "__main__":
    main()
