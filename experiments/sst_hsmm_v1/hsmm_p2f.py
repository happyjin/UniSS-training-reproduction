"""sst_hsmm_v1 change 54 (S8, anticipation-agreement certificate): a copy of hsmm_cons.py (not modified) whose 5th emission is the
P2F agreement bin pb of the position at that read (hsc21_roll rec["draft_p2f"] = the same read drafted by the P2F adapter):
1 = i < LCP(d0, d_p2f) (the anticipating draft agrees up to i), 0 = i < len(d_p2f) but outside the LCP, 2 = the P2F draft ends before i,
3 = NA (no P2F draft, the final read); categorical per state U / S with logits linear in [1, r, r^2, z]. Same objective, inputs, prior,
determinacy constraint. Precheck (80 / 20 split, reference = the same fit with pb masked) is REPORTED, never gating (change 54: the
final fit and the test run regardless). Original header follows.
sst_hsmm_v1 prereg change 38-2: HSC6 (hsmm_anchor.py, not modified) + one more emission: the K-sample consistency bin cb of the
position at that read (hsc10_si_thinker rec["cons"]): 0 = c == 0, 1 = 0 < c <= 0.5, 2 = 0.5 < c < 1, 3 = c == 1, 4 = NA (no sample reached
the position, the position is beyond the current draft, or the final read); categorical per state U / S with logits linear in
[1, r, r^2, z] like v / sq. Same objective, inputs, prior, determinacy constraint. Precheck on the 80 / 20 item split: the same fit with
cb masked to NA (= HSC6 on the same rollouts) is the reference; GO if AUC_cons >= AUC_ref + 0.010 and ECE_cons <= 0.10.
    <qwen3omni python> -m experiments.sst_hsmm_v1.hsmm_cons fit --roots <glob[,glob]> --pre-out <precheck.json> --out <params.json>
Was: sst_hsmm_v1 prereg change 33-A: HSC6 = the HSC4 HSMM (hsmm_joint structure: one model, direction as input, same inputs, dynamic
prior, emissions and determinacy constraint) trained with a hybrid generative-discriminative objective
    L = sum_seq log p(obs seq) + BETA * sum_inst [y log P(S_t | o_1..t) + (1 - y) log(1 - P(S_t | o_1..t))],   BETA = 1,
where P(S_t | o_1..t) is the filtering posterior the online commit uses (computed exactly in the same graph by a cumulative
logsumexp over the switch time), the instances are the reads at which the position was NOT committed by the N-policy rollout
(commit reads and the final read excluded: no self-fulfilling labels), and y = the position's prefix at that read survives
into the final output. Data: h1 rollouts (open UniST); labels from the rollouts themselves. Adam lr 1e-2, mini-batch 4096,
60 epochs, seed 20261125, uniform init. Saves hsmm_joint-format params (the online filter is hsmm_joint.posterior_stable).
    precheck (80 / 20 item split, vs h9; GO: AUC >= AUC9 + 0.010 and ECE <= 0.5 ECE9) + final fit on all data if GO:
      <qwen3omni python> -m experiments.sst_hsmm_v1.hsmm_anchor fit --roots 'eval_outputs/sst_hsmm_v1/h1/roll_*' \
          --pre-out <precheck.json> --out <params.json>          (exit 3 on NO-GO; refuses to overwrite)
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
from experiments.sst_hsmm_v1.hsmm_dep import ece  # noqa: E402
from experiments.sst_hsmm_v1.hsmm_joint import EPS, LMAX, NC, inputs_vec, relv_from_views  # noqa: E402

BETA = 1.0
CB_NA = 4
PB_NA = 3


def pbin(i, lpp, ldp):
    """change 54: P2F agreement bin of position i (lpp = LCP(d0, d_p2f), ldp = len(d_p2f); None = no P2F draft)"""
    if lpp is None:
        return PB_NA
    return 1 if i < lpp else (0 if i < ldp else 2)


class PParams(HJ.NParams):
    """hsmm_joint params + E["pb_U"], E["pb_S"]: (3, 4); observations are 5-tuples (a, v, sq, qN, pb)"""

    def rest(self, o, u, st):
        out = super().rest(o, u, st)
        if len(o) > 4 and o[4] != PB_NA:
            r, z = u[0], u[4]
            out += float(HJ._lsm(self.E[f"pb_{st}"] @ np.array([1.0, r, r * r, z]))[o[4]])
        return out


def cbin(c):
    if c is None or c < 0:
        return CB_NA
    return 0 if c == 0 else (1 if c <= 0.5 else (2 if c < 1 else 3))


class CParams(HJ.NParams):
    """hsmm_joint params + E["cb_U"], E["cb_S"]: (4, 4); observations are 5-tuples (a, v, sq, qN, cb)"""

    def rest(self, o, u, st):
        out = super().rest(o, u, st)
        if len(o) > 4 and o[4] != CB_NA:
            r, z = u[0], u[4]
            out += float(HJ._lsm(self.E[f"cb_{st}"] @ np.array([1.0, r, r * r, z]))[o[4]])
        return out


def walk(blocks_path, zh, relv, held):
    """as hsmm_dep.walk, but every sequence carries per-index labels (survival at an uncommitted non-final read, else -1)"""
    from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units
    seqs, inst = [], []
    for line in open(blocks_path, encoding="utf-8"):
        r = json.loads(line)
        RV = relv.get(r["id"])
        if RV is None:
            continue
        ho = held(r["id"])
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
            dp = units(x["draft_p2f"], zh) if "draft_p2f" in x else None
            lpp, ldp = (H2.lcp(d0, dp), len(dp)) if dp is not None else (None, 0)
            for i, o in H2.observations(d0, prev, d1, ds, lp0, lps, n).items():
                o = o + (pbin(i, lpp, ldp),)
                h = H.setdefault(i, ([], [], []))
                h[1].append(inputs_vec(rr, vv, len(h[0]), x["t"], zh)); h[0].append(o)
                y = int(i < ok) if i >= n + c else -1
                h[2].append(y)
                if ho and y >= 0:
                    inst.append((list(h[0]), list(h[1]), y, len(h[0])))
            for i in range(n, n + c):
                if i in H:
                    seqs.append((r["id"],) + H.pop(i))
            n += c
            prev = d0
        fx = r["drafts"][-1]
        dfin = units(fx["draft"], zh)
        rr, vv = RV.get(fx["t"], (1.0, 1.0))
        for i, o in H2.final_observations(dfin, prev, n).items():
            o = o + (PB_NA,)
            if i in H:
                h = H[i]; h[1].append(inputs_vec(rr, vv, len(h[0]), fx["t"], zh)); h[0].append(o); h[2].append(-1)
        seqs += [(r["id"],) + v for v in H.values()]
    return seqs, inst


def fit(seqs, epochs=60, bs=4096, lr=1e-2, seed=20261125, beta=BETA):
    """seqs: list of (obs, inputs, labels). Returns hsmm_joint-format params, history, and two numpy/torch checks."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)
    B = len(seqs)
    L = min(LMAX, max(len(s[0]) for s in seqs))
    O = torch.zeros(B, L, 5, dtype=torch.long); U = torch.zeros(B, L, 5); M = torch.zeros(B, L); T = torch.zeros(B, dtype=torch.long)
    Y = torch.full((B, L), -1, dtype=torch.long)
    for b, (ob, ip, yl) in enumerate(seqs):
        k = min(len(ob), L)
        O[b, :k] = torch.tensor(ob[:k]); U[b, :k] = torch.tensor(ip[:k], dtype=torch.float32); M[b, :k] = 1; T[b] = k
        Y[b, :k] = torch.tensor(yl[:k])
    NA = torch.tensor([H2.NA[k] for k in H2.KEYS] + [PB_NA])
    NAm = (O == NA) | (M[..., None] == 0)
    Oc = torch.where(NAm, torch.zeros_like(O), O)

    def mlp():
        return nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 1))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.g0, self.g = mlp(), mlp()
            self.E = nn.ParameterDict({f"{k}_{st}": nn.Parameter(torch.zeros(NC[k], 4)) for k in ("v", "sq") for st in ("U", "S")})
            self.E["a_U"] = nn.Parameter(torch.zeros(NC["a"], 4))
            self.E["qN_U"] = nn.Parameter(torch.zeros(NC["qN"], 2)); self.E["qN_S"] = nn.Parameter(torch.zeros(NC["qN"], 2))
            self.E["pb_U"] = nn.Parameter(torch.zeros(3, 4)); self.E["pb_S"] = nn.Parameter(torch.zeros(3, 4))

        def forward(self, o, oc, na, u, t):
            """returns (marginal log-lik (b,), filtering log P(S_t) (b, L), filtering log P(U_t) (b, L))"""
            r = u[..., 0]; zf = u[..., 4]
            feat = torch.stack([torch.ones_like(r), r, r * r, zf], -1)
            featq = torch.stack([torch.ones_like(r), zf], -1)

            def lp(key, j):
                if key == "qN":
                    return lambda st: torch.gather(F.log_softmax(featq @ self.E[f"qN_{st}"].T, -1), -1, oc[..., j:j + 1])[..., 0]
                return lambda st: torch.gather(F.log_softmax(feat @ self.E[f"{key}_{st}"].T, -1), -1, oc[..., j:j + 1])[..., 0]
            rest = {}
            for st in ("U", "S"):
                tot = 0.0
                for j, key in ((1, "v"), (2, "sq"), (3, "qN"), (4, "pb")):
                    tot = tot + torch.where(na[..., j], torch.zeros_like(r), lp(key, j)(st))
                rest[st] = tot
            aU = torch.where(na[..., 0], torch.zeros_like(r), lp("a", 0)("U"))
            aS = torch.where(na[..., 0], torch.zeros_like(r),
                             torch.where(oc[..., 0] == 1, torch.full_like(r, math.log(1 - EPS)), torch.full_like(r, math.log(EPS))))
            z0 = self.g0(u[:, 0])[..., 0]
            zh = self.g(u)[..., 0]
            lh, l1mh = F.logsigmoid(zh), F.logsigmoid(-zh)
            Lb = o.shape[1]
            valid = torch.arange(Lb)[None, :] < t[:, None]
            l1mh = torch.where(valid, l1mh, torch.zeros_like(l1mh))
            l1mh_ = l1mh.clone(); l1mh_[:, 0] = 0.0
            c1mh = torch.cumsum(l1mh_, 1)                                   # c1mh[:, k] = sum_{j=1..k} log(1 - h_j)
            prior = torch.empty_like(zh)
            prior[:, 0] = F.logsigmoid(z0)
            prior[:, 1:] = F.logsigmoid(-z0)[:, None] + c1mh[:, :-1] + lh[:, 1:]
            zU = torch.cumsum(rest["U"] + aU, 1); cU = torch.cat([torch.zeros_like(zU[:, :1]), zU], 1)
            zS = torch.cumsum(rest["S"], 1); cRS = torch.cat([torch.zeros_like(zS[:, :1]), zS], 1)
            zA = torch.cumsum(aS, 1); cAS = torch.cat([torch.zeros_like(zA[:, :1]), zA], 1)
            totRS, totAS = cRS[:, -1:], cAS[:, -1:]
            # marginal likelihood (as hsmm_joint)
            w = prior + cU[:, :-1] + rest["S"] + aU + (totRS - cRS[:, 1:]) + (totAS - cAS[:, 1:])
            w = torch.where(valid, w, torch.full_like(w, -1e30))
            cens = F.logsigmoid(-z0) + torch.gather(c1mh, 1, (t - 1)[:, None])[:, 0] + torch.gather(cU, 1, t[:, None])[:, 0]
            ll = torch.logsumexp(torch.cat([w, cens[:, None]], 1), 1)
            # filtering posterior after k + 1 observations (k = 0..L-1): switch d <= k vs censored at k
            a1 = prior + cU[:, :-1] + rest["S"] + aU - cRS[:, 1:] - cAS[:, 1:]          # (b, L), indexed by d
            num = torch.logcumsumexp(a1, 1) + cRS[:, 1:] + cAS[:, 1:]                  # (b, L), indexed by k
            cen = F.logsigmoid(-z0)[:, None] + c1mh + cU[:, 1:]
            den = torch.logaddexp(num, cen)
            return ll, num - den, cen - den

    m = Model()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(B)
        tl = tb = 0.0; nb = 0
        for s in range(0, B, bs):
            idx = perm[s: s + bs]
            ll, lS, lU = m(O[idx], Oc[idx], NAm[idx], U[idx], T[idx])
            y = Y[idx]; msk = y >= 0
            bce = torch.where(msk, torch.where(y == 1, lS, lU), torch.zeros_like(lS)).sum(1)
            loss = -(ll + beta * bce).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tl += float(ll.sum()); tb += float(bce.sum()); nb += int(msk.sum())
        hist.append({"loglik": tl / B, "bce_per_inst": -tb / max(nb, 1)})
        print(f"  epoch {ep} mean loglik {tl / B:.4f} survival BCE {-tb / max(nb, 1):.4f}", flush=True)
    sd = {k: v.detach().numpy() for k, v in m.state_dict().items()}
    params = {"g0": {"w1": sd["g0.0.weight"].tolist(), "b1": sd["g0.0.bias"].tolist(), "w2": sd["g0.2.weight"][0].tolist(), "b2": float(sd["g0.2.bias"][0])},
              "g": {"w1": sd["g.0.weight"].tolist(), "b1": sd["g.0.bias"].tolist(), "w2": sd["g.2.weight"][0].tolist(), "b2": float(sd["g.2.bias"][0])},
              "E": {k.split(".", 1)[1]: v.tolist() for k, v in sd.items() if k.startswith("E.")}}
    with torch.no_grad():                         # torch filtering posterior == numpy posterior_stable on prefixes
        P = PParams(params)
        k = min(30, B)
        _, lS, _ = m(O[:k], Oc[:k], NAm[:k], U[:k], T[:k])
        dmax = 0.0
        for b in range(k):
            ob, ip, _ = seqs[b]
            for j in range(min(len(ob), L)):
                dmax = max(dmax, abs(math.exp(float(lS[b, j])) - HJ.posterior_stable(ob[:j + 1], ip[:j + 1], P)))
    return params, hist, dmax


def report(name, p, y, Tn):
    p = np.asarray(p); y = np.asarray(y); Tn = np.minimum(np.asarray(Tn), 4)
    low = {int(t): (float(y[(Tn == t) & (p < 0.1)].mean()) if ((Tn == t) & (p < 0.1)).any() else None,
                    float(p[(Tn == t) & (p < 0.1)].mean()) if ((Tn == t) & (p < 0.1)).any() else None,
                    int(((Tn == t) & (p < 0.1)).sum())) for t in (1, 2, 3, 4)}
    bins = [0, .1, .3, .5, .7, .9, 1.0001]
    rel = [(bins[k], float(y[(p >= bins[k]) & (p < bins[k + 1])].mean()) if ((p >= bins[k]) & (p < bins[k + 1])).any() else None,
            int(((p >= bins[k]) & (p < bins[k + 1])).sum())) for k in range(len(bins) - 1)]
    out = {"ece": ece(p, y), "auc": H2.auc(y, p), "mean_p": float(p.mean()), "mean_y": float(y.mean()), "n": int(len(p)),
           "low_bin_by_T(survival, mean_p, n)": low, "reliability(bin_lo, survival, n)": rel}
    print(name, json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("fit",))
    ap.add_argument("--roots", required=True)
    ap.add_argument("--pre-out", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for f in (a.pre_out, a.out):
        if Path(f).exists():
            sys.exit(f"拒绝覆盖: {f}")
    held = lambda i: int(hashlib.sha1(i.encode()).hexdigest(), 16) % 5 == 0       # the change-32 split
    E1 = REPO / "eval_outputs/sst_hsmm_v1/e1"
    seqs, inst = [], []
    for d in ("en2zh", "zh2en"):
        relv = relv_from_views(E1 / "data" / f"items_{d}.jsonl", E1 / "views" / d)
        n0, m0 = len(seqs), len(inst)
        for root in a.roots.split(","):
            for f in sorted(glob.glob(f"{root}/blocks_{d}_w1.jsonl")):
                s, i = walk(f, d.endswith("zh"), relv, held)
                seqs += s; inst += i
        print(d, "sequences", len(seqs) - n0, "held-out instances", len(inst) - m0, flush=True)
    mask = lambda ob: [tuple(o[:4]) + (PB_NA,) for o in ob]
    train = [(o, u, y) for i, o, u, y in seqs if not held(i)]
    cov = float(np.mean([o[4] != PB_NA for _, ob, _, _ in seqs for o in ob]))
    print("train sequences", len(train), "pb observed share", round(cov, 3), flush=True)
    pc, hc, chk = fit(train)
    pr, hr, _ = fit([(mask(o), u, y) for o, u, y in train])
    print("torch filtering vs numpy posterior max abs diff", chk, flush=True)
    Pc, Pr = PParams(pc), PParams(pr)
    y = [x[2] for x in inst]; Tn = [x[3] for x in inst]
    rr = report("ref_masked_80", [HJ.posterior_stable(mask(o), u, Pr) for o, u, _, _ in inst], y, Tn)
    rc = report("p2f_80", [HJ.posterior_stable(o, u, Pc) for o, u, _, _ in inst], y, Tn)
    go = rc["auc"] >= rr["auc"] + 0.010 and rc["ece"] <= 0.10
    res = {"go": bool(go), "rule": "reported only (change 54): AUC_p2f >= AUC_ref(pb masked, same data) + 0.010 and ECE_p2f <= 0.10",
           "ref_masked_80": rr, "p2f_80": rc, "hist_p2f": hc, "hist_ref": hr, "filter_check": chk, "n_train": len(train),
           "pb_observed_share": cov, "beta": BETA, "E_pb": {k: v for k, v in pc["E"].items() if k.startswith("pb_")}}
    Path(a.pre_out).write_text(json.dumps(res, indent=1))
    print("precheck GO" if go else "precheck NO-GO (reported only; final fit continues)", flush=True)
    pall, hall, chk2 = fit([(o, u, y) for _, o, u, y in seqs])
    out = {"lmax": LMAX, "eps": EPS, "joint": True, "anchored": True, "p2f": True, "beta": BETA, "params": pall, "hist": hall,
           "filter_check": chk2, "n_seq": len(seqs)}
    out["dirs"] = {d: {"params": pall, "p2f": True} for d in ("en2zh", "zh2en")}      # the same single model for both directions
    Path(a.out).write_text(json.dumps(out))
    print("final fit written", a.out, flush=True)


if __name__ == "__main__":
    main()
