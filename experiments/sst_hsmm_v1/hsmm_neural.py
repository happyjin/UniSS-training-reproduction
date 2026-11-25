"""sst_hsmm_v1 prereg change 31: end-to-end differentiable input-driven HSMM (no separate EM).

Per draft position: U (unstable) for D reads, then S (absorbing). Exogenous inputs per read u = [r, vad, age/10, t/10]
(r = 20 ms-frame suppression-ratio reliability of the read's new 1 s, enhanced vs raw; vad = share of raw frames above
-40 dBFS; age = reads since the position first appeared; t = read time in s). Dynamic prior: p0 = sigmoid(g0(u_0)),
hazard h_k = sigmoid(g(u_k)) (k >= 1), P(switch at d) = prod_{j<d}(1 - h_j) h_d. Emissions (hsmm_commit2 observations
a, v, sq, qN): v, sq and U's a have logits linear in [1, r, r^2] per state; qN per state; determinacy constraint (S at t-1
=> a_t = 1, P(a = 0) = EPS). The exact marginal log-likelihood (forward over the switch time, censored end) is maximised
by back-propagation through the whole graph (prior networks + emission parameters jointly; its gradient is EM's expected
sufficient statistics). Training data: h1 N-policy rollouts on open UniST (no labels). Saves weights as JSON for the
numpy filter used online (posterior_stable).
    fit: <qwen3omni python> -m experiments.sst_hsmm_v1.hsmm_neural fit --roots 'eval_outputs/sst_hsmm_v1/h1/roll_*' --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1 import hsmm_commit2 as H2  # noqa: E402
from experiments.sst_hsmm_v1 import hsmm_commit3 as H3  # noqa: E402

EPS = 1e-3
LMAX = 16
NC = {"a": 2, "v": 2, "sq": 4, "qN": 3}          # modelled (non-NA) categories


def rel_vad(raw_chunk, enh_chunk):
    raw = np.asarray(raw_chunk, np.float64)
    F = 320; n = len(raw) // F
    vad = float(((raw[: n * F].reshape(n, F) ** 2).mean(1) > 1e-4).mean()) if n else 0.0
    return H3.reliability(raw_chunk, enh_chunk), vad


def inputs_vec(r, vad, age, t_ms):
    return [r, vad, min(age, 15) / 10.0, min(t_ms / 1000.0, 30.0) / 10.0]


# ------------------------------------------------------------------------------------------------ numpy online filter
def _mlp(Wd, x):
    h = np.tanh(Wd["w1"] @ x + Wd["b1"])
    return float(Wd["w2"] @ h + Wd["b2"])


def _lsm(z):
    z = np.asarray(z, float); m = z.max()
    return z - (m + math.log(np.exp(z - m).sum()))


def _logsig(x):
    return -math.log1p(math.exp(-x)) if x > -30 else x


class NParams:
    def __init__(self, d):
        self.g0 = {k: np.array(v, float) for k, v in d["g0"].items()}
        self.g = {k: np.array(v, float) for k, v in d["g"].items()}
        self.E = {k: np.array(v, float) for k, v in d["E"].items()}   # v_U, v_S, sq_U, sq_S, aU: (cats, 3); qN_U, qN_S: (cats,)

    def lp(self, key, st, cat, r):
        if key == "qN":
            return float(_lsm(self.E[f"qN_{st}"])[cat])
        return float(_lsm(self.E[f"{key}_{st}"] @ np.array([1.0, r, r * r]))[cat])

    def rest(self, o, r, st):
        out = 0.0
        if o[1] != H2.NA["v"]:
            out += self.lp("v", st, o[1], r)
        if o[2] != H2.NA["sq"]:
            out += self.lp("sq", st, o[2], r)
        if o[3] != H2.NA["qN"]:
            out += self.lp("qN", st, o[3], r)
        return out

    def la(self, o, r, prev_st):
        if o[0] == 2:
            return 0.0
        if prev_st == "S":
            return math.log(1 - EPS) if o[0] == 1 else math.log(EPS)
        return self.lp("a", "U", o[0], r)


def posterior_stable(seq, inps, P: NParams) -> float:
    """P(S at the last observation | obs, inputs); inps[k] = inputs_vec(...) of sequence index k"""
    T = len(seq)
    if T == 0:
        return 0.0
    rs = [u[0] for u in inps]
    rU = [P.rest(o, r, "U") for o, r in zip(seq, rs)]; rS = [P.rest(o, r, "S") for o, r in zip(seq, rs)]
    aU = [P.la(o, r, "U") for o, r in zip(seq, rs)]; aS = [P.la(o, r, "S") for o, r in zip(seq, rs)]
    z0 = _mlp(P.g0, np.array(inps[0]))
    lp0, l1mp0 = _logsig(z0), _logsig(-z0)
    zs = [_mlp(P.g, np.array(u)) for u in inps]
    cU = np.concatenate([[0.0], np.cumsum(np.array(rU) + np.array(aU))])
    cRS = np.concatenate([[0.0], np.cumsum(rS)]); cAS = np.concatenate([[0.0], np.cumsum(aS)])
    c1mh = 0.0
    lw = []
    for d in range(T):
        if d == 0:
            pr = lp0
        else:
            pr = l1mp0 + c1mh + _logsig(zs[d])
            c1mh += _logsig(-zs[d])
        lw.append(pr + cU[d] + rS[d] + aU[d] + (cRS[T] - cRS[d + 1]) + (cAS[T] - cAS[d + 1]))
    cens = l1mp0 + c1mh + cU[T]
    allw = lw + [cens]
    m = max(allw)
    return sum(math.exp(w - m) for w in lw) / sum(math.exp(w - m) for w in allw)


# --------------------------------------------------------------------------------------------------------- training data
def build_sequences(blocks_path, zh, relv):
    from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units
    out = []
    for line in open(blocks_path, encoding="utf-8"):
        r = json.loads(line)
        RV = relv.get(r["id"])
        if RV is None:
            continue
        n, prev, H = 0, None, {}
        for x in r["drafts"][:-1]:
            if "lp0" not in x:
                continue
            d0 = units(x["draft"], zh); d1 = units(x["draft_v1"], zh); ds = units(x["draft_sap"], zh)
            lp0 = [0.0] * n + x["lp0"]; lps = [0.0] * n + x["lps"]
            rr, vv = RV.get(x["t"], (1.0, 1.0))
            for i, o in H2.observations(d0, prev, d1, ds, lp0, lps, n).items():
                h = H.setdefault(i, ([], []))
                h[1].append(inputs_vec(rr, vv, len(h[0]), x["t"])); h[0].append(o)
            c = x["cv_new"]
            for i in range(n, n + c):
                if i in H:
                    out.append(H.pop(i))
            n += c
            prev = d0
        fx = r["drafts"][-1]
        dfin = units(fx["draft"], zh)
        rr, vv = RV.get(fx["t"], (1.0, 1.0))
        for i, o in H2.final_observations(dfin, prev, n).items():
            if i in H:
                h = H[i]; h[1].append(inputs_vec(rr, vv, len(h[0]), fx["t"])); h[0].append(o)
        out += list(H.values())
    return out


def relv_from_views(items_file, views_dir):
    import soundfile as sf
    idx = {r["id"]: r for r in map(json.loads, open(Path(views_dir) / "index.jsonl", encoding="utf-8"))}
    out = {}
    for it in map(json.loads, open(items_file, encoding="utf-8")):
        r = idx.get(it["id"])
        if r is None or not r["reads"]:
            continue
        x, _ = sf.read(it["audio"], dtype="float32", always_2d=True)
        x = x.mean(axis=1)[: int(it["dur_ms"]) * 16]
        v = np.load(Path(views_dir) / r["file"])
        o, d = 0, {}
        for t in r["reads"]:
            pref = v[o: o + t * 16]; o += t * 16
            d[t] = rel_vad(x[(t - 1000) * 16: t * 16], pref[-16000:])
        d[len(x) // 16] = d[max(r["reads"])]
        out[it["id"]] = d
    return out


# ------------------------------------------------------------------------------------------------------------ torch model
def fit(seqs, epochs=60, bs=4096, lr=1e-2, seed=20261125):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)
    B = len(seqs)
    L = min(LMAX, max(len(s[0]) for s in seqs))
    O = torch.full((B, L, 4), 0, dtype=torch.long); U = torch.zeros(B, L, 4); M = torch.zeros(B, L); T = torch.zeros(B, dtype=torch.long)
    NAm = torch.zeros(B, L, 4, dtype=torch.bool)
    for b, (ob, ip) in enumerate(seqs):
        k = min(len(ob), L)
        O[b, :k] = torch.tensor(ob[:k]); U[b, :k] = torch.tensor(ip[:k], dtype=torch.float32); M[b, :k] = 1; T[b] = k
    NA = torch.tensor([H2.NA[k] for k in H2.KEYS])
    NAm = (O == NA) | (M[..., None] == 0)
    Oc = torch.where(NAm, torch.zeros_like(O), O)

    def mlp():
        return nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.g0, self.g = mlp(), mlp()
            self.E = nn.ParameterDict({f"{k}_{st}": nn.Parameter(torch.zeros(NC[k], 3)) for k in ("v", "sq") for st in ("U", "S")})
            self.E["a_U"] = nn.Parameter(torch.zeros(NC["a"], 3))
            self.E["qN_U"] = nn.Parameter(torch.zeros(NC["qN"])); self.E["qN_S"] = nn.Parameter(torch.zeros(NC["qN"]))

        def forward(self, o, oc, na, u, t):
            r = u[..., 0]
            feat = torch.stack([torch.ones_like(r), r, r * r], -1)                       # (b, L, 3)

            def lp(key, j):
                if key == "qN":
                    return lambda st: F.log_softmax(self.E[f"qN_{st}"], -1)[oc[..., j]]
                return lambda st: torch.gather(F.log_softmax(feat @ self.E[f"{key}_{st}"].T, -1), -1, oc[..., j:j + 1])[..., 0]
            rest = {}
            for st in ("U", "S"):
                tot = 0.0
                for j, key in ((1, "v"), (2, "sq"), (3, "qN")):
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
            c1mh = torch.cumsum(l1mh_, 1)                                                 # sum_{j=1..d} log(1-h_j)
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

    m = Model()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(B)
        tot = 0.0
        for s in range(0, B, bs):
            idx = perm[s: s + bs]
            ll = m(O[idx], Oc[idx], NAm[idx], U[idx], T[idx])
            loss = -ll.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(ll.sum())
        hist.append(tot / B)
        print(f"  epoch {ep} mean loglik {hist[-1]:.4f}", flush=True)
    sd = {k: v.detach().numpy() for k, v in m.state_dict().items()}
    params = {"g0": {"w1": sd["g0.0.weight"].tolist(), "b1": sd["g0.0.bias"].tolist(), "w2": sd["g0.2.weight"][0].tolist(), "b2": float(sd["g0.2.bias"][0])},
              "g": {"w1": sd["g.0.weight"].tolist(), "b1": sd["g.0.bias"].tolist(), "w2": sd["g.2.weight"][0].tolist(), "b2": float(sd["g.2.bias"][0])},
              "E": {k.split(".", 1)[1]: v.tolist() for k, v in sd.items() if k.startswith("E.")}}
    return params, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("fit",))
    ap.add_argument("--roots", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit(f"拒绝覆盖: {a.out}")
    E1 = REPO / "eval_outputs/sst_hsmm_v1/e1"
    res = {"lmax": LMAX, "eps": EPS, "dirs": {}}
    for d in ("en2zh", "zh2en"):
        relv = relv_from_views(E1 / "data" / f"items_{d}.jsonl", E1 / "views" / d)
        seqs = []
        for f in sorted(glob.glob(f"{a.roots}/blocks_{d}_w1.jsonl")):
            seqs += build_sequences(f, d.endswith("zh"), relv)
        print(d, "sequences", len(seqs), flush=True)
        params, hist = fit(seqs)
        P = NParams(params)
        # numpy filter == torch likelihood check on a few sequences (posterior in [0, 1])
        chk = [posterior_stable(s[0], s[1], P) for s in seqs[:200]]
        res["dirs"][d] = {"params": params, "loglik": hist, "n_seq": len(seqs), "post_mean_200": float(np.mean(chk))}
    Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
