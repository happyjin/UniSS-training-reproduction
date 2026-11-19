"""Idea 2 M1b: contraction-aware streaming identity posterior (prereg v2, reports/mist_w3po_v1/voicecond_idea2_prereg.zh-CN.md).

  z_k = lambda_k s_j + (1 - lambda_k) c + eps_k,  eps_k ~ N(0, R / pi_k),  lambda_k = sigmoid(alpha + beta^T v~_k)

    python -I experiments/mist_w3po_v1/voicecond/idea2_shrink.py fit  --which pilot|all --e20 <e20 dir> --out <dir>
    python -I experiments/mist_w3po_v1/voicecond/idea2_shrink.py eval --which pilot|all --split test --e20 <dir> --fit <dir> --out <dir>

ShrinkStream is the causal streaming interface (also used for the esports demo): step(z, slot, pi, vt) per window,
mean(slot) -> window-space posterior mean (add b_M for prompt space). Outputs refuse to overwrite.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_common import (DIAR_DELAY, I2, T_EVAL, cluster_boot, load_map,  # noqa: E402
                                                             load_refs, load_stream, sids, slot_for_speaker,
                                                             speech_clock)

D = 192
LANGS = ("en", "zh")


def load_vt(sid, L, K):
    return np.load(I2 / f"pre/feats2_L{L:g}" / f"{sid}.npz")["vt"][:K].astype(np.float64)


# ------------------------------------------------------------------------------------------------ parameters
class ShrinkParams:
    def __init__(self, alpha, beta, c, R, vt_mean, vt_std, mu0, Sig0, bM, L=2.0, per_lang=False):
        self.alpha, self.beta = float(alpha), np.asarray(beta, np.float64)
        self.c = {k: np.asarray(v, np.float64) for k, v in c.items()}      # {"all": c} or {"en": .., "zh": ..}
        self.R, self.vt_mean, self.vt_std = np.asarray(R), np.asarray(vt_mean), np.asarray(vt_std)
        self.mu0, self.Sig0, self.bM = np.asarray(mu0), np.asarray(Sig0), np.asarray(bM)
        self.L, self.per_lang = float(L), bool(per_lang)

    def cvec(self, lang):
        return self.c[lang] if self.per_lang else self.c["all"]

    def lam(self, vt):
        x = (np.asarray(vt, np.float64) - self.vt_mean) / self.vt_std
        return 1.0 / (1.0 + np.exp(-(self.alpha + x @ self.beta)))

    def save(self, path):
        np.savez(path, alpha=self.alpha, beta=self.beta, c_keys=np.array(list(self.c)),
                 c_vals=np.stack(list(self.c.values())), R=self.R, vt_mean=self.vt_mean, vt_std=self.vt_std,
                 mu0=self.mu0, Sig0=self.Sig0, bM=self.bM, L=self.L, per_lang=self.per_lang)

    @staticmethod
    def load(path):
        z = np.load(path)
        c = {str(k): v for k, v in zip(z["c_keys"], z["c_vals"])}
        return ShrinkParams(z["alpha"], z["beta"], c, z["R"], z["vt_mean"], z["vt_std"], z["mu0"], z["Sig0"], z["bM"],
                            float(z["L"]), bool(z["per_lang"]))


class ShrinkStream:
    """causal per-slot Gaussian posterior. mode: 'full' (lambda from v~), 'const' (params fitted with beta = 0),
    'prior' (lambda = 1, no c). step() after each 0.5 s-hop window."""

    def __init__(self, p: ShrinkParams, lang: str, mode="full", hop=0.5):
        self.p, self.mode = p, mode
        self.c = p.cvec(lang)
        Rf = p.R * (p.L / hop)
        self.Rfi = np.linalg.inv(Rf)
        self.S0i = np.linalg.inv(p.Sig0)
        self.e0 = self.S0i @ p.mu0
        self.acc = {}          # slot -> [sum pi lam^2, sum pi lam (z - (1 - lam) c)]

    def step(self, z, slot, pi, vt):
        if z is None or slot is None or slot < 0 or pi <= 0:
            return
        lam = 1.0 if self.mode == "prior" else float(self.p.lam(vt))
        y = np.asarray(z, np.float64) - (0.0 if self.mode == "prior" else (1 - lam) * self.c)
        a = self.acc.setdefault(int(slot), [0.0, np.zeros(D)])
        a[0] += pi * lam * lam
        a[1] = a[1] + pi * lam * y

    def mean(self, slot):
        a = self.acc.get(int(slot))
        if a is None:
            return self.p.mu0.copy()
        return np.linalg.solve(self.S0i + a[0] * self.Rfi, self.e0 + self.Rfi @ a[1])

    def info(self, slot):
        a = self.acc.get(int(slot))
        return 0.0 if a is None else a[0]


# ------------------------------------------------------------------------------------------------ fit
def fit_one(Z, S, V, W, LG, per_lang, const, log):
    """weighted LS for (alpha, beta, c); returns alpha, beta, c dict, R, objective"""
    dev = torch.device("cpu")
    Zt, St, Vt, Wt = (torch.as_tensor(x, dtype=torch.float64, device=dev) for x in (Z, S, V, W))
    keys = list(LANGS) if per_lang else ["all"]
    gi = torch.as_tensor(np.array([keys.index(l) if per_lang else 0 for l in LG]), device=dev)
    lam0 = 0.63
    c0 = []
    for k in range(len(keys)):
        m = (gi == k)
        y = (Zt[m] - lam0 * St[m]) / (1 - lam0)
        c0.append((Wt[m, None] * y).sum(0) / Wt[m].sum())
    alpha = torch.tensor([np.log(lam0 / (1 - lam0))], dtype=torch.float64, requires_grad=True)
    beta = torch.zeros(V.shape[1], dtype=torch.float64, requires_grad=not const)
    c = torch.stack(c0).clone().requires_grad_(True)
    params = [alpha, c] + ([] if const else [beta])
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-14,
                            history_size=50, line_search_fn="strong_wolfe")
    Wn = Wt / Wt.sum()

    def obj():
        lam = torch.sigmoid(alpha + Vt @ beta)[:, None]
        e = Zt - lam * St - (1 - lam) * c[gi]
        return (Wn * (e * e).sum(1)).sum()

    def closure():
        opt.zero_grad()
        f = obj()
        f.backward()
        return f
    f0 = float(obj())
    opt.step(closure)
    with torch.no_grad():
        f1 = float(obj())
        lam = torch.sigmoid(alpha + Vt @ beta)[:, None]
        e = Zt - lam * St - (1 - lam) * c[gi]
        R = ((Wt[:, None] * e).T @ e / len(Zt)).numpy()
    log(f"[fit] per_lang={per_lang} const={const}: objective {f0:.4f} -> {f1:.4f}; alpha {float(alpha):.4f}"
        f" beta {np.round(beta.detach().numpy(), 4).tolist()}; lambda mean {float(lam.mean()):.3f}"
        f" [p10 {float(lam.quantile(0.1)):.3f}, p90 {float(lam.quantile(0.9)):.3f}]")
    return float(alpha), beta.detach().numpy(), {k: c[i].detach().numpy() for i, k in enumerate(keys)}, R, f1


def cmd_fit(a):
    od = Path(a.out)
    if (od / "full.npz").exists():
        sys.exit(f"refuse to overwrite {od}")
    od.mkdir(parents=True, exist_ok=True)
    logf = open(od / "fit.log", "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    e20 = json.load(open(Path(a.e20) / "e20.json"))
    L = float(e20["L_chosen"])
    Mk = load_map(Path(a.e20) / f"M_L{L:g}.npz")
    assert Mk[0] == "bias", "prereg v2 assumes the bias map"
    bM = Mk[2]
    pr = np.load(Path(a.e20) / f"prior_L{L:g}.npz")
    refs = load_refs()
    step = int(round(L / 0.5))
    Z, S, V, W, LG = [], [], [], [], []
    for sid in sids(a.which, "train"):
        d = load_stream(sid, L)
        K = len(d["end"])
        vt = load_vt(sid, L, K)
        spk = d["meta"]["speakers"]
        for k in range(0, K, step):
            sl = int(d["slot"][k])
            if sl < 0 or sl >= len(d["slot2spk"]) or d["slot2spk"][sl] < 0:
                continue
            j = int(d["slot2spk"][sl])
            Z.append(d["z_mix"][k]); S.append(refs[spk[j]][0] - bM); V.append(vt[k]); W.append(d["pi"][k])
            LG.append(d["meta"]["lang"])
    Z, S, V, W = map(np.asarray, (Z, S, V, W))
    vm, vs = V.mean(0), V.std(0) + 1e-6
    Vn = (V - vm) / vs
    log(f"[fit] {a.which} train windows {len(Z)} (non-overlapping, slot matched to a true speaker); L={L:g}")
    res = {}
    for per_lang in (False, True):
        res[per_lang] = fit_one(Z, S, Vn, W, LG, per_lang, False, log)
    cen = res[True][2]
    dist = float(np.linalg.norm(cen["en"] - cen["zh"]))
    sp = e20["bias"]["train"]["spread"]
    thr = 0.2 * float(np.mean([sp["en"], sp["zh"]]))
    per_lang = dist >= thr
    log(f"[fit] |c_en - c_zh| = {dist:.3f}; threshold 0.2 * mean spread = {thr:.3f} -> per_lang = {per_lang}")
    al, be, cc, R, f = res[per_lang]
    ShrinkParams(al, be, cc, R, vm, vs, pr["mu0"], pr["Sig0"], bM, L, per_lang).save(od / "full.npz")
    al, be, cc, R, f = fit_one(Z, S, Vn, W, LG, per_lang, True, log)
    ShrinkParams(al, be, cc, R, vm, vs, pr["mu0"], pr["Sig0"], bM, L, per_lang).save(od / "const.npz")
    json.dump({"n_windows": len(Z), "c_dist": dist, "c_threshold": thr, "per_lang": bool(per_lang)},
              open(od / "fit.json", "x"), indent=1)


# ------------------------------------------------------------------------------------------------ eval
METHODS = ["first8", "runmean", "median", "prior", "const", "full", "ref_vhat", "ref_clean"]


def run_stream(sid, L, pf, pc, refs, P):
    from experiments.mist_w3po_v1.voicecond.idea2_eval import first8_prompt, solo_runs
    import soundfile as sf
    d = load_stream(sid, L)
    meta = d["meta"]
    lang = meta["lang"]
    K = len(d["end"])
    vt = load_vt(sid, L, K)
    avail = d["end"] + DIAR_DELAY
    post = torch.load(I2 / "pre/diar" / f"{sid}.post.pt", weights_only=False)["post80"].float().numpy()
    mix = sf.read(str(I2 / "streams" / sid / "mix.wav"), dtype="float32")[0]
    J = len(meta["speakers"])
    unit_slot = [slot_for_speaker(d, j) for j in range(J)]
    slots = sorted(set(unit_slot))
    # cumulative sufficient statistics per mode and slot (identical to ShrinkStream.step / .mean, vectorised)
    modes = {"full": (pf, "full"), "const": (pc, "full"), "prior": (pf, "prior")}
    cum = {}
    for m, (pp, mode) in modes.items():
        lam = np.ones(K) if mode == "prior" else pp.lam(vt)
        y = d["z_mix"] - (0.0 if mode == "prior" else (1 - lam)[:, None] * pp.cvec(lang)[None])
        Rfi = np.linalg.inv(pp.R * (pp.L / 0.5))
        S0i = np.linalg.inv(pp.Sig0)
        cum[m] = {"Rfi": Rfi, "S0i": S0i, "e0": S0i @ pp.mu0}
        for s in slots:
            w = np.where((d["slot"] == s) & (d["pi"] > 0), d["pi"], 0.0)
            cum[m][s] = (np.cumsum(w * lam * lam), np.cumsum((w * lam)[:, None] * y, 0))

    def post_mean(m, s, kk):
        if kk < 0 or cum[m][s][0][kk] <= 0:
            return pf.mu0
        C = cum[m]
        return np.linalg.solve(C["S0i"] + C[s][0][kk] * C["Rfi"], C["e0"] + C["Rfi"] @ C[s][1][kk])
    lam_hist = pf.lam(vt)
    bM = pf.bM
    out = []
    for j in range(J):
        s = unit_slot[j]
        spk = meta["speakers"][j]
        u_star, g_star = refs[spk]
        tau, tot = speech_clock(meta, j)
        runs = solo_runs(post, s)
        mine = np.nonzero(d["slot"] == s)[0]
        oracle = np.nonzero(d["single"] == j)[0]
        rec = {"sid": sid, "spk": spk, "lang": lang, "tot": tot, "slot": s,
               "err": {m: np.full(len(T_EVAL), np.nan) for m in METHODS},
               "tok": {m: np.full(len(T_EVAL), np.nan) for m in METHODS},
               "lam_mean": float(np.average(lam_hist[mine], weights=d["pi"][mine])) if len(mine) else np.nan}
        est = {m: [] for m in METHODS}
        for ti, t in enumerate(T_EVAL):
            tt = tau(t)
            if tt is None:
                continue
            kk = int(np.searchsorted(avail, tt, side="right")) - 1
            sel = mine[mine <= kk]
            w = d["pi"][sel]
            e = {"runmean": (w[:, None] * d["z_mix"][sel]).sum(0) / w.sum() if len(sel) else pf.mu0,
                 "median": np.median(d["z_mix"][sel], 0) if len(sel) else pf.mu0,
                 "ref_vhat": (w[:, None] * d["z_vhat"][sel]).sum(0) / w.sum() if len(sel) else pf.mu0}
            oc = oracle[d["end"][oracle] <= tt + 1e-9]
            e["ref_clean"] = d["z_speech"][oc].mean(0) if len(oc) else pf.mu0
            for m in modes:
                e[m] = post_mean(m, s, kk)
            for m in e:
                est[m].append((ti, e[m] + bM))
            pr, _ = first8_prompt(mix, runs, tt - DIAR_DELAY)
            est["first8"].append((ti, P.phi_wave(pr).reshape(-1)))
        for m in METHODS:
            if not est[m]:
                continue
            idx = [ti for ti, _ in est[m]]
            U = np.stack([u for _, u in est[m]])
            rec["err"][m][idx] = np.linalg.norm(U - u_star[None], axis=1)
            g = P.tokens(U.reshape(-1, 32, 6).astype(np.float32))
            rec["tok"][m][idx] = (g == g_star[None]).mean(1)
        out.append(rec)
    return out


def floor_fit(units, m, ts=(8, 10, 15, 20, 30, 40, 50, 60)):
    ti = [T_EVAL.index(t) for t in ts]
    y = np.nanmean(np.array([[u["err"][m][i] ** 2 for i in ti] for u in units]), 0)
    a, b = np.linalg.lstsq(np.c_[np.ones(len(ts)), 1.0 / np.array(ts)], y, rcond=None)[0]
    return float(a), float(b)


def summarize(units, n_boot=2000):
    by = {}
    for u in units:
        by.setdefault(u["sid"], []).append(u)
    i60 = T_EVAL.index(60)
    u60 = [c for c in ([u for u in c if u["tot"] >= 60] for c in by.values()) if c]
    flat = lambda cl: [u for c in cl for u in c]
    S = {"n_units": len(units), "n_streams": len(by), "n_units_60": sum(len(c) for c in u60), "median_err": {},
         "rms_err": {}, "tok": {}}
    for m in METHODS:
        S["median_err"][m] = {str(t): float(np.nanmedian([u["err"][m][i] for u in units])) for i, t in enumerate(T_EVAL)}
        S["rms_err"][m] = {str(t): float(np.sqrt(np.nanmean([u["err"][m][i] ** 2 for u in units]))) for i, t in enumerate(T_EVAL)}
        S["tok"][m] = {str(t): float(np.nanmean([u["tok"][m][i] for u in units])) for i, t in enumerate(T_EVAL)}
    med = lambda m: (lambda cl: float(np.nanmedian([u["err"][m][i60] for u in flat(cl)])))
    dmed = lambda m1, m2: (lambda cl: med(m1)(cl) - med(m2)(cl))
    pmed = lambda m1, m2: (lambda cl: float(np.nanmedian([u["err"][m1][i60] - u["err"][m2][i60] for u in flat(cl)])))
    S["median60"] = {m: cluster_boot(u60, med(m), n_boot) for m in METHODS}
    S["dmedian60"] = {f"full-{m}": cluster_boot(u60, dmed("full", m), n_boot) for m in METHODS if m != "full"}
    S["paired_median60"] = {f"full-{m}": cluster_boot(u60, pmed("full", m), n_boot) for m in ("runmean", "first8", "ref_vhat", "const")}
    S["first_sentence_median"] = {m: [S["median_err"][m]["1"], S["median_err"][m]["2"]] for m in METHODS}
    S["floor"] = {m: floor_fit(flat(u60), m) for m in METHODS}
    S["lam_mean_units"] = float(np.nanmean([u["lam_mean"] for u in units]))
    S["G2.2"] = {"vs_runmean": bool(S["dmedian60"]["full-runmean"][2] < 0),
                 "vs_first8": bool(S["dmedian60"]["full-first8"][2] < 0)}
    S["G2.2"]["pass"] = bool(S["G2.2"]["vs_runmean"] and S["G2.2"]["vs_first8"])
    for lang in LANGS:
        ul = [u for u in units if u["lang"] == lang]
        if ul:
            S[f"median_err_{lang}"] = {m: {str(t): float(np.nanmedian([u["err"][m][T_EVAL.index(t)] for u in ul]))
                                           for t in (2, 8, 20, 60)} for m in METHODS}
    return S


def cmd_eval(a):
    od = Path(a.out)
    if (od / "summary.json").exists():
        sys.exit(f"refuse to overwrite {od}")
    od.mkdir(parents=True, exist_ok=True)
    e20 = json.load(open(Path(a.e20) / "e20.json"))
    L = float(e20["L_chosen"])
    pf, pc = ShrinkParams.load(Path(a.fit) / "full.npz"), ShrinkParams.load(Path(a.fit) / "const.npz")
    refs = load_refs()
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    P = BiCodecPhi(a.device)
    units, t0 = [], time.time()
    ss = sids(a.which, a.split)
    for i, sid in enumerate(ss, 1):
        units += run_stream(sid, L, pf, pc, refs, P)
        print(f"[eval] {i}/{len(ss)} {sid} units {len(units)} {time.time() - t0:.0f}s", flush=True)
    np.savez(od / "units.npz", units=np.array(units, dtype=object))
    S = summarize(units)
    S.update({"which": a.which, "split": a.split, "fit": str(a.fit), "e20": str(a.e20), "T_EVAL": T_EVAL})
    (od / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({k: S[k] for k in ("n_units", "n_units_60", "median60", "dmedian60", "G2.2")}, indent=1))
    for m in METHODS:
        print(f"{m:10s} med", " ".join(f"{S['median_err'][m][str(t)]:.2f}" for t in T_EVAL), "| tok60", f"{S['tok'][m]['60']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "eval"])
    ap.add_argument("--which", choices=["pilot", "all"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--e20", required=True)
    ap.add_argument("--fit", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    cmd_fit(a) if a.cmd == "fit" else cmd_eval(a)
