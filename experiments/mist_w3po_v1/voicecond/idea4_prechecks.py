"""Idea 4 pre-checks (CPU only, analysis only): reports/mist_w3po_v1/idea4_prechecks.zh-CN.md §0 (preregistered).

  1. low rank of the background nuisance d = u(s+b) - u(s) (uncentered SVD energy, top-8 >= 80 %), residual after
     shrinkage, principal angles vs the between-speaker subspace of clean codes;
  2. mixture structure: GMM (full / diag, K = 1..10) on PCA-16 of d, BIC K, ARI / NMI vs background category;
  3. gating upper bound on Idea 2 dev streams: frozen shrink posterior (sha1 25aaf6cd) on all windows vs an oracle
     top-50 % gate (true window SNR; also code-error oracle and a realisable lambda gate) vs the running mean.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 <uniss-eval python> -I experiments/mist_w3po_v1/voicecond/idea4_prechecks.py \
        --out eval_outputs/mist_w3po_v1/idea4_prechecks

No new encodings: D1 = Idea 1 diag dev codes (e10/{en,zh}_dev_u.npz), D2/D3 = Idea 2 dev window codes. Dev only.
Refuses to overwrite <out>/summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_common import (DIAR_DELAY, I2, load_refs, load_stream, sids,  # noqa: E402
                                                             slot_for_speaker, speech_clock)
from experiments.mist_w3po_v1.voicecond.idea2_shrink import ShrinkParams, load_vt  # noqa: E402

E10 = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea1/e10"
DIAG = REPO / "data/open_voicecond_v1/idea1"
FIT = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea2/shrink_fit_all/full.npz"
FIT_SHA1 = "25aaf6cd7f6a69b1e7972b947459d1217744e037"
L, SR, SEED = 2.0, 16000, 20261119
KMAX = 16


# ------------------------------------------------------------------------------------------------ helpers
def energy(X, center=False):
    X = X - X.mean(0) if center else X
    s = np.linalg.svd(X, compute_uv=False) ** 2
    c = np.cumsum(s) / s.sum()
    return c


def ek_report(X, center=False):
    c = energy(X, center)
    return {"E": {str(k): float(c[k - 1]) for k in range(1, min(KMAX, len(c)) + 1)},
            "k80": int(np.searchsorted(c, 0.80) + 1), "k90": int(np.searchsorted(c, 0.90) + 1)}


def null_e8(X, seed=SEED):
    rng = np.random.default_rng(seed)
    Xp = np.stack([rng.permutation(X[:, j]) for j in range(X.shape[1])], 1)
    return float(energy(Xp)[7])


def top_right(X, k):
    return np.linalg.svd(X, full_matrices=False)[2][:k].T          # [D, k] orthonormal


def subspace_report(Dm, Sbasis):
    """Dm: nuisance samples [n, 192] (uncentered); Sbasis [192, 16] speaker PCs"""
    N8 = top_right(Dm, 8)
    S8, S16 = Sbasis[:, :8], Sbasis[:, :16]
    cos = np.cos(subspace_angles(N8, S8))[::-1]
    tot = (Dm ** 2).sum()
    return {"cos_principal_angles_N8_S8": [float(x) for x in np.sort(cos)[::-1]],
            "overlap_N8_S8": float(np.linalg.norm(S8.T @ N8) ** 2 / 8), "overlap_random": 8 / Dm.shape[1],
            "energy_in_S8": float(((Dm @ S8) ** 2).sum() / tot), "energy_in_S16": float(((Dm @ S16) ** 2).sum() / tot),
            "energy_in_S8_random": 8 / Dm.shape[1], "energy_in_S16_random": 16 / Dm.shape[1]}


def eta2(y, lab):
    y = np.asarray(y, float)
    tot = ((y - y.mean()) ** 2).sum()
    return float(sum((lab == g).sum() * (y[lab == g].mean() - y.mean()) ** 2 for g in np.unique(lab)) / tot)


def gmm_report(X, labels: dict, snr, ncomp=16):
    pca = PCA(ncomp, random_state=0).fit(X)
    Y = pca.transform(X)
    out = {"pca_explained": float(pca.explained_variance_ratio_.sum()), "n": int(len(X))}
    for cov in ("full", "diag"):
        bics, models = [], []
        for K in range(1, 11):
            g = GaussianMixture(K, covariance_type=cov, n_init=3, max_iter=500, reg_covar=1e-6, random_state=0).fit(Y)
            bics.append(float(g.bic(Y))); models.append(g)
        kb = int(np.argmin(bics)) + 1
        cl = models[kb - 1].predict(Y)
        r = {"bic": bics, "K_bic": kb, "cluster_sizes": np.bincount(cl, minlength=kb).tolist(),
             "eta2_snr": eta2(snr, cl) if kb > 1 else 0.0}
        for name, lab in labels.items():
            r[f"ARI_{name}"] = float(adjusted_rand_score(lab, cl))
            r[f"NMI_{name}"] = float(normalized_mutual_info_score(lab, cl))
        # forced K = number of categories (information only)
        kc = len(np.unique(labels["cat"]))
        clc = models[kc - 1].predict(Y)
        r[f"ARI_cat_forcedK{kc}"] = float(adjusted_rand_score(labels["cat"], clc))
        r[f"NMI_cat_forcedK{kc}"] = float(normalized_mutual_info_score(labels["cat"], clc))
        out[cov] = r
    rho = spearmanr(np.linalg.norm(X, axis=1), snr)
    out["spearman_normd_snr"] = float(rho.correlation)
    return out


# ------------------------------------------------------------------------------------------------ data
def load_d1():
    us, uy, cat, binv, bvr, lang, spk = [], [], [], [], [], [], []
    for lg in ("en", "zh"):
        z = np.load(E10 / f"{lg}_dev_u.npz")
        assert abs(z["grid"][4] - 1.0) < 1e-9
        man = {r["id"]: r for r in map(json.loads, open(DIAG / f"diag_{lg}/manifest.jsonl"))}
        for i, sid in enumerate(z["ids"]):
            r = man[str(sid)]
            assert r["split"] == "dev"
            us.append(z["ustar"][i].reshape(-1)); uy.append(z["true"][i, 4].reshape(-1))
            cat.append(r["cat"]); binv.append(r["bin"]); bvr.append(r["bvr_true"]); lang.append(lg); spk.append(r["speaker"])
    return {"us": np.asarray(us, np.float64), "uy": np.asarray(uy, np.float64), "cat": np.array(cat),
            "bin": np.array(binv), "snr": -np.asarray(bvr, np.float64), "lang": np.array(lang), "spk": np.array(spk)}


def shrink_s1(us, uy, g):
    """constant lambda + per-group intercept (1 - lambda) c_g; closed-form LS"""
    num = den = 0.0
    mu = {}
    for k in np.unique(g):
        m = g == k
        a, b = us[m] - us[m].mean(0), uy[m] - uy[m].mean(0)
        num += (a * b).sum(); den += (a * a).sum()
        mu[k] = (us[m].mean(0), uy[m].mean(0))
    lam = num / den
    r = uy - lam * us
    for k in np.unique(g):
        m = g == k
        r[m] -= mu[k][1] - lam * mu[k][0]
    return float(lam), r


def shrink_s2(us, uy, g, iters=20):
    """per-pair lambda_i + shared per-group c (alternating LS); start c from S1"""
    lam1, _ = shrink_s1(us, uy, g)
    c = {k: ((uy[g == k] - lam1 * us[g == k]).mean(0)) / (1 - lam1) for k in np.unique(g)}
    for _ in range(iters):
        C = np.stack([c[k] for k in g])
        a, b = us - C, uy - C
        lam = (a * b).sum(1) / (a * a).sum(1)
        for k in np.unique(g):
            m = g == k
            w = 1 - lam[m]
            c[k] = (w[:, None] * (uy[m] - lam[m, None] * us[m])).sum(0) / (w * w).sum()
    C = np.stack([c[k] for k in g])
    r = uy - lam[:, None] * us - (1 - lam[:, None]) * C
    return lam, r


def speaker_basis():
    tr = set()
    for sid in sids("all", "train"):
        tr.update(json.load(open(I2 / "streams" / sid / "meta.json"))["speakers"])
    refs = load_refs()
    U = np.stack([refs[s][0] for s in sorted(tr) if s in refs])
    pca = PCA(16, random_state=0).fit(U)
    return pca.components_.T, len(U), float(pca.explained_variance_ratio_[:8].sum())


def window_snr(sid, end):
    sp = sf.read(str(I2 / "streams" / sid / "speech.wav"), dtype="float32")[0].astype(np.float64)
    bg = sf.read(str(I2 / "streams" / sid / "bg.wav"), dtype="float32")[0].astype(np.float64)
    n = min(len(sp), len(bg))
    cs, cb = np.r_[0, np.cumsum(sp[:n] ** 2)], np.r_[0, np.cumsum(bg[:n] ** 2)]
    e = np.clip(np.round(end * SR).astype(int), 0, n)
    s = np.clip(e - int(L * SR), 0, n)
    return 10 * np.log10((cs[e] - cs[s] + 1e-10) / (cb[e] - cb[s] + 1e-10))


def window_layers(meta, end):
    bgm = meta.get("background", {})
    lab = []
    for e in end:
        s = e - L
        act = [any(ev["t"] < e and ev["t"] + ev["dur"] > s for ev in bgm.get(k, [])) for k in ("crowd", "sfx")]
        lab.append({(False, False): "bed", (True, False): "bed+crowd", (False, True): "bed+sfx",
                    (True, True): "bed+crowd+sfx"}[tuple(act)])
    return np.array(lab)


def load_dev_streams(pf):
    out = []
    for sid in sids("all", "dev"):
        d = load_stream(sid, L)
        K = len(d["end"])
        d["vt"] = load_vt(sid, L, K)
        d["lam"] = pf.lam(d["vt"])
        d["snr"] = window_snr(sid, d["end"])
        d["layer"] = window_layers(d["meta"], d["end"])
        d["sid"] = sid
        out.append(d)
    return out


def d2_arrays(streams, pf):
    zm, zs, lam, C, lab, snr, lang = [], [], [], [], [], [], []
    for d in streams:
        k = np.nonzero((d["single"] >= 0) & (np.arange(len(d["end"])) % 4 == 0))[0]
        zm.append(d["z_mix"][k]); zs.append(d["z_speech"][k]); lam.append(d["lam"][k])
        C.append(np.repeat(pf.cvec(d["meta"]["lang"])[None], len(k), 0))
        lab.append(d["layer"][k]); snr.append(d["snr"][k]); lang += [d["meta"]["lang"]] * len(k)
    zm, zs, lam, C = map(np.concatenate, (zm, zs, lam, C))
    return {"d": zm - zs, "r": zm - lam[:, None] * zs - (1 - lam[:, None]) * C, "zs": zs,
            "cat": np.concatenate(lab), "snr": np.concatenate(snr), "lang": np.array(lang), "lam": lam}


# ------------------------------------------------------------------------------------------------ check 3
class Post:
    def __init__(self, pf, lang):
        self.c = pf.cvec(lang)
        self.Rfi = np.linalg.inv(pf.R * (pf.L / 0.5))
        self.S0i = np.linalg.inv(pf.Sig0)
        self.e0 = self.S0i @ pf.mu0
        self.mu0 = pf.mu0

    def mean(self, d, idx):
        idx = idx[d["pi"][idx] > 0]
        if len(idx) == 0:
            return self.mu0.copy()
        w, lam = d["pi"][idx], d["lam"][idx]
        y = d["z_mix"][idx] - (1 - lam)[:, None] * self.c[None]
        a0 = (w * lam * lam).sum()
        a1 = ((w * lam)[:, None] * y).sum(0)
        return np.linalg.solve(self.S0i + a0 * self.Rfi, self.e0 + self.Rfi @ a1)


def top_half(idx, score):
    """keep ceil(n/2) indices with the largest score"""
    if len(idx) == 0:
        return idx
    n = int(np.ceil(len(idx) / 2))
    return idx[np.argsort(-score, kind="stable")[:n]]


def check3_units(streams, pf, refs, horizons=(8, 20, 60)):
    units = []
    for d in streams:
        meta = d["meta"]
        P = Post(pf, meta["lang"])
        avail = d["end"] + DIAR_DELAY
        for j, spk in enumerate(meta["speakers"]):
            s = slot_for_speaker(d, j)
            u_star = refs[spk][0]
            target = u_star - pf.bM
            tau, tot = speech_clock(meta, j)
            mine = np.nonzero((d["slot"] == s) & (d["pi"] > 0))[0]
            werr = np.linalg.norm((d["z_mix"][mine] - (1 - d["lam"][mine])[:, None] * P.c[None]) / d["lam"][mine, None]
                                  - target[None], axis=1)
            werr_of = dict(zip(mine.tolist(), werr))
            rec = {"sid": d["sid"], "spk": spk, "lang": meta["lang"], "tot": tot, "err": {}, "n_win": {}}
            for h in list(horizons) + ["end"]:
                if h == "end":
                    kk = len(d["end"]) - 1
                else:
                    tt = tau(h)
                    if tt is None:
                        continue
                    kk = int(np.searchsorted(avail, tt, side="right")) - 1
                sel = mine[mine <= kk]
                if len(sel) == 0:
                    continue
                e = {}
                e["a_shrink"] = P.mean(d, sel)
                e["b_oracle_snr"] = P.mean(d, top_half(sel, d["snr"][sel]))
                e["b_oracle_codeerr"] = P.mean(d, top_half(sel, -np.array([werr_of[i] for i in sel])))
                e["b_lambda_gate"] = P.mean(d, top_half(sel, d["lam"][sel]))
                w = d["pi"][sel]
                e["c_runmean"] = (w[:, None] * d["z_mix"][sel]).sum(0) / w.sum()
                rec["err"][str(h)] = {m: float(np.linalg.norm(v + pf.bM - u_star)) for m, v in e.items()}
                rec["n_win"][str(h)] = int(len(sel))
            units.append(rec)
    return units


def boot_R(units, h, mb, ma="a_shrink", n=2000, stat="mean"):
    by = {}
    for u in units:
        if str(h) in u["err"] and (h != 60 or u["tot"] >= 60):
            by.setdefault(u["sid"], []).append((u["err"][str(h)][ma], u["err"][str(h)][mb]))
    cl = list(by.values())
    f = np.mean if stat == "mean" else np.median

    def fn(cs):
        x = np.array([p for c in cs for p in c])
        return 1 - f(x[:, 1]) / f(x[:, 0])
    rng = np.random.default_rng(SEED)
    est = fn(cl)
    bs = [fn([cl[i] for i in rng.integers(0, len(cl), len(cl))]) for _ in range(n)]
    x = np.array([p for c in cl for p in c])
    return {"R": float(est), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n_units": int(len(x)), "n_streams": len(cl), f"{stat}_err_a": float(f(x[:, 0])), f"{stat}_err_b": float(f(x[:, 1]))}


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    od = Path(a.out)
    if (od / "summary.json").exists():
        sys.exit(f"refuse to overwrite {od / 'summary.json'}")
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sha = hashlib.sha1(FIT.read_bytes()).hexdigest()
    assert sha == FIT_SHA1, sha
    pf = ShrinkParams.load(FIT)
    S = {"fit": str(FIT.relative_to(REPO)), "fit_sha1": sha, "prereg": "reports/mist_w3po_v1/idea4_prechecks.zh-CN.md §0"}

    # ---------------- D1
    D1 = load_d1()
    d = D1["uy"] - D1["us"]
    lam1, r1 = shrink_s1(D1["us"], D1["uy"], D1["lang"])
    lam2, r2 = shrink_s2(D1["us"], D1["uy"], D1["lang"])
    Sb, n_spk, s8var = speaker_basis()
    c1 = {"data": "D1 Idea 1 diag dev (1000 pairs, prompt space)", "n": int(len(d)),
          "mean_norm_d": float(np.linalg.norm(d, axis=1).mean()),
          "primary_uncentered": ek_report(d), "centered": ek_report(d, True), "null_perm_E8": null_e8(d),
          "per_lang": {lg: ek_report(d[D1["lang"] == lg]) for lg in ("en", "zh")},
          "per_token_6d_pooled": {str(k): float(v) for k, v in enumerate(energy(d.reshape(-1, 6)), 1)},
          "per_token_6d_pooled_centered": {str(k): float(v) for k, v in enumerate(energy(d.reshape(-1, 6), True), 1)},
          "S1_const_lambda": {"lambda": lam1, "resid_energy_frac": float((r1 ** 2).sum() / (d ** 2).sum()), **ek_report(r1)},
          "S2_perpair_lambda": {"lambda_mean": float(lam2.mean()), "lambda_p10_p90": [float(np.percentile(lam2, 10)), float(np.percentile(lam2, 90))],
                                "resid_energy_frac": float((r2 ** 2).sum() / (d ** 2).sum()), **ek_report(r2),
                                "null_perm_E8": null_e8(r2)},
          "speaker_subspace": {"n_train_speakers": n_spk, "S8_between_speaker_var_frac": s8var},
          "angles_raw": subspace_report(d, Sb), "angles_S2_resid": subspace_report(r2, Sb)}
    c1["E8"] = c1["primary_uncentered"]["E"]["8"]
    c1["pass"] = bool(c1["E8"] >= 0.80)
    print(f"[c1] E8 = {c1['E8']:.3f} pass={c1['pass']} ({time.time() - t0:.0f}s)", flush=True)

    c2 = {"data": "D1", "raw": gmm_report(d, {"cat": D1["cat"], "bin": D1["bin"], "lang": D1["lang"]}, D1["snr"])}
    print(f"[c2] raw done ({time.time() - t0:.0f}s)", flush=True)
    c2["S2_resid"] = gmm_report(r2, {"cat": D1["cat"], "bin": D1["bin"], "lang": D1["lang"]}, D1["snr"])
    c2["K_bic"] = c2["raw"]["full"]["K_bic"]
    c2["ARI_cat"] = c2["raw"]["full"]["ARI_cat"]
    c2["pass"] = bool(c2["K_bic"] >= 2 and c2["ARI_cat"] >= 0.3)
    print(f"[c2] K={c2['K_bic']} ARI={c2['ARI_cat']:.3f} pass={c2['pass']} ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- D2 / D3 (Idea 2 dev streams)
    refs = load_refs()
    streams = load_dev_streams(pf)
    print(f"[d2] {len(streams)} dev streams loaded ({time.time() - t0:.0f}s)", flush=True)
    D2 = d2_arrays(streams, pf)
    c1["D2_windows"] = {"data": "D2 Idea 2 dev oracle single-speaker windows, non-overlapping (window space)",
                        "n": int(len(D2["d"])), "raw_uncentered": ek_report(D2["d"]), "raw_centered": ek_report(D2["d"], True),
                        "null_perm_E8": null_e8(D2["d"]),
                        "frozen_shrink_resid": {"resid_energy_frac": float((D2["r"] ** 2).sum() / (D2["d"] ** 2).sum()),
                                                **ek_report(D2["r"])},
                        "layer_counts": {k: int((D2["cat"] == k).sum()) for k in np.unique(D2["cat"])}}
    c2["D2_windows_raw"] = gmm_report(D2["d"], {"cat": D2["cat"], "lang": D2["lang"]}, D2["snr"])
    c2["D2_windows_frozen_resid"] = gmm_report(D2["r"], {"cat": D2["cat"], "lang": D2["lang"]}, D2["snr"])
    print(f"[c2] D2 done ({time.time() - t0:.0f}s)", flush=True)

    units = check3_units(streams, pf, refs)
    c3 = {"data": "D3 Idea 2 dev streams (80), frozen fit", "n_units": len(units),
          "n_units_60": sum(u["tot"] >= 60 for u in units), "R": {}, "R_median": {}}
    for h in (60, 8, 20, "end"):
        c3["R"][str(h)] = {m: boot_R(units, h, m) for m in ("b_oracle_snr", "b_oracle_codeerr", "b_lambda_gate", "c_runmean")}
        c3["R_median"][str(h)] = {m: boot_R(units, h, m, stat="median") for m in ("b_oracle_snr", "b_oracle_codeerr", "b_lambda_gate", "c_runmean")}
    c3["R_primary"] = c3["R"]["60"]["b_oracle_snr"]["R"]
    c3["pass"] = bool(c3["R_primary"] >= 0.10)
    print(f"[c3] R(b_snr vs a) @60 = {c3['R_primary']:.3f} pass={c3['pass']} ({time.time() - t0:.0f}s)", flush=True)

    S.update({"check1": c1, "check2": c2, "check3": c3, "elapsed_s": time.time() - t0})
    with open(od / "units_check3.json", "x") as f:
        json.dump(units, f)
    with open(od / "summary.json", "x") as f:
        json.dump(S, f, indent=1)
    print("[done]", od / "summary.json", flush=True)


if __name__ == "__main__":
    main()
