"""Idea 2 E2.0 assumption checks and gate G2.0 (prereg §3).

    python -I experiments/mist_w3po_v1/voicecond/idea2_e20.py --which pilot --L 2 4      (decides L, M, r on the pilot)
    python -I experiments/mist_w3po_v1/voicecond/idea2_e20.py --which all --L 2 --fixed eval_outputs/.../e20_pilot  (re-check)

Oracle single-speaker windows only (meta: one speaker, covering >= 90 % of the window). Fit on train, report on dev.
Writes eval_outputs/mist_w3po_v1/voicecond_idea2/e20_<which>/ {e20.json, M_L<L>.npz, prior_L<L>.npz}; refuses to overwrite.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_common import (OUTD, apply_map, fit_maps, load_map, load_refs,  # noqa: E402
                                                             load_stream, save_map, sids, speech_clock)


def units(split, which, L, refs):
    """per (stream, speaker): oracle single windows' z_mix, z_speech, v, end; u*; speech clock"""
    out = []
    for sid in sids(which, split):
        d = load_stream(sid, L)
        for j, spk in enumerate(d["meta"]["speakers"]):
            idx = np.nonzero(d["single"] == j)[0]
            tau, tot = speech_clock(d["meta"], j)
            out.append({"sid": sid, "spk": spk, "lang": d["meta"]["lang"], "idx": idx, "zm": d["z_mix"][idx],
                        "zc": d["z_speech"][idx], "v": d["v"][idx], "end": d["end"][idx], "tau": tau, "tot": tot,
                        "u": refs[spk][0], "bvr": d["meta"]["bvr_db"]})
    return out


def runmean_at(u, t, key="zc"):
    tau = u["tau"](t)
    if tau is None:
        return None
    sel = u["end"] <= tau + 1e-9
    if not sel.any():
        return "none"
    return u[key][sel].mean(0)


def err_at(us, Mk, t, prior, key="zc", need=None):
    e = []
    for u in us:
        if need is not None and u["tot"] < need:
            continue
        m = runmean_at(u, t, key)
        if m is None:
            continue
        m = prior if isinstance(m, str) else m
        e.append(np.linalg.norm(apply_map(Mk, m) - u["u"]))
    return float(np.sqrt(np.mean(np.square(e)))) if e else float("nan"), len(e)


def spread(us):
    """median pairwise distance between clean window-space means of distinct speakers (same language)"""
    by = {}
    for u in us:
        if len(u["zc"]) >= 3:
            by.setdefault((u["lang"], u["spk"]), []).append(u["zc"].mean(0))
    out = {}
    for lang in ("en", "zh"):
        X = np.array([np.mean(v, 0) for (l, _), v in by.items() if l == lang])
        if len(X) < 2:
            continue
        dd = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
        out[lang] = float(np.median(dd[np.triu_indices(len(X), 1)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["pilot", "all"], required=True)
    ap.add_argument("--L", type=float, nargs="+", default=[2.0, 4.0])
    ap.add_argument("--fixed", default="", help="dir of a previous E2.0 whose L / M / r are kept fixed")
    a = ap.parse_args()
    od = OUTD / f"e20_{a.which}"
    if (od / "e20.json").exists():
        sys.exit(f"refuse to overwrite {od}/e20.json")
    od.mkdir(parents=True, exist_ok=True)
    refs = load_refs()
    res = {"which": a.which, "L": {}}
    data = {}
    for L in a.L:
        tr, dv = units("train", a.which, L, refs), units("dev", a.which, L, refs)
        data[L] = (tr, dv)
        # map M: fit on train clean window means -> u*
        X = np.array([u["zc"].mean(0) for u in tr if len(u["zc"]) >= 5])
        Y = np.array([u["u"] for u in tr if len(u["zc"]) >= 5])
        maps = fit_maps(X, Y)
        # speaker prior in window space (train clean speaker means)
        by = {}
        for u in tr:
            if len(u["zc"]) >= 5:
                by.setdefault(u["spk"], []).append(u["zc"].mean(0))
        S = np.array([np.mean(v, 0) for v in by.values()])
        mu0 = S.mean(0)
        Sig0 = np.cov(S.T) + 1e-4 * np.trace(np.cov(S.T)) / 192 * np.eye(192)
        # within-speaker clean window covariance (EM init for R)
        Wc = np.concatenate([u["zc"] - u["zc"].mean(0) for u in tr if len(u["zc"]) >= 5])
        Rc = Wc.T @ Wc / max(1, len(Wc) - len(X))
        np.savez(od / f"prior_L{L:g}.npz", mu0=mu0, Sig0=Sig0, R0=Rc, n_spk=len(S))
        sel = {}
        for name, Mk in maps.items():
            sel[name] = err_at(dv, Mk, 60, mu0, "zc", need=60)
        best = min(sel, key=lambda k: sel[k][0])
        if a.fixed:
            best = json.load(open(Path(a.fixed) / "e20.json"))["L"][f"{L:g}"]["M_chosen"]
        Mk = maps[best]
        save_map(od / f"M_L{L:g}.npz", Mk)
        e8 = err_at(dv, Mk, 8, mu0, "zc")
        res["L"][f"{L:g}"] = {"M_dev_err60": sel, "M_chosen": best, "e_clean_runmean_dev": {
            str(t): err_at(dv, Mk, t, mu0, "zc") for t in (2, 4, 8, 15, 30, 60)}, "e8": e8,
            "n_train_units": len(tr), "n_dev_units": len(dv), "n_prior_speakers": len(S),
            "within_spk_window_var_tr": float(np.trace(Rc)), "between_spk_var_tr": float(np.trace(np.cov(S.T)))}
        print(f"L={L:g}: M errors at 60 s {sel} -> {best}; e(8) = {e8}", flush=True)
    # window-length rule
    if a.fixed:
        Lc = json.load(open(Path(a.fixed) / "e20.json"))["L_chosen"]
    elif len(a.L) == 2:
        e2, e4 = res["L"]["2"]["e8"][0], res["L"]["4"]["e8"][0]
        Lc = 2.0 if e2 <= 1.1 * e4 else 4.0
    else:
        Lc = a.L[0]
    res["L_chosen"] = Lc
    tr, dv = data[Lc]
    Mk = load_map(od / f"M_L{Lc:g}.npz")
    # (i) persistent bias vs inter-speaker spread
    out_b = {}
    for name, us in (("train", tr), ("dev", dv)):
        b = [np.linalg.norm(u["zm"].mean(0) - u["zc"].mean(0)) for u in us if len(u["zc"]) >= 5]
        sp = spread(us)
        rat = []
        for u in us:
            if len(u["zc"]) >= 5 and u["lang"] in sp:
                rat.append(np.linalg.norm(u["zm"].mean(0) - u["zc"].mean(0)) / sp[u["lang"]])
        # bias in prompt space too: |M(mean noisy) - M(mean clean)|
        bp = [np.linalg.norm(apply_map(Mk, u["zm"].mean(0)) - apply_map(Mk, u["zc"].mean(0))) for u in us if len(u["zc"]) >= 5]
        out_b[name] = {"median_bias": float(np.median(b)), "spread": sp, "median_ratio": float(np.median(rat)),
                       "q25_q75_ratio": [float(np.percentile(rat, 25)), float(np.percentile(rat, 75))],
                       "median_bias_prompt_space": float(np.median(bp)), "n": len(rat)}
    res["bias"] = out_b
    # (ii) low rank: PCA of window shifts on train
    dT = np.concatenate([u["zm"] - u["zc"] for u in tr])
    vT = np.concatenate([u["v"] for u in tr])
    dD = np.concatenate([u["zm"] - u["zc"] for u in dv])
    vD = np.concatenate([u["v"] for u in dv])
    ev = np.linalg.eigvalsh(np.cov(dT.T))[::-1]
    cum = np.cumsum(ev) / ev.sum()
    r90 = int(np.searchsorted(cum, 0.9) + 1)
    evu = np.linalg.eigvalsh(dT.T @ dT / len(dT))[::-1]
    r90u = int(np.searchsorted(np.cumsum(evu) / evu.sum(), 0.9) + 1)
    if a.fixed:
        r_fixed = json.load(open(Path(a.fixed) / "e20.json"))["r"]
    else:
        r_fixed = min(r90, 96)
    res["r"] = r_fixed
    res["pca"] = {"r90_centered": r90, "r90_uncentered": r90u, "cum_var": [round(float(x), 4) for x in cum[:100]],
                  "n_windows_train": len(dT)}
    # (iii) R^2 of v -> shift
    Xt = np.c_[vT, np.ones(len(vT))]
    G = np.linalg.lstsq(Xt, dT, rcond=None)[0]
    pred = np.c_[vD, np.ones(len(vD))] @ G
    r2 = 1 - ((dD - pred) ** 2).sum() / ((dD - dD.mean(0)) ** 2).sum()
    G0 = np.linalg.lstsq(vT, dT, rcond=None)[0]
    r2_0 = 1 - ((dD - vD @ G0) ** 2).sum() / ((dD - dD.mean(0)) ** 2).sum()
    # stream-level (persistent part): per (stream, speaker) mean shift vs mean v
    sT = np.array([(u["zm"] - u["zc"]).mean(0) for u in tr if len(u["zc"]) >= 5])
    svT = np.array([u["v"].mean(0) for u in tr if len(u["zc"]) >= 5])
    sD = np.array([(u["zm"] - u["zc"]).mean(0) for u in dv if len(u["zc"]) >= 5])
    svD = np.array([u["v"].mean(0) for u in dv if len(u["zc"]) >= 5])
    lam = 1e-2 * np.trace(svT.T @ svT) / svT.shape[1]
    Gs = np.linalg.solve(np.c_[svT, np.ones(len(svT))].T @ np.c_[svT, np.ones(len(svT))] + lam * np.eye(65),
                         np.c_[svT, np.ones(len(svT))].T @ sT)
    r2s = 1 - ((sD - np.c_[svD, np.ones(len(svD))] @ Gs) ** 2).sum() / ((sD - sD.mean(0)) ** 2).sum()
    res["r2"] = {"window_with_intercept_dev": float(r2), "window_no_intercept_dev": float(r2_0),
                 "stream_level_dev_ridge": float(r2s), "n_dev_windows": len(dD), "n_dev_units": len(sD)}
    # (iv) Gaussianity of residuals on top-10 PCs (dev)
    resid = dD - pred
    w, V = np.linalg.eigh(np.cov(resid.T))
    P = (resid - resid.mean(0)) @ V[:, ::-1][:, :10]
    P = P / P.std(0)
    res["gauss"] = {"skew": [round(float((P[:, i] ** 3).mean()), 3) for i in range(10)],
                    "excess_kurtosis": [round(float((P[:, i] ** 4).mean() - 3), 3) for i in range(10)]}
    # gate
    ratio = out_b["dev"]["median_ratio"]
    res["G2.0"] = {"bias_ratio_dev": float(ratio), "bias_ok": bool(ratio >= 0.2), "r2_dev": float(r2), "r2_ok": bool(r2 >= 0.3),
                   "pass": bool(ratio >= 0.2 and r2 >= 0.3)}
    (od / "e20.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("L_chosen", "r", "bias", "r2", "G2.0")}, indent=1))
    print("pca r90 centered/uncentered", r90, r90u)


if __name__ == "__main__":
    main()
