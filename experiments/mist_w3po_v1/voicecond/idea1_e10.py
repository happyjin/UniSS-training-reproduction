"""Idea 1 (ZNE) E1.0 mechanism diagnostics, family selection and gate G1.0 on the open diagnostic set
(reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md §3, §4).

encode   (GPU, venvs/vllm_omni_030 python) per pair of a split: u* = phi(s); TRUE path phi(s + lam b) and ESTIMATE path
         phi(y + (lam-1) bh), bh = y - vhat, on lam in GRID = {0, .25, .5, sqrt.5, 1, sqrt2, 1.5, 2, 3} (true lam=0 is
         u*, estimate lam=0 is phi(vhat)); anchors phi(b), phi(bh); scale phi(.5 y), phi(2 y).
         -> <out>.npz (refuses an existing file)
    python -m experiments.mist_w3po_v1.voicecond.idea1_e10 encode --diag <diag dir> --split dev --out <npz> [--device cuda:0]
analyze  (CPU) the pre-registered family selection (median ||u_hat - u*||_2 on the estimate path, simplest family
         within 5 % of the best; order R-t < Q-lam < M-anc), G1.0 (>= 70 % pairs with ||2 c2||_inf < ||c1||_inf on the
         TRUE path in the selected family's variable; median ||u_hat_sel - u*|| < median ||phi(vhat) - u*||), Prop. 9
         (||c1|| / ||c2|| of the lam-quadratic on the true path, lam in [0, 2]), scale invariance, separator c / kappa,
         per-category / per-bin breakdowns, token Hamming to g*.
    python -m experiments.mist_w3po_v1.voicecond.idea1_e10 analyze --npz <npz> [<npz> ...] --diag <dir> [<dir> ...]
        --out <json> [--no-select]  (--no-select: report only, e.g. English before Chinese exists)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SQ2 = math.sqrt(2.0)
GRID = (0.0, 0.25, 0.5, math.sqrt(0.5), 1.0, SQ2, 1.5, 2.0, 3.0)
ORDER = ("R-t", "Q-lam", "M-anc")


def gi(lam: float) -> int:
    return min(range(len(GRID)), key=lambda i: abs(GRID[i] - lam))


# ----------------------------------------------------------------------------------------------------------- encode
def encode(a) -> None:
    import soundfile as sf
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    sys.path.insert(0, str(REPO))
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    P = BiCodecPhi(a.device)
    D = Path(a.diag)
    rows = [r for r in map(json.loads, open(D / "manifest.jsonl", encoding="utf-8")) if r["split"] == a.split]
    ids, U = [], {k: [] for k in ("ustar", "true", "est", "ub_true", "ub_est", "scale")}
    for r in rows:
        w = {k: sf.read(str(D / "wav" / f"{r['id']}_{k}.wav"), dtype="float64")[0] for k in ("s", "b", "y", "vhat")}
        s, b, y = w["s"], w["b"], w["y"]
        bh = y - w["vhat"]
        waves = [s] + [s + l * b for l in GRID[1:]] + [y + (l - 1) * bh for l in GRID] + [b, bh, 0.5 * y, 2.0 * y]
        u = P.phi_wave(waves)
        n = len(GRID)
        U["ustar"].append(u[0])
        U["true"].append(np.concatenate([u[:1], u[1:n]]))
        U["est"].append(u[n:2 * n])
        U["ub_true"].append(u[2 * n])
        U["ub_est"].append(u[2 * n + 1])
        U["scale"].append(u[2 * n + 2:2 * n + 4])
        ids.append(r["id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=np.array(ids), grid=np.array(GRID), **{k: np.stack(v).astype(np.float32) for k, v in U.items()})
    print(json.dumps({"out": str(out), "pairs": len(ids)}))


# ---------------------------------------------------------------------------------------------------------- analyze
def lagrange0(xs):
    return [float(np.prod([xj / (xj - xi) for j, xj in enumerate(xs) if j != i])) for i, xi in enumerate(xs)]


QW = lagrange0([1.0, SQ2, 2.0])


def fam_est(F: np.ndarray, ub: np.ndarray, fam: str) -> tuple[np.ndarray, np.ndarray | None]:
    """F: (N, len(GRID), 32, 6) path; ub: (N, 32, 6) anchor -> u_hat (N, 32, 6) [, rho]"""
    f1, f2, f4 = F[:, gi(1.0)].astype(np.float64), F[:, gi(SQ2)].astype(np.float64), F[:, gi(2.0)].astype(np.float64)
    if fam == "R-t":
        return 2 * f1 - f2, None
    if fam == "Q-lam":
        return QW[0] * f1 + QW[1] * f2 + QW[2] * f4, None
    A, B = f1 - ub, f2 - ub
    d = (A - 2 * B).reshape(len(A), -1)
    den = (d * d).sum(1)
    rho = np.where(den > 0, np.maximum(0.0, (d * (B - A).reshape(len(A), -1)).sum(1) / np.maximum(den, 1e-30)), 0.0)
    return ub + (1 + rho)[:, None, None] * A, rho


def l2(x, y):
    return np.linalg.norm((x - y).reshape(len(x), -1), axis=1)


def quad_fit(F: np.ndarray, xs: list, idx: list):
    """per-component least squares c0 + c1 x + c2 x^2 over nodes; F (N, G, 32, 6) -> c1, c2 (N, 192)"""
    X = np.stack([np.ones(len(xs)), np.array(xs), np.array(xs) ** 2], 1)
    Y = F[:, idx].reshape(F.shape[0], len(idx), -1).astype(np.float64)       # (N, k, 192)
    pinv = np.linalg.pinv(X)                                                  # (3, k)
    C = np.einsum("ck,nkd->ncd", pinv, Y)
    return C[:, 1], C[:, 2]


def boot_ci(d, fn=np.median, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = fn(d[rng.integers(0, len(d), size=(n, len(d)))], axis=1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def q(x):
    x = np.asarray(x, dtype=float)
    return {"median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
            "mean": float(x.mean()), "n": int(len(x))}


def analyze(a) -> None:
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    sys.path.insert(0, str(REPO))
    Z = [np.load(p) for p in a.npz]
    ids = np.concatenate([z["ids"] for z in Z])
    cat = {k: np.concatenate([z[k] for z in Z]) for k in ("ustar", "true", "est", "ub_true", "ub_est", "scale")}
    man = {}
    for d in a.diag:
        for r in map(json.loads, open(Path(d) / "manifest.jsonl", encoding="utf-8")):
            man[r["id"]] = r
    M = [man[i] for i in ids]
    us, T, E = cat["ustar"].astype(np.float64), cat["true"], cat["est"]
    res = {"pairs": int(len(ids)), "langs": sorted({m["lang"] for m in M})}
    # family errors on the estimate path (the method) and on the true path (bh = b, mechanism)
    errs, est = {}, {}
    for fam in ORDER:
        uh, rho = fam_est(E, cat["ub_est"].astype(np.float64), fam)
        est[fam] = uh
        errs[fam] = l2(uh, us)
        uh_t, _ = fam_est(T, cat["ub_true"].astype(np.float64), fam)
        res[f"err_true_path_{fam}"] = q(l2(uh_t, us))
        if rho is not None:
            res["mobius_rho_est"] = q(rho)
    raw = l2(E[:, gi(1.0)].astype(np.float64), us)
    vh = l2(E[:, gi(0.0)].astype(np.float64), us)
    res["err_raw"], res["err_vhat"] = q(raw), q(vh)
    res["err_family"] = {f: q(errs[f]) for f in ORDER}
    meds = {f: float(np.median(errs[f])) for f in ORDER}
    best = min(meds.values())
    sel = next(f for f in ORDER if meds[f] <= 1.05 * best)
    res["selection"] = {"medians": meds, "best": best, "selected": sel, "rule": "simplest within 5 % of the best median",
                        "official": not a.no_select}
    music = np.array([m["cat"] == "music" for m in M])
    if music.any():
        mm = {f: float(np.median(errs[f][music])) for f in ORDER}
        res["selection_music_only"] = {"medians": mm, "selected": next(f for f in ORDER if mm[f] <= 1.05 * min(mm.values()))}
    # G1.0 (1): curvature vs slope on the TRUE path in the selected family's variable
    if sel in ("R-t", "M-anc"):
        lams = [0.0, 0.25, 0.5, math.sqrt(0.5), 1.0, SQ2]
        xs = [l * l for l in lams]
    else:
        lams = [0.0, 0.25, 0.5, math.sqrt(0.5), 1.0, SQ2, 1.5, 2.0]
        xs = lams
    c1, c2 = quad_fit(T, xs, [gi(l) for l in lams])
    M2, slope = np.abs(2 * c2).max(1), np.abs(c1).max(1)
    M2_2, slope_2 = np.linalg.norm(2 * c2, axis=1), np.linalg.norm(c1, axis=1)
    pass1 = M2 < slope
    res["G1_0_curvature"] = {"variable": "t" if sel in ("R-t", "M-anc") else "lam", "share_M2_lt_slope_inf": float(pass1.mean()),
                             "share_M2_lt_slope_l2": float((M2_2 < slope_2).mean()), "ratio_inf": q(M2 / slope),
                             "ratio_l2": q(M2_2 / slope_2)}
    # the other variable too (mechanism)
    lt = [0.0, 0.25, 0.5, math.sqrt(0.5), 1.0, SQ2]
    c1t, c2t = quad_fit(T, [l * l for l in lt], [gi(l) for l in lt])
    ll = [0.0, 0.25, 0.5, math.sqrt(0.5), 1.0, SQ2, 1.5, 2.0]
    c1l, c2l = quad_fit(T, ll, [gi(l) for l in ll])
    res["curvature_t"] = {"share_inf": float((np.abs(2 * c2t).max(1) < np.abs(c1t).max(1)).mean()),
                          "ratio_inf": q(np.abs(2 * c2t).max(1) / np.abs(c1t).max(1))}
    res["curvature_lam"] = {"share_inf": float((np.abs(2 * c2l).max(1) < np.abs(c1l).max(1)).mean()),
                            "ratio_inf": q(np.abs(2 * c2l).max(1) / np.abs(c1l).max(1))}
    # Prop. 9: |c1| vs |c2| of the lam-quadratic (true path, lam in [0, 2])
    res["prop9_c1_over_c2_l2"] = q(np.linalg.norm(c1l, axis=1) / np.linalg.norm(c2l, axis=1))
    # G1.0 (2)
    d = vh - errs[sel]
    res["G1_0_vs_vhat"] = {"median_sel": meds[sel], "median_vhat": float(np.median(vh)), "paired_win_rate": float((d > 0).mean()),
                           "paired_median_diff_vhat_minus_sel": float(np.median(d)), "ci": boot_ci(d)}
    res["G1_0_vs_raw"] = {"median_raw": float(np.median(raw)), "paired_win_rate": float((raw - errs[sel] > 0).mean()),
                          "paired_median_diff_raw_minus_sel": float(np.median(raw - errs[sel])), "ci": boot_ci(raw - errs[sel])}
    g1 = res["G1_0_curvature"]["share_M2_lt_slope_inf"] >= 0.70
    g2 = meds[sel] < float(np.median(vh))
    res["G1_0"] = {"curvature_ok": bool(g1), "beats_vhat_ok": bool(g2), "pass": bool(g1 and g2)}
    # scale invariance
    raw_u = E[:, gi(1.0)].astype(np.float64)
    sc = cat["scale"].astype(np.float64)
    res["scale_invariance"] = {f"alpha_{al}": q(l2(sc[:, k], raw_u) / np.maximum(raw, 1e-9)) for k, al in enumerate((0.5, 2.0))}
    res["scale_abs"] = {f"alpha_{al}": q(l2(sc[:, k], raw_u)) for k, al in enumerate((0.5, 2.0))}
    # separator
    res["separator"] = {k: q([m[k] for m in M]) for k in ("c", "kappa", "e_rel")}
    # tokens: Hamming to g*
    try:
        import torch
        from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi  # noqa: F401
        from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer  # noqa: F401
        P = BiCodecPhi("cpu")
        g_star = P.tokens(us.astype(np.float32))
        ham = {"raw": (P.tokens(raw_u.astype(np.float32)) != g_star).sum(1),
               "vhat": (P.tokens(E[:, gi(0.0)]) != g_star).sum(1)}
        for f in ORDER:
            ham[f] = (P.tokens(est[f].astype(np.float32)) != g_star).sum(1)
        res["hamming"] = {k: {**q(v), "exact": float((v == 0).mean())} for k, v in ham.items()}
        del torch
    except Exception as e:  # noqa: BLE001
        res["hamming_error"] = repr(e)[:200]
    # breakdowns
    groups = {}
    for key in ("lang", "cat", "bin"):
        for val in sorted({m[key] for m in M}):
            mk = np.array([m[key] == val for m in M])
            groups[f"{key}={val}"] = {"n": int(mk.sum()), "raw": float(np.median(raw[mk])), "vhat": float(np.median(vh[mk])),
                                      **{f: float(np.median(errs[f][mk])) for f in ORDER},
                                      "curv_share": float(pass1[mk].mean())}
    res["breakdown_median_err"] = groups
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez(out.with_suffix(".perpair.npz"), ids=ids, raw=raw, vhat=vh, **{f"err_{f}": errs[f] for f in ORDER},
             M2=M2, slope=slope)
    print(json.dumps({"pairs": res["pairs"], "selection": res["selection"], "G1_0": res["G1_0"],
                      "curv_share": res["G1_0_curvature"]["share_M2_lt_slope_inf"],
                      "err_raw": res["err_raw"]["median"], "err_vhat": res["err_vhat"]["median"]}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--diag", required=True)
    e.add_argument("--split", default="dev", choices=["dev"])
    e.add_argument("--out", required=True)
    e.add_argument("--device", default="cuda:0")
    n = sub.add_parser("analyze")
    n.add_argument("--npz", nargs="+", required=True)
    n.add_argument("--diag", nargs="+", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--no-select", action="store_true")
    a = ap.parse_args()
    encode(a) if a.cmd == "encode" else analyze(a)


if __name__ == "__main__":
    main()
