"""Idea 1 (ZNE) E1.1 decoder-only resynthesis on the open diagnostic dev pairs
(reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md §5). Semantic tokens of the CLEAN speech s are fixed; only the 32
global tokens change: raw = Q(phi(y)), V = Q(phi(vhat)), ZNE = Q(u_hat of the SELECTED family, estimate path),
oracle = Q(phi(s)) (all from the E1.0 encode npz). Writes <out>/<arm>/wav/<id>.wav plus <out>/<arm>/results.jsonl
({id, speech_path}; measured with idea1_speech_metrics measure) and <out>/tokens.json with per pair and arm:
Hamming to g*, per-dimension FSQ flips, ||u_hat - u*||_inf, the Prop. 5 margin min_r delta_r / h (delta_r = distance of
B(u*)_r to the nearest rounding boundary, h = FSQ half-width), the guarantee check and the per-dimension count of
|u_hat_r - u*_r| >= delta_r / h. Refuses an existing --out.

    python -m experiments.mist_w3po_v1.voicecond.idea1_e11 --npz <e10 npz> [...] --diag <dir> [...] --family R-t|Q-lam|M-anc
        --out <dir> [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--diag", nargs="+", required=True)
    ap.add_argument("--family", required=True, choices=["R-t", "Q-lam", "M-anc"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    sys.path.insert(0, str(REPO))
    import soundfile as sf
    import torch
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_e10 import fam_est, gi
    P = BiCodecPhi(a.device)
    Z = [np.load(p) for p in a.npz]
    ids = np.concatenate([z["ids"] for z in Z])
    us = np.concatenate([z["ustar"] for z in Z]).astype(np.float64)
    E = np.concatenate([z["est"] for z in Z])
    ub = np.concatenate([z["ub_est"] for z in Z]).astype(np.float64)
    wavdir = {}
    for d in a.diag:
        for r in map(json.loads, open(Path(d) / "manifest.jsonl", encoding="utf-8")):
            wavdir[r["id"]] = Path(d) / "wav"
    uz, _ = fam_est(E, ub, a.family)
    U = {"raw": E[:, gi(1.0)].astype(np.float64), "V": E[:, gi(0.0)].astype(np.float64), "ZNE": uz, "oracle": us}
    fsq = P.fsq
    h = float((fsq._levels.max().item() - 1) * (1 + 1e-3) / 2)   # FSQ.bound half-width (eps 1e-3), Lip(B) <= h
    Bstar = P.bounded(us.astype(np.float32)).astype(np.float64)                      # (N, 32, 6)
    delta = 0.5 - np.abs(Bstar - np.round(Bstar))
    g_star = P.tokens(us.astype(np.float32))
    q_star = np.round(Bstar)
    for arm in U:
        (out / arm / "wav").mkdir(parents=True)
    stats = {arm: [] for arm in U}
    res_fh = {arm: open(out / arm / "results.jsonl", "x", encoding="utf-8") for arm in U}
    toks = {arm: P.tokens(U[arm].astype(np.float32)) for arm in U}
    for i, pid in enumerate(ids):
        _, sem = P.tok.tokenize(str(wavdir[pid] / f"{pid}_s.wav"))
        for arm in U:
            g = torch.as_tensor(toks[arm][i][None], dtype=torch.long, device=P.device)
            wav = P.tok.detokenize(g, sem.to(P.device))
            p = out / arm / "wav" / f"{pid}.wav"
            sf.write(str(p), np.asarray(wav, dtype=np.float32), 16000)
            res_fh[arm].write(json.dumps({"id": str(pid), "speech_path": str(p.resolve())}) + "\n")
            Ba = P.bounded(U[arm][i].astype(np.float32)).astype(np.float64)
            err = np.abs(U[arm][i] - us[i])
            stats[arm].append({"id": str(pid), "ham": int((toks[arm][i] != g_star[i]).sum()),
                               "dim_flips": int((np.round(Ba) != q_star[i]).sum()),
                               "err_inf": float(err.max()), "margin": float(delta[i].min() / h),
                               "guarantee": bool(err.max() < delta[i].min() / h),
                               "dims_over_margin": int((err >= delta[i] / h).sum()),
                               "q_dist": float(np.linalg.norm((np.round(Ba) - q_star[i]) / 2))})
    for f in res_fh.values():
        f.close()
    summ = {}
    for arm, L in stats.items():
        ham = np.array([s["ham"] for s in L])
        g = np.array([s["guarantee"] for s in L])
        summ[arm] = {"ham_median": float(np.median(ham)), "ham_mean": float(ham.mean()), "exact": float((ham == 0).mean()),
                     "guarantee_share": float(g.mean()),
                     "guarantee_implies_exact": bool(all(s["ham"] == 0 for s in L if s["guarantee"])),
                     "dims_over_margin_mean": float(np.mean([s["dims_over_margin"] for s in L])),
                     "dim_flips_mean": float(np.mean([s["dim_flips"] for s in L])),
                     "q_dist_median": float(np.median([s["q_dist"] for s in L]))}
    (out / "tokens.json").write_text(json.dumps({"family": a.family, "h": h, "summary": summ, "per_pair": stats}),
                                     encoding="utf-8")
    print(json.dumps({"out": str(out), "family": a.family, "h": h, "summary": summ}))


if __name__ == "__main__":
    main()
