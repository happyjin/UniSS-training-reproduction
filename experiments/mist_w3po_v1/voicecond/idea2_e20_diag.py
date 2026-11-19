"""Idea 2 E2.0 follow-up diagnostics (NOT part of the gate; run after G2.0 failed on R^2): why does v not predict the
window code shift delta = phi(mix window) - phi(clean window)?

    python -I experiments/mist_w3po_v1/voicecond/idea2_e20_diag.py --e20 eval_outputs/mist_w3po_v1/voicecond_idea2/e20_pilot

1. variance decomposition of delta: between (stream, speaker) means vs within;
2. R^2 (fit train, eval dev) of delta on: v (HDemucs, as pre-registered), v* (oracle: TRUE background / TRUE speech power,
   same formula), log-power scalars, and v / v* reduced to 8 PCs; window level, unit level, within-unit;
3. R^2 of delta on the clean code itself (speech-dependence of the shift) and on [v*, z_clean].
Writes <e20>/diag/diag.json (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_common import I2, load_stream, sids  # noqa: E402
from experiments.mist_w3po_v1.voicecond.idea2_feats import melpow  # noqa: E402


def oracle_v(sid, d, L):
    bg = sf.read(str(I2 / "streams" / sid / "bg.wav"), dtype="float32")[0]
    sp = sf.read(str(I2 / "streams" / sid / "speech.wav"), dtype="float32")[0]
    Pb, Ps = melpow(bg), melpow(sp)
    T = min(len(Pb), len(Ps))
    es = Ps[:T].mean(1)
    act = es > es.max() * 1e-4
    cs_e = np.concatenate([[0.0], np.cumsum(es * act)])
    cs_n = np.concatenate([[0], np.cumsum(act)])
    out = np.zeros((len(d["end"]), Pb.shape[1]))
    for k, e in enumerate(d["end"]):
        a, b = int((e - L) * 100), min(T, int(e * 100))
        pbar = cs_e[b] / cs_n[b] if cs_n[b] > 0 else es[a:b].mean() + 1e-10
        out[k] = (Pb[a:b].mean(0) / max(pbar, 1e-10)) ** 0.3
    return out


def r2(Xtr, Ytr, Xte, Yte, lam_rel=1e-3):
    Xtr1, Xte1 = np.c_[Xtr, np.ones(len(Xtr))], np.c_[Xte, np.ones(len(Xte))]
    lam = lam_rel * np.trace(Xtr1.T @ Xtr1) / Xtr1.shape[1]
    G = np.linalg.solve(Xtr1.T @ Xtr1 + lam * np.eye(Xtr1.shape[1]), Xtr1.T @ Ytr)
    return float(1 - ((Yte - Xte1 @ G) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e20", required=True)
    a = ap.parse_args()
    e20 = json.load(open(Path(a.e20) / "e20.json"))
    L = float(e20["L_chosen"])
    od = Path(a.e20) / "diag"
    if (od / "diag.json").exists():
        sys.exit("refuse to overwrite")
    od.mkdir(parents=True, exist_ok=True)
    D = {}
    for split in ("train", "dev"):
        rows = []
        for sid in sids(e20["which"], split):
            d = load_stream(sid, L)
            vo = oracle_v(sid, d, L)
            for j in range(len(d["meta"]["speakers"])):
                idx = np.nonzero(d["single"] == j)[0]
                if len(idx) < 5:
                    continue
                rows.append({"delta": d["z_mix"][idx] - d["z_speech"][idx], "v": d["v"][idx], "vo": vo[idx],
                             "zc": d["z_speech"][idx], "bvr": d["meta"]["bvr_db"]})
        D[split] = rows
    res = {}
    cat = lambda rs, k: np.concatenate([r[k] for r in rs])
    tr, dv = D["train"], D["dev"]
    dT, dD = cat(tr, "delta"), cat(dv, "delta")
    # 1. variance decomposition (dev and train)
    for name, rs in (("train", tr), ("dev", dv)):
        dl = cat(rs, "delta")
        tot = ((dl - dl.mean(0)) ** 2).sum()
        within = sum(((r["delta"] - r["delta"].mean(0)) ** 2).sum() for r in rs)
        res[f"between_unit_fraction_{name}"] = float(1 - within / tot)
        res[f"mean_shift_norm_{name}"] = float(np.linalg.norm(dl.mean(0)))
        res[f"rms_shift_{name}"] = float(np.sqrt((dl ** 2).sum(1).mean()))
    # 2. window-level R^2
    feats = {}
    for k in ("v", "vo"):
        XT, XD = cat(tr, k), cat(dv, k)
        feats[k] = (XT, XD)
        U, s, Vt = np.linalg.svd(XT - XT.mean(0), full_matrices=False)
        feats[k + "_pc8"] = ((XT - XT.mean(0)) @ Vt[:8].T, (XD - XT.mean(0)) @ Vt[:8].T)
        feats[k + "_logpow"] = (np.log(XT.mean(1, keepdims=True) + 1e-6), np.log(XD.mean(1, keepdims=True) + 1e-6))
    res["window_r2"] = {k: r2(*feats[k][:1], dT, feats[k][1], dD) for k in feats}
    res["window_r2"]["zclean"] = r2(cat(tr, "zc"), dT, cat(dv, "zc"), dD)
    res["window_r2"]["vo+zclean"] = r2(np.c_[cat(tr, "vo"), cat(tr, "zc")], dT, np.c_[cat(dv, "vo"), cat(dv, "zc")], dD)
    # within-unit (time-varying part)
    wc = lambda rs, k: np.concatenate([r[k] - r[k].mean(0) for r in rs])
    res["within_unit_r2"] = {k: r2(wc(tr, k), wc(tr, "delta"), wc(dv, k), wc(dv, "delta")) for k in ("v", "vo")}
    # unit level (persistent part), low-dimensional predictors
    um = lambda rs, k: np.array([r[k].mean(0) for r in rs])
    uT, uD = um(tr, "delta"), um(dv, "delta")
    res["unit_r2"] = {}
    for k in ("v", "vo"):
        XT, XD = um(tr, k), um(dv, k)
        U, s, Vt = np.linalg.svd(XT - XT.mean(0), full_matrices=False)
        res["unit_r2"][k + "_pc8"] = r2((XT - XT.mean(0)) @ Vt[:8].T, uT, (XD - XT.mean(0)) @ Vt[:8].T, uD)
        res["unit_r2"][k + "_logpow"] = r2(np.log(XT.mean(1, keepdims=True)), uT, np.log(XD.mean(1, keepdims=True)), uD)
    res["unit_r2"]["bvr_db"] = r2(np.array([[r["bvr"]] for r in tr]), uT, np.array([[r["bvr"]] for r in dv]), uD)
    res["corr_v_vs_oracle_v_window_dev"] = float(np.corrcoef(feats["v"][1].ravel(), feats["vo"][1].ravel())[0, 1])
    res["n"] = {"train_units": len(tr), "dev_units": len(dv), "train_windows": len(dT), "dev_windows": len(dD)}
    (od / "diag.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
