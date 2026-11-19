"""Idea 2 E2.0 diagnostics part 2 (not part of the gate): in-sample R^2 of the pre-registered v -> delta regression on
train, and the speech dependence of the shift: delta = G z_clean + c (is G ~ -alpha I, i.e. a contraction of the speaker
code towards a common point?), and how much of the persistent (unit-mean) bias a speaker-independent affine correction
of the noisy mean removes. Writes <e20>/diag/diag2.json (refuses to overwrite)."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_common import load_stream, sids  # noqa: E402

e20d = Path(sys.argv[1]); out = e20d / "diag/diag2.json"
if out.exists():
    sys.exit("refuse to overwrite")
e20 = json.load(open(e20d / "e20.json")); L = float(e20["L_chosen"])
U = {}
for split in ("train", "dev"):
    rows = []
    for sid in sids(e20["which"], split):
        d = load_stream(sid, L)
        for j in range(len(d["meta"]["speakers"])):
            idx = np.nonzero(d["single"] == j)[0]
            if len(idx) >= 5:
                rows.append((d["z_mix"][idx], d["z_speech"][idx], d["v"][idx]))
    U[split] = rows
cat = lambda rs, i: np.concatenate([r[i] for r in rs])
res = {}
zmT, zcT, vT = cat(U["train"], 0), cat(U["train"], 1), cat(U["train"], 2)
dT = zmT - zcT
X = np.c_[vT, np.ones(len(vT))]
G = np.linalg.lstsq(X, dT, rcond=None)[0]
res["in_sample_r2_v_train"] = float(1 - ((dT - X @ G) ** 2).sum() / ((dT - dT.mean(0)) ** 2).sum())
Xc = np.c_[zcT, np.ones(len(zcT))]
Gc = np.linalg.lstsq(Xc, dT, rcond=None)[0][:192]
res["G_zclean_diag_mean"] = float(np.mean(np.diag(Gc)))
res["G_zclean_offdiag_rms"] = float(np.sqrt(np.mean((Gc - np.diag(np.diag(Gc))) ** 2)))
# scalar contraction: delta = -alpha (zc - c0) + b  <=> z_mix = (1-alpha) zc + const
zc0 = zcT - zcT.mean(0); d0 = dT - dT.mean(0)
alpha = float(-(zc0 * d0).sum() / (zc0 * zc0).sum())
res["contraction_alpha"] = alpha
res["r2_contraction_window_train"] = float(1 - ((d0 + alpha * zc0) ** 2).sum() / (d0 ** 2).sum())
# unit level: noisy mean m = (1-alpha) s + c  -> corrected s_hat = (m - c) / (1 - alpha); fit alpha, c on train units
mT = np.array([r[0].mean(0) for r in U["train"]]); sT = np.array([r[1].mean(0) for r in U["train"]])
mD = np.array([r[0].mean(0) for r in U["dev"]]); sD = np.array([r[1].mean(0) for r in U["dev"]])
a1 = float(((mT - mT.mean(0)) * (sT - sT.mean(0))).sum() / ((sT - sT.mean(0)) ** 2).sum())
c1 = mT.mean(0) - a1 * sT.mean(0)
res["unit_slope_noisy_on_clean"] = a1
res["unit_bias_dev_raw"] = float(np.median(np.linalg.norm(mD - sD, axis=1)))
res["unit_bias_dev_after_affine_inverse"] = float(np.median(np.linalg.norm((mD - c1) / a1 - sD, axis=1)))
res["unit_bias_dev_after_mean_offset"] = float(np.median(np.linalg.norm(mD - (mT - sT).mean(0) - sD, axis=1)))
out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
