"""Idea 1 M2 evaluation on the ID-1 TEST split of the open diagnostic set (reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md
v3.4 item 1). Arms (u0* = phi_0(s), the frozen encoder on the clean speech):
  raw     phi_0(y)                      V       phi_0(vhat)                 Rt      2 phi_0(y) - phi_0(y + (sqrt2-1) bh)
  ctrl    phi_C(y)  (denoising encoder) zne     2 phi_Z(y) - phi_Z(y + (sqrt2-1) bh)  (ZNE-in-the-loop encoder)
  oracle  phi_0(s)
plus records (not arms): phi_Z(y) without extrapolation, and the clean-input no-harm checks |phi_C(s) - u0*|, |phi_Z(s) - u0*|.
Per pair: |u - u0*|_2, Hamming of Q(u) to g0* = Q(u0*); decoder-only resynthesis with the CLEAN speech's semantic tokens and
each arm's 32 global tokens -> <out>/<arm>/wav/<id>.wav + <out>/<arm>/results.jsonl (measured by idea1_speech_metrics).
Writes <out>/codes.json (summary + per pair). Refuses an existing --out.
    venvs/vllm_omni_030/bin/python -m experiments.mist_w3po_v1.voicecond.idea1_m2_id1 --diag <diag_en> <diag_zh>
        --zne <encoder.pt> --ctrl <encoder.pt> --out <dir> [--split test] [--device cuda:0]
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
ARMS = ("raw", "V", "Rt", "ctrl", "zne", "oracle")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag", nargs="+", required=True)
    ap.add_argument("--zne", required=True)
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    sys.path.insert(0, str(REPO))
    import soundfile as sf
    import torch
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    P0 = BiCodecPhi(a.device)
    PZ = load_encoder(BiCodecPhi(a.device), a.zne)
    PC = load_encoder(BiCodecPhi(a.device), a.ctrl)
    for arm in ARMS:
        (out / arm / "wav").mkdir(parents=True)
    fh = {arm: open(out / arm / "results.jsonl", "x", encoding="utf-8") for arm in ARMS}
    per = []
    for d in a.diag:
        D = Path(d)
        rows = [r for r in map(json.loads, open(D / "manifest.jsonl", encoding="utf-8")) if r["split"] == a.split]
        for r in rows:
            w = {x: sf.read(str(D / "wav" / f"{r['id']}_{x}.wav"), dtype="float64")[0] for x in ("s", "y", "vhat")}
            s, y = w["s"], w["y"]
            bh = y - w["vhat"]
            y2 = y + (SQ2 - 1) * bh
            u0, uy, uv, u2 = P0.phi_wave([s, y, w["vhat"], y2])
            zy, z2, zs = PZ.phi_wave([y, y2, s])
            cy, cs = PC.phi_wave([y, s])
            U = {"raw": uy, "V": uv, "Rt": 2 * uy - u2, "ctrl": cy, "zne": 2 * zy - z2, "oracle": u0}
            g0 = P0.tokens(u0)
            rec = {"id": r["id"], "lang": r["lang"], "cat": r["cat"], "bin": r["bin"],
                   "zne_noextrap_err": float(np.linalg.norm(zy - u0)), "ctrl_clean_err": float(np.linalg.norm(cs - u0)),
                   "zne_clean_err": float(np.linalg.norm(zs - u0))}
            _, sem = P0.tok.tokenize(str(D / "wav" / f"{r['id']}_s.wav"))
            for arm, u in U.items():
                ids = P0.tokens(u)
                rec[f"{arm}_err"] = float(np.linalg.norm(u - u0))
                rec[f"{arm}_ham"] = int((ids != g0).sum())
                g = torch.as_tensor(ids[None], dtype=torch.long, device=P0.device)
                wav = P0.tok.detokenize(g, sem.to(P0.device))
                p = out / arm / "wav" / f"{r['id']}.wav"
                sf.write(str(p), np.asarray(wav, dtype=np.float32), 16000)
                fh[arm].write(json.dumps({"id": r["id"], "speech_path": str(p.resolve())}) + "\n")
            per.append(rec)
    for f in fh.values():
        f.close()

    def summ(sel):
        o = {"n": len(sel)}
        for arm in ARMS:
            e = np.array([x[f"{arm}_err"] for x in sel])
            h = np.array([x[f"{arm}_ham"] for x in sel])
            o[arm] = {"err_median": float(np.median(e)), "err_mean": float(e.mean()), "ham_median": float(np.median(h)),
                      "exact": float((h == 0).mean())}
        for k in ("zne_noextrap_err", "ctrl_clean_err", "zne_clean_err"):
            o[k] = float(np.median([x[k] for x in sel]))
        rng = np.random.default_rng(0)
        for A, Bm in (("zne", "V"), ("ctrl", "V"), ("zne", "ctrl"), ("zne", "raw"), ("ctrl", "raw")):
            dd = np.array([x[f"{Bm}_err"] - x[f"{A}_err"] for x in sel])
            bs = np.median(dd[rng.integers(0, len(dd), (10000, len(dd)))], 1)
            o[f"{Bm}_minus_{A}_err"] = {"median": float(np.median(dd)), "ci": [float(np.percentile(bs, 2.5)),
                                                                                float(np.percentile(bs, 97.5))],
                                        "win_rate": float((dd > 0).mean())}
        return o

    res = {"split": a.split, "all": summ(per)}
    for key in ("lang", "cat", "bin"):
        for val in sorted({x[key] for x in per}):
            res[f"{key}={val}"] = summ([x for x in per if x[key] == val])
    (out / "codes.json").write_text(json.dumps({"summary": res, "per_pair": per}), encoding="utf-8")
    s = res["all"]
    print(json.dumps({"n": s["n"], **{arm: round(s[arm]["err_median"], 3) for arm in ARMS},
                      "zne_noextrap": round(s["zne_noextrap_err"], 3), "clean_ctrl": round(s["ctrl_clean_err"], 3),
                      "clean_zne": round(s["zne_clean_err"], 3)}))


if __name__ == "__main__":
    main()
