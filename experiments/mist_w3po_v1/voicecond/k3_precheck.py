"""Robust K3 precheck (reports/mist_w3po_v1/robust_k3_discrim_prereg.zh-CN.md section 4): dev codes of the frozen
encoder phi0, V1 (voicecond_idea1/m2/ctrl/encoder.pt) and K3 on the ID-1 diag dev pairs (diag_en / diag_zh, split dev;
clean s, noisy y, speaker label). Per language and encoder:
  spread_noisy / spread_clean : tr(Cov of per-speaker mean codes of phi(y) / phi(s)) / tr(same for u* = phi0(s))
  cos_noisy / cos_clean       : median cos(phi(y), u*) / cos(phi(s), u*) on the flat 192-d pre-bound code
  spkid_noisy                 : nearest clean-speaker-centroid accuracy of phi(y) (record only)
Stop rule: go = K3 spread_noisy > V1 spread_noisy in at least one language.
    vllm_omni_030 python -m experiments.mist_w3po_v1.voicecond.k3_precheck --k3 <encoder.pt> --out <json> [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
V1 = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea1/m2/ctrl/encoder.pt"
DIAG = REPO / "data/open_voicecond_v1/idea1"


def between_trace(U, spk):
    keys = sorted(set(spk))
    M = np.stack([U[[k == s for s in spk]].mean(0) for k in keys])
    return float(np.trace(np.cov(M.T)))


def cos(A, B):
    return np.sum(A * B, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N dev rows per language")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    sys.path.insert(0, str(REPO))
    import soundfile as sf
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    enc = {"phi0": BiCodecPhi(a.device), "v1": load_encoder(BiCodecPhi(a.device), str(V1)),
           "k3": load_encoder(BiCodecPhi(a.device), a.k3)}
    for P in enc.values():
        P.model.eval()
    res = {"k3": a.k3, "langs": {}}
    for lang in ("en", "zh"):
        d = DIAG / f"diag_{lang}"
        rows = [r for r in map(json.loads, open(d / "manifest.jsonl", encoding="utf-8")) if r["split"] == "dev"]
        rows = rows[: a.limit] if a.limit else rows
        spk = [r["speaker"] for r in rows]
        S = [sf.read(str(d / "wav" / f"{r['id']}_s.wav"), dtype="float64")[0] for r in rows]
        Y = [sf.read(str(d / "wav" / f"{r['id']}_y.wav"), dtype="float64")[0] for r in rows]
        codes = {}
        for k, P in enc.items():
            codes[k] = {c: np.concatenate([P.phi_wave(W[i:i + 50]).reshape(-1, 192) for i in range(0, len(W), 50)])
                        for c, W in (("y", Y), ("s", S))}
        u = codes["phi0"]["s"]
        tu = between_trace(u, spk)
        keys = sorted(set(spk))
        C = np.stack([u[[k == s for s in spk]].mean(0) for k in keys])
        lab = np.array([keys.index(s) for s in spk])
        st = {"n": len(rows), "n_speakers": len(keys), "trace_between_ustar": tu}
        for k in enc:
            Uy, Us = codes[k]["y"], codes[k]["s"]
            pred = np.argmin(((Uy[:, None, :] - C[None]) ** 2).sum(-1), 1)
            st[k] = {"spread_noisy": round(between_trace(Uy, spk) / tu, 4),
                     "spread_clean": round(between_trace(Us, spk) / tu, 4),
                     "cos_noisy": round(float(np.median(cos(Uy, u))), 4),
                     "cos_clean": round(float(np.median(cos(Us, u))), 4),
                     "spkid_noisy": round(float((pred == lab).mean()), 4)}
        res["langs"][lang] = st
        print(json.dumps({lang: st}), flush=True)
    res["go"] = any(res["langs"][l]["k3"]["spread_noisy"] > res["langs"][l]["v1"]["spread_noisy"] for l in ("en", "zh"))
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"go": res["go"]}), flush=True)


if __name__ == "__main__":
    main()
