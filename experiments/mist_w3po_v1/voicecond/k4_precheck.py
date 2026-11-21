"""Robust K4 precheck (reports/mist_w3po_v1/robust_k4_tokencls_prereg.zh-CN.md section 3): global-token agreement with the
clean prompt on the ID-1 diag dev pairs (diag_en / diag_zh, split dev; clean s, noisy y) for phi0 (record), V1
(voicecond_idea1/m2/ctrl/encoder.pt), K3 (k3/encoder/encoder.pt) and K4. Clean tokens t* = Q(phi0(s)). Per language/encoder:
  pos_noisy / pos_clean     : fraction of the 32 positions where Q(phi(y)) / Q(phi(s)) == t* exactly
  level_noisy / level_clean : fraction of the 32 x 6 FSQ level digits that agree
  dvec_cos_noisy            : median cos(detokenize(Q(phi(y))), detokenize(t*)) (BiCodec decoder speaker condition; record)
Stop rule: go = K4 pos_noisy >= V1 pos_noisy + 0.05 in BOTH languages.
    vllm_omni_030 python -m experiments.mist_w3po_v1.voicecond.k4_precheck --k4 <encoder.pt> --out <json> [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
V1 = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea1/m2/ctrl/encoder.pt"
K3 = REPO / "eval_outputs/mist_w3po_v1/k3/encoder/encoder.pt"
DIAG = REPO / "data/open_voicecond_v1/idea1"
MARGIN = 0.05


def digits(ids):
    """token ids (N,32) -> FSQ level digits (N,32,6), base-4 little endian (FSQ _basis = cumprod([1,4,4,4,4,4]))"""
    ids = np.asarray(ids)
    return np.stack([(ids // 4 ** k) % 4 for k in range(6)], -1)


def cos(A, B):
    return np.sum(A * B, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k4", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N dev rows per language")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    sys.path.insert(0, str(REPO))
    import soundfile as sf
    import torch
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    enc = {"phi0": BiCodecPhi(a.device), "v1": load_encoder(BiCodecPhi(a.device), str(V1)),
           "k3": load_encoder(BiCodecPhi(a.device), str(K3)), "k4": load_encoder(BiCodecPhi(a.device), a.k4)}
    for P in enc.values():
        P.model.eval()
    P0 = enc["phi0"]
    res = {"k4": a.k4, "margin": MARGIN, "langs": {}}
    for lang in ("en", "zh"):
        d = DIAG / f"diag_{lang}"
        rows = [r for r in map(json.loads, open(d / "manifest.jsonl", encoding="utf-8")) if r["split"] == "dev"]
        rows = rows[: a.limit] if a.limit else rows
        S = [sf.read(str(d / "wav" / f"{r['id']}_s.wav"), dtype="float64")[0] for r in rows]
        Y = [sf.read(str(d / "wav" / f"{r['id']}_y.wav"), dtype="float64")[0] for r in rows]
        toks = {}
        for k, P in enc.items():
            toks[k] = {}
            for c, W in (("y", Y), ("s", S)):
                u = np.concatenate([P.phi_wave(W[i:i + 50]) for i in range(0, len(W), 50)])
                toks[k][c] = np.concatenate([P.tokens(u[i:i + 100]) for i in range(0, len(u), 100)])
                if k == "phi0" and c == "s":   # check: digits of the ids == round(bound(u)) + 2 (training labels)
                    lab = np.rint(P.bounded(u)).astype(np.int64) + 2
                    assert (digits(toks[k][c]) == lab).all(), "token digits != FSQ level labels"
        tstar = toks["phi0"]["s"]
        dstar = P0.d_vector(torch.as_tensor(tstar)[:, None, :])  # (B,1,32): detokenize transposes to (B,32,1)
        st = {"n": len(rows)}
        for k in enc:
            e = {}
            for c, name in (("y", "noisy"), ("s", "clean")):
                t = toks[k][c]
                e[f"pos_{name}"] = round(float((t == tstar).mean()), 4)
                e[f"level_{name}"] = round(float((digits(t) == digits(tstar)).mean()), 4)
                e[f"dvec_cos_{name}"] = round(float(np.median(cos(P0.d_vector(torch.as_tensor(t)[:, None, :]).reshape(len(t), -1),
                                                                   dstar.reshape(len(t), -1)))), 4)
            st[k] = e
        res["langs"][lang] = st
        print(json.dumps({lang: st}), flush=True)
    res["go"] = all(res["langs"][l]["k4"]["pos_noisy"] >= res["langs"][l]["v1"]["pos_noisy"] + MARGIN for l in ("en", "zh"))
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"go": res["go"]}), flush=True)


if __name__ == "__main__":
    main()
