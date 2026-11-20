"""Robust wave 9: fit the anchor discretisation (PCA + quantile bins) on CLEAN wave-5 training audio, once, fixed.

Rows: every zh2en training row (7004) plus as many en2zh rows, the first by sha1("20261120|w9|pca|<id>") -- the two
directions weighted 1:1 as in the training batches. u = C(first 1.0 s of the clip) (wave9_common.Anchor.u, the V1
encoder's pre-FSQ (32, 6) code, flattened). PCA by SVD of the centred 192-dim codes (float64), the first 8 components,
each signed so that its largest-|loading| entry is positive; per component the 31 inner bin edges are the
k/32 quantiles (k = 1..31) of the projected training codes. Output (refuses to overwrite): wave9/anchor/pca_bins.npz
(mean, comps, edges, evr) + fit.json (rows, explained variance, bin occupancy, tokenizer check).
    CUDA_VISIBLE_DEVICES=<c> <qwen3omni python> -I experiments/mist_w3po_v1/voicecond/wave9_anchor_fit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import wave9_common as C  # noqa: E402

DATA = REPO / "eval_outputs/mist_w3po_v1/wave5/M/data"


def main() -> None:
    if C.PCA_FILE.exists():
        sys.exit(f"拒绝覆盖已有输出: {C.PCA_FILE}")
    from transformers import AutoTokenizer
    from experiments.mist_w3po_v1.voicecond import wave1_data as W1D
    tokchk = C.check_unused(AutoTokenizer.from_pretrained(str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")))
    held = W1D.held_speakers()
    rows = {d: [json.loads(l) for l in open(DATA / f"rows_{d}.jsonl", encoding="utf-8")] for d in ("zh2en", "en2zh")}
    for d in rows:
        assert not {r["spk"] for r in rows[d]} & held, d
    n = len(rows["zh2en"])
    en = sorted(rows["en2zh"], key=lambda r: C.h(f"20261120|w9|pca|{r['id']}"))[:n]
    sel = rows["zh2en"] + en
    A = C.Anchor("cuda:0", pca=False)
    U = []
    for k in range(0, len(sel), 64):
        wavs = [sf.read(str(REPO / r["audio"]), dtype="float32")[0] for r in sel[k: k + 64]]
        U.append(A.u(wavs))
        if k % 2048 == 0:
            print(f"{k}/{len(sel)}", flush=True)
    U = np.concatenate(U)
    mean = U.mean(0)
    _, S, Vt = np.linalg.svd(U - mean, full_matrices=False)
    comps = Vt[: C.N_DIM].copy()
    for k in range(C.N_DIM):
        if comps[k, np.argmax(np.abs(comps[k]))] < 0:
            comps[k] *= -1
    evr = (S ** 2 / (S ** 2).sum())[: C.N_DIM]
    Z = (U - mean) @ comps.T
    edges = np.stack([np.quantile(Z[:, k], np.arange(1, C.N_BIN) / C.N_BIN) for k in range(C.N_DIM)])
    C.ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.PCA_FILE, "xb") as fh:
        np.savez(fh, mean=mean, comps=comps, edges=edges, evr=evr)
    A.mean, A.comps, A.edges = mean, comps, edges
    ids = A.ids_from_u(U)
    occ = [np.bincount(ids[:, k] - C.ANCHOR_LO - C.N_BIN * k, minlength=C.N_BIN).tolist() for k in range(C.N_DIM)]
    info = {"rows": {"zh2en": n, "en2zh": len(en)}, "n": len(sel), "evr": [round(float(v), 4) for v in evr],
            "evr_sum": round(float(evr.sum()), 4), "occupancy": occ, "tokenizer": tokchk,
            "ids_range": [C.ANCHOR_LO, C.VOCAB - 1], "pca_file": str(C.PCA_FILE.relative_to(REPO))}
    with open(C.ANCHOR_DIR / "fit.json", "x", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "occupancy"}), flush=True)


if __name__ == "__main__":
    main()
