"""Robust R10 (reports/mist_w3po_v1/robust_r10_tsvproj_prereg.zh-CN.md): remove the whole tau_clean subspace from tau_noisy
(two-sided singular-subspace projection, in the spirit of Task Singular Vectors / TSV-Merge, CVPR 2025) instead of R1's
subtraction. Per adapted weight: dW_c = U S V' (rank <= 16, singular values > 1e-6 * max kept), dW_rob = (I - U U') dW_n
(I - V V'); W' = W + coef * dW_rob, coef = 1 (prereg, not swept). Copy of r1_merge.py otherwise (hard-linked shards,
bf16 rounding report). Refuses to overwrite.

--- r1_merge.py docstring ---
Robust R1 (reports/mist_w3po_v1/robust_r1_taskdiff_prereg.zh-CN.md): merge the style-cancelling task-vector difference
into a full Qwen3-Omni copy for si_thinker_stream (--thinker):

    W' = W + c * (dW_noisy - dW_clean),   dW_x = (alpha_x / rank_x) * b_x @ a_x   (per q/k/v/o of every text layer)

LoRA factors are never subtracted directly: each dW is formed in fp32 first. c = 1 (prereg, not swept). Shards without an
adapted weight are hard-linked from the release; shards holding one are rewritten (copy of wave5_m_merge.py's layout).
bf16 rounding check (as wave 16's addendum): for every adapted tensor, the merged bf16 weight minus the bf16 base weight is
compared with the fp32 delta; the json reports the mean / max relative error ||(W'_bf16 - W) - c*dD|| / ||c*dD|| and
||c*dD|| / ||W||, so a difference too small to survive bf16 is visible. Refuses to overwrite.

    <qwen3omni python> experiments/mist_w3po_v1/voicecond/r1_merge.py --noisy <run>/adapters_final.pt \
        --clean <run>/adapters_final.pt --out <dir> [--coef 1]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"


def deltas(path: str) -> dict:
    meta = json.load(open(Path(path).with_suffix(".json")))
    scale = meta["alpha"] / meta["rank"]
    st = torch.load(path, map_location="cpu")
    return {f"thinker.model.{k[:-2]}.weight": scale * (st[k[:-2] + ".b"].float() @ st[k].float())
            for k in st if k.endswith(".a")}, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--coef", type=float, default=1.0)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    dn, mn = deltas(a.noisy)
    dc, mc = deltas(a.clean)
    assert set(dn) == set(dc), "adapter keys differ"
    assert mc.get("clean_only") is True, f"--clean must be a clean-only run: {mc}"
    assert mn["steps"] == mc["steps"] and mn["rank"] == mc["rank"] and mn["alpha"] == mc["alpha"], (mn, mc)
    delta, kept, ranks = {}, [], []
    for k in dn:
        U, S, Vh = torch.linalg.svd(dc[k].double(), full_matrices=False)
        r = int((S > 1e-6 * S.max()).sum())
        U, V = U[:, :r], Vh[:r].T
        x = dn[k].double()
        x = x - U @ (U.T @ x)
        x = x - (x @ V) @ V.T
        delta[k] = (a.coef * x).float()
        kept.append(float(x.norm() / dn[k].double().norm().clamp(min=1e-12)))
        ranks.append(r)
    index = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
    miss = [k for k in delta if k not in index]
    assert not miss, miss[:3]
    shards = {index[k] for k in delta}
    out.mkdir(parents=True)
    for f in BASE.iterdir():
        if f.is_file() and f.name not in shards:
            try:
                os.link(f, out / f.name)
            except OSError:
                shutil.copy2(f, out / f.name)
    n, rel_err, rel_size, cos_nc = 0, [], [], []
    for sh in sorted(shards):
        state = load_file(str(BASE / sh))
        for k, d in delta.items():
            if index[k] != sh:
                continue
            w0 = state[k]
            w1 = (w0.float() + d).to(w0.dtype).contiguous()
            got = w1.float() - w0.float()
            rel_err.append(float((got - d).norm() / d.norm().clamp(min=1e-12)))
            rel_size.append(float(d.norm() / w0.float().norm().clamp(min=1e-12)))
            cos_nc.append(float(torch.nn.functional.cosine_similarity(dn[k].reshape(1, -1), dc[k].reshape(1, -1)).item()))
            state[k] = w1
            n += 1
        save_file(state, str(out / sh), metadata={"format": "pt"})
    nd = [float(d.norm()) for d in delta.values()]
    nn_ = [float(dn[k].norm()) for k in dn]
    info = {"out": str(out), "adapted": n, "shards_rewritten": len(shards), "coef": a.coef,
            "norm_diff_mean": round(sum(nd) / len(nd), 5), "norm_noisy_mean": round(sum(nn_) / len(nn_), 5),
            "cos_noisy_clean_mean": round(sum(cos_nc) / len(cos_nc), 4),
            "bf16_rel_err_mean": round(sum(rel_err) / len(rel_err), 4), "bf16_rel_err_max": round(max(rel_err), 4),
            "delta_over_w_mean": round(sum(rel_size) / len(rel_size), 6),
            "kept_norm_share_mean": round(sum(kept) / len(kept), 4), "kept_norm_share_min": round(min(kept), 4),
            "clean_rank_mean": round(sum(ranks) / len(ranks), 2), "projection": "two-sided (I-UU') dW_n (I-VV')"}
    with open(out / "r10_merge.json", "x") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
