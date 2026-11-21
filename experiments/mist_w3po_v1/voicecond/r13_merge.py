"""Robust R13 (reports/mist_w3po_v1/robust_r13_fishermask_prereg.zh-CN.md, section 3): Fisher-weighted reversion of the
ERM-WiSE update. dW = unist_wise05 dense increment (--adapters, alpha/rank from its json); alpha_j = F_noisy / (F_noisy +
F_clean + 1e-30) from r13_fisher.py (--fisher); W' = W + alpha (.) dW. Copy of r1_merge.py otherwise (hard-linked shards,
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
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--fisher", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    dn, mn = deltas(a.adapters)
    FI = torch.load(a.fisher, map_location="cpu")
    assert set(FI) == set(dn), (len(FI), len(dn))
    delta, amean, kept = {}, [], []
    for k in dn:
        fc, fn = FI[k]["clean"].double(), FI[k]["noisy"].double()
        al = fn / (fn + fc + 1e-30)
        delta[k] = (al * dn[k].double()).float()
        amean.append(float(al.mean()))
        kept.append(float(delta[k].norm() / dn[k].norm().clamp(min=1e-12)))
    dc = dn
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
            state[k] = w1
            n += 1
        save_file(state, str(out / sh), metadata={"format": "pt"})
    nd = [float(d.norm()) for d in delta.values()]
    nn_ = [float(dn[k].norm()) for k in dn]
    info = {"out": str(out), "adapted": n, "shards_rewritten": len(shards), "alpha_mean": round(sum(amean) / len(amean), 4),
            "kept_norm_share_mean": round(sum(kept) / len(kept), 4), "kept_norm_share_min": round(min(kept), 4),
            "norm_diff_mean": round(sum(nd) / len(nd), 5), "norm_noisy_mean": round(sum(nn_) / len(nn_), 5),
            
            "bf16_rel_err_mean": round(sum(rel_err) / len(rel_err), 4), "bf16_rel_err_max": round(max(rel_err), 4),
            "delta_over_w_mean": round(sum(rel_size) / len(rel_size), 6)}
    with open(out / "r13_merge.json", "x") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
