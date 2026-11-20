"""R6 (reports/mist_w3po_v1/robust_r6_cleannull_prereg.zh-CN.md, section 3): per-module clean null spaces of the
INPUTS of the Thinker audio encoder's 128 LoRA targets (layers.<i>.self_attn.{q,k,v,out}_proj; q/k/v share an input).
Copy of r3_subspace.py (same 2000 UniST training rows, same clean / noisy mixing) with forward-pre hooks on the module
inputs instead of output hooks. For each unique input (layer i: qkv, out): K_c = E[x_c x_c'] (clean frames, uncentred,
fp64), K_d = E[D D'], D = x_n - x_c (diagnostic only). Rule: eigenvalues of K_c ascending; U0 = eigenvectors of the
largest number r0 of smallest eigenvalues whose sum <= 1% tr(K_c). Output <out>/U0.pt {module name: U0 [1280, r0] fp32}
(q/k/v of a layer share one tensor) + stats.json (r0, clean energy share, noise-residual energy share in U0).
Refuses to overwrite.

    CUDA_VISIBLE_DEVICES=0 <qwen3omni python> experiments/mist_w3po_v1/voicecond/r6_nullspace.py --out <dir> [--n 1000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
QWEN3 = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
DATA = REPO / "eval_outputs/mist_w3po_v1/wave14/data"
POOL = {"music": "musan_music", "crowd": "fsd_crowd", "sfx": "fsd_sfx_musan_noise"}
ENVN = ("music", "crowd", "sfx")


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def log(m: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}", flush=True)


def mix(x, d, k, pools, bv):
    from experiments.mist_w3po_v1.voicecond import bench_build as BB
    env = ENVN[k % 3]
    rng = random.Random(h(f"20261120|r3sub|mix|{d}|{k}"))
    bvr = float(np.quantile(bv, rng.uniform(0.05, 0.95)))
    P = pools[POOL[env]]
    f = P[int(rng.random() * len(P)) % len(P)]
    b = BB.load(f["file"])
    n = len(x)
    if len(b) < n:
        b = np.tile(b, 1 + n // max(1, len(b)))
    off = int(rng.random() * max(0, len(b) - n))
    seg = BB.fade(BB.unit_rms(b[off: off + n]), 0.02, 0.02)
    es, eb = float((x.astype(np.float64) ** 2).sum()), float((seg.astype(np.float64) ** 2).sum())
    alpha = math.sqrt(10 ** (bvr / 10) * es / max(eb, 1e-12))
    y = x + alpha * seg
    return (y * min(1.0, 0.99 / float(np.abs(y).max()))).astype(np.float32), env, bvr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000, help="rows per direction")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from experiments.mist_w3po_v1.voicecond import wave1_data as W1D
    pools, bv = W1D.bg_pools("train"), W1D.bvr_values()
    dev = torch.device("cuda", 0)
    # memory cap (coordinator: R4 stacks on this card): load on CPU, move only the audio encoder (~1.3 GB) to the GPU
    omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(str(QWEN3), torch_dtype=torch.bfloat16,
                                                                device_map={"": "cpu"}, attn_implementation="sdpa")
    enc = omni.thinker.audio_tower.to(dev).eval()
    del omni
    fe = Qwen3OmniMoeProcessor.from_pretrained(str(QWEN3)).feature_extractor
    mods = {}
    for i, L in enumerate(enc.layers):
        for p in ("q_proj", "k_proj", "v_proj", "out_proj"):
            mods[f"layers.{i}.self_attn.{p}"] = getattr(L.self_attn, p)
    assert len(mods) == 128
    d = 1280
    cap = {}
    uniq = {}
    for i, L in enumerate(enc.layers):
        uniq[f"{i}.qkv"] = L.self_attn.q_proj
        uniq[f"{i}.out"] = L.self_attn.out_proj
    hooks = [m.register_forward_pre_hook((lambda nm: lambda _m, inp: cap.__setitem__(nm, inp[0].detach()))(nm)) for nm, m in uniq.items()]
    SD = {nm: torch.zeros(d, d, dtype=torch.float64, device=dev) for nm in uniq}     # 64 x 2 x 12.5 MiB
    SC = {nm: torch.zeros(d, d, dtype=torch.float64, device=dev) for nm in uniq}
    nf, used, envs = 0, [], {e: 0 for e in ENVN}
    t0 = time.time()

    def run(x):
        f = fe(x, sampling_rate=16000, return_attention_mask=True, padding="longest", truncation=False, return_tensors="pt")
        feats = f["input_features"].to(dev, torch.bfloat16)
        mask = f["attention_mask"].to(dev)
        cap.clear()
        with torch.no_grad():                            # = Thinker.get_audio_features
            lens = mask.sum(1)
            x_ = feats.permute(0, 2, 1)[mask.bool()].permute(1, 0)
            enc(x_, feature_lens=lens, return_dict=True)
        return {nm: v.reshape(-1, d).double() for nm, v in cap.items()}

    for dr in ("zh2en", "en2zh"):
        rows = [json.loads(l) for l in open(DATA / f"rows_{dr}.jsonl", encoding="utf-8")]
        for k in range(a.n):
            r = rows[h(f"20261120|r3sub|row|{dr}|{k}") % len(rows)]
            x = sf.read(str(REPO / r["audio"]), dtype="float32")[0]
            xn, env, bvr = mix(x, dr, k, pools, bv)
            yc, yn = run(x), run(xn)
            for nm in uniq:
                assert yc[nm].shape == yn[nm].shape, (nm, yc[nm].shape, yn[nm].shape)
                D = yn[nm] - yc[nm]
                SD[nm] += D.T @ D
                SC[nm] += yc[nm].T @ yc[nm]
            nf += yc["0.qkv"].shape[0]
            used.append({"dir": dr, "id": r["id"], "env": env, "bvr": round(bvr, 2)})
            envs[env] += 1
            if k % 200 == 0:
                log(f"{dr} {k}/{a.n}: frames {nf}, {time.time() - t0:.0f}s")
    for hk in hooks:
        hk.remove()
    U, st = {}, {"n_pairs": len(used), "n_frames": nf, "envs": envs,
                 "rule": "U0 = eigvecs of the smallest eigenvalues of K_c with cumulative sum <= 1% tr(K_c)",
                 "r0": {}, "clean_share": {}, "noise_share": {}}
    for nm in uniq:
        Kd, Kc = (SD[nm] / nf).cpu(), (SC[nm] / nf).cpu()
        Kc = (Kc + Kc.T) / 2
        lam, V = torch.linalg.eigh(Kc)                    # ascending
        lam = lam.clamp(min=0)
        cs = torch.cumsum(lam, 0)
        r0 = int((cs <= 0.01 * float(lam.sum())).sum())
        U0 = V[:, :r0]
        st["r0"][nm] = r0
        st["clean_share"][nm] = round(float(cs[r0 - 1] / lam.sum()) if r0 else 0.0, 5)
        st["noise_share"][nm] = round(float(torch.trace(U0.T @ Kd @ U0) / torch.trace(Kd)) if r0 else 0.0, 5)
        U[nm] = U0.float().contiguous()
    out_U = {}
    for i in range(len(enc.layers)):
        for p in ("q_proj", "k_proj", "v_proj"):
            out_U[f"layers.{i}.self_attn.{p}"] = U[f"{i}.qkv"]
        out_U[f"layers.{i}.self_attn.out_proj"] = U[f"{i}.out"]
    assert set(out_U) == set(mods)
    for kind in ("qkv", "out"):
        rr = [st["r0"][f"{i}.{kind}"] for i in range(len(enc.layers))]
        ns = [st["noise_share"][f"{i}.{kind}"] for i in range(len(enc.layers))]
        st[f"r0_{kind}"] = {"mean": round(float(np.mean(rr)), 1), "min": min(rr), "max": max(rr)}
        st[f"noise_share_{kind}"] = {"mean": round(float(np.mean(ns)), 4), "min": round(min(ns), 4), "max": round(max(ns), 4)}
    out.mkdir(parents=True)
    torch.save(out_U, out / "U0.pt")
    st["peak_mem_gib"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
    json.dump(st, open(out / "stats.json", "x"), indent=1)
    with open(out / "used.jsonl", "x", encoding="utf-8") as fh:
        for u in used:
            fh.write(json.dumps(u) + "\n")
    log(json.dumps({k: st[k] for k in ("n_pairs", "n_frames", "peak_mem_gib", "r0_qkv", "r0_out", "noise_share_qkv", "noise_share_out")}))


if __name__ == "__main__":
    main()
