"""R3 (reports/mist_w3po_v1/robust_r3_noisesub_prereg.zh-CN.md, section 2): per-module noise subspaces of the Thinker's
audio encoder, for the 128 LoRA targets of wave 17 (layers.<i>.self_attn.{q,k,v,out}_proj).

2000 UniST training rows (1000 per direction, sha1("20261120|r3sub|row|<dir>|<k>")), each as a clean and a noisy copy
(env music / crowd / sfx by k % 3; mixing = wave 14/17 Sampler: Idea 2 background index split "train", BVR at a
U(0.05, 0.95) quantile of the OpenBroadcast-SI values, unit RMS, 0.02 s fades, limiter; rng sha1("20261120|r3sub|mix|..")).
For every module m: base output y_c (clean), y_n (noisy), frame-aligned (same length); D = y_n - y_c;
Sigma_D = E[D D'], Sigma_c = E[y_c y_c'] (frame level, uncentred). Generalised eigenproblem
Sigma_D w = lam (Sigma_c + eps I) w, eps = 1e-6 tr(Sigma_c)/d; keep lam > 1; Q_m = orthonormal basis (QR) of the kept w.
Output <out>/Q.pt {module name: Q [1280, r] float32} + <out>/stats.json. Refuses to overwrite.

    CUDA_VISIBLE_DEVICES=0 <qwen3omni python> experiments/mist_w3po_v1/voicecond/r3_subspace.py --out <dir> [--n 1000]
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
    hooks = [m.register_forward_hook((lambda nm: lambda _m, _i, o: cap.__setitem__(nm, o.detach()))(nm)) for nm, m in mods.items()]
    SD = {nm: torch.zeros(d, d, dtype=torch.float64, device=dev) for nm in mods}     # 128 x 2 x 12.5 MiB = 3.3 GiB
    SC = {nm: torch.zeros(d, d, dtype=torch.float64, device=dev) for nm in mods}
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
            for nm in mods:
                assert yc[nm].shape == yn[nm].shape, (nm, yc[nm].shape, yn[nm].shape)
                D = yn[nm] - yc[nm]
                SD[nm] += D.T @ D
                SC[nm] += yc[nm].T @ yc[nm]
            nf += yc["layers.0.self_attn.q_proj"].shape[0]
            used.append({"dir": dr, "id": r["id"], "env": env, "bvr": round(bvr, 2)})
            envs[env] += 1
            if k % 200 == 0:
                log(f"{dr} {k}/{a.n}: frames {nf}, {time.time() - t0:.0f}s")
    for hk in hooks:
        hk.remove()
    Q, st = {}, {"n_pairs": len(used), "n_frames": nf, "envs": envs, "rule": "lam > 1 of Sigma_D w = lam (Sigma_c + eps I) w",
                 "rank": {}, "lam_max": {}, "lam_med": {}}
    for nm in mods:
        Sd, Sc = (SD[nm] / nf).cpu(), (SC[nm] / nf).cpu()
        eps = 1e-6 * float(torch.trace(Sc)) / d
        Lc = torch.linalg.cholesky(Sc + eps * torch.eye(d, dtype=torch.float64))
        Li = torch.linalg.inv(Lc)
        M = Li @ Sd @ Li.T
        M = (M + M.T) / 2
        lam, U = torch.linalg.eigh(M)
        W = Li.T @ U                                      # generalised eigenvectors (columns)
        keep = lam > 1.0
        r = int(keep.sum())
        if r:
            q, _ = torch.linalg.qr(W[:, keep])
            Q[nm] = q.float().contiguous()
        else:
            Q[nm] = torch.zeros(d, 0)
        st["rank"][nm], st["lam_max"][nm], st["lam_med"][nm] = r, round(float(lam.max()), 4), round(float(lam.median()), 5)
    ranks = list(st["rank"].values())
    for p in ("q_proj", "k_proj", "v_proj", "out_proj"):
        rr = [st["rank"][f"layers.{i}.self_attn.{p}"] for i in range(32)]
        st[f"rank_{p}"] = {"mean": round(float(np.mean(rr)), 2), "min": min(rr), "max": max(rr), "zero": sum(x == 0 for x in rr)}
    st["rank_total"], st["rank_mean"], st["modules_rank0"] = sum(ranks), round(float(np.mean(ranks)), 2), sum(x == 0 for x in ranks)
    out.mkdir(parents=True)
    torch.save(Q, out / "Q.pt")
    json.dump(st, open(out / "stats.json", "x"), indent=1)
    with open(out / "used.jsonl", "x", encoding="utf-8") as fh:
        for u in used:
            fh.write(json.dumps(u) + "\n")
    st["peak_mem_gib"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
    log(json.dumps({k: st[k] for k in ("n_pairs", "n_frames", "peak_mem_gib", "rank_total", "rank_mean", "modules_rank0", "rank_q_proj",
                                        "rank_k_proj", "rank_v_proj", "rank_out_proj")}))


if __name__ == "__main__":
    main()
