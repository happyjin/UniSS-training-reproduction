"""Idea 1 M2 training pairs (reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md v3.3): per language N pairs =
4 background categories x 5 BVR bins (esports quintiles) x N/20, 6 s each, built exactly as the diagnostic set
(idea1_build_diag helpers) but from TRAINING speakers and only the DEV-parity background pool (held-out lists, the
benchmark rule and the MUSAN fma subsets excluded as in idea1_build_diag.bg_pools):
  en  LibriSpeech train-clean-100 (all speakers)      zh  AISHELL-1 wav/{train,dev,test}/<speaker> (all speakers)
Writes <out>/{s,y}.npy (float16, N x 96000, memmap) + <out>/manifest.jsonl; vhat.npy is added by idea1_hdemucs_gpu.py.
CPU pool; each pair has its own seed (20261119 + 1_000_003 * lang + index). Refuses an existing --out.
    conda_envs/uniss-eval/bin/python -m experiments.mist_w3po_v1.voicecond.idea1_build_train --lang en|zh --speech <root>
        --musan <musan/music> --fsd-audio <dir> --fsd-gt <eval.csv> --out <dir> [--n 20000] [--procs 40]
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.voicecond import idea1_build_diag as B

G = {}


def speakers(lang: str, root: str) -> dict:
    out = {}
    if lang == "en":
        for spk in sorted(os.listdir(root)):
            fs = sorted(glob.glob(os.path.join(root, spk, "*", "*.flac")))
            if fs:
                out[spk] = fs
    else:
        for split in ("train", "dev", "test"):
            for d in sorted(glob.glob(os.path.join(root, split, "*"))):
                fs = sorted(glob.glob(os.path.join(d, "*.wav")))
                if fs:
                    out[Path(d).name] = fs
    return out


def init(state):
    G.update(state)


def job(task):
    k, cat, bi, target, spk, seed = task
    rng = np.random.default_rng(seed)
    P = G["pools"]
    s, utts = B.speech6(rng, G["spk"][spk], set())
    if cat == "music":
        b, src = B.music6(rng, P["music"])
    elif cat in ("crowd", "sfx"):
        b, src = B.take6(rng, P[cat])
    else:
        m, s1 = B.music6(rng, P["music"])
        c, s2 = B.take6(rng, P["crowd"])
        x, s3 = B.take6(rng, P["sfx"])
        b, src = B.unit(m) + 0.7 * B.unit(c) + 0.7 * B.unit(x), s1 + s2 + s3
    g = np.sqrt((s ** 2).sum() / ((b ** 2).sum() + 1e-12) * 10 ** (target / 10))
    b = (g * b).astype(np.float32)
    y = s + b
    pk = float(np.abs(y).max())
    if pk > 0.99:
        s, b = s * (0.99 / pk), b * (0.99 / pk)
        y = s + b
    S = np.load(G["out"] + "/s.npy", mmap_mode="r+")
    Y = np.load(G["out"] + "/y.npy", mmap_mode="r+")
    S[k] = s.astype(np.float16)
    Y[k] = y.astype(np.float16)
    S.flush()
    Y.flush()
    return {"k": k, "cat": cat, "bin": bi, "bvr_target": round(target, 3),
            "bvr_true": round(float(10 * np.log10((b ** 2).sum() / (s ** 2).sum())), 3), "speaker": spk,
            "utts": utts, "bg": src, "peak_scaled": pk > 0.99}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["en", "zh"])
    ap.add_argument("--speech", required=True)
    ap.add_argument("--musan", required=True)
    ap.add_argument("--fsd-audio", required=True)
    ap.add_argument("--fsd-gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--procs", type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    pools = B.bg_pools(a, B.heldout())["dev"]
    spk = speakers(a.lang, a.speech)
    names = sorted(spk)
    bv = B.esports_bvr()
    edges = np.percentile(bv, np.linspace(0, 100, B.NBIN + 1))
    per = a.n // (len(B.CATS) * B.NBIN)
    base = 20261119 + 1_000_003 * (0 if a.lang == "en" else 1)
    rng = np.random.default_rng(base)
    tasks, k = [], 0
    for cat in B.CATS:
        for bi in range(B.NBIN):
            vals = bv[(bv >= edges[bi]) & (bv <= edges[bi + 1])]
            for _ in range(per):
                tasks.append((k, cat, bi, float(vals[rng.integers(len(vals))]), names[k % len(names)], base + k))
                k += 1
    out.mkdir(parents=True)
    for nm in ("s", "y"):
        np.lib.format.open_memmap(str(out / f"{nm}.npy"), mode="w+", dtype=np.float16, shape=(len(tasks), B.N6)).flush()
    with mp.get_context("fork").Pool(a.procs, initializer=init, initargs=({"pools": pools, "spk": spk, "out": str(out)},)) as pool:
        rows = pool.map(job, tasks, chunksize=8)
    with open(out / "manifest.jsonl", "x", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda r: r["k"]):
            fh.write(json.dumps({"lang": a.lang, **r}) + "\n")
    print(json.dumps({"out": str(out), "pairs": len(rows), "speakers": len(names),
                      "bg_pool": {c: len(v) for c, v in pools.items()}}))


if __name__ == "__main__":
    main()
