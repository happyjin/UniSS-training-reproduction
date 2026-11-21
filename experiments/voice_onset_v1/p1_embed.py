"""voice_onset_v1 P1 (reports/voice_onset_v1/P1_P3_prereg.zh-CN.md section 1-2): precompute TitaNet-large embeddings on
the open data. Per 6 s clip and condition c in {y (noisy), s (clean)}: chunk embeddings of [k-1, k) s (k = 1..6),
prefix embeddings of [0, k) s (k = 1..6), chunk RMS dB. Train = idea1 train_en + train_zh (y.npy / s.npy, 40000 clips);
held-out = diag_en + diag_zh (wav/<id>_{y,s}.wav, dev + test). CPU only (nemo_diar).
    python -m experiments.voice_onset_v1.p1_embed --out eval_outputs/voice_onset_v1/p1/emb [--workers 48]
Outputs <out>/<split>_{chunk,prefix}_{y,s}.npy (N, 6, 192) float16, <split>_rms_{y,s}.npy (N, 6), <split>_meta.jsonl.
Refuses existing outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
I = REPO / "data/open_voicecond_v1/idea1"
SR = 16000
NK = 6


def items():
    """(split, list of dicts {src, ...meta})"""
    out = {"train": [], "heldout": []}
    for d in ("train_en", "train_zh"):
        for k, l in enumerate(open(I / d / "manifest.jsonl", encoding="utf-8")):
            r = json.loads(l)
            out["train"].append({"src": ["npy", d, k], "lang": r["lang"], "speaker": f"{r['lang']}:{r['speaker']}",
                                 "cat": r["cat"], "bin": r["bin"], "bvr": r["bvr_true"]})
    for d in ("diag_en", "diag_zh"):
        for l in open(I / d / "manifest.jsonl", encoding="utf-8"):
            r = json.loads(l)
            out["heldout"].append({"src": ["wav", d, r["id"]], "lang": r["lang"], "speaker": f"{r['lang']}:{r['speaker']}",
                                   "cat": r["cat"], "bin": r["bin"], "bvr": r["bvr_true"], "split": r["split"]})
    return out


def worker(args):
    split, rows, lo = args
    import soundfile as sf
    import torch
    torch.set_num_threads(4)
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    from experiments.mist_w3po_v1.voicecond.idea1_spksim import NEMO
    m = EncDecSpeakerLabelModel.restore_from(NEMO, map_location="cpu").eval()
    arrs = {}

    def get(src, c):
        kind, d, k = src
        if kind == "npy":
            if (d, c) not in arrs:
                arrs[(d, c)] = np.load(I / d / f"{c}.npy", mmap_mode="r")
            x = np.asarray(arrs[(d, c)][k], dtype=np.float32)
        else:
            x, sr = sf.read(str(I / d / "wav" / f"{k}_{c}.wav"), dtype="float32", always_2d=True)
            assert sr == SR
            x = x.mean(1)
        x = x[: NK * SR]
        if len(x) < NK * SR:
            x = np.pad(x, (0, NK * SR - len(x)))
        return x

    @torch.no_grad()
    def emb(batch):   # (B, T) same length
        t = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32))
        _, e = m.forward(input_signal=t, input_signal_length=torch.full((t.shape[0],), t.shape[1]))
        e = e.float().numpy()
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    n = len(rows)
    out = {f"{a}_{c}": np.zeros((n, NK, 192), np.float16) for a in ("chunk", "prefix") for c in ("y", "s")}
    out.update({f"rms_{c}": np.zeros((n, NK), np.float32) for c in ("y", "s")})
    B = 8
    for b0 in range(0, n, B):
        rs = rows[b0:b0 + B]
        for c in ("y", "s"):
            X = np.stack([get(r["src"], c) for r in rs])                       # (b, 6 s)
            ch = X.reshape(len(rs), NK, SR)
            out[f"rms_{c}"][b0:b0 + len(rs)] = 10 * np.log10((ch.astype(np.float64) ** 2).mean(2) + 1e-10)
            out[f"chunk_{c}"][b0:b0 + len(rs)] = emb(ch.reshape(-1, SR)).reshape(len(rs), NK, 192)
            for k in range(1, NK + 1):
                out[f"prefix_{c}"][b0:b0 + len(rs), k - 1] = emb(X[:, : k * SR])
        if (b0 // B) % 50 == 0:
            print(f"[p1_embed] {split} @{lo} {b0 + len(rs)}/{n}", flush=True)
    return split, lo, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "train_meta.jsonl").exists():
        sys.exit(f"拒绝覆盖已有输出: {out}")
    out.mkdir(parents=True, exist_ok=True)
    S = items()
    tasks = []
    for split, rows in S.items():
        step = -(-len(rows) // (a.workers if split == "train" else 8))
        tasks += [(split, rows[lo:lo + step], lo) for lo in range(0, len(rows), step)]
    res = {s: {} for s in S}
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for split, lo, o in pool.imap_unordered(worker, tasks):
            res[split][lo] = o
            print(f"[p1_embed] shard {split}@{lo} done", flush=True)
    for split, rows in S.items():
        los = sorted(res[split])
        for key in res[split][los[0]]:
            np.save(out / f"{split}_{key}.npy", np.concatenate([res[split][lo][key] for lo in los]))
        with open(out / f"{split}_meta.jsonl", "x", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    print("[p1_embed] done", flush=True)


if __name__ == "__main__":
    main()
