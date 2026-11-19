"""Robust wave 7 (reports/mist_w3po_v1/robust_wave7_taskfe_prereg.zh-CN.md): train the causal front-end, one arm.

  --arm pseudo   FE-pseudo: noisy input -> target x* (wave7_fe_pseudo), clean input -> target x_clean (identity)
  --arm clean    FE-clean:  noisy input -> target x_clean,               clean input -> target x_clean
Both arms see identical samples: step s, slot j of batch 16: sentence = rows[sha1("20261119|w7fe|row|s|j") mod 10000]
(both directions pooled, one front-end), crop of CROP = 4 s at an offset from the same hash (shorter clips are zero
padded at the end and the padding is masked out of the loss), condition clean with probability 0.2, otherwise
env in {music, crowd, sfx} uniformly (wave7_fe_common.Mixer, wave 5's single-layer recipe with wave-7 seeds) mixed onto
x_clean; the target is scaled by the same limiter gain as the mixture. Loss = wave7_fe_common.mrstft_l1 (multi-resolution
STFT magnitude L1 + waveform L1) on the valid samples. Adam lr 3e-4 (betas 0.9 / 0.999), 20000 steps, constant lr,
grad-norm clip 5; fp32. Saves the final state only (and a resume checkpoint every 2000 steps); no selection.

    CUDA_VISIBLE_DEVICES=<card> python -I experiments/mist_w3po_v1/voicecond/wave7_fe_train.py --arm pseudo|clean \
        --out eval_outputs/mist_w3po_v1/wave7/FE/train_<arm> [--steps 20000] [--workers 12]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from experiments.mist_w3po_v1.voicecond.wave7_fe_common import OUT, SR, FrontEnd, Mixer, h, mrstft_l1  # noqa: E402

BATCH, CROP, P_CLEAN = 16, 4.0, 0.2
ENVS = ("music", "crowd", "sfx")


def log(m: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}", flush=True)


class Data(torch.utils.data.IterableDataset):
    def __init__(self, arm, start, steps):
        self.arm, self.start, self.steps = arm, start, steps

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        nw, wid = (wi.num_workers, wi.id) if wi else (1, 0)
        rows = []
        for d in ("zh2en", "en2zh"):
            rows += [json.loads(l) for l in open(OUT / "data" / f"train_{d}.jsonl", encoding="utf-8")]
        pse = {}
        for p in sorted((OUT / "runs").glob("*.jsonl")):
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                pse[r["id"]] = r["pseudo"]
        if self.arm == "pseudo":
            missing = [r["id"] for r in rows if r["id"] not in pse]
            assert not missing, f"{len(missing)} sentences without a pseudo target"
        mx = Mixer()
        n = int(CROP * SR)
        for s in range(self.start, self.steps):
            if (s - self.start) % nw != wid:
                continue
            X, T, M = [], [], []
            for j in range(BATCH):
                hh = h(f"20261119|w7fe|row|{s}|{j}")
                r = rows[hh % len(rows)]
                xc = sf.read(str(REPO / r["audio"]), dtype="float32")[0]
                xt = sf.read(str(REPO / pse[r["id"]]), dtype="float32")[0] if self.arm == "pseudo" else xc
                assert len(xt) == len(xc), r["id"]
                off = (hh >> 64) % max(1, len(xc) - n + 1)
                xc, xt = xc[off: off + n], xt[off: off + n]
                m = np.zeros(n, np.float32)
                m[: len(xc)] = 1
                u = ((hh >> 96) % 10000) / 10000.0
                if u < P_CLEAN:
                    x, t = xc, xc
                else:
                    env = ENVS[(hh >> 128) % 3]
                    x, lim = mx.mix(xc, env, f"{s}|{j}")
                    t = xt * lim
                X.append(np.pad(x, (0, n - len(x))))
                T.append(np.pad(t, (0, n - len(t))))
                M.append(m)
            yield s, torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(T)), torch.from_numpy(np.stack(M))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("pseudo", "clean"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "final.pt").exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out / 'final.pt'}")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(20261119)
    fe = FrontEnd().to(dev)
    opt = torch.optim.Adam(fe.parameters(), lr=3e-4)
    start = 0
    ck = out / "resume.pt"
    if ck.exists():
        st = torch.load(ck, map_location=dev)
        fe.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = st["step"]
        log(f"resume at step {start}")
    log(f"arm {a.arm}: {sum(p.numel() for p in fe.parameters()) / 1e6:.2f} M params on {dev}, steps {start}..{a.steps}")
    dl = torch.utils.data.DataLoader(Data(a.arm, start, a.steps), batch_size=None, num_workers=a.workers,
                                     prefetch_factor=4 if a.workers else None, persistent_workers=False)
    fe.train()
    t0, acc = time.time(), []
    with open(out / "train_log.jsonl", "a", encoding="utf-8") as lf:
        for s, X, T, M in dl:
            X, T, M = X.to(dev), T.to(dev), M.to(dev)
            Y = fe(X) * M
            loss = mrstft_l1(Y, T * M)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(fe.parameters(), 5.0))
            opt.step()
            acc.append(float(loss))
            if s % 100 == 0 or s == a.steps - 1:
                rec = {"step": s, "loss": float(np.mean(acc)), "grad_norm": gn, "sec": round(time.time() - t0, 1)}
                lf.write(json.dumps(rec) + "\n")
                lf.flush()
                log(json.dumps(rec))
                acc = []
            if (s + 1) % 2000 == 0 and s + 1 < a.steps:
                torch.save({"model": fe.state_dict(), "opt": opt.state_dict(), "step": s + 1}, str(ck) + ".tmp")
                os.replace(str(ck) + ".tmp", ck)
    torch.save({"model": fe.state_dict(), "arm": a.arm, "steps": a.steps}, out / "final.pt")
    log(f"done: {out / 'final.pt'}")


if __name__ == "__main__":
    main()
