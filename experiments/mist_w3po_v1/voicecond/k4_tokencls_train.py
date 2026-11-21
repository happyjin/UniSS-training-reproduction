"""Robust K4 (reports/mist_w3po_v1/robust_k4_tokencls_prereg.zh-CN.md): the Idea 1 M2 denoising-encoder control
(idea1_e2e_train.py --mode ctrl via k3_discrim_train.py, copied line by line; data, init, steps, optimizer, seed unchanged)
with ONLY the head / target / loss changed: the clean prompt's discrete global tokens as labels.
  labels   l*_{k,d} = round(bound(u0*_{k,d})) + 2 in {0..3}   (FSQ levels [4]*6; u0* = phi_0(s), frozen tokenizer)
  head     parameter-free distance classifier on b = fsq.bound(phi(x)): logit_c = -(b - c)^2 / TAU, c in {-2,-1,0,1},
           TAU = 0.25; argmax == FSQ round, so deployment tokens are Q(phi(y)) exactly as for V1.
  L = CE(phi(y), l*) + 1.0 CE(phi(s), l*)      (mean over the 32 x 6 levels; the 2nd term = V1's clean-input anchor as CE)
Full fine-tune of the BiCodec speaker encoder (ECAPA-TDNN + Perceiver + project_in; BN / dropout in eval mode; mel front
end, FSQ, decoder and Talker untouched). AdamW lr 5e-5, 200 warm-up steps, cosine to 0, 6000 steps, 32 pairs per step,
weight decay 0, clip 1.0, fp32, seed 20261119. Only the final checkpoint is saved (<out>/encoder.pt); every 1000 steps the
losses, the code error and the token position / level match on 200 ID-1 dev pairs are logged (record only, no selection).
Refuses an existing --out.
    venvs/vllm_omni_030/bin/python -m experiments.mist_w3po_v1.voicecond.k4_tokencls_train
        --data <train_en> <train_zh> --dev <diag_en> <diag_zh> --out <dir> [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
SQ2 = math.sqrt(2.0)
LR, WARM, STEPS, BS, W_ANC, W_CURV, SEED = 5e-5, 200, 6000, 32, 1.0, 0.1, 20261119
TAU = 0.25


def levels(fsq, u):
    """FSQ level indices (0..3) of pre-bound codes u (..., 6): round(bound(u)) + 2 (== the token's per-dim digits)"""
    return (torch.round(fsq.bound(u)) + 2).long()


def level_ce(fsq, u, lab):
    """distance-head cross-entropy of pre-bound codes u (B,32,6) against level labels lab (B,32,6)"""
    b = fsq.bound(u)
    c = torch.arange(-2, 2, device=u.device, dtype=u.dtype)
    logits = -((b[..., None] - c) ** 2) / TAU
    return torch.nn.functional.cross_entropy(logits.reshape(-1, 4), lab.reshape(-1))


class Pairs(torch.utils.data.Dataset):
    def __init__(self, dirs, mode):
        self.dirs, self.mode = [str(d) for d in dirs], mode
        self.n = [np.load(Path(d) / "y.npy", mmap_mode="r").shape[0] for d in self.dirs]
        self.idx = [(i, k) for i, n in enumerate(self.n) for k in range(n)]
        self.mm = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        from uniss.speech_tokenizer.bicodec.utils.audio import audio_volume_normalize as vn
        if self.mm is None:
            self.mm = [{x: np.load(Path(d) / f"{x}.npy", mmap_mode="r") for x in ("s", "y", "vhat")} for d in self.dirs]
        i, k = self.idx[j]
        m = self.mm[i]
        s, y, v = (np.asarray(m[x][k], dtype=np.float64) for x in ("s", "y", "vhat"))
        bh = y - v
        ws = [y, y + (SQ2 - 1) * bh, y + bh, s] if self.mode == "zne" else [y, s]
        return torch.from_numpy(np.stack([vn(w) for w in ws]).astype(np.float32)), j


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    a.mode = "ctrl"
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.mkdir(parents=True)
    sys.path.insert(0, str(REPO))
    import soundfile as sf
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device(a.device)
    P0 = BiCodecPhi(a.device)
    P = BiCodecPhi(a.device)
    ds = Pairs(a.data, a.mode)
    # frozen targets u0* = phi_0(s) for every training pair
    t0 = time.time()
    U0 = np.zeros((len(ds), 32, 6), dtype=np.float32)
    off = 0
    for d in a.data:
        S = np.load(Path(d) / "s.npy", mmap_mode="r")
        for i in range(0, S.shape[0], 256):
            batch = [np.asarray(r, dtype=np.float64) for r in S[i:i + 256]]
            U0[off + i: off + i + len(batch)] = P0.phi_wave(batch)
        off += S.shape[0]
    np.save(out / "u0_targets.npy", U0)
    U0t = torch.from_numpy(U0).to(dev)
    with torch.no_grad():
        LAB = levels(P0.fsq, U0t)
    print(f"targets {len(ds)} in {time.time() - t0:.0f}s", flush=True)
    # ID-1 dev monitor: 100 pairs per language (record only)
    mon = []
    for d in a.dev:
        rows = [r for r in map(json.loads, open(Path(d) / "manifest.jsonl", encoding="utf-8")) if r["split"] == "dev"]
        for r in rows[:: max(1, len(rows) // 100)][:100]:
            w = {x: sf.read(str(Path(d) / "wav" / f"{r['id']}_{x}.wav"), dtype="float64")[0] for x in ("s", "y", "vhat")}
            mon.append(w)
    mon_u0 = torch.from_numpy(P0.phi_wave([w["s"] for w in mon])).to(dev)
    with torch.no_grad():
        mon_lab = levels(P0.fsq, mon_u0)
    del P0
    train_mods = [P.se.speaker_encoder, P.se.perceiver_sampler, P.rq.project_in]
    for p in P.model.parameters():
        p.requires_grad_(False)
    params = [p for m in train_mods for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    P.model.eval()
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: (st + 1) / WARM if st < WARM else 0.5 * (1 + math.cos(math.pi * (st - WARM) / (STEPS - WARM))))
    g = torch.Generator()
    g.manual_seed(SEED)
    dl = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=True, drop_last=True, num_workers=10, generator=g,
                                     persistent_workers=True, prefetch_factor=4)
    mse = lambda x, y: ((x - y) ** 2).mean()  # noqa: E731

    def losses(W, u0, lab):
        B, K, T = W.shape
        U = P._u_from_ref(W.reshape(B * K, T)).reshape(B, K, 32, 6)
        if a.mode == "zne":
            uh = 2 * U[:, 0] - U[:, 1]
            l_main = mse(uh, u0)
            curv = ((U[:, 2] - U[:, 1]) / 2 - (U[:, 1] - U[:, 0])) / 3
            l_curv = (curv ** 2).mean()
            l_anc = mse(U[:, 3], u0)
            return l_main + W_ANC * l_anc + W_CURV * l_curv, {"main": l_main, "anchor": l_anc, "curv": l_curv}, uh
        l_main = level_ce(P.fsq, U[:, 0], lab)
        l_anc = level_ce(P.fsq, U[:, 1], lab)
        return l_main + W_ANC * l_anc, {"main": l_main, "anchor": l_anc, "mse_noisy": mse(U[:, 0].detach(), u0)}, U[:, 0]

    def monitor(step):
        from uniss.speech_tokenizer.bicodec.utils.audio import audio_volume_normalize as vn
        errs, raw, tot, pos, lev = [], [], [], [], []
        with torch.no_grad():
            for i in range(0, len(mon), 25):
                ws = []
                for w in mon[i:i + 25]:
                    y, s, bh = w["y"], w["s"], w["y"] - w["vhat"]
                    ws.append(np.stack([vn(x) for x in ([y, y + (SQ2 - 1) * bh, y + bh, s] if a.mode == "zne" else [y, s])]))
                W = torch.from_numpy(np.stack(ws).astype(np.float32)).to(dev)
                u0 = mon_u0[i:i + 25]
                lab = mon_lab[i:i + 25]
                L, parts, uh = losses(W, u0, lab)
                tot.append(float(L))
                hit = levels(P.fsq, uh) == lab
                pos += hit.all(-1).float().mean(-1).tolist()
                lev += hit.float().mean((-1, -2)).tolist()
                errs += torch.linalg.norm((uh - u0).reshape(len(u0), -1), dim=1).tolist()
        rec = {"step": step, "dev_loss": float(np.mean(tot)), "dev_err_median": float(np.median(errs)),
               "dev_pos_match": float(np.mean(pos)), "dev_level_match": float(np.mean(lev))}
        print(json.dumps(rec), flush=True)
        return rec

    log = open(out / "train_log.jsonl", "x", encoding="utf-8")
    log.write(json.dumps(monitor(0)) + "\n")
    step, t0 = 0, time.time()
    while step < STEPS:
        for W, j in dl:
            W = W.to(dev, non_blocking=True)
            u0 = U0t[j.to(dev)]
            L, parts, _ = losses(W, u0, LAB[j.to(dev)])
            opt.zero_grad(set_to_none=True)
            L.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                rec = {"step": step, "loss": float(L), **{k: float(v) for k, v in parts.items()}, "grad_norm": float(gn),
                       "lr": sched.get_last_lr()[0], "s_per_step": (time.time() - t0) / step}
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if step % 1000 == 0:
                log.write(json.dumps(monitor(step)) + "\n")
                log.flush()
            if step >= STEPS:
                break
    torch.save({"speaker_encoder": P.se.speaker_encoder.state_dict(), "perceiver_sampler": P.se.perceiver_sampler.state_dict(),
                "project_in": P.rq.project_in.state_dict(),
                "meta": {"mode": "k4_ctrl_token_level_ce", "tau": TAU, "steps": STEPS, "bs": BS, "lr": LR, "w_anchor": W_ANC, "w_curv": W_CURV,
                         "seed": SEED, "data": a.data}}, out / "encoder.pt")
    log.close()
    print(json.dumps({"out": str(out / "encoder.pt"), "steps": step, "minutes": (time.time() - t0) / 60}), flush=True)


if __name__ == "__main__":
    main()
