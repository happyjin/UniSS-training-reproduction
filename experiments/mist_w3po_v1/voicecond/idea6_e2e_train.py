"""Idea 6 end-to-end training (prereg reports/mist_w3po_v1/voicecond_idea6_bayes_prereg.zh-CN.md, "端到端训练预登记"):
learned heteroscedastic likelihood + closed-form streaming posterior.

  encoder phi_theta (init: Idea 1 denoising encoder C; ECAPA + Perceiver + project_in, BN / dropout eval) -> u_k (192)
  precision head h: [sg(mean_32 perceiver output) (128), u_k (192)] -> LayerNorm -> 128 -> GELU -> alpha_k (192), last
      layer zero-initialised (step 0 == precheck A's C_post)
  Lambda_k = pi_k D_k^1/2 G D_k^1/2, D_k = diag(exp(alpha_k)), G = inv(R_C L / hop) (R_C: precheck A pilot estimate, fixed)
  m = (Sig0^-1 + sum Lambda_k)^-1 (Sig0^-1 (mu0 + bM) + sum Lambda_k u_k)   (prior: Idea 2 full fit sha1 25aaf6cd)
  L = |m - u*|^2 + 0.1 mean_k |u_k - u*|^2

Data: Idea 2 train (640 streams). A sample = (unit = stream x true speaker, slot_for_speaker; t uniform in
{2, 4, 8, 15, 30, 60} s of that speaker's speech), rejection-sampled (pairs that do not reach t or have no own-slot
pi > 0 window available at tt + causal diarizer delay are redrawn); > 48 windows -> 48 uniformly without replacement,
pi * n / 48. 4-GPU DDP (gradients all-reduced by hand), 16 samples / GPU / step, 3000 steps, AdamW (wd 0) lr 2e-5
(encoder) / 1e-3 (head), 200 warm-up + cosine to 0, clip 1.0 per group, fp32 (posterior solve fp64), seed 20261119,
cuDNN SDPA off, non-finite step guard. Checkpoints (encoder + head + optimizer) at step 0, every 500 and at the end.

    torchrun --nproc_per_node 4 -m experiments.mist_w3po_v1.voicecond.idea6_e2e_train --out <dir>
Refuses an existing --out. Smoke flags: --steps --bs-per-gpu --ckpt-every.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_common import (DIAR_DELAY, I2, load_refs, load_stream, sids,  # noqa: E402
                                                             slot_for_speaker, speech_clock)

CKPT_C = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea1/m2/ctrl/encoder.pt"
FIT = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea2/shrink_fit_all/full.npz"
FIT_SHA1 = "25aaf6cd7f6a69b1e7972b947459d1217744e037"
RC_PATH = REPO / "eval_outputs/mist_w3po_v1/idea6_precheckA/R_C.npz"
PRE_ENC = REPO / "eval_outputs/mist_w3po_v1/idea6_precheckA/enc"
SR, HOP, L = 16000, 0.5, 2.0
WS = int(L * SR)
T_SET = (2, 4, 8, 15, 30, 60)
NMAX, W_ANC = 48, 0.1
LR_ENC, LR_HEAD, WARM, STEPS, BS_GPU, CLIP, SEED = 2e-5, 1e-3, 200, 3000, 16, 1.0, 20261119
ENC_NAMES = ("speaker_encoder", "perceiver_sampler", "project_in")


# ------------------------------------------------------------------------------------------------ model pieces
class PrecHead(nn.Module):
    """[sg(pooled perceiver output) 128, u_k 192] -> alpha_k 192; last layer zero -> alpha = 0 at init"""

    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(320)
        self.l1 = nn.Linear(320, 128)
        self.l2 = nn.Linear(128, 192)
        nn.init.zeros_(self.l2.weight)
        nn.init.zeros_(self.l2.bias)

    def forward(self, pooled, u):
        return self.l2(F.gelu(self.l1(self.ln(torch.cat([pooled.detach(), u], -1)))))


def enc_forward(P, ref):
    """same computation as BiCodecPhi._u_from_ref; ref (n, 96000) -> u (n, 192), pooled perceiver output (n, 128)"""
    mel = P.model.mel_transformer(ref).squeeze(1)
    _, feats = P.se.speaker_encoder(mel.transpose(1, 2), True)
    x = P.se.perceiver_sampler(feats.transpose(1, 2)).transpose(1, 2)      # (n, 128, 32)
    x = x.transpose(1, 2)                                                   # (n, 32, 128)
    return P.rq.project_in(x).reshape(len(ref), 192), x.mean(1)


def prior_terms(pf, RC):
    """numpy fp64: S0i, e0C = S0i (mu0 + bM), G = inv(R_C L / hop), prior mean"""
    S0i = np.linalg.inv(pf.Sig0)
    return S0i, S0i @ (pf.mu0 + pf.bM), np.linalg.inv(RC * (L / HOP)), pf.mu0 + pf.bM


def post_numpy(u, alpha, w, S0i, e0C, G, m0):
    """closed-form posterior mean for one unit (numpy fp64); u, alpha (n, 192), w (n,) pi"""
    if len(w) == 0 or w.sum() <= 0:
        return m0.copy()
    dd = np.exp(0.5 * alpha)
    wd = w[:, None] * dd
    A = S0i + G * (wd.T @ dd)
    rhs = e0C + (wd * ((dd * u) @ G)).sum(0)
    return np.linalg.solve(A, rhs)


def load_trained(P, ck):
    """encoder state of a training checkpoint (or C's encoder.pt) into BiCodecPhi P; returns the head (or None)"""
    for name, mod in (("speaker_encoder", P.se.speaker_encoder), ("perceiver_sampler", P.se.perceiver_sampler),
                      ("project_in", P.rq.project_in)):
        mod.load_state_dict(ck[name], strict=True)
    if "head" not in ck:
        return None
    h = PrecHead().to(P.device)
    h.load_state_dict(ck["head"])
    return h.eval()


# ------------------------------------------------------------------------------------------------ data
def build_units(ss, refs):
    units = []
    for sid in ss:
        d = load_stream(sid, L)
        meta = d["meta"]
        avail = d["end"] + DIAR_DELAY
        for j, spk in enumerate(meta["speakers"]):
            s = slot_for_speaker(d, j)
            tau, _ = speech_clock(meta, j)
            mine = np.nonzero((d["slot"] == s) & (d["pi"] > 0))[0]
            n_t = {}
            for t in T_SET:
                tt = tau(t)
                if tt is None:
                    continue
                kk = int(np.searchsorted(avail, tt, side="right")) - 1
                n_t[t] = int((mine <= kk).sum())       # usable windows are a prefix of mine
            units.append({"sid": sid, "spk": spk, "ustar": refs[spk][0].astype(np.float64), "mine": mine,
                          "pi": d["pi"][mine].astype(np.float64), "n_t": n_t})
    return units


def draw(units, rng):
    while True:
        i = int(rng.integers(len(units)))
        t = T_SET[int(rng.integers(len(T_SET)))]
        n = units[i]["n_t"].get(t, 0)
        if n > 0:
            break
    if n > NMAX:
        pick = np.sort(rng.choice(n, NMAX, replace=False))
        return i, t, pick, n / NMAX, n
    return i, t, np.arange(n), 1.0, n


class StepSampler(torch.utils.data.Sampler):
    def __init__(self, units, steps, bs, rank, start=0):
        self.units, self.steps, self.bs, self.rank, self.start = units, steps, bs, rank, start

    def __iter__(self):
        for st in range(self.start, self.steps):
            rng = np.random.default_rng([SEED, st, self.rank])
            yield [draw(self.units, rng) for _ in range(self.bs)]

    def __len__(self):
        return self.steps - self.start


def vnorm(x):
    from uniss.speech_tokenizer.bicodec.utils.audio import audio_volume_normalize as vn
    return vn(np.asarray(x, np.float64)).astype(np.float32)


class Windows(torch.utils.data.Dataset):
    def __init__(self, units):
        self.units = units

    def __len__(self):
        return len(self.units)

    def __getitem__(self, key):
        import soundfile as sf
        i, t, pick, scale, n = key
        u = self.units[i]
        k = u["mine"][pick]
        st = (k * HOP * SR).astype(int)
        lo, hi = int(st.min()), int(st.max()) + WS
        x = sf.read(str(I2 / "streams" / u["sid"] / "mix.wav"), dtype="float32", start=lo, stop=hi)[0]
        win = np.stack([vnorm(x[s - lo:s - lo + WS]) for s in st])
        return {"i": i, "t": t, "n": n, "win": torch.from_numpy(win), "pi": torch.from_numpy(u["pi"][pick] * scale),
                "ustar": torch.from_numpy(u["ustar"]), "k": k}


def collate(batch):
    return {"i": [b["i"] for b in batch], "t": [b["t"] for b in batch], "n": [b["n"] for b in batch],
            "k": [b["k"] for b in batch], "win": torch.cat([b["win"] for b in batch]),
            "wit": torch.cat([torch.full((len(b["win"]),), n, dtype=torch.long) for n, b in enumerate(batch)]),
            "pi": torch.cat([b["pi"] for b in batch]), "ustar": torch.stack([b["ustar"] for b in batch])}


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--bs-per-gpu", type=int, default=BS_GPU)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--enc-chunk", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-bad", type=int, default=10)
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    rank, world = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])) if ddp else (0, 1)
    lrank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lrank)
    dev = torch.device("cuda", lrank)
    if ddp:
        dist.init_process_group("nccl", device_id=dev)
    bs = a.bs_per_gpu
    out = Path(a.out)
    if rank == 0:
        if out.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {out}")
        out.mkdir(parents=True)
    if ddp:
        dist.barrier()
    logf = open(out / "train.log", "a", encoding="utf-8") if rank == 0 else None

    def log(s):
        if rank == 0:
            s = f"{time.strftime('%F %T')} {s}"
            print(s, flush=True)
            logf.write(s + "\n")
            logf.flush()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.enable_cudnn_sdp(False)      # Idea 2 e2e lesson (cuDNN SDPA backward NaN)
    log(f"[cfg] {json.dumps(vars(a))} world={world} bs_per_gpu={bs} global_bs={bs * world}")

    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea2_shrink import ShrinkParams
    sha = hashlib.sha1(FIT.read_bytes()).hexdigest()
    assert sha == FIT_SHA1, sha
    pf = ShrinkParams.load(FIT)
    RC = np.load(RC_PATH)["R_C"]
    S0i_n, e0C_n, G_n, m0_n = prior_terms(pf, RC)
    refs = load_refs()
    tr_sids = sids("all", "train")
    assert len(tr_sids) == 640 and all(s.startswith("train_") for s in tr_sids)
    units = build_units(tr_sids, refs)
    cnt = {t: sum(1 for u in units if u["n_t"].get(t, 0) > 0) for t in T_SET}
    reach = {t: sum(1 for u in units if t in u["n_t"]) for t in T_SET}
    log(f"[data] {len(tr_sids)} train streams, {len(units)} units; reach t {reach}; with >= 1 usable window {cnt}")

    P = BiCodecPhi(f"cuda:{lrank}")
    load_trained(P, torch.load(CKPT_C, map_location="cpu"))
    for p in P.model.parameters():
        p.requires_grad_(False)
    P.model.eval()
    enc_mods = {"speaker_encoder": P.se.speaker_encoder, "perceiver_sampler": P.se.perceiver_sampler,
                "project_in": P.rq.project_in}
    enc_params = [p for m in enc_mods.values() for p in m.parameters()]
    for p in enc_params:
        p.requires_grad_(True)
    torch.manual_seed(SEED)
    head = PrecHead().to(dev)
    head_params = list(head.parameters())
    all_params = enc_params + head_params
    log(f"[model] encoder params {sum(p.numel() for p in enc_params)}, head params {sum(p.numel() for p in head_params)}")
    t64 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64, device=dev)   # noqa: E731
    S0i, e0C, G = t64(S0i_n), t64(e0C_n), t64(G_n)

    opt = torch.optim.AdamW([{"params": enc_params, "lr": LR_ENC}, {"params": head_params, "lr": LR_HEAD}],
                            weight_decay=0.0)
    steps = a.steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: (st + 1) / WARM if st < WARM else 0.5 * (1 + math.cos(math.pi * (st - WARM) / max(1, steps - WARM))))

    def encode(win, grad=True):
        from torch.utils.checkpoint import checkpoint
        us, ps = [], []
        for i in range(0, len(win), a.enc_chunk):
            r = win[i:i + a.enc_chunk].repeat(1, 3)            # tiled to 6 s exactly as BiCodecPhi.ref_clip
            if grad:
                u, p = checkpoint(enc_forward, P, r, use_reentrant=False)
            else:
                with torch.no_grad():
                    u, p = enc_forward(P, r)
            us.append(u); ps.append(p)
        return torch.cat(us), torch.cat(ps)

    def posterior(u, alpha, pi, wit, B):
        d = torch.exp(0.5 * alpha.double())
        w = pi.double()
        wd = w[:, None] * d
        M = torch.zeros(B, 192, 192, dtype=torch.float64, device=dev).index_add(0, wit, wd[:, :, None] * d[:, None, :])
        Y = torch.zeros(B, 192, dtype=torch.float64, device=dev).index_add(0, wit, wd * ((d * u.double()) @ G))
        return torch.linalg.solve(S0i[None] + G[None] * M, e0C[None] + Y)

    def forward(bt, grad=True):
        win = bt["win"].to(dev, non_blocking=True)
        wit, pi, ustar = bt["wit"].to(dev), bt["pi"].to(dev), bt["ustar"].to(dev)
        B = len(bt["i"])
        u, pooled = encode(win, grad)
        alpha = head(pooled, u)
        m = posterior(u, alpha, pi, wit, B)
        post = ((m - ustar) ** 2).sum(1)
        cnt = torch.bincount(wit, minlength=B).double()
        anc = torch.zeros(B, dtype=torch.float64, device=dev).index_add(0, wit, ((u.double() - ustar[wit]) ** 2).sum(1)) / cnt
        return {"loss": post.mean() + W_ANC * anc.mean(), "post": post.mean(), "anc": anc.mean(), "u": u, "alpha": alpha,
                "m": m, "err": post.sqrt(), "nwin": len(win)}

    def save(step, final=False):
        if rank != 0:
            return
        ck = {name: m.state_dict() for name, m in enc_mods.items()}
        ck.update({"step": step, "head": head.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                   "fit_sha1": FIT_SHA1, "R_C": RC, "args": vars(a), "world": world, "bs": bs})
        p = out / f"ckpt_step{step:05d}.pt"
        assert not p.exists()
        torch.save(ck, out / f".{p.name}.partial")
        os.rename(out / f".{p.name}.partial", p)
        log(f"[ckpt] {p}{' (final)' if final else ''}")

    @torch.no_grad()
    def init_check(n=6):
        """step 0: trainer u_k == precheck A's stored u_C (pilot train streams) and torch posterior == numpy C_post"""
        rng = np.random.default_rng([SEED, 999999])
        pil = [i for i, u in enumerate(units) if (PRE_ENC / f"{u['sid']}.npz").exists()]
        batch = []
        while len(batch) < n:
            i, t, pick, scale, nn_ = draw(units, rng)
            if i in pil:
                batch.append(Windows(units)[(i, t, pick, scale, nn_)])
        bt = collate(batch)
        r = forward(bt, grad=False)
        du, dm = 0.0, 0.0
        o = 0
        for b, (i, k) in enumerate(zip(bt["i"], bt["k"])):
            uC = np.load(PRE_ENC / f"{units[i]['sid']}.npz")["uC"][k].astype(np.float64)
            uk = r["u"][o:o + len(k)].double().cpu().numpy()
            du = max(du, float(np.abs(uk - uC).max()))
            w = bt["pi"][o:o + len(k)].double().numpy()
            mC = np.linalg.solve(S0i_n + w.sum() * G_n, e0C_n + G_n @ (w[:, None] * uC).sum(0))
            dm = max(dm, float(np.abs(r["m"][b].cpu().numpy() - mC).max()))
            o += len(k)
        return {"u_maxabs_vs_precheckA": du, "post_maxabs_vs_numpy_C_post": dm,
                "alpha_maxabs": float(r["alpha"].abs().max()), "n_windows": r["nwin"]}

    log("[init_check] " + json.dumps(init_check()))
    save(0)
    if rank == 0:
        jl = open(out / "train_log.jsonl", "x", encoding="utf-8")
    dl = torch.utils.data.DataLoader(Windows(units), batch_sampler=StepSampler(units, steps, bs, rank),
                                     collate_fn=collate, num_workers=a.workers, prefetch_factor=4, pin_memory=True)
    n_bad, n_bad_run = 0, 0
    t_start = t_last = time.time()
    step = 0
    for bt in dl:
        t_data = time.time() - t_last
        torch.cuda.reset_peak_memory_stats(dev)
        r = forward(bt, grad=True)
        opt.zero_grad(set_to_none=True)
        r["loss"].backward()
        for p in all_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        if ddp:
            flat = torch.cat([p.grad.reshape(-1) for p in all_params])
            dist.all_reduce(flat)
            flat /= world
            o = 0
            for p in all_params:
                p.grad.copy_(flat[o:o + p.numel()].view_as(p.grad))
                o += p.numel()
        gn_enc = float(torch.nn.utils.clip_grad_norm_(enc_params, CLIP))
        gn_head = float(torch.nn.utils.clip_grad_norm_(head_params, CLIP))
        al = r["alpha"].detach().double()
        terms = torch.stack([r["loss"].detach(), r["post"].detach(), r["anc"].detach(), r["err"].detach().mean(),
                             al.mean(), al.std(), al.mean(1).std()])
        if ddp:
            dist.all_reduce(terms)
            terms /= world
        ok = math.isfinite(gn_enc) and math.isfinite(gn_head) and bool(torch.isfinite(terms).all())
        if not ok:
            n_bad, n_bad_run = n_bad + 1, n_bad_run + 1
            log(f"[nonfinite] step {step + 1}: update skipped (total {n_bad}, consecutive {n_bad_run}); grad_norm enc "
                f"{gn_enc} head {gn_head}; terms {terms.tolist()}")
            opt.zero_grad(set_to_none=True)
            if n_bad_run >= a.max_bad:
                raise SystemExit(f"{n_bad_run} consecutive non-finite steps: abort")
        else:
            n_bad_run = 0
            opt.step()
        sched.step()
        step += 1
        now = time.time()
        if rank == 0:
            rec = {"step": step, "loss": float(terms[0]), "post": float(terms[1]), "anc": float(terms[2]),
                   "err_mean": float(terms[3]), "alpha_mean": float(terms[4]), "alpha_std": float(terms[5]),
                   "alpha_winmean_std": float(terms[6]), "grad_norm_enc": gn_enc, "grad_norm_head": gn_head,
                   "skipped": int(not ok), "n_skipped": n_bad, "lr_enc": sched.get_last_lr()[0],
                   "lr_head": sched.get_last_lr()[1], "nwin": r["nwin"], "s_step": round(now - t_last, 3),
                   "s_data": round(t_data, 3), "s_avg": round((now - t_start) / step, 3),
                   "peak_gib": round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)}
            jl.write(json.dumps(rec) + "\n")
            jl.flush()
            if step <= 10 or step % 10 == 0:
                log(f"[step {step}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                  for k, v in rec.items() if k != "step"))
        t_last = now
        if step % a.ckpt_every == 0 or step == steps:
            save(step, final=step == steps)
            t_last = time.time()
    if rank == 0:
        (out / "DONE").write_text(json.dumps({"steps": step, "minutes": (time.time() - t_start) / 60,
                                              "n_skipped": n_bad}) + "\n")
        log(f"[done] {step} steps in {(time.time() - t_start) / 60:.1f} min")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
