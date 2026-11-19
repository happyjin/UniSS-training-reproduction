"""Idea 2 end-to-end training (reports/mist_w3po_v1/voicecond_idea2_e2e_prereg.zh-CN.md v1, §1-§6): window encoder
phi_theta (BiCodec ECAPA + Perceiver + project_in, full fine-tune, BN / dropout eval) -> lambda / log r head h_omega
(5 -> 32 -> 2 MLP on v~, initialised to the fitted alpha + beta^T v~ and log r = 0) -> closed-form contraction posterior
(c_lang trainable from the full fit sha1 25aaf6cd; R, Sigma_0, mu_0, b_M fixed) -> u_hat -> FSQ tokens g(u_hat) with the
straight-through linear bridge e = E[g] + (q_c - sg(q_c)) W^T -> T1c Talker + LoRA (r 16, alpha 32, q_proj / v_proj).

  L = KL(p0(.|g*) || p_psi(.|g(u_hat))) + KL(p0(.|g*) || p_psi(.|g*)) + w_code |u_hat - u*|^2 + w_anc E_clean |phi_theta(s) - phi_0(s)|^2

KLs: teacher-forced on the cached T1c(g*) semantic codes (cvt teacher_out, --dump-seq train_ids / label_from), per position
over the 8192 semantic codes only, mean over positions per sequence, mean over sequences. Weight rule (prereg §3, fixed):
w_code / w_anc = mean(KL1) / mean(term) over the first 50 batches (global means over all ranks), then frozen; during
those 50 batches the two weights are not yet defined and are 0 (implementation detail, see the LOG entry).

Windows of item i (segment k of stream sid, true speaker j): the diarizer slot matched to j (slot_for_speaker), windows
of that slot with end + 1.04 s <= segment start + write_ms (the inference rule of idea2_stream_tokens.py); items without
any usable window (inference falls back to V tokens) are excluded. Clean anchor s: one random clean 6 s prompt
(cvt/refs/<spk>/p*.wav) of the item's speaker per sequence; phi_0(s) from a frozen copy of the encoder in the same
step (exactly 0 at initialisation).

Setting (prereg §6): 4-GPU DDP (gradients all-reduced by hand), 64 sequences / step (16 / GPU), Talker bf16 with gradient
checkpointing (encoder and posterior in fp32 / fp64 so that at initialisation u_hat equals the frozen shrink method),
AdamW (weight decay 0), lr 1e-4 (LoRA, h_omega, c) / 5e-5 (encoder), 200 warm-up steps, cosine to 0, 3000 steps, clip 1.0,
seed 20261119. Checkpoints of the trainable parts only (LoRA, head, c_lang, encoder, W, weights, optimizer) every 500
steps and at the end; dev (cvt dev: both KLs, code error, token recovery; Idea 2 synthetic pilot dev: code error and
token recovery with all slot windows) every 250 steps -- logging only, no selection.

--ablation (prereg §4, same data / steps, NOT launched yet): attnpool (2-layer attention pooling regresses u_hat from the
windows), constlam (lambda and log r trainable constants, init from the const fit), frozen_talker (no LoRA; KL2 == 0).

    torchrun --nproc_per_node 4 -m experiments.mist_w3po_v1.voicecond.idea2_e2e_train --out <dir> [--ablation main]
Refuses an existing --out. Smoke flags: --steps --bs-per-gpu --calib --dev-every --ckpt-every --dev-limit.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_common import DIAR_DELAY, slot_for_speaker  # noqa: E402
from experiments.mist_w3po_v1.voicecond.idea2_shrink import ShrinkParams  # noqa: E402

T1C = REPO / "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss"
I2 = REPO / "data/open_voicecond_v1/idea2"
CV = I2 / "cvt"
FIT = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea2/shrink_fit_all"
FIT_SHA1 = "25aaf6cd7f6a69b1e7972b947459d1217744e037"
GLOBAL_0, SEMANTIC_0, START_GLOBAL, END_GLOBAL = 151665, 155761, 180387, 180391
SR, HOP, L = 16000, 0.5, 2.0
WS = int(L * SR)
TRAIN_SETS, DEV_SETS = ("zh2en_train", "en2zh_train"), ("zh2en_dev", "en2zh_dev")
LR_MAIN, LR_ENC, WARM, STEPS, BS, CLIP, SEED, CALIB = 1e-4, 5e-5, 200, 3000, 64, 1.0, 20261119, 50


# ------------------------------------------------------------------------------------------------ data
def latest_refs(root):
    k, p = 2, root / "refs/ustar.npz"
    while (root / f"refs/ustar_v{k}.npz").exists():
        p, k = root / f"refs/ustar_v{k}.npz", k + 1
    z = np.load(p, allow_pickle=True)
    return {str(s): (u.reshape(-1).astype(np.float64), np.asarray(g)) for s, u, g in zip(z["spk"], z["ustar"], z["gstar"])}


def load_stream(root, sid):
    meta = json.load(open(root / "streams" / sid / "meta.json"))
    w = np.load(root / "pre/win_L2" / f"{sid}.npz")
    f = np.load(root / "pre/feats_L2" / f"{sid}.npz")
    vt = np.load(root / "pre/feats2_L2" / f"{sid}.npz")["vt"]
    K = min(len(w["end_s"]), len(f["end_s"]), len(vt))
    return {"meta": meta, "end": w["end_s"][:K].astype(np.float64), "slot": f["slot"][:K], "pi": f["pi"][:K],
            "slot2spk": f["slot2spk"], "ov": f["ov"], "vt": vt[:K].astype(np.float64), "z_mix": w["z_mix"][:K]}


def build_items(sets, pf, refs, log):
    """teacher records -> training / dev items (windows by the inference rule); items without windows are dropped"""
    items, cache, dropped = [], {}, {}
    for s in sets:
        parts = sorted(glob.glob(str(CV / "teacher_out" / s / "part_*.jsonl")))
        n0 = len(items)
        for p in parts:
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                assert r["error"] is None and len(r["blocks"]) == 1, r["id"]
                b = r["blocks"][0]
                sid, k = r["id"].rsplit("__", 1)
                if sid not in cache:
                    cache[sid] = load_stream(CV, sid)
                d = cache[sid]
                seg = d["meta"]["segments"][int(k)]
                j = d["meta"]["speakers"].index(seg["spk"])
                sl = slot_for_speaker(d, j)
                t = seg["start"] + b["write_ms"] / 1000.0
                sel = np.nonzero((d["slot"] == sl) & (d["end"] + DIAR_DELAY <= t + 1e-9))[0]
                if len(sel) == 0:
                    dropped[s] = dropped.get(s, 0) + 1
                    continue
                ids = b["train_ids"]
                lf, nc = int(b["label_from"]), len(b["code_ids"])
                assert ids[3] == START_GLOBAL and ids[36] == END_GLOBAL and len(ids) == lf + nc + 1, r["id"]
                u, g = refs[seg["spk"]]
                assert [x - GLOBAL_0 for x in ids[4:36]] == [int(x) for x in g], r["id"]
                lang = d["meta"]["lang"]
                items.append({"id": r["id"], "set": s, "ids": ids, "lf": lf, "nc": nc, "sid": sid, "sel": sel,
                              "vt": ((d["vt"][sel] - pf.vt_mean) / pf.vt_std).astype(np.float32),
                              "pi": d["pi"][sel].astype(np.float32), "lang": lang, "spk": seg["spk"], "ustar": u,
                              "gstar": np.asarray(g, np.int64), "z0": d["z_mix"][sel].reshape(len(sel), -1)})
        log(f"[data] {s}: {len(items) - n0} items ({len(parts)} parts), dropped without usable windows: {dropped.get(s, 0)}")
    return items, dropped


def vnorm_windows(x, starts):
    from uniss.speech_tokenizer.bicodec.utils.audio import audio_volume_normalize as vn
    return np.stack([vn(np.asarray(x[s:s + WS], np.float64)) for s in starts]).astype(np.float32)


class Items(torch.utils.data.Dataset):
    """item -> windows (n, 32000; volume-normalised exactly as BiCodecPhi.ref_clip, tiled to 6 s on the GPU), anchor"""

    def __init__(self, items, root, prompts=None, seed=SEED):
        self.items, self.root, self.prompts, self.seed = items, root, prompts, seed
        self.mix = OrderedDict()

    def __len__(self):
        return len(self.items)

    def _mix(self, sid):
        if sid not in self.mix:
            import soundfile as sf
            self.mix[sid] = sf.read(str(self.root / "streams" / sid / "mix.wav"), dtype="float32")[0]
            while len(self.mix) > 12:
                self.mix.popitem(last=False)
        return self.mix[sid]

    def __getitem__(self, key):
        i, step = key if isinstance(key, tuple) else (key, 0)
        it = self.items[i]
        x = self._mix(it["sid"])
        out = {"i": i, "win": torch.from_numpy(vnorm_windows(x, (it["sel"] * HOP * SR).astype(int)))}
        if self.prompts is not None:
            ps = self.prompts[it["spk"]]
            k = int(np.random.default_rng([self.seed, step, i]).integers(len(ps)))
            out["anc"], out["anc_key"] = ps[k]
        return out


def collate(batch, items):
    B = len(batch)
    Lm = max(len(items[b["i"]]["ids"]) for b in batch)
    ids = torch.zeros(B, Lm, dtype=torch.long)
    mask = torch.zeros(B, Lm, dtype=torch.long)
    bi, pos, wit = [], [], []
    for n, b in enumerate(batch):
        it = items[b["i"]]
        ids[n, :len(it["ids"])] = torch.tensor(it["ids"])
        mask[n, :len(it["ids"])] = 1
        bi += [n] * it["nc"]
        pos += list(range(it["lf"] - 1, it["lf"] - 1 + it["nc"]))
        wit += [n] * len(it["sel"])
    out = {"idx": [b["i"] for b in batch], "ids": ids, "mask": mask, "bi": torch.tensor(bi), "pos": torch.tensor(pos),
           "win": torch.cat([b["win"] for b in batch]), "wit": torch.tensor(wit),
           "vt": torch.from_numpy(np.concatenate([items[b["i"]]["vt"] for b in batch])),
           "pi": torch.from_numpy(np.concatenate([items[b["i"]]["pi"] for b in batch])),
           "lang": torch.tensor([0 if items[b["i"]]["lang"] == "en" else 1 for b in batch]),
           "ustar": torch.from_numpy(np.stack([items[b["i"]]["ustar"] for b in batch])),
           "gstar": torch.from_numpy(np.stack([items[b["i"]]["gstar"] for b in batch]))}
    if "anc" in batch[0]:
        out["anc"] = torch.from_numpy(np.stack([b["anc"] for b in batch]))
        out["anc_key"] = [b["anc_key"] for b in batch]
    return out


class StepSampler(torch.utils.data.Sampler):
    """global shuffle per epoch (seed + epoch); step s takes perm[64 s: 64 (s + 1)], rank r its slice of bs_gpu"""

    def __init__(self, n, steps, bs, rank, world, start=0):
        self.n, self.steps, self.bs, self.rank, self.world, self.start = n, steps, bs, rank, world, start

    def __iter__(self):
        per = self.n // (self.bs * self.world)
        for st in range(self.start, self.steps):
            ep, k = divmod(st, per)
            g = torch.Generator()
            g.manual_seed(SEED + ep)
            perm = torch.randperm(self.n, generator=g)
            b = perm[(k * self.world + self.rank) * self.bs:(k * self.world + self.rank + 1) * self.bs]
            yield [(int(i), st) for i in b]

    def __len__(self):
        return self.steps - self.start


# ------------------------------------------------------------------------------------------------ modules
class LamHead(nn.Module):
    """h_omega: v~ (normalised, 5) -> [lambda logit, log r]; init exactly alpha + beta^T v~ and 0"""

    def __init__(self, alpha, beta):
        super().__init__()
        self.l1, self.l2 = nn.Linear(5, 32), nn.Linear(32, 2)
        g = torch.Generator().manual_seed(SEED)
        with torch.no_grad():
            self.l1.weight.copy_(torch.randn(32, 5, generator=g) * 1e-3)
            self.l1.bias.zero_()
            self.l1.weight[:5] = torch.eye(5)
            self.l1.weight[5:10] = -torch.eye(5)
            self.l2.weight.zero_()
            self.l2.weight[0, :5] = torch.as_tensor(beta, dtype=torch.float32)
            self.l2.weight[0, 5:10] = -torch.as_tensor(beta, dtype=torch.float32)
            self.l2.bias.zero_()
            self.l2.bias[0] = float(alpha)

    def forward(self, vt):
        o = self.l2(torch.relu(self.l1(vt)))
        return o[:, 0], o[:, 1]


class ConstHead(nn.Module):
    """ablation 2: lambda and log r trainable constants (no v~)"""

    def __init__(self, alpha):
        super().__init__()
        self.a, self.r = nn.Parameter(torch.tensor(float(alpha))), nn.Parameter(torch.tensor(0.0))

    def forward(self, vt):
        n = vt.shape[0]
        return self.a.expand(n), self.r.expand(n)


class AttnPool(nn.Module):
    """ablation 1: black-box 2-layer attention pooling of the windows [z, v~, log pi] -> u_hat (= mu0 + b_M at init)"""

    def __init__(self, base):
        super().__init__()
        self.inp = nn.Linear(192 + 5 + 1, 256)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, 512, dropout=0.0, batch_first=True), 2,
                                         enable_nested_tensor=False)
        self.q = nn.Parameter(torch.randn(1, 1, 256) * 0.02)
        self.pool = nn.MultiheadAttention(256, 4, batch_first=True)
        self.out = nn.Linear(256, 192)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("base", torch.as_tensor(base, dtype=torch.float32))

    def forward(self, z, vt, pi, wit, B):
        x = self.inp(torch.cat([z.float(), vt, torch.log(pi.clamp_min(1e-4))[:, None]], 1))
        n = torch.bincount(wit, minlength=B)
        X = x.new_zeros(B, int(n.max()), 256)
        pad = torch.ones(B, int(n.max()), dtype=torch.bool, device=x.device)
        off = torch.cumsum(n, 0) - n
        j = torch.arange(len(wit), device=x.device) - off[wit]
        X[wit, j] = x
        pad[wit, j] = False
        h = self.enc(X, src_key_padding_mask=pad)
        p, _ = self.pool(self.q.expand(B, 1, 256), h, h, key_padding_mask=pad)
        return self.out(p[:, 0]).double() + self.base.double()


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ablation", default="main", choices=["main", "attnpool", "constlam", "frozen_talker"])
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--bs-per-gpu", type=int, default=0, help="default 64 / world size")
    ap.add_argument("--calib", type=int, default=CALIB)
    ap.add_argument("--dev-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--dev-limit", type=int, default=0, help="smoke only: first n dev items per set")
    ap.add_argument("--synth-dev", type=int, default=1)
    ap.add_argument("--enc-chunk", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--load-check", default="", help="smoke only: load this checkpoint and compare dev outputs")
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    rank, world = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])) if ddp else (0, 1)
    lrank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lrank)
    dev = torch.device("cuda", lrank)
    if ddp:
        dist.init_process_group("nccl", device_id=dev)
    bs = a.bs_per_gpu or BS // world
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
    log(f"[cfg] {json.dumps(vars(a))} world={world} bs_per_gpu={bs} global_bs={bs * world}")
    assert hashlib.sha1((FIT / "full.npz").read_bytes()).hexdigest() == FIT_SHA1
    pf = ShrinkParams.load(FIT / "full.npz")
    pc = ShrinkParams.load(FIT / "const.npz")
    p0 = pc if a.ablation == "constlam" else pf
    assert pf.per_lang and abs(pf.L - L) < 1e-9
    refs = latest_refs(CV)
    tr_items, tr_drop = build_items(TRAIN_SETS, pf, refs, log)
    dv_items, dv_drop = build_items(DEV_SETS, pf, refs, log)
    if a.dev_limit:
        dv_items = [it for s in DEV_SETS for it in [x for x in dv_items if x["set"] == s][:a.dev_limit]]

    # ---- models
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    P = BiCodecPhi(f"cuda:{lrank}")
    for p in P.model.parameters():
        p.requires_grad_(False)
    P.model.eval()
    enc_mods = {"speaker_encoder": P.se.speaker_encoder, "perceiver_sampler": P.se.perceiver_sampler,
                "project_in": P.rq.project_in}
    enc_params = [p for m in enc_mods.values() for p in m.parameters()]
    import copy
    enc0 = {k: copy.deepcopy(m).eval() for k, m in enc_mods.items()}       # frozen phi_0 for the anchor term
    for p in enc_params:
        p.requires_grad_(True)

    @torch.no_grad()
    def phi0(ref):
        """frozen phi_0, same computation as BiCodecPhi._u_from_ref; run in the same step on the same batch as phi_theta,
        so that the anchor term is exactly 0 at initialisation (no batch-composition numerical floor)"""
        mel = P.model.mel_transformer(ref).squeeze(1)
        _, feats = enc0["speaker_encoder"](mel.transpose(1, 2), True)
        x = enc0["perceiver_sampler"](feats.transpose(1, 2)).transpose(1, 2).transpose(1, 2)
        return enc0["project_in"](x)

    # frozen phi_0 of the clean anchor prompts (cvt/refs/<spk>/p*.wav) of the training speakers
    import soundfile as sf
    spks = sorted({it["spk"] for it in tr_items})
    prompts, keys, wavs = {}, [], []
    for s in spks:
        dd = CV / "refs" / s.replace(":", "_")
        n = json.load(open(dd / "info.json"))["n_prompts"]
        prompts[s] = []
        for k in range(n):
            w = P.ref_clip(sf.read(str(dd / f"p{k}.wav"), dtype="float32")[0]).astype(np.float32)
            prompts[s].append((w, len(keys)))
            keys.append((s, k))
            wavs.append(w)
    with torch.no_grad():
        A0 = torch.cat([P._u_from_ref(torch.from_numpy(np.stack(wavs[i:i + 256])).to(dev)).reshape(-1, 192)
                        for i in range(0, len(wavs), 256)])
    log(f"[anchor] {len(keys)} clean prompts of {len(spks)} training speakers, phi_0 precomputed")

    model = AutoModelForCausalLM.from_pretrained(str(T1C), torch_dtype=torch.bfloat16).to(dev)
    model.config.use_cache = False
    emb = model.get_input_embeddings()
    E = emb.weight[GLOBAL_0:GLOBAL_0 + 4096].detach().float()
    Q = torch.as_tensor(P.codes(np.arange(4096)), dtype=torch.float32, device=dev).reshape(4096, -1)
    X = torch.cat([Q, torch.ones(4096, 1, device=dev)], 1)
    Wb = torch.linalg.lstsq(X, E).solution
    r2 = float(1 - ((E - X @ Wb) ** 2).sum() / ((E - E.mean(0)) ** 2).sum())
    Wst = Wb[:6].T.contiguous().to(torch.bfloat16)                    # (1536, 6)
    log(f"[bridge] linear surrogate R^2 = {r2:.4f}")
    use_lora = a.ablation != "frozen_talker"
    if use_lora:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
        body, head_w = model.base_model.model.model, model.base_model.model.lm_head.weight
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        body, head_w = model.model, model.lm_head.weight
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    lm_sem = head_w[SEMANTIC_0:SEMANTIC_0 + 8192].detach()
    lora_params = [p for p in model.parameters() if p.requires_grad]
    log(f"[talker] LoRA params {sum(p.numel() for p in lora_params)}; encoder params {sum(p.numel() for p in enc_params)}")

    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64, device=dev)   # noqa: E731
    Rfi, S0i = t(np.linalg.inv(pf.R * (pf.L / HOP))), t(np.linalg.inv(pf.Sig0))
    mu0, bM = t(pf.mu0), t(pf.bM)
    c_lang = nn.Parameter(torch.stack([t(p0.c["en"]), t(p0.c["zh"])]).float())      # trainable c_en, c_zh
    if a.ablation == "attnpool":
        head = AttnPool((pf.mu0 + pf.bM)).to(dev)
    elif a.ablation == "constlam":
        head = ConstHead(pc.alpha).to(dev)
    else:
        head = LamHead(pf.alpha, pf.beta).to(dev)
    head_params = list(head.parameters()) + ([c_lang] if a.ablation != "attnpool" else [])
    main_params = lora_params + head_params
    all_params = main_params + enc_params
    opt = torch.optim.AdamW([{"params": main_params, "lr": LR_MAIN}, {"params": enc_params, "lr": LR_ENC}],
                            weight_decay=0.0)
    steps = a.steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: (st + 1) / WARM if st < WARM else 0.5 * (1 + math.cos(math.pi * (st - WARM) / max(1, steps - WARM))))

    def phi(win, grad=True):
        """volume-normalised 2 s windows (n, 32000) -> z (n, 192); tiled to 6 s exactly as ref_clip"""
        from torch.utils.checkpoint import checkpoint
        outs = []
        for i in range(0, len(win), a.enc_chunk):
            r = win[i:i + a.enc_chunk].repeat(1, 3)
            if grad:
                outs.append(checkpoint(P._u_from_ref, r, use_reentrant=False))
            else:
                with torch.no_grad():
                    outs.append(P._u_from_ref(r))
        return torch.cat(outs).reshape(len(win), 192)

    def posterior(z, vt, pi, wit, lang, B):
        """closed-form contraction posterior per item -> u_hat (B, 192) fp64, lambda, log r"""
        if a.ablation == "attnpool":
            return head(z, vt, pi, wit, B), None, None
        lg, lr_ = head(vt)
        lam, er = torch.sigmoid(lg).double(), torch.exp(-lr_.double())
        c = c_lang.double()[lang[wit]]
        pid = pi.double()
        y = z.double() - (1 - lam)[:, None] * c
        A = torch.zeros(B, dtype=torch.float64, device=dev).index_add(0, wit, pid * lam * lam * er)
        Y = torch.zeros(B, 192, dtype=torch.float64, device=dev).index_add(0, wit, (pid * lam * er)[:, None] * y)
        prec = S0i[None] + A[:, None, None] * Rfi[None]
        rhs = (S0i @ mu0)[None] + Y @ Rfi.T
        return torch.linalg.solve(prec, rhs) + bM, lam, lr_

    def st_embed(u):
        uf = u.float().reshape(-1, 32, 6)
        qc = P.fsq.bound(uf) / 2
        with torch.no_grad():
            _, g = P.fsq(uf.detach())
            g = g.reshape(-1, 32).long()
        e = emb.weight[GLOBAL_0 + g] + ((qc - qc.detach()).to(torch.bfloat16) @ Wst.T)
        return e, g

    def sem_logp(embeds, mask, bi, pos):
        hs = body(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        return torch.log_softmax((hs[bi, pos] @ lm_sem.T).float(), -1)

    def seq_kl(lt, ls, bi, B):
        k = (lt.exp() * (lt - ls)).sum(-1)
        n = torch.bincount(bi, minlength=B).double()
        return (torch.zeros(B, dtype=torch.float64, device=dev).index_add(0, bi, k.double()) / n)

    PROF = os.environ.get("IDEA2_PROF") == "1"
    prof = {}

    def tick(name):
        if PROF:
            torch.cuda.synchronize(dev)
            prof[name] = time.time()

    def forward(bt, train=True):
        tick("t0")
        ids, mask = bt["ids"].to(dev), bt["mask"].to(dev)
        bi, pos = bt["bi"].to(dev), bt["pos"].to(dev)
        B = ids.shape[0]
        win = bt["win"].to(dev, non_blocking=True)
        wit, vt, pi = bt["wit"].to(dev), bt["vt"].to(dev), bt["pi"].to(dev)
        lang, ustar, gstar = bt["lang"].to(dev), bt["ustar"].to(dev), bt["gstar"].to(dev)
        z = phi(win, grad=train)
        tick("phi")
        u, lam, lr_ = posterior(z, vt, pi, wit, lang, B)
        base = emb(ids)
        with torch.no_grad():
            if use_lora:
                with model.disable_adapter():
                    lt = sem_logp(base, mask, bi, pos)
            else:
                lt = sem_logp(base, mask, bi, pos)
        tick("teacher")
        e_st, g = st_embed(u)
        stu = base.clone()
        stu[:, 4:36] = e_st
        M = len(bi)
        if use_lora:
            ls = sem_logp(torch.cat([stu, base]), torch.cat([mask, mask]), torch.cat([bi, bi + B]), torch.cat([pos, pos]))
            k1, k2 = seq_kl(lt, ls[:M], bi, B), seq_kl(lt, ls[M:], bi, B)
        else:
            ls = sem_logp(stu, mask, bi, pos)
            k1 = seq_kl(lt, ls, bi, B)
            k2 = torch.zeros_like(k1)
        tick("student")
        csq = ((u - ustar) ** 2).sum(1)
        tok = (g == gstar).double().mean(1)
        res = {"kl1": k1.mean(), "kl2": k2.mean(), "code": csq.mean(), "tok": tok.mean(), "u": u, "lam": lam, "logr": lr_,
               "nwin": len(win), "seq": {"kl1": k1, "kl2": k2, "code": csq, "tok": tok}}
        if train and "anc" in bt:
            za = P._u_from_ref(bt["anc"].to(dev)).reshape(B, 192)
            key = torch.tensor(bt["anc_key"], device=dev)
            za0 = phi0(bt["anc"].to(dev)).reshape(B, 192)
            res["anc"] = ((za - za0) ** 2).sum(1).mean().double()
            res["anc_vs_pre"] = float(((za0 - A0[key]) ** 2).sum(1).mean())
        return res

    # ---- dev
    synth = []
    if a.synth_dev:
        srefs = latest_refs(I2)
        for sid in [f"dev_{lg}_{i:04d}" for lg in ("en", "zh") for i in range(4)]:
            d = load_stream(I2, sid)
            for j, spk in enumerate(d["meta"]["speakers"]):
                sl = slot_for_speaker(d, j)
                sel = np.nonzero(d["slot"] == sl)[0]
                if len(sel):
                    synth.append({"sid": sid, "sel": sel, "vt": ((d["vt"][sel] - pf.vt_mean) / pf.vt_std).astype(np.float32),
                                  "pi": d["pi"][sel].astype(np.float32), "lang": d["meta"]["lang"], "ustar": srefs[spk][0],
                                  "gstar": np.asarray(srefs[spk][1], np.int64)})
        log(f"[dev] Idea 2 synthetic pilot dev: {len(synth)} units (all slot windows to stream end)")
    log(f"[dev] cvt dev items: {len(dv_items)}")

    @torch.no_grad()
    def run_dev(step):
        t0 = time.time()
        acc = torch.zeros(2, 5, dtype=torch.float64, device=dev)    # per direction: n, kl1, kl2, code, tok
        mine = dv_items[rank::world]
        ds = Items(mine, CV)
        for i in range(0, len(mine), bs):
            bt = collate([ds[k] for k in range(i, min(i + bs, len(mine)))], mine)
            r = forward(bt, train=False)
            per = r["seq"]
            dirs = torch.tensor([0 if mine[k]["set"].startswith("zh2en") else 1 for k in range(i, min(i + bs, len(mine)))],
                                device=dev)
            for dd in (0, 1):
                m = dirs == dd
                if m.any():
                    acc[dd] += torch.stack([m.sum().double(), per["kl1"][m].sum(), per["kl2"][m].sum(),
                                            per["code"][m].sum(), per["tok"][m].sum()])
        if ddp:
            dist.all_reduce(acc)
        rec = {"step": step, "dev": {}}
        for dd, name in enumerate(("zh2en", "en2zh")):
            n = float(acc[dd, 0])
            if n:
                rec["dev"][name] = {"n": int(n), "kl1": float(acc[dd, 1] / n), "kl2": float(acc[dd, 2] / n),
                                    "code_sq": float(acc[dd, 3] / n), "tok": float(acc[dd, 4] / n)}
        if synth:
            sacc = torch.zeros(3, dtype=torch.float64, device=dev)
            dsy = Items(synth, I2)
            for k in range(rank, len(synth), world):
                it = synth[k]
                bt = collate_units([dsy[k]], synth)
                z = phi(bt["win"].to(dev), grad=False)
                u, _, _ = posterior(z, bt["vt"].to(dev), bt["pi"].to(dev), bt["wit"].to(dev), bt["lang"].to(dev), 1)
                sacc += torch.stack([torch.tensor(1.0, dtype=torch.float64, device=dev),
                                     ((u[0] - t(it["ustar"])) ** 2).sum(),
                                     (st_tokens(u)[0] == torch.as_tensor(it["gstar"], device=dev)).double().mean()])
            if ddp:
                dist.all_reduce(sacc)
            rec["synth_dev"] = {"n": int(sacc[0]), "code_sq": float(sacc[1] / sacc[0]), "tok": float(sacc[2] / sacc[0])}
        rec["sec"] = round(time.time() - t0, 1)
        return rec

    def st_tokens(u):
        _, g = P.fsq(u.float().reshape(-1, 32, 6))
        return g.reshape(-1, 32).long()

    def collate_units(batch, units):
        b = batch[0]
        it = units[b["i"]]
        return {"win": b["win"], "wit": torch.zeros(len(it["sel"]), dtype=torch.long), "vt": torch.from_numpy(it["vt"]),
                "pi": torch.from_numpy(it["pi"]), "lang": torch.tensor([0 if it["lang"] == "en" else 1])}

    def save(step, final=False):
        if rank != 0:
            return
        ck = {"step": step, "ablation": a.ablation, "fit_sha1": FIT_SHA1, "W_bridge": Wst.float().cpu(), "r2_bridge": r2,
              "w_code": wts["code"], "w_anc": wts["anc"], "calib": wts["calib"],
              "encoder": {k: m.state_dict() for k, m in enc_mods.items()}, "head": head.state_dict(),
              "c_lang": {"en": c_lang.detach()[0].cpu(), "zh": c_lang.detach()[1].cpu()},
              "lora": get_peft_model_state_dict(model) if use_lora else None,
              "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "args": vars(a), "world": world, "bs": bs}
        p = out / f"ckpt_step{step:05d}.pt"
        assert not p.exists()
        torch.save(ck, out / f".{p.name}.partial")
        os.rename(out / f".{p.name}.partial", p)
        log(f"[ckpt] {p}{' (final)' if final else ''}")

    def load_check(path):
        """roundtrip: load a checkpoint into the live modules and verify parameters match exactly"""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        diffs = {}
        for k, m in enc_mods.items():
            sd = m.state_dict()
            diffs[f"enc.{k}"] = max(float((sd[n].cpu().float() - v.float()).abs().max()) for n, v in ck["encoder"][k].items())
        sd = head.state_dict()
        diffs["head"] = max(float((sd[n].cpu() - v).abs().max()) for n, v in ck["head"].items())
        diffs["c"] = float((c_lang.detach().cpu() - torch.stack([ck["c_lang"]["en"], ck["c_lang"]["zh"]])).abs().max())
        if use_lora:
            cur = get_peft_model_state_dict(model)
            diffs["lora"] = max(float((cur[n].cpu().float() - v.float()).abs().max()) for n, v in ck["lora"].items())
            diffs["lora_keys"] = [len(cur), len(ck["lora"])]
            res = set_peft_model_state_dict(model, ck["lora"])
            diffs["lora_unexpected_on_reload"] = len(res.unexpected_keys)
        diffs["optimizer_groups"] = len(ck["optimizer"]["param_groups"])
        return diffs

    # ---- training
    wts = {"code": 0.0, "anc": 0.0, "calib": None}
    dev_rec = run_dev

    @torch.no_grad()
    def init_check(n=4):
        """at initialisation phi_theta(window) must equal the stored frozen z_mix and u_hat the numpy shrink posterior"""
        ds0 = Items(dv_items[:n], CV)
        bt = collate([ds0[k] for k in range(n)], dv_items[:n])
        z = phi(bt["win"].to(dev), grad=False)
        z0 = torch.from_numpy(np.concatenate([it["z0"] for it in dv_items[:n]])).to(dev)
        u, _, _ = posterior(z, bt["vt"].to(dev), bt["pi"].to(dev), bt["wit"].to(dev), bt["lang"].to(dev), n)
        du = []
        for k, it in enumerate(dv_items[:n]):
            vt = it["vt"].astype(np.float64) * pf.vt_std + pf.vt_mean
            lam = pf.lam(vt)
            y = it["z0"].astype(np.float64) - (1 - lam)[:, None] * pf.cvec(it["lang"])[None]
            w = it["pi"].astype(np.float64)
            Rf_i, S0_i = np.linalg.inv(pf.R * (pf.L / HOP)), np.linalg.inv(pf.Sig0)
            m = np.linalg.solve(S0_i + (w * lam * lam).sum() * Rf_i, S0_i @ pf.mu0 + Rf_i @ ((w * lam)[:, None] * y).sum(0))
            du.append(float(np.abs(u[k].cpu().numpy() - (m + pf.bM)).max()))
        return {"z_maxabs": float((z - z0).abs().max()), "z_scale": float(z0.abs().mean()), "u_maxabs_vs_numpy": max(du)}

    if a.ablation == "main":
        log("[init_check] " + json.dumps(init_check()))
    rec0 = dev_rec(0)
    log("[dev] " + json.dumps(rec0))
    dev_kl1_0 = {k: v["kl1"] for k, v in rec0["dev"].items()}
    if rank == 0:
        jl = open(out / "train_log.jsonl", "x", encoding="utf-8")
        jl.write(json.dumps(rec0) + "\n")
        jl.flush()
    ds = Items(tr_items, CV, prompts)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=StepSampler(len(tr_items), steps, bs, rank, world),
                                     collate_fn=lambda b: collate(b, tr_items), num_workers=a.workers,
                                     prefetch_factor=4, pin_memory=True, persistent_workers=False)
    calib = torch.zeros(3, dtype=torch.float64, device=dev)
    t_start, t_last = time.time(), time.time()
    step = 0
    for bt in dl:
        t_data = time.time() - t_last
        torch.cuda.reset_peak_memory_stats(dev)
        r = forward(bt, train=True)
        loss = r["kl1"] + r["kl2"] + wts["code"] * r["code"] + wts["anc"] * r["anc"]
        opt.zero_grad(set_to_none=True)
        tick("fwd_end")
        loss.backward()
        tick("bwd")
        if PROF and rank == 0:
            ks = ["t0", "phi", "teacher", "student", "fwd_end", "bwd"]
            log("[prof] " + " ".join(f"{b}={prof[b] - prof[a]:.2f}" for a, b in zip(ks, ks[1:])) + f" nwin={r['nwin']}")
        for p in all_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        if ddp:
            flat = torch.cat([p.grad.reshape(-1).float() for p in all_params])
            dist.all_reduce(flat)
            flat /= world
            o = 0
            for p in all_params:
                n = p.numel()
                p.grad.copy_(flat[o:o + n].view_as(p.grad))
                o += n
        # clip 1.0 per parameter group (Talker side: LoRA + head + c; encoder): a global clip would let the encoder's
        # gradient (incl. the anchor term) scale the LoRA gradients below Adam's eps
        gn = float(torch.nn.utils.clip_grad_norm_(main_params, CLIP))
        gn_enc = float(torch.nn.utils.clip_grad_norm_(enc_params, CLIP))
        opt.step()
        sched.step()
        step += 1
        terms = torch.stack([r["kl1"].detach(), r["kl2"].detach(), r["code"].detach(), r["anc"].detach(),
                             r["tok"].detach().double(), loss.detach().double()])
        if ddp:
            dist.all_reduce(terms)
            terms /= world
        if step <= a.calib:
            calib += terms[[0, 2, 3]]
            if step == a.calib:
                m = calib / a.calib
                wts["code"] = float(m[0] / m[1])
                wts["anc"] = float(m[0] / m[2]) if float(m[2]) > 0 else float("inf")
                wts["calib"] = {"kl1_mean": float(m[0]), "code_mean": float(m[1]), "anc_mean": float(m[2]), "batches": a.calib}
                log(f"[weights] fixed after {a.calib} batches: w_code = {wts['code']:.6g}, w_anc = {wts['anc']:.6g} "
                    f"({json.dumps(wts['calib'])})")
                if not math.isfinite(wts["anc"]):
                    raise SystemExit("anchor mean over the calibration batches is 0: weight rule undefined")
        now = time.time()
        if rank == 0:
            rec = {"step": step, "loss": float(terms[5]), "kl1": float(terms[0]), "kl2": float(terms[1]),
                   "code": float(terms[2]), "anc": float(terms[3]), "tok": float(terms[4]),
                   "w_code": wts["code"], "w_anc": wts["anc"], "grad_norm": gn, "grad_norm_enc": gn_enc,
                   "lr": sched.get_last_lr()[0], "lr_enc": sched.get_last_lr()[1], "nwin": r["nwin"],
                   "anc_phi0_vs_precomputed": r["anc_vs_pre"],
                   "s_step": round(now - t_last, 3), "s_data": round(t_data, 3), "s_avg": round((now - t_start) / step, 3),
                   "peak_gib": round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)}
            if r["lam"] is not None:
                lam, lr_ = r["lam"].detach().float(), r["logr"].detach().float()
                rec.update({"lam_mean": float(lam.mean()), "lam_p10": float(lam.quantile(0.1)),
                            "lam_p90": float(lam.quantile(0.9)), "logr_mean": float(lr_.mean()), "logr_std": float(lr_.std())})
            jl.write(json.dumps(rec) + "\n")
            jl.flush()
            if step <= 10 or step % 10 == 0:
                log(f"[step {step}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                  for k, v in rec.items() if k != "step"))
        t_last = now
        if step % a.dev_every == 0 or step == steps:
            rd = dev_rec(step)
            if step >= 1000 and step - a.dev_every < 1000 <= step:
                rd["kl1_rel_drop_vs_step0"] = {k: 1 - v["kl1"] / dev_kl1_0[k] for k, v in rd["dev"].items()}
            log("[dev] " + json.dumps(rd))
            if rank == 0:
                jl.write(json.dumps(rd) + "\n")
                jl.flush()
            t_last = time.time()
        if step % a.ckpt_every == 0 or step == steps:
            save(step, final=step == steps)
            t_last = time.time()
    if a.load_check and rank == 0:
        log("[load_check] " + json.dumps(load_check(a.load_check)))
    if rank == 0:
        (out / "DONE").write_text(json.dumps({"steps": step, "minutes": (time.time() - t_start) / 60, "weights": wts}) + "\n")
        log(f"[done] {step} steps in {(time.time() - t_start) / 60:.1f} min")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
