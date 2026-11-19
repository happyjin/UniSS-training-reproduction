"""Robust wave 6 (reports/mist_w3po_v1/robust_wave6_disen_prereg.zh-CN.md): DisenBooth-style identity / irrelevant
voice-condition disentanglement for the T1c Talker, and its plain LoRA control T-ft. One script, --arm disen|tft.

Rows: the UniST pilot subset (data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source, the open corpus T1c
was trained on; T1c's own shards were not restored after the server move): source audio, source GLM tokens, translation,
target BiCodec semantic codes. Rows are ranked by sha1("20261119|w6|id"), kept if 2 s <= source <= 12 s, 1 <= codes <= 600
and the translation is non-empty, and drawn direction-balanced (16 en->zh + 16 zh->en a step). Both arms read the very same
rows, crops and backgrounds (all randomness keyed by row id and step).

Sequence: T1c's exactly (training/si_batch.py streaming_row with block speeds, or e2_batch offline_row), si share 0.7 as
T1c. Blocks of an si row (the word-time index T1c used is not on this server): 1-4 blocks cut at word (en) / character
(zh) boundaries at equal unit counts, codes cut in proportion, written at the matching share of the source + 200 ms, GLM
up to that time less a holdback of 12 tokens; the last block takes the rest. Labels: every code and closing token, plain
CE (no teacher, no KL, no anchor).

Voice: g = 32 BiCodec global ids from the Idea 1 C encoder (voicecond_idea1/m2/ctrl/encoder.pt) on prompt A = the source
up to L ~ U(1, min(6, source)) s; with p = 0.8 mixed with a non-held-out background (wave1_data.mix_one over the Idea 2
"train" background split, BVR quantile q ~ U(0.05, 0.95) of the benchmark's BVR values), else clean. disen also uses
prompt B = another crop (>= 1 s, from the second half when long enough) of the same utterance with an independent
background draw (UniST carries no speaker ids, so the positive is the same speaker in another stretch and background).

disen:  c_id = sigmoid(m) * E(g) at the 32 voice positions (m: 1536, init 3.0, sigma 0.953); c_irr = h_psi(log-mel of
        prompt A): 4 soft vectors inserted after <end_global>, training only.
        L = NLL(x | c_id, c_irr) + 1.0 NLL(x | c_id) + 0.1 InfoNCE(cbar_id(A), cbar_id(B); tau 0.07, in-batch negatives,
            symmetric) + 0.1 cos^2(cbar_id(A), cbar_irr(A))
tft:    L = NLL(x | E(g)).
Trainable: LoRA r 16 alpha 32 on q/k/v/o of all 28 layers (lr 1e-4); m and h_psi (lr 1e-3, new modules). AdamW (wd 0),
100 warm-up steps then cosine, 3000 steps, global batch 32 (2 GPUs x 16), grad clip 1.0, bf16 base, fp32 adapters,
seed 20261021. Saves lora.pt (+ m, h_psi) at steps 1000, 2000 and the end; only the last is evaluated.

    torchrun --nproc_per_node 2 experiments/mist_w3po_v1/voicecond/wave6_disen_train.py --arm disen|tft --out <dir>
        [--steps 3000] [--smoke]
    python experiments/mist_w3po_v1/voicecond/wave6_disen_train.py --arm tft --out x --select-only   (row cache, CPU)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.uniss_qwen3_omni_v1.training.e2_batch import (END_GLOBAL, END_SEMANTIC, GLOBAL_0, LANG3,  # noqa: E402
                                                                speed_token)
from experiments.uniss_qwen3_omni_v1.training.si_batch import (WAIT_READ, block_speeds, offline_row,  # noqa: E402
                                                               streaming_row)

MAN = REPO / "data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source/stage_a_source_manifest.jsonl"
TALKER = REPO / "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss"
CENC = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea1/m2/ctrl/encoder.pt"
QWEN3 = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
SR = 16000
N_VOICE = 32
VOICE_AT = 4            # [TASK_S2S, BALANCE, lang, START_GLOBAL] then 32 voice ids, then END_GLOBAL at 36
IRR_AT = VOICE_AT + N_VOICE + 1
N_IRR = 4
SEED = 20261021


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


# ---------------------------------------------------------------- rows
def select_rows(n_per_dir: int, cache: Path) -> list[int]:
    """manifest line numbers: sha1-ranked, filtered, n_per_dir per direction (cached as json)."""
    if cache.exists():
        return json.load(open(cache))
    offs = np.fromfile(str(MAN) + ".offsets.bin", dtype="<u8")
    order = sorted(range(len(offs)), key=lambda i: h(f"20261119|w6|{i}"))
    out, got = [], {"eng": 0, "cmn": 0}
    with open(MAN, "rb") as fh:
        for i in order:
            fh.seek(int(offs[i]))
            r = json.loads(fh.readline())
            tgt = r["tgt_lang"]
            if got.get(tgt, n_per_dir) >= n_per_dir:
                if all(v >= n_per_dir for v in got.values()):
                    break
                continue
            if not (2000 <= r["source_duration_ms"] <= 12000 and 1 <= len(r["target_bicodec"]) <= 600
                    and r["translation"].strip() and len(r["source_glm"]) > 0 and len(r["bicodec_global"]) == 32):
                continue
            out.append(i)
            got[tgt] += 1
    cache.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(cache, "w"))
    return out


class Rows:
    def __init__(self):
        self.offs = np.fromfile(str(MAN) + ".offsets.bin", dtype="<u8")
        self.fh = None

    def get(self, i: int) -> dict:
        if self.fh is None:
            self.fh = open(MAN, "rb")
        self.fh.seek(int(self.offs[i]))
        return json.loads(self.fh.readline())


def audio_path(r: dict) -> str:
    p = r["source_audio"]
    if os.path.exists(p):
        return p
    hits = [x for x in os.listdir(os.path.dirname(p)) if x.startswith(os.path.basename(p) + "_")]
    assert len(hits) == 1, p
    return os.path.join(os.path.dirname(p), hits[0])


_POOLS = {}


def mixer():
    if not _POOLS:
        from experiments.mist_w3po_v1.voicecond import wave1_data as W1
        _POOLS["W1"], _POOLS["pools"], _POOLS["bv"] = W1, W1.bg_pools("train"), W1.bvr_values()
    return _POOLS


def maybe_noisy(x: np.ndarray, rng: random.Random) -> tuple[np.ndarray, float | None]:
    if rng.random() >= 0.8:
        return x.astype(np.float32), None
    P = mixer()
    m = P["W1"].mix_one(x.astype(np.float32), rng, P["pools"], P["bv"], with_context=False)
    return m["noisy"], m["bvr_db"]


def units(text: str, tgt: str):
    """unit start char offsets: words (en) or non-space characters (zh)"""
    if tgt == "cmn":
        return [i for i, c in enumerate(text) if not c.isspace()]
    return [m.start() for m in re.finditer(r"\S+", text)]


def prop_blocks(r: dict, rng: random.Random) -> list[dict]:
    text, tgt = r["translation"], r["tgt_lang"]
    st = units(text, tgt)
    n, n_codes, n_glm, src = len(st), len(r["target_bicodec"]), len(r["source_glm"]), int(r["source_duration_ms"])
    nb = min(rng.choice([1, 2, 3, 4]), max(1, n))

    def boundary(w: int) -> int:
        if w >= n:
            return len(text)
        b = st[w]
        while b > 0 and text[b - 1].isspace():
            b -= 1
        return b
    cuts = sorted({round(k * n / nb) for k in range(1, nb)} - {0, n})
    out, c_prev, g_prev, done = [], 0, 0, 0
    for c in cuts:
        code_end = round(n_codes * c / n)
        if code_end <= c_prev or code_end >= n_codes:
            continue
        w = min(src, round(src * c / n) + 200)
        g = max(g_prev, max(0, min(n_glm, round(w / 80)) - 12))
        out.append({"write_ms": w, "glm_end": g, "code_end": code_end, "text": text[boundary(done) if done else 0: boundary(c)],
                    "final": False})
        done, c_prev, g_prev = c, code_end, g
    out.append({"write_ms": src, "glm_end": n_glm, "code_end": n_codes,
                "text": text[boundary(done) if done else 0:], "final": True})
    return out


class DS(torch.utils.data.Dataset):
    """index = step * 32 + slot; slots 0..15 en->zh, 16..31 zh->en; rank r reads slots [16r, 16r + 16)."""

    def __init__(self, rows: list[int], steps: int):
        self.rows = {"cmn": [], "eng": []}
        R = Rows()
        for i in rows:
            self.rows[R.get(i)["tgt_lang"]].append(i)
        self.steps, self.R = steps, None

    def __len__(self):
        return self.steps * 32

    def __getitem__(self, idx: int) -> dict:
        if self.R is None:
            self.R = Rows()
        step, slot = divmod(idx, 32)
        tgt = "cmn" if slot < 16 else "eng"
        pool = self.rows[tgt]
        li = pool[(step * 16 + slot % 16) % len(pool)]
        r = self.R.get(li)
        rng = random.Random(h(f"{SEED}|w6|{li}|{step}"))
        x, sr = sf.read(audio_path(r), dtype="float32")
        assert sr == SR
        if x.ndim > 1:
            x = x.mean(1)
        T = len(x) / SR
        L = rng.uniform(1.0, min(6.0, T))
        a = x[: int(L * SR)]
        LB = rng.uniform(1.0, min(6.0, T))
        lo = max(0.0, min(T - LB, T / 2 if T - LB >= T / 2 else rng.uniform(0, T - LB)))
        b = x[int(lo * SR): int((lo + LB) * SR)]
        a_n, bvr_a = maybe_noisy(a, rng)
        b_n, _ = maybe_noisy(b, rng)
        kind = "si" if rng.random() < 0.7 else "offline"
        return {"li": li, "tgt": LANG3[r["tgt_lang"]], "glm": r["source_glm"], "codes": r["target_bicodec"],
                "text": r["translation"], "src_s": r["source_duration_ms"] / 1000, "kind": kind,
                "blocks": prop_blocks(r, rng) if kind == "si" else None, "a": a_n, "b": b_n, "bvr_a": bvr_a}


def collate(s):
    return s


# ---------------------------------------------------------------- model pieces
class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (F.linear(F.linear(x.float(), self.A), self.B) * self.scale).to(x.dtype)


def add_lora(model) -> dict:
    mods = {}
    for li, layer in enumerate(model.model.layers):
        at = layer.self_attn
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lo = LoRA(getattr(at, nm)).to(getattr(at, nm).weight.device)
            setattr(at, nm, lo)
            mods[f"{li}.{nm}"] = lo
    return mods


class IrrEnc(nn.Module):
    """h_psi: log-mel (80) -> 2 conv layers -> 4 learned queries attend -> 4 x d_model soft vectors"""

    def __init__(self, d_model: int, hid: int = 256):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(80, hid, 5, padding=2), nn.GELU(), nn.Conv1d(hid, hid, 5, stride=2, padding=2),
                                  nn.GELU())
        self.q = nn.Parameter(torch.randn(N_IRR, hid) * 0.02)
        self.att = nn.MultiheadAttention(hid, 4, batch_first=True)
        self.out = nn.Linear(hid, d_model)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def forward(self, mel, mask):   # mel (B, 80, T), mask (B, T) True = pad
        z = self.conv(mel).transpose(1, 2)
        m = mask[:, ::2][:, : z.shape[1]]
        o, _ = self.att(self.q[None].expand(z.shape[0], -1, -1), z, z, key_padding_mask=m)
        return self.out(o)


class Disen(nn.Module):
    """everything trainable besides the LoRA: the mask m and h_psi"""

    def __init__(self, d_model: int):
        super().__init__()
        self.m = nn.Parameter(torch.full((d_model,), 3.0))
        self.irr = IrrEnc(d_model)


# ---------------------------------------------------------------- batch
def build(samples, gids: list[list[int]], tok, device):
    """sequences (T1c format) with the given voice ids; labels as (row, position, label)"""
    seqs, lb, lp, ll = [], [], [], []
    for i, s in enumerate(samples):
        codes = list(s["codes"])
        if s["kind"] == "si":
            blocks = s["blocks"]
            ids = tok([b["text"] for b in blocks], add_special_tokens=False)["input_ids"]
            seq, labels = streaming_row(s["tgt"], gids[i], s["glm"], blocks, ids, codes, block_speeds(blocks, len(codes)))
            for pos, lab, _ in labels:
                lb.append(i); lp.append(pos); ll.append(lab)
        else:
            text_ids = tok(s["text"], add_special_tokens=False)["input_ids"]
            seq, start, target = offline_row(s["tgt"], gids[i], s["glm"], text_ids, codes, speed_token(s["src_s"], len(codes)))
            for k, lab in enumerate(target):
                lb.append(i); lp.append(start + k); ll.append(lab)
        assert seq[VOICE_AT - 1] != END_GLOBAL and seq[VOICE_AT + N_VOICE] == END_GLOBAL
        seqs.append(seq)
    n = max(len(s) for s in seqs)
    ids = torch.zeros((len(seqs), n), dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.as_tensor(s)
        att[i, : len(s)] = 1
    t = lambda v: torch.as_tensor(v, dtype=torch.long, device=device)  # noqa: E731
    return ids.to(device), att.to(device), t(lb), t(lp), t(ll)


def nll(model, emb, att, lb, lp, ll, shift: int = 0):
    out = model.model(inputs_embeds=emb, attention_mask=att)
    hsel = out.last_hidden_state[lb, lp + shift]
    logits = model.lm_head(hsel).float()
    return F.cross_entropy(logits, ll)


def voice_embeds(model, ids, disen: Disen | None):
    emb = model.model.embed_tokens(ids)
    if disen is not None:
        sig = torch.sigmoid(disen.m).to(emb.dtype)
        emb = torch.cat([emb[:, :VOICE_AT], emb[:, VOICE_AT: VOICE_AT + N_VOICE] * sig, emb[:, VOICE_AT + N_VOICE:]], 1)
    return emb


def logmel(wavs, device, mel_fn):
    n = max(len(w) for w in wavs)
    x = torch.zeros((len(wavs), n), device=device)
    mask_s = torch.ones((len(wavs), n), dtype=torch.bool, device=device)
    for i, w in enumerate(wavs):
        x[i, : len(w)] = torch.as_tensor(w, device=device)
        mask_s[i, : len(w)] = False
    m = torch.log(mel_fn(x) + 1e-5)                         # (B, 80, F)
    F_ = m.shape[-1]
    pad = mask_s[:, ::160][:, :F_]
    if pad.shape[1] < F_:
        pad = torch.cat([pad, pad[:, -1:].expand(-1, F_ - pad.shape[1])], 1)
    m = m.masked_fill(pad[:, None], 0.0)
    return m, pad


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["disen", "tft"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true", help="20 steps, no saving except a log")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--select-only", action="store_true", help="build the row cache (CPU) and exit")
    a = ap.parse_args()
    cache = REPO / "eval_outputs/mist_w3po_v1/wave6_disen/rows_3000x32.json"
    if a.select_only:
        rows = select_rows(3000 * 16, cache)
        print(f"rows {len(rows)} -> {cache}")
        return
    import torch.distributed as dist
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2, "global batch 32 = 2 GPUs x 16"
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dev = torch.device("cuda", torch.cuda.current_device())
    out = Path(a.out)
    if rank == 0:
        if (out / "lora_final.pt").exists():
            raise SystemExit(f"拒绝覆盖已有输出: {out}")
        out.mkdir(parents=True, exist_ok=True)
    steps = 20 if a.smoke else a.steps
    torch.manual_seed(SEED)

    rows = json.load(open(cache))   # built beforehand with --select-only

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(QWEN3))
    model = AutoModelForCausalLM.from_pretrained(str(TALKER), dtype=torch.bfloat16).to(dev)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    loras = add_lora(model)
    d_model = model.config.hidden_size
    disen = Disen(d_model).to(dev) if a.arm == "disen" else None
    lora_params = [p for m in loras.values() for p in (m.A, m.B)]
    groups = [{"params": lora_params, "lr": 1e-4}]
    if disen is not None:
        groups.append({"params": list(disen.parameters()), "lr": 1e-3})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    base_lr = [g["lr"] for g in groups]

    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    P = load_encoder(BiCodecPhi(str(dev)), str(CENC))
    import torchaudio
    mel_fn = torchaudio.transforms.MelSpectrogram(SR, n_fft=400, hop_length=160, n_mels=80).to(dev)

    ds = DS(rows, steps)
    idx = [s * 32 + 16 * rank + k for s in range(steps) for k in range(16)]
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=16, shuffle=False, num_workers=a.workers,
                                     collate_fn=collate, prefetch_factor=4, persistent_workers=True)
    trainable = lora_params + (list(disen.parameters()) if disen is not None else [])
    logf = open(out / f"train_{'smoke' if a.smoke else 'main'}_r{rank}.jsonl", "a")
    t0 = time.time()
    for step, samples in enumerate(dl):
        lr_mult = (step + 1) / 100 if step < 100 else 0.5 * (1 + math.cos(math.pi * (step - 100) / max(1, steps - 100)))
        for g, b in zip(opt.param_groups, base_lr):
            g["lr"] = b * lr_mult
        ga = [list(map(int, v)) for v in P.tokens(P.phi_wave([s["a"] for s in samples]))]
        ids, att, lb, lp, ll = build(samples, ga, tok, dev)
        rec = {"step": step}
        if disen is None:
            loss = nll(model, voice_embeds(model, ids, None), att, lb, lp, ll)
            rec["nll"] = float(loss)
        else:
            emb = voice_embeds(model, ids, disen)
            mel, pad = logmel([s["a"] for s in samples], dev, mel_fn)
            c_irr = disen.irr(mel, pad).to(emb.dtype)                                   # (B, 4, d)
            emb_irr = torch.cat([emb[:, :IRR_AT], c_irr, emb[:, IRR_AT:]], 1)
            att_irr = torch.cat([att[:, :IRR_AT], torch.ones_like(att[:, :N_IRR]), att[:, IRR_AT:]], 1)
            l_full = nll(model, emb_irr, att_irr, lb, lp, ll, shift=N_IRR)
            l_id = nll(model, emb, att, lb, lp, ll)
            gb = [list(map(int, v)) for v in P.tokens(P.phi_wave([s["b"] for s in samples]))]
            sig = torch.sigmoid(disen.m)
            E = model.model.embed_tokens.weight
            ca = (E[GLOBAL_0 + torch.as_tensor(ga, device=dev)].float() * sig).mean(1)
            cb = (E[GLOBAL_0 + torch.as_tensor(gb, device=dev)].float() * sig).mean(1)
            za, zb = F.normalize(ca, dim=-1), F.normalize(cb, dim=-1)
            sim = za @ zb.T / 0.07
            tgt = torch.arange(len(samples), device=dev)
            l_nce = 0.5 * (F.cross_entropy(sim, tgt) + F.cross_entropy(sim.T, tgt))
            l_orth = (F.cosine_similarity(ca, c_irr.float().mean(1), dim=-1) ** 2).mean()
            loss = l_full + 1.0 * l_id + 0.1 * l_nce + 0.1 * l_orth
            rec.update(nll_full=float(l_full), nll_id=float(l_id), nce=float(l_nce), orth=float(l_orth),
                       sig_mean=float(sig.mean()), sig_min=float(sig.min()))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p in trainable:   # DDP-free all-reduce of the (few) trainable grads
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        rec.update(loss=float(loss), gnorm=float(gn), lr=opt.param_groups[0]["lr"], n_lab=int(ll.numel()),
                   seq=int(ids.shape[1]), sec=round(time.time() - t0, 2), mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if rank == 0 and (step % 20 == 0 or step == steps - 1):
            print(json.dumps(rec), flush=True)
        if rank == 0 and not a.smoke and ((step + 1) % 1000 == 0 or step + 1 == steps):
            sd = {"lora": {k: {"A": m.A.detach().cpu(), "B": m.B.detach().cpu()} for k, m in loras.items()},
                  "disen": disen.state_dict() if disen is not None else None, "arm": a.arm, "step": step + 1,
                  "r": 16, "alpha": 32}
            torch.save(sd, out / (f"lora_s{step + 1:05d}.pt"))
            if step + 1 == steps:
                torch.save(sd, out / "lora_final.pt")
    if rank == 0:
        (out / ("SMOKE_DONE" if a.smoke else "TRAIN_DONE")).write_text(json.dumps({"steps": steps, "sec": time.time() - t0}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
