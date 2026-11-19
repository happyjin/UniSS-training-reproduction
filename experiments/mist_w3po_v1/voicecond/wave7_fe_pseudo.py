"""Robust wave 7 (reports/mist_w3po_v1/robust_wave7_taskfe_prereg.zh-CN.md): task-driven pseudo-clean targets.

x* = PGD_Linf(x_clean): 3 signed-gradient steps of size eps / 3 DOWN the Thinker's cross-entropy of the reference
translation, eps = 0.1 x RMS(x_clean) (whole clip), projected to the eps-ball around x_clean and to [-0.99, 0.99];
deterministic start at x_clean. Thinker = the base Qwen3-Omni-30B-A3B-Instruct Thinker (HF, bf16, eval, frozen);
prompt = si_thinker_stream's base prompt (E1 instruction of the target language, chat template with the audio),
target = reference + <|im_end|> (wave5_m_train's labels); loss of a microbatch = sum over items of each item's mean CE,
so every item's sign-gradient is its own. The gradient reaches the waveform through wave7_fe_common.logmel_torch, which
replaces the processor's (numpy) input_features; at step 0 the two are compared and the max |diff| is logged.
Labels (reference translations of CoVoST 2 train) are the only supervision; no model output is a target.

Units: (dir, shard) of wave7/FE/data/train_<dir>.jsonl, 16 shards per direction; workers claim units with O_EXCL files
(claims/), write pseudo/<dir>/<id>.wav (float32) and runs/<dir>_s<k>.jsonl ({id, dir, eps, ce_clean, ce_star,
mel_maxdiff}) via .partial -> rename; never overwrite.

    CUDA_VISIBLE_DEVICES=<card> python -I experiments/mist_w3po_v1/voicecond/wave7_fe_pseudo.py --worker k [--mb 4]
    python -I .../wave7_fe_pseudo.py --device cpu --smoke 2 --crop 1.5      (CPU smoke: 2 sentences, nothing kept
                                                                             beyond smoke/)
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from experiments.mist_w3po_v1.voicecond.wave7_fe_common import (IGNORE, IM_END, IM_START, OUT, QWEN3, SR, TGT,  # noqa: E402
                                                               logmel_torch)

NSH = 16
STEPS, EPS_REL = 3, 0.1

try:    # CPU smoke with no visible CUDA device: this torch build raises instead of returning None, transformers' import trips
    torch.accelerator.current_accelerator()
except RuntimeError:
    torch.accelerator.current_accelerator = lambda *a, **k: None


def log(m: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}", flush=True)


def load_thinker(dev: str):
    from transformers import Qwen3OmniMoeForConditionalGeneration
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as M
    omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(str(QWEN3), torch_dtype=torch.bfloat16,
                                                                device_map={"": dev}, attn_implementation="sdpa")
    th = omni.thinker
    for attr in ("talker", "code2wav", "token2wav"):
        if hasattr(omni, attr):
            delattr(omni, attr)
    del omni
    th.requires_grad_(False)
    th.eval()
    if dev == "cpu":   # the fused expert kernel is CUDA-only: the eager loop everywhere on CPU
        eager = M.Qwen3OmniMoeThinkerTextExperts.forward.__wrapped__
        for lay in th.model.layers:
            ex = lay.mlp.experts
            ex.forward = types.MethodType(eager, ex)
    return th


class Prompt:
    def __init__(self):
        from transformers import Qwen3OmniMoeProcessor
        from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
        self.proc = Qwen3OmniMoeProcessor.from_pretrained(str(QWEN3))
        self.proc.tokenizer.padding_side = "right"
        ins = inference_instruction()
        self.head = {}
        for d in ("zh2en", "en2zh"):
            conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": ins[TGT[d]]}]}]
            self.head[d] = self.proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]

    def batch(self, d: str, refs: list[str], wavs: list[np.ndarray]) -> dict:
        texts = [self.head[d] + r + "<|im_end|>" for r in refs]
        inp = self.proc(text=texts, audio=wavs, return_tensors="pt", padding=True)
        ids = inp["input_ids"]
        lab = torch.full_like(ids, IGNORE)
        for b in range(ids.shape[0]):
            row = ids[b].tolist()
            st = max(i for i, t in enumerate(row) if t == IM_START) + 3
            en = max(i for i, t in enumerate(row) if t == IM_END)
            assert en > st, "empty target"
            lab[b, st: en + 1] = ids[b, st: en + 1]
        return {**dict(inp), "labels": lab}


def item_ce(th, inp: dict, feats: torch.Tensor, dev: str) -> torch.Tensor:
    """per-item mean CE over the target tokens, [B]"""
    kw = {k: v.to(dev) for k, v in inp.items() if k not in ("labels", "input_features")}
    out = th(**kw, input_features=feats.to(dev))
    logits = out.logits[:, :-1].float()
    lab = inp["labels"][:, 1:].to(dev)
    ce = torch.nn.functional.cross_entropy(logits.transpose(1, 2), lab.clamp_min(0), reduction="none")
    m = (lab != IGNORE).float()
    return (ce * m).sum(1) / m.sum(1)


def pgd(th, P: Prompt, d: str, rows: list[dict], wavs: list[np.ndarray], dev: str):
    inp = P.batch(d, [r["ref"] for r in rows], wavs)
    x0 = [torch.from_numpy(w).to(dev) for w in wavs]
    eps = [EPS_REL * float(torch.sqrt((w.double() ** 2).mean())) for w in x0]
    feats0, att = logmel_torch(x0)
    diff = float((feats0.cpu() - inp["input_features"].float()).abs().max())
    assert torch.equal(att.cpu(), inp["feature_attention_mask"].long()), "feature attention mask differs from the processor"
    x = [w.clone() for w in x0]
    ce_clean = None
    for k in range(STEPS):
        for w in x:
            w.requires_grad_(True)
        f, _ = logmel_torch(x)
        ce = item_ce(th, inp, f, dev)
        if k == 0:
            ce_clean = ce.detach().cpu().tolist()
        g = torch.autograd.grad(ce.sum(), x)
        with torch.no_grad():
            x = [torch.clamp(torch.minimum(torch.maximum(w - (e / 3) * gi.sign(), w0 - e), w0 + e), -0.99, 0.99)
                 for w, gi, w0, e in zip(x, g, x0, eps)]
    with torch.no_grad():
        f, _ = logmel_torch(x)
        ce_star = item_ce(th, inp, f, dev).cpu().tolist()
    return [w.detach().cpu().numpy().astype(np.float32) for w in x], eps, ce_clean, ce_star, diff


def run_unit(th, P, d, k, dev, mb, outdir: Path, rows_all):
    rows = [r for i, r in enumerate(rows_all) if i % NSH == k]
    rows.sort(key=lambda r: r["dur"])     # length-sorted microbatches (less padding); order is deterministic
    part = outdir / "runs" / f"{d}_s{k}.jsonl.partial"
    final = outdir / "runs" / f"{d}_s{k}.jsonl"
    if final.exists():
        return
    part.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if part.exists():
        done = {json.loads(l)["id"] for l in open(part, encoding="utf-8")}
    t0 = time.time()
    with open(part, "a", encoding="utf-8") as fh:
        for i in range(0, len(rows), mb):
            chunk = [r for r in rows[i: i + mb] if r["id"] not in done]
            if not chunk:
                continue
            wavs = [sf.read(str(REPO / r["audio"]), dtype="float32")[0] for r in chunk]
            xs, eps, c0, c1, diff = pgd(th, P, d, chunk, wavs, dev)
            for r, x, e, a, b in zip(chunk, xs, eps, c0, c1):
                p = outdir / "pseudo" / d / f"{r['id']}.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                if not p.exists():
                    sf.write(str(p), x, SR, subtype="FLOAT")
                fh.write(json.dumps({"id": r["id"], "dir": d, "eps": round(e, 7), "ce_clean": round(a, 5),
                                     "ce_star": round(b, 5), "mel_maxdiff": diff, "pseudo": str(p.relative_to(REPO))}) + "\n")
            fh.flush()
            if (i // mb) % 20 == 0:
                log(f"{d} s{k}: {i + len(chunk)}/{len(rows)} ({(time.time() - t0) / max(1, i + len(chunk)):.2f} s/sent)")
    os.rename(part, final)
    log(f"{d} s{k} done in {time.time() - t0:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--crop", type=float, default=0.0)
    a = ap.parse_args()
    dev = a.device
    P = Prompt()
    data = {d: [json.loads(l) for l in open(OUT / "data" / f"train_{d}.jsonl", encoding="utf-8")] for d in ("zh2en", "en2zh")}
    th = load_thinker(dev)
    log(f"thinker on {dev}")
    if a.smoke:
        sm = OUT / "smoke"
        sm.mkdir(parents=True, exist_ok=True)
        for d in ("zh2en", "en2zh"):
            rows = data[d][: a.smoke]
            wavs = [sf.read(str(REPO / r["audio"]), dtype="float32")[0] for r in rows]
            if a.crop:
                wavs = [w[: int(a.crop * SR)] for w in wavs]
            t0 = time.time()
            xs, eps, c0, c1, diff = pgd(th, P, d, rows, wavs, dev)
            for r, w, x, e, u, v in zip(rows, wavs, xs, eps, c0, c1):
                dl = np.abs(x - w)
                log(f"smoke {d} {r['id']}: len {len(w)} -> {len(x)}, eps {e:.5f}, max|x*-x| {dl.max():.5f} "
                    f"(<= eps: {dl.max() <= e + 1e-6}), changed samples {float((dl > 0).mean()):.3f}, "
                    f"CE {u:.4f} -> {v:.4f}, mel max|diff| vs processor {diff:.2e}")
            log(f"smoke {d}: {time.time() - t0:.1f} s")
        return
    claims = OUT / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    for d in ("zh2en", "en2zh"):
        for k in range(NSH):
            c = claims / f"pseudo_{d}_s{k}"
            if (OUT / "runs" / f"{d}_s{k}.jsonl").exists():
                continue
            try:
                fd = os.open(str(c), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{socket.gethostname()} {os.getpid()} w{a.worker}\n".encode())
                os.close(fd)
            except FileExistsError:
                continue
            log(f"worker {a.worker} claims {d} s{k}")
            run_unit(th, P, d, k, dev, a.mb, OUT, data[d])
    log(f"worker {a.worker}: no unit left")


if __name__ == "__main__":
    main()
