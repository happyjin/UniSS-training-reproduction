"""Robust wave 11 (reports/mist_w3po_v1/robust_wave11_laplace_prereg.zh-CN.md): diagonal empirical Fisher of the adapted
matrices (q/k/v/o of all 48 Thinker text layers, the matrices ERM's LoRA adapts) for the clean-anchored Laplace merge.

  --which c : F_c at W0 (base Thinker; LoRA disabled) on N = 512 clean samples (zh2en clean 256, en2zh clean 256)
  --which n : F_n at W_e = W0 + ERM LoRA (alpha 1, unmerged, the exact training forward) on N = 512 samples of ERM's
              training mixture (8 environments x 64)
F[w] = mean_i (d CE_i / d w)^2, CE_i = per-sample mean cross-entropy over the target tokens of the reference translation
(+ <|im_end|>), with wave5_m_train.py's prompt and label mask; per-sample backward, batch 1, eval mode, fp32 accumulation.

Sample i (0..511) is one of the samples of wave 5's ERM training stream (wave5_m_train.Sampler, identical rows, noise
files, BVR and mixing): k = sha1("20261120|w11|fisher|<c|n>|i"), step s = k mod 2300, slot within the env
m = (k // 2300) mod 8, slot j = 8 e + m with e = 0 (zh2en clean) for c, i < 256, e = 4 (en2zh clean) for c, i >= 256,
and e = i // 64 for n (ENVS order of wave5_m_train).
Gradient w.r.t. the full weight W: every adapted Linear gets a zero fp32 matrix D next to it,
out = base(x) [+ (alpha/r) b a x] + (D x).to(bf16); D = 0 leaves the forward unchanged and dL/dD = dL/dW at that point.
Output (refuses to overwrite): <out>.pt = {thinker.model.<name>.weight: F (fp32, shape of W)}, <out>.json (meta),
<out>.samples.jsonl (one line per sample: i, s, j, env, id, n_tok, ce). --limit K (smoke) stops after K samples and
writes only <out>.samples.jsonl and the timing/memory log.
    CUDA_VISIBLE_DEVICES=0 <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave11_fisher.py --which c --out <dir>/F_c
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import wave5_m_train as W5  # noqa: E402

ERM = REPO / "eval_outputs/mist_w3po_v1/wave5/M/run_erm/adapters_final.pt"
N, STEPS, PER_ENV = 512, 2300, 8


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def plan(which: str) -> list[tuple[int, int, int, int]]:
    """[(i, s, j, env)] of the N Fisher samples"""
    out = []
    for i in range(N):
        k = h(f"20261120|w11|fisher|{which}|{i}")
        e = (0 if i < N // 2 else 4) if which == "c" else i // (N // len(W5.ENVS))
        s, m = k % STEPS, (k // STEPS) % PER_ENV
        out.append((i, s, e * PER_ENV + m, e))
    return out


class Probe(nn.Module):
    """wrapped LoRALinear + a zero fp32 matrix D whose gradient is the gradient of the full weight"""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        w = inner.base.weight
        self.D = nn.Parameter(torch.zeros(w.shape, device=w.device, dtype=torch.float32))

    def forward(self, x):
        out = self.inner(x)
        return out + F.linear(x.float(), self.D).to(out.dtype)


class Items(torch.utils.data.Dataset):
    def __init__(self, pl):
        self.pl, self.S, self.proc, self.head = pl, None, None, None

    def __len__(self):
        return len(self.pl)

    def __getitem__(self, k):
        if self.S is None:
            from transformers import Qwen3OmniMoeProcessor
            from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
            self.proc = Qwen3OmniMoeProcessor.from_pretrained(str(W5.QWEN3))
            ins = inference_instruction()
            self.head = {}
            for d in ("zh2en", "en2zh"):
                conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"},
                                                     {"type": "text", "text": ins[W5.TGT[d]]}]}]
                self.head[d] = self.proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
            self.S = W5.Sampler()
        i, s, j, e = self.pl[k]
        it = self.S.sample(s, j)
        assert it["env"] == e
        inp = self.proc(text=[self.head[it["dir"]] + it["ref"] + "<|im_end|>"], audio=[it["wav"]], return_tensors="pt",
                        padding=True)
        ids = inp["input_ids"]
        lab = torch.full_like(ids, W5.IGNORE)
        row = ids[0].tolist()
        st = max(q for q, t in enumerate(row) if t == W5.IM_START) + 3
        en = max(q for q, t in enumerate(row) if t == W5.IM_END)
        assert en > st, "empty target"
        lab[0, st: en + 1] = ids[0, st: en + 1]
        return {**dict(inp), "labels": lab, "meta": {"i": i, "s": s, "j": j, "env": e, "id": it["id"], "dir": it["dir"]}}


def build(dev, which):
    from transformers import Qwen3OmniMoeForConditionalGeneration
    from experiments.uniss_streaming_p2st_traj_v1.training import lora
    omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(str(W5.QWEN3), torch_dtype=torch.bfloat16,
                                                                device_map={"": dev}, attn_implementation="sdpa")
    thinker = omni.thinker
    for attr in ("talker", "code2wav", "token2wav"):
        if hasattr(omni, attr):
            delattr(omni, attr)
    del omni
    thinker.requires_grad_(False)
    text = thinker.model
    torch.manual_seed(20261119)
    ad = lora.inject(text, rank=W5.RANK_LORA, alpha=W5.ALPHA, targets=("q_proj", "k_proj", "v_proj", "o_proj"))
    assert len(ad) == 4 * len(text.layers)
    if which == "n":
        meta = json.load(open(ERM.with_suffix(".json")))
        assert meta["arm"] == "erm" and meta["rank"] == W5.RANK_LORA and meta["alpha"] == W5.ALPHA, meta
        st = torch.load(ERM, map_location="cpu")
        assert len(st) == 2 * len(ad)
        for L in ad:
            L.a.data.copy_(st[L.lora_name + ".a"].to(dev))
            L.b.data.copy_(st[L.lora_name + ".b"].to(dev))
            L.enabled = True
    else:
        for L in ad:
            L.enabled = False                                 # exactly W0
    probes = {}
    for L in ad:
        L.a.requires_grad_(False)
        L.b.requires_grad_(False)
        parent_name, _, attr = L.lora_name.rpartition(".")
        parent = text.get_submodule(parent_name)
        P = Probe(L)
        setattr(parent, attr, P)
        probes[f"thinker.model.{L.lora_name}.weight"] = P
    thinker.eval()
    return thinker, probes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("c", "n"), required=True)
    ap.add_argument("--out", required=True, help="output stem (no suffix)")
    ap.add_argument("--limit", type=int, default=0, help="smoke: stop after K samples, no Fisher written")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    outs = [out.with_suffix(x) for x in (".pt", ".json", ".samples.jsonl")]
    for p in outs:
        if p.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {p}")
    out.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda", 0)
    pl = plan(a.which)
    dup = len(pl) - len({(s, j) for _, s, j, _ in pl})
    if a.limit:
        pl = pl[: a.limit]
    t0 = time.time()
    thinker, probes = build(dev, a.which)
    Fis = {k: torch.zeros_like(P.D) for k, P in probes.items()}
    nel = sum(x.numel() for x in Fis.values())
    W5.log(f"which {a.which}: {len(probes)} matrices, {nel / 1e6:.1f} M elements; duplicate (s, j) in plan {dup}; "
           f"load {time.time() - t0:.0f} s; mem {torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB")
    dl = torch.utils.data.DataLoader(Items(pl), batch_size=None, shuffle=False, num_workers=a.workers, prefetch_factor=4)
    sj = open(out.with_suffix(".samples.jsonl"), "x", encoding="utf-8")
    t1, n = time.time(), 0
    for b in dl:
        kw = {k: b[k].to(dev) for k in ("input_ids", "attention_mask", "feature_attention_mask")}
        kw["input_features"] = b["input_features"].to(dev, torch.bfloat16)
        lab = b["labels"].to(dev)
        o = thinker(**kw, use_cache=False, return_dict=True)
        tl = lab[:, 1:]
        sel = tl != W5.IGNORE
        ce = F.cross_entropy(o.logits[:, :-1][sel].float(), tl[sel])          # per-sample mean over target tokens
        ce.backward()
        bad = 0
        for k, P in probes.items():
            g = P.D.grad
            bad += int((~torch.isfinite(g)).sum())
            Fis[k].addcmul_(g, g)
            P.D.grad = None
        if bad:
            raise SystemExit(f"non-finite gradient at sample {b['meta']}: {bad} entries")
        n += 1
        sj.write(json.dumps({**b["meta"], "n_tok": int(sel.sum()), "ce": round(float(ce), 5)}, ensure_ascii=False) + "\n")
        sj.flush()
        del o, ce
        if n % 32 == 0 or n <= 3:
            W5.log(f"{n}/{len(pl)} samples, {(time.time() - t1) / n:.2f} s/sample, "
                   f"peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB")
    sj.close()
    W5.log(f"done {n} samples in {time.time() - t1:.0f} s; peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB")
    if a.limit:
        W5.log("smoke: no Fisher written")
        return
    assert n == N, n
    res = {k: (v / n).cpu() for k, v in Fis.items()}
    tot = sum(float(v.double().sum()) for v in res.values())
    zero = sum(int((v == 0).sum()) for v in res.values())
    torch.save(res, open(out.with_suffix(".pt"), "xb"))
    json.dump({"which": a.which, "point": "W0" if a.which == "c" else "W0 + ERM LoRA (alpha 1)", "N": n,
               "duplicate_sj": dup, "elements": nel, "mean": tot / nel, "zero_frac": zero / nel,
               "erm": str(ERM.relative_to(REPO)) if a.which == "n" else None, "sec": round(time.time() - t0)},
              open(out.with_suffix(".json"), "x"), indent=1)
    W5.log(f"Fisher written: mean {tot / nel:.4e}, zero fraction {zero / nel:.2e}")


if __name__ == "__main__":
    main()
