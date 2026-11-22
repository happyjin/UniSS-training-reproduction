"""sst_hsmm_v1 P1a checkpoint diagnostic, GPU part (reports/sst_hsmm_v1/prereg.zh-CN.md change 3; report only).
Prefix-forward emissions of the UE Thinker + the LoRA of one P1a checkpoint on P0b's 1200 held-out items (first 300 per
direction x clean / noisy), shard k of N -> <out>/em/<cond>__<id>.npz (lp_tok, lp_wait fp16; same format as p0/em).
Stacks on the training cards: talker / code2wav never built (enable_audio_output False), no grads, and the process is
capped at (free memory at start - HEADROOM GiB) so that exceeding it OOMs this process, not the training.
    qwen3omni PYTHONPATH=. python -m experiments.sst_hsmm_v1.p1_diag --ckpt <ckpt_XXXXX.pt> --shard k --num-shards N --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1 import model as MD  # noqa: E402

DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
N_P0B = 300
HEADROOM = 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    stat_p = out / f"shard_{a.shard}.json"
    if stat_p.exists():
        sys.exit(f"拒绝覆盖已有输出: {stat_p}")
    (out / "em").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    dev = "cuda:0"
    free, total = torch.cuda.mem_get_info(0)
    cap = free - HEADROOM * 2 ** 30
    if cap < 61 * 2 ** 30:
        sys.exit(f"not enough free memory: free {free / 2 ** 30:.1f} GiB")
    torch.cuda.set_per_process_memory_fraction(cap / total, 0)
    print(f"free {free / 2 ** 30:.1f} GiB, cap {cap / 2 ** 30:.1f} GiB", flush=True)

    from transformers import Qwen3OmniMoeForConditionalGeneration
    from experiments.uniss_streaming_p2st_traj_v1.training import lora as L
    omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(str(MD.THINKER), torch_dtype=torch.bfloat16,
                                                                device_map={"": dev}, attn_implementation="sdpa",
                                                                enable_audio_output=False)
    thinker = omni.thinker
    del omni
    thinker.requires_grad_(False)
    text = thinker.model
    ad = L.inject(text, rank=MD.RANK_LORA, alpha=MD.ALPHA, targets=("q_proj", "k_proj", "v_proj", "o_proj"))
    assert len(ad) == 4 * len(text.layers), len(ad)
    names = {nm: p for nm, p in text.named_parameters() if p.requires_grad}
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    assert set(ck["adapters"]) == set(names), (len(ck["adapters"]), len(names))
    with torch.no_grad():
        for k, p in names.items():
            p.copy_(ck["adapters"][k].to(p.dtype))
    thinker.requires_grad_(False)
    thinker.eval()
    proc, head = MD.processor_and_heads()
    print(f"loaded step {ck['step']}, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)

    jobs = []
    for d in ("en2zh", "zh2en"):
        rows = [json.loads(l) for l in open(DATA / f"heldout_{d}.jsonl", encoding="utf-8")][:N_P0B]
        for r in rows:
            for cond in ("clean", "noisy"):
                jobs.append((r, cond))
    jobs = jobs[a.shard:: a.num_shards]
    st = {"step": ck["step"], "n": 0, "mismatch": 0}
    t0 = time.time()
    for k, (r, cond) in enumerate(jobs):
        wav = sf.read(str(DATA / "heldout_wav" / r["dir"] / f"{cond}__{r['id']}.wav"), dtype="float32")[0]
        inp, pos = MD.prefix_batch(proc, head, r, wav)
        if inp is None:
            st["mismatch"] += 1
            continue
        with torch.no_grad():
            lp, lw = MD.emissions(thinker, inp, pos, dev)
        np.savez(out / "em" / f"{cond}__{r['id']}.npz", lp_tok=lp.float().cpu().numpy().astype(np.float16),
                 lp_wait=lw.float().cpu().numpy().astype(np.float16))
        st["n"] += 1
        if k % 50 == 0:
            print(f"{k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    st["sec_total"] = round(time.time() - t0, 1)
    st["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    with open(stat_p, "x") as fh:
        json.dump(st, fh, indent=1)
    print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
