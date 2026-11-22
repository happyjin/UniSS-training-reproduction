"""sst_hsmm_v1 P0b (reports/sst_hsmm_v1/prereg.zh-CN.md 3.1): prefix-forward emissions of the UE Thinker (LoRA injected, b = 0,
so the forward equals the base) on the first 300 held-out rows per direction, clean and noisy; shard k of N. Saves
<out>/em/<cond>__<id>.npz (lp_tok, lp_wait fp16). Shard 0 additionally times forward + lattice + backward (gate on) on its
first 8 items and records peak memory. One process per card.
    qwen3omni PYTHONPATH=. python -m experiments.sst_hsmm_v1.p0b --shard k --num-shards N --out <dir>
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
from experiments.sst_hsmm_v1 import hsmm as H  # noqa: E402
from experiments.sst_hsmm_v1 import model as MD  # noqa: E402

DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
N_P0B = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "em").mkdir(parents=True, exist_ok=True)
    stat_p = out / f"shard_{a.shard}.json"
    if stat_p.exists():
        sys.exit(f"拒绝覆盖已有输出: {stat_p}")
    dev = "cuda:0"
    proc, head = MD.processor_and_heads()
    thinker, names = MD.build_thinker(dev, lora=True)
    gate = H.Gate()
    jobs = []
    for d in ("en2zh", "zh2en"):
        rows = [json.loads(l) for l in open(DATA / f"heldout_{d}.jsonl", encoding="utf-8")][:N_P0B]
        for r in rows:
            for cond in ("clean", "noisy"):
                jobs.append((r, cond))
    jobs = jobs[a.shard:: a.num_shards]
    st = {"n": 0, "mismatch": 0, "delta_g": [], "timing": []}
    t0 = time.time()
    for k, (r, cond) in enumerate(jobs):
        wav = sf.read(str(DATA / "heldout_wav" / r["dir"] / f"{cond}__{r['id']}.wav"), dtype="float32")[0]
        inp, pos = MD.prefix_batch(proc, head, r, wav)
        if inp is None:
            st["mismatch"] += 1
            continue
        timed = a.shard == 0 and len(st["timing"]) < 8
        if timed:
            torch.cuda.synchronize(); t1 = time.time()
            lp, lw = MD.emissions(thinker, inp, pos, dev)
            tot, _ = H.lattice(lp.cpu(), lw.cpu(), r["g"], [int(round(x / 100)) for x in r["d_ms"]], r["punct"], r["X"],
                               0.1, 1.0, gate)
            loss = -tot / len(r["tok"])
            loss.backward()
            torch.cuda.synchronize()
            st["timing"].append({"C": r["C"], "n": len(r["tok"]), "sec": round(time.time() - t1, 3),
                                 "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)})
            thinker.zero_grad(set_to_none=True)
            lp, lw = lp.detach(), lw.detach()
        else:
            with torch.no_grad():
                lp, lw = MD.emissions(thinker, inp, pos, dev)
        lp, lw = lp.float().cpu().numpy(), lw.float().cpu().numpy()
        np.savez(out / "em" / f"{cond}__{r['id']}.npz", lp_tok=lp.astype(np.float16), lp_wait=lw.astype(np.float16))
        for i, gi in enumerate(r["g"]):
            if 2 <= gi <= r["C"]:
                st["delta_g"].append(float(lp[gi - 1, i] - lp[gi - 2, i]))
        st["n"] += 1
        if k % 50 == 0:
            print(f"{k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    dg = np.array(st.pop("delta_g"))
    st["delta_g_median"] = float(np.median(dg)) if len(dg) else None
    st["delta_g_n"] = int(len(dg))
    st["delta_g_mean"] = float(dg.mean()) if len(dg) else None
    st["sec_total"] = round(time.time() - t0, 1)
    with open(stat_p, "x") as fh:
        json.dump(st, fh, indent=1)
    print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
