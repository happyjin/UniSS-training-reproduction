#!/usr/bin/env python
"""A Megatron G-STAR-SI checkpoint (run_thinker_gstar.sh, training/thinker_gstar_model.py; iter_NNNNNNN, torch dist
checkpoint) as a train_crf_sft-format checkpoint ON ITS BASE (the merged model the run started from with
INIT_MERGED=1, e.g. O2), so that stage-1 checkpoints of the joint bidirectional v8 (reports/mist_w3po_v1/
joint_bidir_plan_prereg_v1.zh-CN.md) are decoded by the unchanged CRF pipeline without a 66 GB merge each:
  --thinker <base merged> --crf-ckpt <this dir>   ==   --thinker <merge_thinker_gstar(iter, base)> (identity ckpt)
Reads exactly the tensors merge_thinker_gstar.py reads (keys ending .a / .b, sd_proj.*, new_emb, new_head):
  lora.pt   {model.layers.N.self_attn.X_proj.{a,b}: tensor as stored}  (crf/decode_crf_ckpt.fold_into computes
            W + (alpha / rank) b @ a in fp32 rounded once = merge_thinker_gstar's W + scale (b @ a), same order)
  extra.pt  new_emb / new_head (float, as merge writes them before the dtype cast), sd_proj (float), lora_rank /
            lora_alpha of the run, step = the iteration, objective "megatron_v8", base, extra_params [], lam 0.0
  DONE
CPU only; refuses to overwrite.

    python -m experiments.mist_w3po_v1.joint.megatron_to_crf_ckpt --checkpoint-dir <iter dir> --base <merged base> \\
        --rank 32 --alpha 64 --out <new dir>
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


def convert(ckdir: str, base: str, rank: int, alpha: float, out: str) -> dict:
    o = Path(out)
    if o.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {o}")
    reader = dcp.FileSystemReader(ckdir)
    meta = reader.read_metadata()
    keys = [k for k in meta.state_dict_metadata
            if not k.startswith("optimizer.")   # SAVE_OPTIM checkpoints: optimizer.state.<exp_avg|...>.<model key>
            and (k.endswith((".a", ".b")) or "sd_proj." in k or k.endswith(("new_emb", "new_head")))]
    w = {k: torch.empty(meta.state_dict_metadata[k].size, dtype=meta.state_dict_metadata[k].properties.dtype)
         for k in keys}
    dcp.load(state_dict=w, storage_reader=reader)
    lo = {}
    for k in keys:
        if k.endswith((".a", ".b")):
            rel = k[k.index("thinker.") + len("thinker."):]
            if not re.match(r"model\.layers\.\d+\.self_attn\.[qkvo]_proj\.[ab]$", rel):
                raise SystemExit(f"unexpected adapter key {k}")
            lo[rel] = w[k]
    emb = next(v for k, v in w.items() if k.endswith("new_emb")).float()
    head = next(v for k, v in w.items() if k.endswith("new_head")).float()
    sd = {k[k.index("sd_proj.") + len("sd_proj."):]: v.float() for k, v in w.items() if "sd_proj." in k}
    it = int(re.search(r"iter_(\d+)", ckdir).group(1)) if re.search(r"iter_(\d+)", ckdir) else -1
    o.mkdir(parents=True)
    torch.save(lo, o / "lora.pt")
    torch.save({"sd_proj": sd, "new_emb": emb, "new_head": head, "extra_params": [], "lam": 0.0, "step": it,
                "objective": "megatron_v8", "base": base, "lora_rank": rank, "lora_alpha": alpha,
                "megatron_ckpt": ckdir}, o / "extra.pt")
    (o / "DONE").write_text(json.dumps({"step": it, "time": time.time(), "from": ckdir}))
    return {"out": out, "adapters": len(lo) // 2, "iteration": it}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(convert(a.checkpoint_dir, a.base, a.rank, a.alpha, a.out)))


if __name__ == "__main__":
    main()
