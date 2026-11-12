#!/usr/bin/env python
"""Identity checkpoint of a merged Thinker (reports/mist_w3po_v1/joint_bidir_plan_prereg_v1.zh-CN.md §stage 1): a
train_crf_sft-format directory (lora.pt, extra.pt, DONE) that applies NOTHING, so a merged v8-type model (the separate
v8 Simul SFTs, the joint bidirectional v8) is decoded by the unchanged CRF dev / test pipeline
(crf/decode_crf_ckpt_finalpost --crf-ckpt <this> --thinker <merged> ..., thinker_t_eval_orch / en2zh_orch --ckpt):
  lora.pt   {}                           crf/decode_crf_ckpt.fold_into folds no adapter
  extra.pt  new_emb / new_head = the merged model's own rows NEW0:NEW_END (thinker.model.embed_tokens / thinker.lm_head,
            read from the safetensors shards in their stored dtype), sd_proj = the model's sd_proj.pt state dict,
            extra_params [], lam 0.0, step 0, objective "identity", base, lora_rank 32, lora_alpha 64.0
  DONE
apply_ckpt then writes back exactly the values the engine already holds (same dtype), i.e. the decode is the merged
model's own. CPU only; refuses to overwrite. (No optim.pt: nothing is trained; not resumable by train_crf_sft.)

    python -m experiments.mist_w3po_v1.joint.identity_ckpt --base <merged dir> --out <new dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

NEW0, NEW_END = 151676, None   # NEW_END from thinker_gstar_model (imported lazily: heavy module)
EMB, HEAD = "thinker.model.embed_tokens.weight", "thinker.lm_head.weight"


def read_rows(base: Path, key: str, lo: int, hi: int) -> torch.Tensor:
    from safetensors import safe_open

    wm = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(base / wm[key]), framework="pt", device="cpu") as f:
        return f.get_slice(key)[lo:hi].clone()


def build(base: str, out: str, new0: int, new_end: int) -> dict:
    b, o = Path(base), Path(out)
    if o.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {o}")
    ex = {"sd_proj": torch.load(b / "sd_proj.pt", map_location="cpu"),
          "new_emb": read_rows(b, EMB, new0, new_end), "new_head": read_rows(b, HEAD, new0, new_end),
          "extra_params": [], "lam": 0.0, "step": 0, "objective": "identity", "base": base,
          "lora_rank": 32, "lora_alpha": 64.0}
    o.mkdir(parents=True)
    torch.save({}, o / "lora.pt")
    torch.save(ex, o / "extra.pt")
    (o / "DONE").write_text(json.dumps({"step": 0, "time": time.time(), "identity_of": base}))
    return {"out": out, "base": base, "rows": [new0, new_end], "dtype": str(ex["new_emb"].dtype)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import NEW0 as N0, NEW_END as NE

    print(json.dumps(build(a.base, a.out, N0, NE)))


if __name__ == "__main__":
    main()
