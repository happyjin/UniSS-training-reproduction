#!/usr/bin/env python
"""Does a Megatron G-STAR-SI checkpoint (run_thinker_gstar.sh with SAVE_OPTIM, torch dist format) hold a usable
optimizer state? (reports/mist_w3po_v1/joint_bidir_plan_prereg_v1.zh-CN.md: every checkpoint keeps the optimizer, user
requirement; stage 1 of (b) is the first Megatron run here that saves it.) CPU only, reads the checkpoint, writes nothing.

model keys  the tensors merge_thinker_gstar reads (.a / .b, sd_proj.*, new_emb, new_head), none under "optimizer."
required    for every model key K: optimizer.state.exp_avg.K and optimizer.state.exp_avg_sq.K with K's shape
            (Megatron's dist_checkpointing.optimizer.make_sharded_optimizer_tensor names them f"{prefix}.{K}")
non-empty   the sum of |exp_avg_sq| over all keys > 0 (Adam's second moment after >= 1 step)
recorded    the other state names (fp32_param = the fp32 main copies of the bf16 parameters), bytes on disk

    python -m experiments.mist_w3po_v1.joint.check_megatron_optim <iter dir>   -> one json line; exit 0 ok / 1 not ok
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

REQUIRED = ("exp_avg", "exp_avg_sq")
PREFIX = "optimizer.state."


def is_model_key(k: str) -> bool:
    return not k.startswith("optimizer.") and (k.endswith((".a", ".b")) or "sd_proj." in k
                                               or k.endswith(("new_emb", "new_head")))


def check(ckdir: str) -> dict:
    reader = dcp.FileSystemReader(ckdir)
    md = reader.read_metadata().state_dict_metadata
    model = sorted(k for k in md if is_model_key(k))
    opt = {k for k in md if k.startswith(PREFIX)}
    states = sorted({k[len(PREFIX):].split(".", 1)[0] for k in opt})
    missing, shape = [], []
    for k in model:
        for s in REQUIRED:
            o = f"{PREFIX}{s}.{k}"
            if o not in opt:
                missing.append(o)
            elif tuple(md[o].size) != tuple(md[k].size):
                shape.append(o)
    sq_sum, n_sq = 0.0, 0
    if model and not missing and not shape:
        want = {f"{PREFIX}exp_avg_sq.{k}": torch.empty(md[f"{PREFIX}exp_avg_sq.{k}"].size,
                                                        dtype=md[f"{PREFIX}exp_avg_sq.{k}"].properties.dtype)
                for k in model}
        dcp.load(state_dict=want, storage_reader=reader)
        sq_sum = float(sum(v.double().abs().sum() for v in want.values()))
        n_sq = len(want)
    nbytes = sum(f.stat().st_size for f in Path(ckdir).iterdir() if f.is_file())
    ok = bool(model) and not missing and not shape and sq_sum > 0
    return {"ckpt": ckdir, "ok": ok, "model_keys": len(model), "optimizer_keys": len(opt), "states": states,
            "missing": missing[:5], "n_missing": len(missing), "shape_mismatch": shape[:5], "exp_avg_sq_keys": n_sq,
            "exp_avg_sq_abs_sum": sq_sum, "bytes": nbytes}


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: check_megatron_optim <iter dir>")
    r = check(argv[0])
    print(json.dumps(r))
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
