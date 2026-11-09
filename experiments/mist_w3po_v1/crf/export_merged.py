"""Write a crf/train_crf_sft.py checkpoint as a merged Thinker directory (the format of v8's thinker_merged), so that
every existing tool takes it as --thinker / THINKER= / --base unchanged: scripts/eval_pipeline.sh, launch_decode.sh,
evaluation/online_decode.py, rl/train_w3po.py, crf/precheck_sched.py.

Same formula as experiments/uniss_qwen3_omni_v1/training/merge_thinker_gstar.py and crf/decode_crf_ckpt.apply_ckpt,
with the v8 merged directory as the base:
  thinker.<stem>.weight += (alpha / rank) b @ a   (fp32 sum, one rounding to bf16) for every adapter in lora.pt
  thinker.model.embed_tokens.weight / thinker.lm_head.weight rows NEW0..NEW_END = new_emb / new_head (extra.pt)
  sd_proj.pt = the checkpoint's projector (float32)
Only the safetensors shards holding a changed tensor are rewritten; every other file is copied (no links). The base
is never touched; an existing --out is refused. ~66 GB per export (the LoRA touches all 48 layers, so nearly every
shard is rewritten). --dry-run prints the plan only.

    python -m experiments.mist_w3po_v1.crf.export_merged --ckpt <out>/ckpt/step_NNNNN --out <new dir>/thinker_merged
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import NEW0, NEW_END, special_tokens

V8 = "eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged"


def deltas(ck: Path) -> tuple[dict, dict, dict, dict]:
    ex = torch.load(ck / "extra.pt", map_location="cpu")
    st = torch.load(ck / "lora.pt", map_location="cpu")
    scale = float(ex["lora_alpha"]) / int(ex["lora_rank"])
    delta = {}
    for k in st:
        if k.endswith(".a"):
            stem = k[:-2]
            delta["thinker." + stem + ".weight"] = scale * (st[stem + ".b"].float() @ st[stem + ".a"].float())
    rows = {"thinker.model.embed_tokens.weight": ex["new_emb"].float(), "thinker.lm_head.weight": ex["new_head"].float()}
    assert all(v.shape[0] == NEW_END - NEW0 for v in rows.values())
    return delta, rows, {k: v.float() for k, v in ex["sd_proj"].items()}, ex


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="<train out>/ckpt/step_NNNNN (lora.pt, extra.pt, DONE)")
    ap.add_argument("--base", default=V8, help="the merged Thinker the run started from (v8)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    ck, base, out = Path(args.ckpt), Path(args.base), Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    if not (ck / "DONE").exists():
        raise SystemExit(f"{ck} has no DONE")
    delta, rows, sd_proj, ex = deltas(ck)
    if ex.get("extra_params"):
        print(f"WARNING: {len(ex['extra_params'])} hook modules are not part of a merged Thinker", flush=True)
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    missing = [k for k in list(delta) + list(rows) if k not in index]
    if missing:
        raise SystemExit(f"{len(missing)} tensors not in the base, e.g. {missing[:3]}")
    shards = sorted({index[k] for k in list(delta) + list(rows)})
    plan = {"ckpt": str(ck), "step": ex["step"], "objective": ex["objective"], "base": str(base), "out": str(out),
            "adapted": len(delta), "new_rows": NEW_END - NEW0, "shards_rewritten": len(shards),
            "files_copied": sum(1 for f in base.iterdir() if f.is_file() and f.name not in shards
                                and f.name not in ("sd_proj.pt", "gstar_tokens.json")),
            "bytes": sum(f.stat().st_size for f in base.iterdir() if f.is_file())}
    print(json.dumps(plan), flush=True)
    if args.dry_run:
        return
    from safetensors.torch import load_file, save_file

    out.mkdir(parents=True)
    for f in base.iterdir():
        if f.is_file() and f.name not in shards and f.name not in ("sd_proj.pt", "gstar_tokens.json"):
            shutil.copy2(f, out / f.name)
    for shard in shards:
        state = load_file(str(base / shard))
        for k, d in delta.items():
            if index[k] == shard:
                state[k] = (state[k].float() + d).to(state[k].dtype).contiguous()
        for k, r in rows.items():
            if index[k] == shard:
                w = state[k].clone()
                w[NEW0:NEW_END] = r.to(w.dtype)
                state[k] = w.contiguous()
        save_file(state, str(out / shard), metadata={"format": "pt"})
    torch.save(sd_proj, out / "sd_proj.pt")
    (out / "gstar_tokens.json").write_text(json.dumps({"first_id": NEW0, "tokens": special_tokens()}), encoding="utf-8")
    (out / "crf_export.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    print(f"CRF_EXPORT_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
