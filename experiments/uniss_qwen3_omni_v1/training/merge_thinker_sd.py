"""Merge Plan 4A's adapters into a full Qwen3-Omni copy that vLLM serves as it serves the release.

W' = W + (alpha / rank) * b @ a for every adapted projection. The release directory is copied file by file; only
the safetensors shards holding an adapted weight are rewritten, and the release itself is never touched.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--base", default=str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"))
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out, base = Path(args.out), Path(args.base)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    reader = dcp.FileSystemReader(args.checkpoint_dir)
    meta = reader.read_metadata()
    keys = [k for k in meta.state_dict_metadata if k.endswith(".a") or k.endswith(".b")]
    if not keys:
        raise SystemExit(f"no adapter tensors in {args.checkpoint_dir}; keys look like {list(meta.state_dict_metadata)[:5]}")
    wanted = {k: torch.empty(meta.state_dict_metadata[k].size, dtype=meta.state_dict_metadata[k].properties.dtype) for k in keys}
    dcp.load(state_dict=wanted, storage_reader=reader)
    scale = args.alpha / args.rank
    delta = {}
    for k in keys:
        if k.endswith(".a"):
            stem = k[:-2]                      # thinker.model.layers.N.self_attn.q_proj
            a, b = wanted[stem + ".a"].float(), wanted[stem + ".b"].float()
            delta[stem + ".weight"] = scale * (b @ a)
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    missing = [k for k in delta if k not in index]
    if missing:
        raise SystemExit(f"{len(missing)} adapted weights not in the release, e.g. {missing[:3]}")
    out.mkdir(parents=True)
    shards = {index[k] for k in delta}
    from safetensors.torch import load_file, save_file

    for f in base.iterdir():
        if f.is_file() and f.name not in shards:
            shutil.copy2(f, out / f.name)
    changed = 0
    for shard in sorted(shards):
        state = load_file(str(base / shard))
        for k, d in delta.items():
            if index[k] == shard:
                state[k] = (state[k].float() + d).to(state[k].dtype).contiguous()
                changed += 1
        save_file(state, str(out / shard), metadata={"format": "pt"})
    norms = [float(d.norm()) for d in delta.values()]
    print(json.dumps({"out": str(out), "adapted": changed, "shards_rewritten": len(shards),
                      "delta_norm_mean": round(sum(norms) / len(norms), 4)}), flush=True)


if __name__ == "__main__":
    main()
