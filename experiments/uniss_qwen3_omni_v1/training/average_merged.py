"""Average merged Thinker copies that share one base (e.g. the three ABC adapter seeds merged into Qwen3-Omni): every
tensor is the mean of the copies, so only the adapted layers change -- base + mean of the adapters' low-rank deltas,
a weight-space approximation to averaging the three posteriors. Shard by shard (one shard of each copy in memory at a
time); the other files (config, tokenizer, index) are copied from the first copy. Refuses to overwrite.

    python -m experiments.uniss_qwen3_omni_v1.training.average_merged --out <new dir> <merged dir> <merged dir> ...
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("copies", nargs="+")
    args = ap.parse_args()
    out, src = Path(args.out), [Path(c) for c in args.copies]
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    shards = sorted(p.name for p in src[0].glob("*.safetensors"))
    for c in src[1:]:
        assert sorted(p.name for p in c.glob("*.safetensors")) == shards, c
    out.mkdir(parents=True)
    for f in src[0].iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, out / f.name)
    changed = 0
    for name in shards:
        hs = [safe_open(str(c / name), "pt") for c in src]
        tensors = {}
        for key in hs[0].keys():
            ts = [h.get_tensor(key) for h in hs]
            if all(torch.equal(ts[0], t) for t in ts[1:]):
                tensors[key] = ts[0]
            else:
                tensors[key] = torch.stack([t.float() for t in ts]).mean(0).to(ts[0].dtype)
                changed += 1
        save_file(tensors, str(out / name), metadata=hs[0].metadata() or {"format": "pt"})
        print(f"{name}: {len(tensors)} tensors", flush=True)
    print(f"AVERAGE_DONE {changed} tensors averaged -> {out}", flush=True)


if __name__ == "__main__":
    main()
