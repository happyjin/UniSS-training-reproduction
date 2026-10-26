"""Pull the commit head out of a Megatron save as the checkpoint si_commit_head.load_head reads.

The save holds the head's layers; its normalisation (the train rows' mean and std) and the
held-out items come from the prep npz it was trained on. --check scores the held-out rows (AUC)
so a broken export shows at once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed.checkpoint as dcp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True, help="an iter_NNNNNNN directory")
    ap.add_argument("--rows", required=True, help="the prep npz it trained on")
    ap.add_argument("--label", choices=("cons", "final"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    reader = dcp.FileSystemReader(args.checkpoint_dir)
    meta = reader.read_metadata()
    keys = [k for k in meta.state_dict_metadata if k.startswith("head.net.")]
    if not keys:
        raise SystemExit(f"no head tensors in {args.checkpoint_dir}; keys look like {list(meta.state_dict_metadata)[:5]}")
    wanted = {k: torch.empty(meta.state_dict_metadata[k].size, dtype=meta.state_dict_metadata[k].properties.dtype)
              for k in keys}
    dcp.load(state_dict=wanted, storage_reader=reader)
    data = np.load(args.rows)
    state = {"mean": torch.as_tensor(data["mean"], dtype=torch.float32),
             "std": torch.as_tensor(data["std"], dtype=torch.float32)}
    state.update({k[len("head."):]: v.float() for k, v in wanted.items()})
    torch.save({"state": state, "label": args.label, "val_ids": json.loads(str(data["val_ids"])),
                "source": args.checkpoint_dir, "rows": args.rows}, out)
    res = {"out": str(out), "tensors": sorted(state)}
    if args.check:
        from experiments.uniss_qwen3_omni_v1.evaluation.si_commit_head import auc_score, load_head

        head = load_head(str(out))
        val = data["val"]
        with torch.no_grad():
            p = torch.sigmoid(head(torch.from_numpy(np.asarray(data["x"][val], dtype=np.float32)))).numpy()
        res["val_auc"] = round(auc_score(np.asarray(data[f"y_{args.label}"][val], dtype=np.float32), p), 4)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
