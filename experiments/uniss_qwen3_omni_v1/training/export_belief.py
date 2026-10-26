"""Pull the commit belief out of a Megatron save as the checkpoint si_belief.load_belief reads, with the prep
npz's normalisation; --check runs si_belief check on the held-out trajectories so a broken export shows at once.
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    reader = dcp.FileSystemReader(args.checkpoint_dir)
    meta = reader.read_metadata()
    keys = [k for k in meta.state_dict_metadata if k.startswith("belief.g.") or k.startswith("belief.h.")]
    if not keys:
        raise SystemExit(f"no belief tensors in {args.checkpoint_dir}; keys look like {list(meta.state_dict_metadata)[:5]}")
    wanted = {k: torch.empty(meta.state_dict_metadata[k].size, dtype=meta.state_dict_metadata[k].properties.dtype)
              for k in keys}
    dcp.load(state_dict=wanted, storage_reader=reader)
    d = np.load(args.rows)
    state = {k: torch.as_tensor(d[k], dtype=torch.float32) for k in ("mx", "sx", "mo", "so")}
    state.update({k[len("belief."):]: v.float() for k, v in wanted.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": state, "source": args.checkpoint_dir, "rows": args.rows}, out)
    print(json.dumps({"out": str(out), "tensors": sorted(state)}), flush=True)
    if args.check:
        from experiments.uniss_qwen3_omni_v1.evaluation.si_belief import cmd_check

        cmd_check(argparse.Namespace(belief=str(out), rows=args.rows))


if __name__ == "__main__":
    main()
