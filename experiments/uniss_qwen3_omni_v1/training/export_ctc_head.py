"""Pull the CTC head out of a Megatron save as the checkpoint si_ctc_head.load_ctc reads.

--check decodes the held-out rows greedily and reports character error rate per target language,
so a broken export (or a head that has not learnt to read) shows at once.
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
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--rows", required=True, help="the prep npz it trained on")
    ap.add_argument("--dims", required=True, help='json, e.g. {"d": 512, "layers": 8, "heads": 8, "ff": 2048}')
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", type=int, default=2000, help="held-out rows to decode (0: none)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from experiments.uniss_qwen3_omni_v1.evaluation.si_ctc_head import PAD_CODE, CTCHead, greedy, load_ctc

    reader = dcp.FileSystemReader(args.checkpoint_dir)
    meta = reader.read_metadata()
    keys = [k for k in meta.state_dict_metadata if k.startswith("ctc.")]
    if not keys:
        raise SystemExit(f"no ctc tensors in {args.checkpoint_dir}; keys look like {list(meta.state_dict_metadata)[:5]}")
    wanted = {k: torch.empty(meta.state_dict_metadata[k].size, dtype=meta.state_dict_metadata[k].properties.dtype)
              for k in keys}
    dcp.load(state_dict=wanted, storage_reader=reader)
    data = np.load(args.rows)
    chars = json.loads(str(data["chars"]))
    cfg = {"n_chars": len(chars), **json.loads(args.dims)}
    state = {k[len("ctc."):]: v.float() for k, v in wanted.items()}
    CTCHead(**cfg).load_state_dict(state)   # every tensor present and shaped right
    torch.save({"cfg": cfg, "state": state, "chars": chars, "source": args.checkpoint_dir, "rows": args.rows}, out)
    res = {"out": str(out), "tensors": len(state)}
    if args.check:
        from rapidfuzz.distance import Levenshtein

        model, chars = load_ctc(str(out), "cuda" if torch.cuda.is_available() else "cpu")
        dev = next(model.parameters()).device
        val = np.flatnonzero(data["val"])[: args.check]
        co, to = data["code_off"], data["text_off"]
        err = {0: [0, 0], 1: [0, 0]}
        with torch.no_grad():
            for r in val:
                c = torch.from_numpy(data["codes"][co[r]: co[r + 1]].astype(np.int64))[None].to(dev)
                hyp = greedy(model(c, c == PAD_CODE)[0], chars)
                ref = "".join(chars[i - 1] for i in data["text"][to[r]: to[r + 1]])
                e = err[int(data["lang"][r])]
                e[0] += Levenshtein.distance(ref, hyp)
                e[1] += len(ref)
        res["val_cer"] = {"cmn": round(err[0][0] / max(err[0][1], 1), 4), "eng": round(err[1][0] / max(err[1][1], 1), 4)}
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
