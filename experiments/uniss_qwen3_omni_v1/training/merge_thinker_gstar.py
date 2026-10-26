"""Merge a G-STAR-SI checkpoint (training/thinker_gstar_model.py) into a full Qwen3-Omni copy.

W' = W + (alpha / rank) * b @ a for every adapted projection (as merge_thinker_sd.py); the 259 new tokens' input
and output rows (new_emb, new_head) are written into the unused rows NEW0.. of thinker.model.embed_tokens and
thinker.lm_head; the tokenizer gets the new tokens; the speaker-stream projector is saved beside the weights as
sd_proj.pt. Only the safetensors shards holding a changed tensor are rewritten; the release is never touched.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import NEW0, NEW_END, special_tokens

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
    keys = [k for k in meta.state_dict_metadata
            if k.endswith((".a", ".b")) or "sd_proj." in k or k.endswith(("new_emb", "new_head"))]
    wanted = {k: torch.empty(meta.state_dict_metadata[k].size, dtype=meta.state_dict_metadata[k].properties.dtype)
              for k in keys}
    dcp.load(state_dict=wanted, storage_reader=reader)
    scale = args.alpha / args.rank
    delta = {}
    for k in keys:
        if k.endswith(".a"):
            stem = k[:-2]
            delta[stem[stem.index("thinker."):] + ".weight"] = scale * (wanted[stem + ".b"].float() @ wanted[stem + ".a"].float())
    emb = next(v for k, v in wanted.items() if k.endswith("new_emb")).float()
    head = next(v for k, v in wanted.items() if k.endswith("new_head")).float()
    assert emb.shape[0] == NEW_END - NEW0 == head.shape[0], (emb.shape, head.shape)
    rows = {"thinker.model.embed_tokens.weight": emb, "thinker.lm_head.weight": head}
    sd_proj = {k[k.index("sd_proj.") + len("sd_proj."):]: v.float() for k, v in wanted.items() if "sd_proj." in k}
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    missing = [k for k in list(delta) + list(rows) if k not in index]
    if missing:
        raise SystemExit(f"{len(missing)} tensors not in the release, e.g. {missing[:3]}")
    out.mkdir(parents=True)
    shards = {index[k] for k in list(delta) + list(rows)}
    from safetensors.torch import load_file, save_file

    for f in base.iterdir():
        if f.is_file() and f.name not in shards:
            shutil.copy2(f, out / f.name)
    for shard in sorted(shards):
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
    # The tokenizer files stay the release's: readers add special_tokens() at load (pretrain_thinker_gstar.processor),
    # which gives them the same ids NEW0.. (importing AutoTokenizer here, with the cards hidden, fails in this env).
    (out / "gstar_tokens.json").write_text(json.dumps({"first_id": NEW0, "tokens": special_tokens()}), encoding="utf-8")
    print(json.dumps({"out": str(out), "adapted": len(delta), "new_rows": NEW_END - NEW0, "sd_proj": sorted(sd_proj),
                      "shards_rewritten": len(shards)}), flush=True)


if __name__ == "__main__":
    main()
