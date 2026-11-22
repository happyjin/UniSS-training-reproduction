"""sst_hsmm_v1 P1a merge (copy of wave5_m_merge.py with BASE = wave17 merged_enc_wise05). Was: Robust wave 5, main line M: merge an arm's final LoRA adapters (wave5_m_train.py adapters_final.pt) into a full
Qwen3-Omni copy that si_thinker_stream's vLLM serves (--thinker). W' = W + (alpha / rank) * b @ a for the q/k/v/o
projection of every text layer (key thinker.model.<name>.weight). Files without an adapted weight are hard-linked from
the release (never modified); shards holding one are rewritten. Refuses to overwrite.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave5_m_merge.py --adapters <run>/adapters_final.pt --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05"   # sst_hsmm_v1: the UE Thinker is the base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    meta = json.load(open(Path(a.adapters).with_suffix(".json")))
    scale = meta["alpha"] / meta["rank"]
    st = torch.load(a.adapters, map_location="cpu")
    delta = {}
    for k in st:
        if k.endswith(".a"):
            stem = k[:-2]
            delta[f"thinker.model.{stem}.weight"] = scale * (st[stem + ".b"].float() @ st[stem + ".a"].float())
    index = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
    miss = [k for k in delta if k not in index]
    assert not miss, miss[:3]
    shards = {index[k] for k in delta}
    out.mkdir(parents=True)
    for f in BASE.iterdir():
        if f.is_file() and f.name not in shards:
            try:
                os.link(f, out / f.name)
            except OSError:
                shutil.copy2(f, out / f.name)
    n = 0
    for sh in sorted(shards):
        state = load_file(str(BASE / sh))
        for k, d in delta.items():
            if index[k] == sh:
                state[k] = (state[k].float() + d).to(state[k].dtype).contiguous()
                n += 1
        save_file(state, str(out / sh), metadata={"format": "pt"})
    norms = [float(d.norm()) for d in delta.values()]
    print(json.dumps({"out": str(out), "adapted": n, "shards_rewritten": len(shards), "scale": scale,
                      "delta_norm_mean": round(sum(norms) / len(norms), 5)}), flush=True)


if __name__ == "__main__":
    main()
