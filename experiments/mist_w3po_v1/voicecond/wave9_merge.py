"""Robust wave 9: merge an arm's final adapters (wave9_train.py adapters_final.pt) into a full Qwen3-Omni copy for
si_thinker_stream's vLLM (--thinker). A copy of wave5_m_merge.py plus the anchor rows:
  LoRA:   W' = W + (alpha / rank) * b @ a for q/k/v/o of every text layer (alpha from the json next to the adapters; the
          WiSE-0.5 run dirs halve it: W0 + 0.5 (W_arm - W0), as wave 5's run_erm_wise05);
  anchor: (arm A) rows ANCHOR_LO .. ANCHOR_LO + 255 of thinker.model.embed_tokens.weight := E0 + w (E_A - E0), E0 = the
          training init (mean embedding row), w = json "anchor_wise" (0.5 for WiSE). thinker.lm_head.weight is a separate
          tensor and is left untouched (asserted bit-identical).
Files without a changed tensor are hard-linked from the release (never modified); shards holding one are rewritten.
Refuses to overwrite.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave9_merge.py --adapters <run>/adapters_final.pt --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import wave9_common as C  # noqa: E402

BASE = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
EMB, HEAD = "thinker.model.embed_tokens.weight", "thinker.lm_head.weight"


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
    rows = None
    if "anchor.E" in st:
        w = float(meta["anchor_wise"])
        E0, EA = st["anchor.E0"].float(), st["anchor.E"].float()
        assert E0.shape == EA.shape == (C.ANCHOR_N, E0.shape[1])
        rows = E0 + w * (EA - E0)
    else:
        assert not meta.get("anchor"), "anchor arm without anchor rows"
    index = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
    miss = [k for k in delta if k not in index]
    assert not miss, miss[:3]
    shards = {index[k] for k in delta} | ({index[EMB]} if rows is not None else set())
    out.mkdir(parents=True)
    for f in BASE.iterdir():
        if f.is_file() and f.name not in shards:
            try:
                os.link(f, out / f.name)
            except OSError:
                shutil.copy2(f, out / f.name)
    n, info = 0, {}
    for sh in sorted(shards):
        state = load_file(str(BASE / sh))
        head0 = state[HEAD].clone() if HEAD in state else None
        for k, d in delta.items():
            if index[k] == sh:
                state[k] = (state[k].float() + d).to(state[k].dtype).contiguous()
                n += 1
        if rows is not None and index[EMB] == sh:
            Wt = state[EMB]
            assert Wt.shape[0] == C.VOCAB and Wt.shape[1] == rows.shape[1]
            old = Wt[C.ANCHOR_LO:].float().clone()
            Wt = Wt.clone()
            Wt[C.ANCHOR_LO:] = rows.to(Wt.dtype)
            state[EMB] = Wt.contiguous()
            info = {"anchor_rows": int(rows.shape[0]), "anchor_wise": float(meta["anchor_wise"]),
                    "anchor_row_norm_mean": round(float(rows.norm(dim=1).mean()), 5),
                    "replaced_row_norm_mean": round(float(old.norm(dim=1).mean()), 5),
                    "anchor_moved_from_init": round(float((rows - st["anchor.E0"].float()).norm(dim=1).mean()), 5)}
        if head0 is not None:
            assert torch.equal(head0, state[HEAD]), "lm_head changed"
        save_file(state, str(out / sh), metadata={"format": "pt"})
    norms = [float(d.norm()) for d in delta.values()]
    print(json.dumps({"out": str(out), "adapted": n, "shards_rewritten": len(shards), "scale": scale,
                      "delta_norm_mean": round(sum(norms) / len(norms), 5), **info}), flush=True)


if __name__ == "__main__":
    main()
