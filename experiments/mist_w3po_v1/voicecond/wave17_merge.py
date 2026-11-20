"""Robust wave 17 (reports/mist_w3po_v1/robust_wave17_enclora_prereg.zh-CN.md, implementation note 4): merge the
encoder LoRA of wave17_enc_train.py (adapters_final.pt, audio_tower-relative keys "layers.<i>.self_attn.<proj>.{a,b}")
into a full Qwen3-Omni copy that si_thinker_stream's vLLM serves (--thinker). Copy of wave5_m_merge.py with the key
prefix taken from adapters_final.json "module" (thinker.audio_tower): W' = W + (alpha / rank) * b @ a for the 128
q/k/v/out_proj weights of the 32 encoder layers. Files without an adapted weight are hard-linked from the release
(never modified); shards holding one are rewritten. Refuses to overwrite.
Self-check (after writing; on failure the script exits non-zero without printing the "adapted" line the orchestrator
looks for): every tensor outside the adapted keys is byte-identical to the release (uint8 views; same dtype / shape;
same key sets per shard); every other file is the same inode or byte-identical; each adapted tensor equals
(W.float() + delta).to(dtype) and differs from W.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave17_merge.py --adapters <run>/adapters_final.pt --out <dir>
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
N_ADAPTED = 32 * 4


def same_bytes(x: torch.Tensor, y: torch.Tensor) -> bool:
    if x.dtype != y.dtype or x.shape != y.shape:
        return False
    return torch.equal(x.contiguous().reshape(-1).view(torch.uint8), y.contiguous().reshape(-1).view(torch.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    meta = json.load(open(Path(a.adapters).with_suffix(".json")))
    prefix = meta["module"]
    assert prefix == "thinker.audio_tower", prefix
    scale = meta["alpha"] / meta["rank"]
    st = torch.load(a.adapters, map_location="cpu")
    delta = {}
    for k in st:
        if k.endswith(".a"):
            stem = k[:-2]
            delta[f"{prefix}.{stem}.weight"] = scale * (st[stem + ".b"].float() @ st[stem + ".a"].float())
    assert len(delta) == N_ADAPTED and len(st) == 2 * N_ADAPTED, (len(delta), len(st))
    assert all(k.startswith("thinker.audio_tower.layers.") and ".self_attn." in k for k in delta), list(delta)[:3]
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
    # ---- self-check: everything outside the encoder adapters is bit-identical to the release
    n_same, n_files = 0, 0
    for f in BASE.iterdir():
        if not f.is_file():
            continue
        g = out / f.name
        assert g.exists(), g
        if f.name in shards:
            b, m = load_file(str(f)), load_file(str(g))
            assert set(b) == set(m), f"{f.name}: key sets differ"
            for k in b:
                if k in delta:
                    want = (b[k].float() + delta[k]).to(b[k].dtype)
                    assert same_bytes(m[k], want), f"{k}: merged != W + delta"
                    assert not same_bytes(m[k], b[k]), f"{k}: unchanged by the merge"
                else:
                    assert same_bytes(m[k], b[k]), f"{k}: changed although not adapted"
                    n_same += 1
        else:
            assert os.path.samefile(f, g) or filecmp.cmp(f, g, shallow=False), f"{f.name} differs"
            n_files += 1
    n_out = sum(1 for k in index if k not in delta)
    norms = [float(d.norm()) for d in delta.values()]
    print(json.dumps({"out": str(out), "module": prefix, "adapted": n, "shards_rewritten": sorted(shards), "scale": scale,
                      "delta_norm_mean": round(sum(norms) / len(norms), 5),
                      "self_check": {"unchanged_tensors_in_rewritten_shards": n_same, "identical_other_files": n_files,
                                     "non_adapted_tensors_in_index": n_out, "ok": True}}), flush=True)


if __name__ == "__main__":
    main()
