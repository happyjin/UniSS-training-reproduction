"""evaluation/online_decode.py with a crf/train_crf_sft.py checkpoint applied in memory (no merged copy on disk).

A train_crf_sft checkpoint moves three things of v8: the LoRA (lora.pt), the speaker projector and the 259 new token
rows (extra.pt). online_decode's own --lora folds only the first, so this wrapper loads --thinker (v8 merged) with the
unchanged Engine.from_merged and then applies the whole checkpoint exactly as crf/export_merged.py writes it (and as
experiments/uniss_qwen3_omni_v1/training/merge_thinker_gstar.py merges a Megatron run):
  W' = bf16(fp32(W) + (alpha / rank) b @ a)  for every adapted q/k/v/o of the 48 text layers,
  embed_tokens / lm_head rows NEW0..NEW_END = new_emb / new_head (bf16),
  sd_proj = the checkpoint's projector (the engine keeps it in bf16, as for v8).
Every other argument goes to online_decode unchanged (output format, sharding, refusal to overwrite, ...).

    python -m experiments.mist_w3po_v1.crf.decode_crf_ckpt --crf-ckpt <out>/ckpt/step_NNNNN \
        --thinker <v8 merged> --windows <jsonl> --out <new jsonl> --policy wait --context 0 --dir zh2en [--limit 2]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from experiments.mist_w3po_v1.crf.train_crf_sft import LAYER_RE
from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import ATTN, NEW0, NEW_END
from experiments.uniss_streaming_p2st_traj_v1.training import lora


@torch.no_grad()
def fold_into(thinker, lora_state: dict, rank: int, alpha: float) -> int:
    """W += (alpha / rank) b @ a in fp32, rounded once to the weight's dtype (merge_thinker_gstar.py's formula), for
    every key model.layers.N.self_attn.X_proj.{a,b} of lora_state (names relative to the thinker)."""
    mods = dict(thinker.named_modules())
    stems = sorted({k.rsplit(".", 1)[0] for k in lora_state})
    for stem in stems:
        if LAYER_RE.search(stem + ".") is None or stem.rsplit(".", 1)[1] not in ATTN:
            raise SystemExit(f"unexpected adapter {stem}")
        lin = mods[stem]
        lin = getattr(lin, "base", lin)
        W = lin.weight
        d = (alpha / rank) * (lora_state[stem + ".b"].float() @ lora_state[stem + ".a"].float())
        W.copy_((W.float() + d.to(W.device)).to(W.dtype))
    return len(stems)


@torch.no_grad()
def apply_ckpt(eng, ck: str | Path) -> dict:
    """Apply a train_crf_sft checkpoint to an engine.Engine loaded from v8 merged (in place)."""
    ck = Path(ck)
    ex = torch.load(ck / "extra.pt", map_location="cpu")
    st = torch.load(ck / "lora.pt", map_location="cpu")
    n = fold_into(eng.thinker, st, int(ex["lora_rank"]), float(ex["lora_alpha"]))
    emb = eng.thinker.get_input_embeddings().weight
    head = eng.thinker.lm_head.weight
    if emb.data_ptr() == head.data_ptr():
        raise SystemExit("tied embeddings: cannot write new_emb and new_head separately")
    emb[NEW0:NEW_END] = ex["new_emb"].to(emb.device, emb.dtype)
    head[NEW0:NEW_END] = ex["new_head"].to(head.device, head.dtype)
    eng.sd_proj.load_state_dict(ex["sd_proj"])
    if ex.get("extra_params"):
        print(f"WARNING: {len(ex['extra_params'])} hook modules in {ck} are NOT applied by this decoder "
              f"(scheme 2 needs its own inference switch)", flush=True)
    return {"ckpt": str(ck), "step": ex["step"], "objective": ex["objective"], "adapters_folded": n,
            "new_rows": NEW_END - NEW0, "base_at_train": ex.get("base")}


def main() -> None:
    argv = sys.argv[1:]
    if "--crf-ckpt" not in argv:
        raise SystemExit("--crf-ckpt <ckpt/step_N dir> is required")
    i = argv.index("--crf-ckpt")
    ck = Path(argv[i + 1])
    del argv[i: i + 2]
    if "--lora" in argv:
        raise SystemExit("--lora and --crf-ckpt are exclusive (the checkpoint already holds the LoRA)")
    for f in ("lora.pt", "extra.pt", "DONE"):
        if not (ck / f).exists():
            raise SystemExit(f"{ck / f} missing")
    from experiments.mist_w3po_v1.core import engine as E

    orig = E.Engine.from_merged.__func__

    def from_merged(cls, path, direction, max_gib=40):
        eng = orig(cls, path, direction, max_gib)
        info = apply_ckpt(eng, ck)
        print("crf ckpt applied " + json.dumps(info), flush=True)
        return eng

    E.Engine.from_merged = classmethod(from_merged)
    from experiments.mist_w3po_v1.evaluation import online_decode as OD

    sys.argv = [sys.argv[0]] + argv
    OD.main()


if __name__ == "__main__":
    main()
