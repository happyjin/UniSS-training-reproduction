"""Merge a saved adapter into a fresh copy of the base and write an HF model.

Kept separate from training so a mid-run snapshot can be exported and measured
without re-running anything, and so the exported model is an ordinary dense
checkpoint that the existing evaluation chain loads with no new code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.uniss_streaming_p2st_traj_v1.training import lora


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    blob = torch.load(args.adapter, map_location="cpu", weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    layers = lora.inject(model, rank=blob["rank"], alpha=blob["alpha"])
    lora.load_state_dict(layers, blob["state"])
    lora.merge(layers)

    # Unwrap: put each base Linear back where its adapter sat, so the saved
    # model has exactly the architecture the evaluator expects.
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, lora.LoRALinear):
                setattr(parent, name, child.base)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True).save_pretrained(out)
    print(f"step {blob['step']} -> {out}")


if __name__ == "__main__":
    main()
