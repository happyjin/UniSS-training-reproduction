"""Load the frozen Phase3 HF base plus the selected streaming LoRA adapter."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_phase3_prefix_streaming_full198_v1.lora import inject_lora


def load_model_and_tokenizer(
    adapter_dir: Path,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
):
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    base = Path(config["base_model"])
    dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
    )
    injection = inject_lora(
        model,
        target_modules=config["target_modules"],
        rank=int(config["rank"]),
        alpha=float(config["alpha"]),
        dropout=0.0,
    )
    adapter = load_file(adapter_dir / config["weights"]["file"])
    missing, unexpected = model.load_state_dict(adapter, strict=False)
    unexpected = [key for key in unexpected if key in adapter]
    missing_adapter = [
        name
        for name, _ in model.named_parameters()
        if (".lora_A." in name or ".lora_B." in name) and name not in adapter
    ]
    if unexpected or missing_adapter:
        raise ValueError(
            f"adapter load mismatch: unexpected={unexpected}, missing_adapter={missing_adapter}"
        )
    # LoRAInjection intentionally creates float32 trainable matrices for
    # training stability.  In inference they must match the frozen base dtype;
    # `generate()` does not keep every cached decode step inside autocast.
    model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, config, injection
