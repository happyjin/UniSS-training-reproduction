#!/usr/bin/env python3
"""Explain the ~25 ms cost of a single Qwen forward found by the Step 0 decomposition.

The decomposition shows that 97% of the streaming wall clock is Qwen, and that a one-token
forward costs the same at KV-cache length 0 and 4096. A flat cost curve means the loop is
not attention-bound, so the follow-up question is what the fixed cost actually is. This
script splits it into: batch scaling (launch-bound versus compute-bound), the expanded
180k-entry LM head, and the 48 unmerged LoRA adapters left in the inference graph.

Read-only with respect to the shared tree: the LoRA merge builds new modules on a private
copy of the loaded model inside this process.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TREE = ROOT / "experiments/uniss_streamspeech_ctc_v1"
for _path in (
    ROOT,
    TREE / "stage02_ctc_probe",
    TREE / "stage03_multitask_encoder",
    TREE / "stage03_multitask_encoder/ar_s2tt_v1",
    TREE / "stage04_b2_discrete_bridge",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from experiments.uniss_streamspeech_ctc_v1.stage08_emformer_phase3_joint.step2_qwen_lora_replay_v1.lora import (  # noqa: E402
    LoRALinear,
)
from experiments.uniss_streamspeech_ctc_v1.stage09_online_runtime.config import (  # noqa: E402
    Stage09Config,
)
from experiments.uniss_streamspeech_ctc_v1.stage09_online_runtime.model_loader import (  # noqa: E402
    load_stage09_bundle,
)
from experiments.uniss_streamspeech_ctc_v1.stage10_cached_micro_write.adapter import (  # noqa: E402
    apply_repetition_penalty,
)
from training import constants_uniss as c  # noqa: E402

SCHEMA_VERSION = "simul_s2st_route_v1_step0_qwen_forward_profile_v1"


def bench(call, *, repeats: int, warmup: int, device: torch.device) -> dict[str, float]:
    for _ in range(warmup):
        call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
    }


def merge_lora(model: nn.Module) -> int:
    """Fold every LoRA residual into its frozen base weight.

    ``y = Wx + scaling * B(A(x))`` is exactly ``(W + scaling * B @ A) x`` once dropout is
    disabled, so an inference-time merge is an algebraic identity, not an approximation.
    """

    merged = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if not isinstance(child, LoRALinear):
                continue
            base = child.base
            delta = child.lora_B.weight.float() @ child.lora_A.weight.float()
            weight = base.weight.float() + child.scaling * delta
            replacement = nn.Linear(
                base.in_features,
                base.out_features,
                bias=base.bias is not None,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
            replacement.weight.data.copy_(weight.to(base.weight.dtype))
            if base.bias is not None:
                replacement.bias.data.copy_(base.bias.data)
            replacement.requires_grad_(False)
            setattr(parent, name, replacement)
            merged += 1
    return merged


def vectorized_repetition_penalty(
    logits: torch.Tensor, token_ids, penalty: float
) -> torch.Tensor:
    """Same semantics as the Stage10 helper, with one gather/scatter instead of a Python loop."""

    if penalty == 1.0 or not token_ids:
        return logits
    unique = torch.tensor(
        sorted({int(value) for value in token_ids}), dtype=torch.long, device=logits.device
    )
    unique = unique[(unique >= 0) & (unique < logits.shape[-1])]
    if unique.numel() == 0:
        return logits
    output = logits.clone()
    values = output.index_select(-1, unique)
    updated = torch.where(values < 0, values * penalty, values / penalty)
    output.index_copy_(-1, unique, updated)
    return output


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--cache-length", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for output in (args.output_json, args.output_md):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite: {output}")

    bundle = load_stage09_bundle(Stage09Config(device=args.device))
    model = bundle.qwen
    device = bundle.device
    autocast = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    def build_cache(batch: int, length: int):
        if length <= 0:
            return None
        ids = torch.randint(0, 1000, (batch, length), dtype=torch.long, device=device)
        with torch.autocast(**autocast):
            return model(input_ids=ids, use_cache=True).past_key_values

    def forward_full(batch: int):
        cache = build_cache(batch, args.cache_length)
        ids = torch.randint(0, 1000, (batch, 1), dtype=torch.long, device=device)

        def call():
            with torch.autocast(**autocast):
                model(input_ids=ids, past_key_values=cache, use_cache=True)

        return call

    lora_modules = sum(isinstance(module, LoRALinear) for module in model.modules())
    batch_sizes = (1, 2, 4, 8, 16, 32)

    with_lora = {
        str(batch): bench(
            forward_full(batch), repeats=args.repeats, warmup=args.warmup, device=device
        )
        for batch in batch_sizes
    }

    # Component split at batch 1: backbone alone, then the expanded-vocabulary LM head alone.
    cache = build_cache(1, args.cache_length)
    ids = torch.randint(0, 1000, (1, 1), dtype=torch.long, device=device)

    def backbone_call():
        with torch.autocast(**autocast):
            model.model(input_ids=ids, past_key_values=cache, use_cache=True)

    hidden = torch.zeros(
        1, 1, model.config.hidden_size, device=device, dtype=torch.bfloat16
    )

    def head_call():
        with torch.autocast(**autocast):
            model.lm_head(hidden)

    components = {
        "backbone_only": bench(
            backbone_call, repeats=args.repeats, warmup=args.warmup, device=device
        ),
        "lm_head_only": bench(head_call, repeats=args.repeats, warmup=args.warmup, device=device),
    }

    # Numerical check plus timing for the merged-LoRA variant.
    probe_ids = torch.randint(0, 1000, (1, 8), dtype=torch.long, device=device)
    with torch.autocast(**autocast):
        reference = model(input_ids=probe_ids, use_cache=False).logits.float()
    merged_count = merge_lora(model)
    with torch.autocast(**autocast):
        after = model(input_ids=probe_ids, use_cache=False).logits.float()
    merge_error = {
        "merged_modules": merged_count,
        "max_abs_logit_delta": float((after - reference).abs().max()),
        "reference_logit_abs_max": float(reference.abs().max()),
        "argmax_agreement": float((after.argmax(-1) == reference.argmax(-1)).float().mean()),
    }
    without_lora = {
        str(batch): bench(
            forward_full(batch), repeats=args.repeats, warmup=args.warmup, device=device
        )
        for batch in batch_sizes
    }

    # Repetition penalty: the Stage10 helper is O(unique tokens) CUDA launches.
    penalty_rows = []
    logits = torch.randn(1, c.VOCAB_SIZE, device=device)
    for history in (16, 64, 256, 512):
        tokens = list(range(history))
        expected = apply_repetition_penalty(logits, tokens, 1.1)
        observed = vectorized_repetition_penalty(logits, tokens, 1.1)
        penalty_rows.append(
            {
                "history_tokens": history,
                "loop_ms": bench(
                    lambda: apply_repetition_penalty(logits, tokens, 1.1),
                    repeats=args.repeats,
                    warmup=args.warmup,
                    device=device,
                )["mean_ms"],
                "vectorized_ms": bench(
                    lambda: vectorized_repetition_penalty(logits, tokens, 1.1),
                    repeats=args.repeats,
                    warmup=args.warmup,
                    device=device,
                )["mean_ms"],
                "max_abs_delta": float((observed - expected).abs().max()),
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "research_only": True,
        "run_name": args.run_name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "config": {
            "cache_length": args.cache_length,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "hidden_size": int(model.config.hidden_size),
            "num_hidden_layers": int(model.config.num_hidden_layers),
            "vocab_size": int(model.config.vocab_size),
            "lora_modules": lora_modules,
        },
        "batch_scaling_with_lora": with_lora,
        "batch_scaling_merged_lora": without_lora,
        "components_batch1": components,
        "lora_merge": merge_error,
        "repetition_penalty": penalty_rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Step 0b — what the 25 ms Qwen forward is made of",
        "",
        f"> Run `{payload['run_name']}` · {payload['generated_at']} · {payload['gpu']} · research only.",
        "",
        f"Model: {payload['config']['num_hidden_layers']} layers, hidden "
        f"{payload['config']['hidden_size']}, vocab {payload['config']['vocab_size']}, "
        f"{payload['config']['lora_modules']} unmerged LoRA adapters. "
        f"KV cache held at {args.cache_length} positions.",
        "",
        "## 1. Batch scaling of a one-token forward",
        "",
        "A cost that barely moves from batch 1 to batch 32 is dominated by fixed per-call",
        "overhead rather than by arithmetic.",
        "",
        "| Batch | With LoRA adapters (ms) | With LoRA merged (ms) | Merged speed-up | Per-sequence merged (ms) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for batch in batch_sizes:
        left = with_lora[str(batch)]["mean_ms"]
        right = without_lora[str(batch)]["mean_ms"]
        lines.append(
            f"| {batch} | {left:.2f} | {right:.2f} | {left / right:.2f}x | {right / batch:.2f} |"
        )
    lines += [
        "",
        "## 2. Where one forward goes at batch 1",
        "",
        "| Component | Mean ms | Median ms |",
        "|---|---:|---:|",
    ]
    for name, block in components.items():
        lines.append(f"| `{name}` | {block['mean_ms']:.2f} | {block['median_ms']:.2f} |")
    lines += [
        "",
        "## 3. LoRA merge is an identity, not an approximation",
        "",
        f"- merged modules: {merge_error['merged_modules']}",
        f"- max absolute logit change: {merge_error['max_abs_logit_delta']:.4g} "
        f"(logit magnitude up to {merge_error['reference_logit_abs_max']:.4g})",
        f"- argmax agreement: {merge_error['argmax_agreement'] * 100:.2f}%",
        "",
        "## 4. Repetition penalty over the expanded vocabulary",
        "",
        "The Stage10 helper issues three CUDA operations per distinct generated token.",
        "",
        "| History tokens | Python loop (ms) | Vectorised (ms) | Speed-up | Max abs delta |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in penalty_rows:
        speedup = row["loop_ms"] / max(row["vectorized_ms"], 1e-9)
        lines.append(
            f"| {row['history_tokens']} | {row['loop_ms']:.2f} | {row['vectorized_ms']:.2f} | "
            f"{speedup:.1f}x | {row['max_abs_delta']:.3g} |"
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "stage": "done",
                "batch1_with_lora_ms": with_lora["1"]["mean_ms"],
                "batch1_merged_ms": without_lora["1"]["mean_ms"],
                "batch32_merged_ms": without_lora["32"]["mean_ms"],
                "lm_head_ms": components["lm_head_only"]["mean_ms"],
                "backbone_ms": components["backbone_only"]["mean_ms"],
                "lora_merge_argmax_agreement": merge_error["argmax_agreement"],
                "report": str(args.output_md),
            }
        )
    )


if __name__ == "__main__":
    main()
