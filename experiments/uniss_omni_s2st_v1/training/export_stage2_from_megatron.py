"""Pull the Stage 2 weights out of a Megatron torch_dist checkpoint.

Stage 2 trains three things: the Talker in full, the speaker-prefix
embedding, and a rank-32 LoRA on the Thinker's attention projections. The
Thinker's 4.7B frozen base is not among them -- it comes from the original
Qwen2.5-Omni checkpoint at evaluation time, so writing it here would cost
nine gigabytes a checkpoint to say nothing new.

Selecting the adapter is the only part that needs care. ``lora.LoRALinear``
registers its factors as ``a`` and ``b`` and keeps the base weight in a
``base`` submodule, so the adapter tensors are the ``thinker.*`` keys whose
last segment is ``a`` or ``b`` and whose parent is one of the four targeted
projections. A looser match -- every short-named tensor under ``thinker.``
-- would quietly sweep in whatever else happens to be named that way.

The optimiser-moment hazard from Stage 1 applies unchanged: the checkpoint
also holds ``optimizer.state.exp_avg.talker.*``, and a substring match
collapses it onto the weight of the same name. Every match below is
anchored at the start of the key, and the RMSNorm check at the end is the
backstop that caught it the first time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from experiments.uniss_omni_s2st_v1.training.stage2_tasks import THINKER_TARGETS


def is_adapter(key: str) -> bool:
    parts = key.split(".")
    return (
        parts[0] == "thinker"
        and len(parts) >= 3
        and parts[-1] in ("a", "b")
        and parts[-2] in THINKER_TARGETS
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True, help="a Megatron iter_XXXXXXX directory")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    source = Path(args.checkpoint_dir)
    if not (source / "metadata.json").is_file() and not any(source.glob("*.distcp")):
        raise SystemExit(f"{source} does not look like a torch_dist checkpoint")

    reader = dcp.FileSystemReader(str(source))
    metadata = reader.read_metadata()
    wanted = {
        key: torch.empty(
            value.size,
            dtype=value.properties.dtype if hasattr(value, "properties") else torch.bfloat16,
        )
        for key, value in metadata.state_dict_metadata.items()
        if key.startswith("talker.") or key.startswith("global_embed.") or is_adapter(key)
    }
    if not wanted:
        sample = list(metadata.state_dict_metadata)[:5]
        raise SystemExit(f"no stage 2 tensors in {source}; keys look like {sample}")

    dcp.load(state_dict=wanted, storage_reader=reader)

    talker: dict[str, torch.Tensor] = {}
    adapter: dict[str, torch.Tensor] = {}
    extras: dict[str, torch.Tensor] = {}
    for key, tensor in wanted.items():
        if key.startswith("global_embed."):
            extras[key] = tensor
        elif is_adapter(key):
            # lora.load_state_dict matches on the qualified module name it
            # stored as lora_name, which is relative to the Thinker.
            adapter[key[len("thinker.") :]] = tensor
        else:
            stripped = key[len("talker.") :]
            if stripped in talker:
                raise SystemExit(f"two checkpoint keys map to {stripped!r}; refusing to guess")
            talker[stripped] = tensor

    if not adapter:
        raise SystemExit("no LoRA tensors found; the Thinker would load unadapted")
    if len(adapter) % 2:
        raise SystemExit(f"{len(adapter)} adapter tensors is odd; every layer needs an a and a b")

    norm = talker.get("model.norm.weight")
    if norm is not None and float(norm.float().abs().mean()) < 0.01:
        raise SystemExit(
            f"model.norm.weight has mean |w| {float(norm.float().abs().mean()):.2e};"
            " that is an optimiser moment, not a trained norm"
        )
    # B starts at zero and only moves if the adapter trained. All-zero B is
    # a Thinker that is exactly the frozen one, which would read as a
    # believable but wrong Stage 2 result.
    b_moved = sum(
        1 for key, t in adapter.items() if key.endswith(".b") and float(t.float().abs().max()) > 0
    )
    if not b_moved:
        raise SystemExit("every LoRA B is still zero; this checkpoint did not train")

    iteration = int(source.name.split("_")[-1])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "talker": talker,
        "thinker_lora": adapter,
        "step": iteration,
        "source": str(source),
    }
    if extras:
        payload["global_embed"] = extras["global_embed.weight"]
    torch.save(payload, out)
    print(
        json.dumps(
            {
                "talker_tensors": len(talker),
                "adapter_layers": len(adapter) // 2,
                "adapter_b_nonzero": b_moved,
                "global_embed": bool(extras),
                "parameters_m": round(
                    sum(t.numel() for t in list(talker.values()) + list(adapter.values()) + list(extras.values()))
                    / 1e6,
                    1,
                ),
                "iteration": iteration,
                "output": str(out),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
