"""Pull Phase D's trained weights out of a Megatron torch_dist checkpoint.

Three things train here and none of them is the Thinker: the Talker, the
speaker-prefix embedding, and the two vectors that end the text stream.
Qwen3-Omni's 31.7B comes from its own checkpoint at evaluation time, so
writing it alongside would cost sixty gigabytes a save to repeat what is
already on disk.

The traps are stage 1's, unchanged. Keys are matched anchored at the
start, because the checkpoint also holds optimizer.state.exp_avg.talker.*
and a substring match collapses all of them onto one name -- Adam's first
moment exported as model.norm.weight produces a Talker that predicts a
constant and nothing downstream says so. The RMSNorm check at the end is
the backstop that caught it the first time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

PREFIXES = ("talker.", "global_embed.", "markers.")


def _is_adapter(key: str) -> bool:
    """A phase F LoRA tensor. Phase D and E have none and this stays empty.

    Without the adapter a phase F probe would run the frozen Thinker and
    score a model that was never trained -- believable, and wrong.
    """
    parts = key.split(".")
    return (parts[0] == "thinker" and len(parts) >= 3
            and parts[-1] in ("a", "b")
            and parts[-2] in ("q_proj", "k_proj", "v_proj", "o_proj"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True)
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
        if key.startswith(PREFIXES) or _is_adapter(key)
    }
    if not wanted:
        sample = list(metadata.state_dict_metadata)[:5]
        raise SystemExit(f"no phase D tensors in {source}; keys look like {sample}")
    dcp.load(state_dict=wanted, storage_reader=reader)

    talker: dict[str, torch.Tensor] = {}
    markers: dict[str, torch.Tensor] = {}
    adapter: dict[str, torch.Tensor] = {}
    global_embed = None
    for key, tensor in wanted.items():
        if key.startswith("global_embed."):
            global_embed = tensor
        elif _is_adapter(key):
            # lora.load_state_dict matches on lora_name, which is relative
            # to the Thinker.
            adapter[key[len("thinker."):]] = tensor
        elif key.startswith("markers."):
            markers[key[len("markers."):]] = tensor
        else:
            stripped = key[len("talker."):]
            if stripped in talker:
                raise SystemExit(f"two checkpoint keys map to {stripped!r}; refusing to guess")
            talker[stripped] = tensor

    if global_embed is None:
        raise SystemExit("no global_embed; the speaker prefix would be random")
    for name in ("eos", "pad"):
        if name not in markers:
            raise SystemExit(f"markers.{name} is missing")
    # The eval side has to build the same module before loading, and a
    # strict=False load would otherwise drop the MLP in silence.
    proj_variant = ("mlp" if any(k.startswith("thinker_to_talker_proj.base")
                                 for k in talker) else "linear")
    norm = talker.get("model.norm.weight")
    if norm is not None and float(norm.float().abs().mean()) < 0.01:
        raise SystemExit(
            f"model.norm.weight has mean |w| {float(norm.float().abs().mean()):.2e};"
            " that is an optimiser moment, not a trained norm")

    iteration = int(source.name.split("_")[-1])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if adapter:
        # B starts at zero; an all-zero B is the frozen Thinker wearing a
        # trained model's name.
        if not any(k.endswith(".b") and float(t.float().abs().max()) > 0
                   for k, t in adapter.items()):
            raise SystemExit("every LoRA B is still zero; this checkpoint did not train")
        if len(adapter) % 2:
            raise SystemExit(f"{len(adapter)} adapter tensors is odd; each layer needs a and b")
    torch.save({"talker": talker, "proj_variant": proj_variant,
                "thinker_lora": adapter, "global_embed": global_embed,
                "markers": markers, "step": iteration, "source": str(source)}, out)
    print(json.dumps({
        "talker_tensors": len(talker), "markers": sorted(markers),
        "global_embed": list(global_embed.shape),
        "norm_mean_abs": round(float(norm.float().abs().mean()), 3) if norm is not None else None,
        "iteration": iteration, "output": str(out)}, indent=1))


if __name__ == "__main__":
    main()
