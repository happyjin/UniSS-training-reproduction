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
        if key.startswith(PREFIXES)
    }
    if not wanted:
        sample = list(metadata.state_dict_metadata)[:5]
        raise SystemExit(f"no phase D tensors in {source}; keys look like {sample}")
    dcp.load(state_dict=wanted, storage_reader=reader)

    talker: dict[str, torch.Tensor] = {}
    markers: dict[str, torch.Tensor] = {}
    global_embed = None
    for key, tensor in wanted.items():
        if key.startswith("global_embed."):
            global_embed = tensor
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
    norm = talker.get("model.norm.weight")
    if norm is not None and float(norm.float().abs().mean()) < 0.01:
        raise SystemExit(
            f"model.norm.weight has mean |w| {float(norm.float().abs().mean()):.2e};"
            " that is an optimiser moment, not a trained norm")

    iteration = int(source.name.split("_")[-1])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"talker": talker, "global_embed": global_embed,
                "markers": markers, "step": iteration, "source": str(source)}, out)
    print(json.dumps({
        "talker_tensors": len(talker), "markers": sorted(markers),
        "global_embed": list(global_embed.shape),
        "norm_mean_abs": round(float(norm.float().abs().mean()), 3) if norm is not None else None,
        "iteration": iteration, "output": str(out)}, indent=1))


if __name__ == "__main__":
    main()
