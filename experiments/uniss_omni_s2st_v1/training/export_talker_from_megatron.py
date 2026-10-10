"""Pull the Talker weights out of a Megatron torch_dist checkpoint.

Megatron saves sharded ``.distcp`` files, which the Stage 1 evaluation
scripts cannot read -- they expect a plain ``{"talker": state_dict}``. This
loads the distributed checkpoint onto CPU and writes that, so the acceptance
path is identical whether the run came from Megatron or from the earlier
DDP trainer.

The Talker and, when the run used one, the speaker-prefix embedding. The
Thinker was frozen for the whole run and the vocoder was deleted before
training started, so both come from the original Qwen2.5-Omni checkpoint at
evaluation time.

Missing the prefix embedding would not fail loudly: generation would simply
run without the conditioning it was trained with, and the audio would be
worse for a reason nothing reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


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
    # Megatron nests the model under "model"; the wrapper's own prefix is
    # "talker." because that is the attribute name on OmniTalkerWarmupModel.
    # Anchored at the start of the key, not matched anywhere in it. The
    # checkpoint also holds optimizer.state.exp_avg.talker.*,
    # exp_avg_sq.talker.* and fp32_param.talker.*, and a substring match
    # pulls all four in -- which then collapse onto one name when the
    # prefix is stripped, so the last one written wins. Adam's first moment
    # is near zero, and a Talker exported with exp_avg in place of
    # model.norm.weight predicts a constant.
    wanted = {
        key: torch.empty(
            value.size,
            dtype=value.properties.dtype
            if hasattr(value, "properties")
            else torch.bfloat16,
        )
        for key, value in metadata.state_dict_metadata.items()
        if key.startswith("talker.") or key.startswith("global_embed.")
    }
    if not wanted:
        sample = list(metadata.state_dict_metadata)[:5]
        raise SystemExit(f"no talker tensors in {source}; keys look like {sample}")

    dcp.load(state_dict=wanted, storage_reader=reader)

    talker: dict[str, torch.Tensor] = {}
    extras: dict[str, torch.Tensor] = {}
    for key, tensor in wanted.items():
        if key.startswith("global_embed."):
            extras[key] = tensor
            continue
        stripped = key[len("talker.") :]
        if stripped in talker:
            raise SystemExit(
                f"two checkpoint keys map to {stripped!r}; refusing to guess"
            )
        talker[stripped] = tensor
    iteration = int(source.name.split("_")[-1])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"talker": talker, "step": iteration, "source": str(source)}
    if extras:
        payload["global_embed"] = extras["global_embed.weight"]
    # A trained RMSNorm sits near 1. Near zero means an optimiser moment
    # was exported in its place, which produces a constant-output model
    # that nothing downstream would flag.
    norm = talker.get("model.norm.weight")
    if norm is not None and float(norm.float().abs().mean()) < 0.01:
        raise SystemExit(
            f"model.norm.weight has mean |w| {float(norm.float().abs().mean()):.2e};"
            " that is an optimiser moment, not a trained norm"
        )
    torch.save(payload, out)
    total = sum(t.numel() for t in talker.values()) + sum(
        t.numel() for t in extras.values()
    )
    print(
        json.dumps(
            {
                "tensors": len(talker),
                "global_embed": bool(extras),
                "parameters_m": round(total / 1e6, 1),
                "iteration": iteration,
                "output": str(out),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
