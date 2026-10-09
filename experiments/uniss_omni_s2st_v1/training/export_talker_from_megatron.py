"""Pull the Talker weights out of a Megatron torch_dist checkpoint.

Megatron saves sharded ``.distcp`` files, which the Stage 1 evaluation
scripts cannot read -- they expect a plain ``{"talker": state_dict}``. This
loads the distributed checkpoint onto CPU and writes that, so the acceptance
path is identical whether the run came from Megatron or from the earlier
DDP trainer.

Only the Talker is written. The Thinker was frozen for the whole run and the
vocoder was deleted before training started, so both come from the original
Qwen2.5-Omni checkpoint at evaluation time.
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
    wanted = {
        key: torch.empty(
            value.size, dtype=value.properties.dtype if hasattr(value, "properties") else torch.bfloat16
        )
        for key, value in metadata.state_dict_metadata.items()
        if ".talker." in key or key.startswith("talker.")
    }
    if not wanted:
        sample = list(metadata.state_dict_metadata)[:5]
        raise SystemExit(f"no talker tensors in {source}; keys look like {sample}")

    dcp.load(state_dict=wanted, storage_reader=reader)

    talker: dict[str, torch.Tensor] = {}
    for key, tensor in wanted.items():
        stripped = key.split("talker.", 1)[1]
        talker[stripped] = tensor
    iteration = int(source.name.split("_")[-1])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"talker": talker, "step": iteration, "source": str(source)}, out)
    total = sum(t.numel() for t in talker.values())
    print(
        json.dumps(
            {
                "tensors": len(talker),
                "parameters_m": round(total / 1e6, 1),
                "iteration": iteration,
                "output": str(out),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
